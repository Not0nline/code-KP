import math
import requests
import time
import os
import logging
import sys
from kubernetes import client, config
from prometheus_client import start_http_server, Gauge
import json

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Prometheus metrics
cpu_utilization_metric = Gauge("cpu_utilization", "CPU Utilization")
request_rate_metric = Gauge("request_rate", "Request Rate")
scaling_events_metric = Gauge("scaling_events", "Scaling Events")
current_replicas_metric = Gauge("current_replicas", "Current number of replicas")

# Load configuration from environment variables
PREDICTIVE_SCALER_SERVICE = os.getenv("PREDICTIVE_SCALER_SERVICE", "http://predictive-scaler:5000")
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "product-app")
TARGET_NAMESPACE = os.getenv("TARGET_NAMESPACE", "default")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
SCALING_COOLDOWN = int(os.getenv("SCALING_COOLDOWN", "120"))
SCALING_THRESHOLD = float(os.getenv("SCALING_THRESHOLD", "0.7"))
CPU_SCALE_UP_THRESHOLD = float(os.getenv("CPU_SCALE_UP_THRESHOLD", "70"))
CPU_SCALE_DOWN_THRESHOLD = float(os.getenv("CPU_SCALE_DOWN_THRESHOLD", "30"))
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
PROMETHEUS_SERVER = os.getenv("PROMETHEUS_SERVER", "http://prometheus:9090")
TARGET_CPU_UTILIZATION = float(os.getenv("TARGET_CPU_UTILIZATION", "0.7"))
TARGET_REQUESTS_PER_REPLICA = int(os.getenv("TARGET_REQUESTS_PER_REPLICA", "100"))
PREDICTIVE_ENDPOINT = os.getenv("PREDICTIVE_ENDPOINT", "/predict_combined")
PREDICTIVE_TARGET_DEPLOYMENT = os.getenv("PREDICTIVE_TARGET_DEPLOYMENT", TARGET_DEPLOYMENT)

# Initialize Kubernetes client
try:
    config.load_incluster_config()
except:
    config.load_kube_config()  # For local testing
    
apps_v1 = client.AppsV1Api()

# Track last scaling time and current replicas
last_scaling_time = 0
current_replicas = 0

def get_current_replicas():
    """Get current number of replicas for the target deployment."""
    try:
        deployment = apps_v1.read_namespaced_deployment(
            name=TARGET_DEPLOYMENT,
            namespace=TARGET_NAMESPACE
        )
        return deployment.spec.replicas
    except client.rest.ApiException as e:
        logger.error(f"Error getting current replicas: {e}")
        return MIN_REPLICAS

def get_scaling_recommendation():
    """Get scaling recommendation from predictive scaler."""
    try:
        predictive_url = f"{PREDICTIVE_SCALER_SERVICE.rstrip('/')}{PREDICTIVE_ENDPOINT}"
        response = requests.get(predictive_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            reported_target = data.get("target_deployment")
            if reported_target and reported_target != PREDICTIVE_TARGET_DEPLOYMENT:
                logger.warning(
                    "Ignoring predictive recommendation for deployment '%s'; controller targets '%s'",
                    reported_target,
                    PREDICTIVE_TARGET_DEPLOYMENT,
                )
                return None
            if not reported_target:
                logger.debug("Predictive scaler response missing target_deployment; assuming compatibility")
            logger.info(f"Received prediction: {data}")
            return data
        else:
            logger.error(f"Error fetching prediction: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with predictive scaler: {e}")
        return None

def determine_replica_count(current_replicas, cpu_utilization, request_rate, prediction_data):
    """Determine the desired number of replicas based on current metrics and predictions."""
    
    # Get predicted replicas from the predictive scaler
    predicted_replicas = prediction_data.get("recommended_replicas", current_replicas) if prediction_data else current_replicas
    
    # Calculate reactive scaling based on current metrics
    # CPU hysteresis: scale up aggressively above 70%, down conservatively below 30%
    cpu_replicas = current_replicas
    if cpu_utilization >= CPU_SCALE_UP_THRESHOLD:
        # scale proportionally above threshold
        factor = cpu_utilization / max(CPU_SCALE_UP_THRESHOLD, 1)
        cpu_replicas = max(current_replicas + 1, math.ceil(current_replicas * factor))
    elif cpu_utilization <= CPU_SCALE_DOWN_THRESHOLD:
        # scale down one step, but not below min and not below proportional target
        proportional = max(1, math.floor(current_replicas * (cpu_utilization / max(CPU_SCALE_DOWN_THRESHOLD, 1))))
        cpu_replicas = min(current_replicas - 1, proportional)
    
    # Use request rate if available
    request_replicas = math.ceil(request_rate / TARGET_REQUESTS_PER_REPLICA) if request_rate > 0 else current_replicas
    
    # Take the maximum of reactive and predictive scaling
    desired_replicas = max(predicted_replicas, cpu_replicas, request_replicas)
    
    # Apply min/max constraints
    desired_replicas = min(max(desired_replicas, MIN_REPLICAS), MAX_REPLICAS)
    
    logger.info(f"Scaling decision: current={current_replicas}, predicted={predicted_replicas}, "
                f"cpu_based={cpu_replicas}, request_based={request_replicas}, final={desired_replicas}")
    
    return desired_replicas

def scale_deployment(desired_replicas):
    """Scale the deployment to the desired number of replicas."""
    global last_scaling_time, current_replicas
    
    current_time = time.time()
    
    # Check cooldown period
    if current_time - last_scaling_time < SCALING_COOLDOWN:
        logger.info(f"Scaling cooldown in effect. Time remaining: {SCALING_COOLDOWN - (current_time - last_scaling_time):.0f}s")
        return False
    
    # Check if scaling is needed
    if desired_replicas == current_replicas:
        logger.info(f"No scaling needed. Current replicas: {current_replicas}")
        return False
    
    try:
        # For k3s compatibility, patch the deployment directly
        body = {"spec": {"replicas": desired_replicas}}
        apps_v1.patch_namespaced_deployment(
            name=TARGET_DEPLOYMENT,
            namespace=TARGET_NAMESPACE,
            body=body
        )
        
        logger.info(f"Successfully scaled deployment from {current_replicas} to {desired_replicas} replicas")
        last_scaling_time = current_time
        current_replicas = desired_replicas
        scaling_events_metric.inc()
        current_replicas_metric.set(desired_replicas)
        return True
        
    except client.rest.ApiException as e:
        logger.error(f"Error scaling deployment: {e}")
        return False

def get_cpu_utilization():
    """Get CPU utilization from Prometheus."""
    try:
        # Query for average CPU utilization across all pods
        query = f'avg(rate(container_cpu_usage_seconds_total{{pod=~"{TARGET_DEPLOYMENT}-.*", container!="POD"}}[1m])) * 100'
        
        response = requests.get(f"{PROMETHEUS_SERVER}/api/v1/query", 
                              params={"query": query},
                              timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data["data"]["result"]:
                cpu_utilization = float(data["data"]["result"][0]["value"][1])
                return cpu_utilization
            else:
                logger.warning("No CPU utilization data available")
                return 0.0
        else:
            logger.error(f"Error fetching CPU utilization: {response.text}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error communicating with Prometheus: {e}")
        return 0.0

def get_request_rate():
    """Get per-second request rate from Prometheus."""
    try:
        # Query for request rate
        query = f'sum(rate(app_http_requests_total{{app="{TARGET_DEPLOYMENT}"}}[1m]))'
        
        response = requests.get(f"{PROMETHEUS_SERVER}/api/v1/query", 
                              params={"query": query},
                              timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data["data"]["result"]:
                request_rate = float(data["data"]["result"][0]["value"][1])
                return request_rate
            else:
                logger.warning("No request rate data available")
                return 0.0
        else:
            logger.error(f"Error fetching request rate: {response.text}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error communicating with Prometheus: {e}")
        return 0.0

def main_loop():
    """Main control loop."""
    global current_replicas
    
    # Get initial replica count
    current_replicas = get_current_replicas()
    current_replicas_metric.set(current_replicas)
    logger.info(f"Starting controller. Initial replicas: {current_replicas}")
    
    while True:
        try:
            # Get current metrics
            cpu_utilization = get_cpu_utilization()
            request_rate = get_request_rate()
            
            # Update Prometheus metrics
            cpu_utilization_metric.set(cpu_utilization)
            request_rate_metric.set(request_rate)
            
            # Get current replica count (in case it was changed externally)
            current_replicas = get_current_replicas()
            
            # Get prediction from predictive scaler
            prediction_data = get_scaling_recommendation()
            
            # Determine desired replicas
            desired_replicas = determine_replica_count(
                current_replicas, 
                cpu_utilization, 
                request_rate, 
                prediction_data
            )
            
            # Scale if needed
            scale_deployment(desired_replicas)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        # Wait before next iteration
        time.sleep(POLL_INTERVAL)

def main():
    """Main entry point."""
    # Start the Prometheus HTTP server
    start_http_server(8000)
    logger.info("Started Prometheus metrics server on port 8000")
    
    # Run the main control loop
    main_loop()

if __name__ == "__main__":
    main()