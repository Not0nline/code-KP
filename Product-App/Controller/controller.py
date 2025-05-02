import math
import requests
import time
import os
import logging
import sys
from kubernetes import client, config
from prometheus_client import start_http_server, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Prometheus metrics
cpu_utilization_metric = Gauge("cpu_utilization", "CPU Utilization")
request_rate_metric = Gauge("request_rate", "Request Rate")
scaling_events_metric = Gauge("scaling_events", "Scaling Events")

# Load configuration from environment variables
PREDICTIVE_SCALER_SERVICE = os.getenv("PREDICTIVE_SCALER_SERVICE", "http://predictive-scaler:5000/predict")
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "product-app")
TARGET_NAMESPACE = os.getenv("TARGET_NAMESPACE", "default")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
SCALING_COOLDOWN = int(os.getenv("SCALING_COOLDOWN", "120"))
SCALING_THRESHOLD = float(os.getenv("SCALING_THRESHOLD", "0.7"))
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
PROMETHEUS_SERVER = os.getenv("PROMETHEUS_SERVER", "http://prometheus:9090")

# Initialize Kubernetes client
config.load_incluster_config()
apps_v1 = client.AppsV1Api()

# Track last scaling time and current replicas
last_scaling_time = 0
current_replicas = 0

# Initialize cpu utilization history
cpu_utilization_history = []

def get_scaling_recommendation():
    try:
        response = requests.get(PREDICTIVE_SCALER_SERVICE)
        if response.status_code == 200:
            data = response.json()
            return data["recommended_replicas"]
        else:
            logger.error(f"Error fetching prediction: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with predictive scaler: {e}")
        return None
    
def determine_replica_count(self, current_replicas, cpu_utilization, request_rate, predicted_requests):
    # Define the CPU utilization threshold for scaling up
    cpu_threshold = 80  # Adjust this value based on your requirements

    # Define the scaling factor based on CPU utilization
    if cpu_utilization >= cpu_threshold:
        cpu_scaling_factor = cpu_utilization / cpu_threshold
    else:
        cpu_scaling_factor = 1.0

    # Calculate the replica count based on the current request rate and CPU utilization
    request_rate_replicas = math.ceil(request_rate / self.target_requests_per_replica)
    cpu_replicas = math.ceil(cpu_utilization / self.target_cpu_utilization)

    # Use the maximum of the request rate and CPU-based replica counts
    reactive_replicas = max(request_rate_replicas, cpu_replicas)

    # Apply the CPU scaling factor to the reactive replicas
    scaled_replicas = math.ceil(reactive_replicas * cpu_scaling_factor)

    # Compare the scaled replicas with the predicted replicas
    desired_replicas = max(scaled_replicas, predicted_requests)

    # Ensure the desired replicas fall within the min and max limits
    desired_replicas = min(max(desired_replicas, self.min_replicas), self.max_replicas)

    # Apply the cooldown period to prevent rapid scaling
    if self.cooldown_timestamp is not None and time.time() < self.cooldown_timestamp:
        desired_replicas = current_replicas

    return desired_replicas

def scale_deployment(cpu_utilization, request_rate, predicted_replicas):
    global last_scaling_time, current_replicas

    desired_replicas = determine_replica_count(current_replicas, cpu_utilization, request_rate, predicted_replicas)

    current_time = time.time()
    if current_time - last_scaling_time < SCALING_COOLDOWN:
        logger.info(f"Scaling cooldown in effect. Skipping scaling.")
        return

    try:
        apps_v1.patch_namespaced_deployment_scale(
            name=TARGET_DEPLOYMENT,
            namespace=TARGET_NAMESPACE,
            body={"spec": {"replicas": desired_replicas}}
        )
        last_scaling_time = current_time
        current_replicas = desired_replicas
        logger.info(f"Scaled deployment to {desired_replicas} replicas")
        scaling_events_metric.inc()
    except client.rest.ApiException as e:
        logger.error(f"Error scaling deployment: {e}")

def monitor_cpu_utilization():
    while True:
        try:
            cpu_utilization = get_cpu_utilization()
            if cpu_utilization is not None:
                cpu_utilization_history.append(cpu_utilization)
                cpu_utilization_metric.set(cpu_utilization)

                # Keep the history within the desired length
                if len(cpu_utilization_history) > 60:
                    cpu_utilization_history.pop(0)

            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Error monitoring CPU utilization: {e}")

def get_cpu_utilization():
    try:
        response = requests.get(f"{PROMETHEUS_SERVER}/api/v1/query", params={
            "query": f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{TARGET_DEPLOYMENT}-.*", container!="POD"}}[1m]))'
        })
        if response.status_code == 200:
            data = response.json()
            cpu_utilization = float(data["data"]["result"][0]["value"][1])
            return cpu_utilization
        else:
            logger.error(f"Error fetching CPU utilization: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Prometheus: {e}")
        return None

def monitor_request_rate():
    while True:
        try:
            request_rate = get_request_rate()
            if request_rate is not None:
                request_rate_metric.set(request_rate)
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error(f"Error monitoring request rate: {e}")

def get_request_rate():
    try:
        response = requests.get(f"{PROMETHEUS_SERVER}/api/v1/query", params={
            "query": f'sum(rate(http_requests_total{{pod=~"{TARGET_DEPLOYMENT}-.*"}}[1m]))'
        })
        if response.status_code == 200:
            data = response.json()
            request_rate = float(data["data"]["result"][0]["value"][1])
            return request_rate
        else:
            logger.error(f"Error fetching request rate: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Prometheus: {e}")
        return None

def main():
    # Start the Prometheus HTTP server
    start_http_server(8000)

    while True:
        replicas = get_scaling_recommendation()
        if replicas is not None:
            scale_deployment(replicas)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    # Start CPU utilization and request rate monitoring in separate threads
    import threading
    threading.Thread(target=monitor_cpu_utilization).start()
    threading.Thread(target=monitor_request_rate).start()

    main()