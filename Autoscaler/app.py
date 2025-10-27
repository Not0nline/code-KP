from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import time
import threading
import json
import socket
import struct
from datetime import datetime, timedelta
import logging
import joblib
import traceback
from collections import deque
from prometheus_client import Counter, Gauge, Summary, Histogram
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_api_client import PrometheusConnect
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout
import heapq
import subprocess
import warnings
try:
    from kubernetes import client as k8s_client, config as k8s_config
    K8S_AVAILABLE = True
except Exception:
    K8S_AVAILABLE = False
warnings.filterwarnings('ignore')


def _env_float(name, default):
    """Safely read a float value from the environment."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

# Flask app initialization
app = Flask(__name__)
metrics = PrometheusMetrics(app)  # Exposes /metrics endpoint

# Define Prometheus metrics
SERVICE_LABEL = os.getenv('APP_NAME', 'predictive-scaler')

# Ensure a consistent 'service' label on custom metrics
def _labels_with_service(**labels):
        labels = dict(labels)
        labels['service'] = SERVICE_LABEL
        return labels

prediction_requests = Counter('prediction_requests_total', 'Number of prediction requests', 
                                                         labelnames=['service','method'])
scaling_decisions = Counter('scaling_decisions_total', 'Number of scaling decisions', 
                                                    labelnames=['service','decision'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency in seconds',
                                                         labelnames=['service','method'])
cpu_utilization = Gauge('current_cpu_utilization', 'Current CPU utilization')
traffic_gauge = Gauge('current_traffic', 'Current traffic level')
recommended_replicas = Gauge('recommended_replicas', 'Number of recommended replicas')
prediction_accuracy = Summary('prediction_accuracy', 'Accuracy of predictions vs actual',
                                                        labelnames=['service','method'])
http_requests = Counter('http_requests_total', 'Total HTTP requests', 
                                             labelnames=['service','app', 'status_code', 'path'])
http_duration = Histogram('http_request_duration_seconds', 'HTTP request duration',
                                                 labelnames=['service','app'])

# Changed MSE from Gauge to Histogram for better graphing
prediction_mse = Histogram('predictive_scaler_mse', 'Prediction Mean Squared Error', 
                          labelnames=['service','model'], 
                          buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

training_time = Gauge('predictive_scaler_training_time_ms', 'Model training time in ms',
                     labelnames=['service','model'])
prediction_time = Gauge('predictive_scaler_prediction_time_ms', 'Prediction time in ms',
                      labelnames=['service','model'])
predictive_scaler_recommended_replicas = Gauge('predictive_scaler_recommended_replicas', 
                                             'Recommended number of replicas', 
                                             ['service','method'])
predicted_load_gauge = Gauge('predictive_scaler_predicted_load',
                             'Predicted request load (req/s)',
                             labelnames=['service','model'])
current_actual_replicas = Gauge('predictive_scaler_actual_replicas', 
                              'Current actual number of replicas',
                              labelnames=['service'])

# Enhanced MSE metrics with -1 for invalid states
current_mse = Gauge('current_model_mse', 'Current MSE for each model (-1 = invalid/insufficient data)', 
                   labelnames=['service','model'])
model_selection = Counter('model_selection_total', 'Number of times each model was selected',
                         labelnames=['service','model', 'reason'])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables - Paper-based architecture
# Separate persistent data from temporary config
DATA_DIR = "/data"  # Persistent volume for data storage
CONFIG_DIR = "/tmp"  # Temporary directory for config (always use Docker image defaults)

# Data files (persistent) - keep your collected data
DATA_FILE = os.path.join(DATA_DIR, "traffic_data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "gru_model.h5")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions_history.json")
STATUS_FILE = os.path.join(DATA_DIR, "collection_status.json")
SCALER_X_FILE = os.path.join(DATA_DIR, "scaler_X.pkl")
SCALER_Y_FILE = os.path.join(DATA_DIR, "scaler_y.pkl")

# Config file (temporary) - always use fresh config from Docker image
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
MIN_DATA_POINTS_FOR_GRU = 2000  # Based on your hyperparameters
PREDICTION_WINDOW = 24  # Based on your hyperparameters
SCALING_THRESHOLD = _env_float('SCALING_THRESHOLD', 0.7)  # CPU threshold for scaling decision
SCALING_COOLDOWN = 60  # Reduced to 1 minute for faster scale down

# Paper-based Two-Coroutine Configuration
PREDICTION_INTERVAL = 60  # Main coroutine: prediction every 60 seconds
UPDATE_INTERVAL = 300     # Update coroutine: model updates every 5 minutes
GRU_RETRAINING_INTERVAL = 3600  # GRU retraining every hour
HOLT_WINTERS_UPDATE_INTERVAL = 180  # HW MSE updates every 3 minutes

# Default configuration with your original hyperparameters
config = {
    "use_gru": False,
    "collection_interval": 60,  # seconds
    "cpu_threshold": 5.0,  # Skip collection below this CPU % (avoid idle periods)
    "training_threshold_minutes": 3,  # Switch to GRU after this (reduced from 5)
    "prometheus_server": os.getenv('PROMETHEUS_SERVER', 
        'http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090'),
    "target_deployment": os.getenv('TARGET_DEPLOYMENT', 'product-app-combined'),
    "target_namespace": os.getenv('TARGET_NAMESPACE', 'default'),
    "cost_optimization": {
        "enabled": True,
    "target_cpu_utilization": _env_float('TARGET_CPU_UTILIZATION', 65),
    "scale_up_threshold": _env_float('CPU_SCALE_UP_THRESHOLD', 70),
    "scale_down_threshold": _env_float('CPU_SCALE_DOWN_THRESHOLD', 50),  
        "min_replicas": 1,
        "max_replicas": 10,
        "aggressive_scaling": False,  # More conservative scaling
        "scale_down_delay_minutes": 1,  # Faster scale down when predictive
        "zero_traffic_threshold": 0.1,  # Consider traffic zero below this
        "idle_scale_down_minutes": 3  # Faster idle scale down when predictive
    },
    "mse_config": {
        "enabled": True,
        "min_samples_for_mse": 1,  # Minimum samples needed to calculate MSE
        "mse_window_size": 10,  # Increased window for better MSE calculation
        "mse_threshold_difference": 0.5,  # Min difference between MSEs to switch models
        "prediction_match_tolerance_minutes": 5,  # How long to wait for actual values
        "max_prediction_age_hours": 2,  # Remove predictions older than this
        "debug_mse_matching": True  # Enable detailed MSE matching logs
    },
    "models": {
        "holt_winters": {
            "slen": 30,  # 30 data points per season (30 minutes @ 60s per data point)
            "look_forward": 6,   # Predict 6 data points ahead; with 60s points this is 6 minutes
            "look_backward": 90, # Look back 90 data points (~90 minutes = 3 seasons @ 30 min/season)
            "alpha": 0.3,        # Reduced for more stable level smoothing
            "beta": 0.1,         # Reduced for stable trend
            "gamma": 0.9,        # High seasonal smoothing for strong patterns
            "needTrain": False
        },
        "gru": {
            "address": "/tmp/uds_socket",
            "resp_recv_address": "/tmp/rra.socket",
            "look_back": 100, 
            "look_forward": 24,
            "train_size": 2000,
            "batch_size": 10,
            "epochs": 200,
            "n_layers": 1,
            "needTrain": True,
            "preTrained": True
        }
    }
}

# Tracking variables
last_scaling_time = 0
last_scale_up_time = 0
last_scale_down_time = 0
# Use a fixed-size ring buffer (deque) to enforce exactly 240 most-recent points
# This prevents unbounded growth and keeps memory usage predictable.
traffic_data = deque(maxlen=240)
training_dataset = []  # Permanent 4-hour dataset for model training
is_collecting = False
collection_thread = None
data_collection_complete = False  # Flag to track if initial 4-hour data collection is complete
data_collection_start_time = None  # Track when data collection started
gru_model = None
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
is_model_trained = False
last_training_time = None
start_time = datetime.now()
prometheus_client = None
_last_cpu_value = 0.0
_last_cpu_time = None

# Initialization guards
_init_lock = threading.Lock()

# Traffic monitoring variables
low_traffic_start_time = None  # Track when low traffic period started
consecutive_low_cpu_count = 0  # Track consecutive low CPU measurements

# Enhanced MSE tracking variables with MinHeap system
# The MinHeap system uses MSE (Mean Squared Error) to rank models and make scaling decisions
# Lower MSE = Better model performance = Higher priority in the heap
predictions_history = {
    'gru': [],
    'holt_winters': []
}
gru_mse = float('inf')
holt_winters_mse = float('inf')
last_mse_calculation = None
last_cleanup_time = None  # Track cleanup operations to prevent too frequent cleanup

# Training state management to prevent Flask blocking
is_training = False
training_thread = None

# Paper-based Two-Coroutine System Management
main_coroutine_thread = None      # Main coroutine: predictions and scaling
update_coroutine_thread = None    # Update coroutine: model training and MSE updates
is_main_coroutine_running = False
is_update_coroutine_running = False

# Legacy MSE update thread (will be replaced by update coroutine)
mse_update_thread = None
is_mse_updating = False
training_lock = threading.Lock()
file_lock = threading.Lock()  # For file I/O operations

# Model retraining tracking (per paper architecture)
last_gru_retraining = None
last_holt_winters_update = None
gru_needs_retraining = False

# Track last scaling decision for observability in /status
last_scaling_decision_info = {
    'decision': 'maintain',
    'replicas': None,
    'reason': None,
    'time': None,
}

# Hybrid initialization configuration
SYNTHETIC_DATA_ENABLED = False  # Disabled - using real 24-hour collection
SYNTHETIC_DATA_FILE = 'synthetic_dataset_24h.json'  # Default 24-hour dataset
FALLBACK_SYNTHETIC_HOURS = 24  # Generate if file not found
using_synthetic_data = False
synthetic_to_real_transition_time = None
models_performance_history = {
    'gru': [],
    'holt_winters': []
}


def ensure_traffic_buffer_integrity():
    """Guard the shared traffic data buffer against accidental list conversions or overflow."""
    global traffic_data

    buffer_type = type(traffic_data).__name__
    current_maxlen = traffic_data.maxlen if isinstance(traffic_data, deque) else None

    # Rebuild as deque if someone replaced it with a list or changed/removed maxlen
    if not isinstance(traffic_data, deque) or current_maxlen != 240:
        logger.warning(
            "Traffic buffer integrity check: rebuilding %s into deque(maxlen=240)",
            buffer_type
        )
        try:
            snapshot = list(traffic_data)
        except TypeError:
            snapshot = []
        traffic_data = deque(snapshot, maxlen=240)

    # Enforce the maxlen manually if the deque temporarily exceeded it (should be rare)
    maxlen = traffic_data.maxlen or 240
    overflow = max(0, len(traffic_data) - maxlen)
    if overflow > 0:
        for _ in range(overflow):
            traffic_data.popleft()
        logger.warning("Traffic buffer integrity check: trimmed %s excess points", overflow)

    return len(traffic_data)


def get_traffic_buffer_diagnostics():
    """Return current ring buffer state to verify the 240-point cap."""
    ensure_traffic_buffer_integrity()
    max_size = traffic_data.maxlen or 240
    current_size = len(traffic_data)
    return {
        'current_size': current_size,
        'max_size': max_size,
        'within_bounds': current_size <= max_size
    }

def main_coroutine():
    """
    Main coroutine: Handles periodic predictions and scaling decisions
    Based on paper architecture - runs every PREDICTION_INTERVAL seconds
    """
    global is_main_coroutine_running, traffic_data
    
    is_main_coroutine_running = True
    logger.info("🚀 Main coroutine started - handling predictions and scaling")
    
    while is_main_coroutine_running:
        try:
            # Collect current metrics
            if prometheus_client is None:
                initialize_prometheus()
            
            # Get current metrics for scaling decision
            current_metrics = collect_single_metric_point()
            if current_metrics:
                # Always append to traffic_data to keep a continuous history for matching/predictions
                with file_lock:  # Thread-safe data access
                    traffic_data.append(current_metrics)
                    # traffic_data is a deque with maxlen=240 so it will auto-trim oldest entries
                    # Ensure we don't accidentally convert it to a list elsewhere
                if current_metrics['cpu_utilization'] >= config['cpu_threshold']:
                    logger.debug(f"Main coroutine: Data collected - CPU={current_metrics['cpu_utilization']:.1f}% (above threshold {config['cpu_threshold']}%)")
                else:
                    logger.debug(f"Main coroutine: Data collected (idle) - CPU={current_metrics['cpu_utilization']:.1f}% (below threshold {config['cpu_threshold']}%)")
                
                # Make predictions using best model from min-heap (only if we have data)
                best_model = select_best_model_with_minheap()
                if best_model:
                    logger.info(f"🎯 Using best model: {best_model}")
                
                # Try to make predictions with available models
                predictions = {}
                try:
                    # Try Holt-Winters prediction (reduced threshold for faster testing)
                    if len(traffic_data) >= 5:  # Reduced from 10 to 5
                        hw_pred = predict_with_holtwinters(steps=1)
                        if hw_pred and len(hw_pred) > 0:
                            predictions['holt_winters'] = int(hw_pred[0])
                            logger.info(f"✅ Holt-Winters prediction: {predictions['holt_winters']}")
                except Exception as e:
                    logger.debug(f"Holt-Winters prediction failed: {e}")
                
                try:
                    # Try GRU prediction (reduced threshold for faster testing)
                    if len(traffic_data) >= 20:  # Reduced from 105 to 20
                        gru_pred = predict_with_gru(steps=1)
                        if gru_pred and len(gru_pred) > 0:
                            predictions['gru'] = int(gru_pred[0])
                            logger.info(f"✅ GRU prediction: {predictions['gru']}")
                except Exception as e:
                    logger.debug(f"GRU prediction failed: {e}")
                
                # Ensemble removed per simplification request
                
                # Update Prometheus metrics with all predictions and add to MSE tracking
                current_time = datetime.now()
                # Record predictions for the NEXT interval to avoid self-matching
                future_timestamp_str = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
                
                if 'gru' in predictions:
                    predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='gru').set(predictions['gru'])
                    logger.info(f"📊 Main loop - Updated GRU metric: {predictions['gru']}")
                    # Add GRU prediction to history for MSE tracking
                    add_prediction_to_history('gru', predictions['gru'], future_timestamp_str)
                else:
                    predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='gru').set(0)
                    logger.info(f"📊 Main loop - Set GRU metric to 0 (need 20+ data points, have {len(traffic_data)})")
                
                if 'holt_winters' in predictions:
                    predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='holt_winters').set(predictions['holt_winters'])
                    logger.info(f"📊 Main loop - Updated Holt-Winters metric: {predictions['holt_winters']}")
                    # Add Holt-Winters prediction to history for MSE tracking
                    add_prediction_to_history('holt_winters', predictions['holt_winters'], future_timestamp_str)
                else:
                    predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='holt_winters').set(0)
                    logger.info(f"📊 Main loop - Set Holt-Winters metric to 0 (need 5+ data points, have {len(traffic_data)})")
                
                # Ensemble metrics removed
                
                # Automatically trigger MSE calculation after making predictions
                update_mse_metrics()
                
                # Make scaling decision AND execute it (even if no model predictions available)
                decision, replicas = make_scaling_decision_with_minheap(predictions)
                logger.info(f"📊 Main coroutine - Predictions: {predictions}, Decision: {decision}, Replicas: {replicas}")
                
                # ACTUALLY SCALE THE PODS if decision is not "maintain"
                if decision != "maintain":
                    target_deployment = config.get('target_deployment', 'product-app-combined')
                    target_namespace = config.get('target_namespace', 'default')
                    
                    logger.info(f"🚀 EXECUTING SCALING: {decision} {target_deployment} to {replicas} replicas")
                    
                    scaling_success = scale_kubernetes_deployment(
                        deployment_name=target_deployment,
                        namespace=target_namespace,
                        target_replicas=replicas
                    )
                    
                    if scaling_success:
                        logger.info(f"✅ SCALING SUCCESS: {target_deployment} scaled to {replicas} replicas")
                        # Update scaling metrics for Prometheus
                        scaling_decisions.labels(service=SERVICE_LABEL, decision=decision).inc()
                        last_scaling_decision_info.update({
                            'decision': decision,
                            'replicas': replicas,
                            'reason': 'main_coroutine',
                            'time': datetime.now().isoformat(),
                        })
                    else:
                        logger.error(f"❌ SCALING FAILED: Could not scale {target_deployment}")
                else:
                    logger.info(f"🔄 MAINTAINING: Keeping {replicas} replicas (no scaling needed)")
                    last_scaling_decision_info.update({
                        'decision': 'maintain',
                        'replicas': replicas,
                        'reason': 'main_coroutine',
                        'time': datetime.now().isoformat(),
                    })
            
            # Sleep until next prediction interval
            # Sleep for the collection interval so each data point is spaced by 60s
            time.sleep(int(config.get('collection_interval', PREDICTION_INTERVAL)))
            
        except Exception as e:
            logger.error(f"❌ Main coroutine error: {e}")
            time.sleep(10)
    
    logger.info("🛑 Main coroutine stopped")

def update_coroutine():
    """
    Update coroutine: Responsible for continuous model training and updates
    Based on paper architecture - runs every UPDATE_INTERVAL seconds
    """
    global is_update_coroutine_running, last_gru_retraining, last_holt_winters_update
    
    is_update_coroutine_running = True
    logger.info("🔄 Update coroutine started - handling model training and updates")
    
    while is_update_coroutine_running:
        try:
            current_time = datetime.now()
            
            # Handle GRU model retraining (paper: "user-configured retraining time point")
            if should_retrain_gru(current_time):
                logger.info("🧠 Update coroutine - Starting GRU retraining")
                retrain_gru_model_async()
                last_gru_retraining = current_time
            
            # Handle Holt-Winters continuous updates (paper: "continuously updates by running simulated predictions")
            if should_update_holt_winters(current_time):
                logger.info("📈 Update coroutine - Updating Holt-Winters MSE")
                update_holt_winters_performance()
                last_holt_winters_update = current_time
            
            # Update model rankings in min-heap based on current performance
            update_model_heap_rankings()
            
            # Periodically save data for persistence (every update cycle)
            if len(traffic_data) > 0:
                save_data()
                logger.debug(f"💾 Auto-saved {len(traffic_data)} data points")
            
            # Sleep until next update interval
            time.sleep(UPDATE_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ Update coroutine error: {e}")
            time.sleep(30)
    
    logger.info("🛑 Update coroutine stopped")

def collect_single_metric_point():
    """Collect a single data point for the main coroutine"""
    try:
        # Get current CPU utilization using improved detection
        current_cpu = get_current_cpu_from_prometheus()
        
        # Memory usage
        memory_query = f'avg(container_memory_usage_bytes{{pod=~"{config["target_deployment"]}-.*"}}) / 1024 / 1024'
        memory_result = prometheus_client.custom_query(memory_query)
        current_memory = float(memory_result[0]['value'][1]) if memory_result else 0
        
        # Request rate (traffic)
        request_query = f'sum(rate(app_http_requests_total{{app="{config["target_deployment"]}"}}[1m]))'
        request_result = prometheus_client.custom_query(request_query)
        current_traffic = float(request_result[0]['value'][1]) if request_result else 0
        
        # Current replicas
        replicas_query = f'kube_deployment_status_replicas{{deployment="{config["target_deployment"]}", namespace="{config["target_namespace"]}"}}'
        replicas_result = prometheus_client.custom_query(replicas_query)
        current_replicas = int(float(replicas_result[0]['value'][1])) if replicas_result else 1
        
        # Update Prometheus metrics
        cpu_utilization.set(current_cpu)
        traffic_gauge.set(current_traffic)
        current_actual_replicas.labels(service=SERVICE_LABEL).set(current_replicas)
        
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'traffic': current_traffic,
            'cpu_utilization': current_cpu,
            'memory_usage': current_memory,
            'replicas': current_replicas,
            'synthetic': False  # Real data from Prometheus
        }
        
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
        return None

def should_retrain_gru(current_time):
    """Determine if GRU should be retrained based on paper criteria"""
    global last_gru_retraining, is_model_trained, traffic_data
    
    # Never trained
    if not is_model_trained:
        return len(traffic_data) >= MIN_DATA_POINTS_FOR_GRU
    
    # Time-based retraining
    if last_gru_retraining is None:
        return True
    
    time_since_last = (current_time - last_gru_retraining).total_seconds()
    return time_since_last >= GRU_RETRAINING_INTERVAL

def should_update_holt_winters(current_time):
    """Determine if Holt-Winters MSE should be updated"""
    global last_holt_winters_update
    
    if last_holt_winters_update is None:
        return len(traffic_data) >= 12  # Minimum data for HW
    
    time_since_last = (current_time - last_holt_winters_update).total_seconds()
    return time_since_last >= HOLT_WINTERS_UPDATE_INTERVAL

def update_holt_winters_performance():
    """Paper: 'Continuously updates by running simulated predictions'"""
    try:
        # Run simulated predictions to calculate current MSE
        if len(traffic_data) >= 30:  # Need sufficient data
            # Use recent data window for HW performance evaluation
            traffic_list = list(traffic_data)
            recent_data = traffic_list[-180:] if len(traffic_list) > 180 else traffic_list
            
            # Calculate MSE using recent predictions vs actual values
            hw_mse = calculate_mse_robust('holt_winters')
            
            # Update performance history
            models_performance_history['holt_winters'].append({
                'timestamp': datetime.now(),
                'mse': hw_mse,
                'data_points_used': len(recent_data)
            })
            
            # Keep only recent performance history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            models_performance_history['holt_winters'] = [
                p for p in models_performance_history['holt_winters'] 
                if p['timestamp'] > cutoff_time
            ]
            
            logger.info(f"📈 Holt-Winters MSE updated: {hw_mse:.4f}")
            
    except Exception as e:
        logger.error(f"Error updating Holt-Winters performance: {e}")

def update_model_heap_rankings():
    """Paper: 'Adjusts model rankings in the min-heap based on performance'"""
    try:
        # This will trigger the existing min-heap logic with updated MSE values
        get_ranked_model_predictions()
        logger.debug("🔄 Model heap rankings updated")
        
    except Exception as e:
        logger.error(f"Error updating model heap rankings: {e}")

def make_prediction_with_model(model_name):
    """Make prediction using specified model"""
    try:
        if model_name == 'gru' and gru_model is not None:
            return predict_with_gru(6)
        elif model_name == 'holt_winters':
            return predict_with_holtwinters(6)
        else:
            return predict_with_holtwinters(6)  # Fallback
            
    except Exception as e:
        logger.error(f"Error making prediction with {model_name}: {e}")
        return None

def retrain_gru_model_async():
    """Asynchronous GRU retraining for update coroutine"""
    global is_training, traffic_data
    try:
        if not is_training and len(traffic_data) >= MIN_DATA_POINTS_FOR_GRU:
            # Use sliding window of recent data for training
            traffic_list = list(traffic_data)
            training_data_copy = traffic_list[-MIN_DATA_POINTS_FOR_GRU:]
            
            # Start async training
            training_thread = threading.Thread(
                target=async_model_training,
                args=("update_coroutine_retraining", training_data_copy, True)
            )
            training_thread.daemon = True
            training_thread.start()
            
            logger.info(f"🚀 Update coroutine - GRU retraining started with {len(training_data_copy)} data points")
        
    except Exception as e:
        logger.error(f"Error starting GRU retraining: {e}")

def start_two_coroutine_system():
    """Start the paper-based two-coroutine system"""
    global main_coroutine_thread, update_coroutine_thread
    
    # Start main coroutine
    main_coroutine_thread = threading.Thread(target=main_coroutine)
    main_coroutine_thread.daemon = True
    main_coroutine_thread.start()
    logger.info("✅ Main coroutine thread started")
    
    # Start update coroutine  
    update_coroutine_thread = threading.Thread(target=update_coroutine)
    update_coroutine_thread.daemon = True
    update_coroutine_thread.start()
    logger.info("✅ Update coroutine thread started")

def stop_two_coroutine_system():
    """Stop the two-coroutine system"""
    global is_main_coroutine_running, is_update_coroutine_running
    
    is_main_coroutine_running = False
    is_update_coroutine_running = False
    logger.info("🛑 Two-coroutine system stopping...")

def async_model_training(training_reason, traffic_data_copy, advanced_fallback=True):
    """Perform model training in a separate thread to prevent Flask blocking"""
    global is_training, is_model_trained, last_training_time, models, gru_model
    
    try:
        with training_lock:
            if is_training:
                logger.warning("⚠️ Training already in progress, skipping...")
                return
            is_training = True
        
        logger.info(f"🚀 Starting ASYNC model training: {training_reason}")
        training_start = datetime.now()
        
        # Configure TensorFlow to use limited resources to prevent blocking
        try:
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(2)  # Limit threads
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except:
            pass  # If TensorFlow config fails, continue anyway
        
        success = False
        
        # Use the thread-safe copy for training without modifying global state
        try:
            # Try advanced model first if we have enough data
            if advanced_fallback and len(traffic_data_copy) >= 30:
                logger.info("🧠 Attempting advanced GRU model training...")
                success = build_gru_model_with_data(traffic_data_copy)
                if success:
                    logger.info("✅ Advanced GRU model training succeeded!")
            
            # Fallback to basic model if advanced failed or not attempted
            if not success:
                logger.info("🔧 Attempting basic GRU model training...")
                success = build_gru_model_with_data(traffic_data_copy)
                if success:
                    logger.info("✅ Basic GRU model training succeeded!")
                    
        except Exception as training_error:
            logger.error(f"❌ Model training execution failed: {training_error}")
            success = False
        
        if success:
            is_model_trained = True
            last_training_time = datetime.now()
            
            # Update config to enable GRU usage
            if not config['use_gru']:
                config['use_gru'] = True
                save_config()
                logger.info("✅ GRU model enabled in configuration")
            
            duration = (datetime.now() - training_start).total_seconds()
            logger.info(f"🎯 ASYNC training completed successfully in {duration:.1f}s")
            
            # 🆘 BOOTSTRAP: Auto-create emergency predictions for first training
            # Note: was_first_training was determined before training started
            if len(predictions_history.get('gru', [])) == 0:
                logger.info("🆘 AUTO-BOOTSTRAP: Creating emergency predictions for first-time training...")
                try:
                    create_emergency_gru_predictions()
                    logger.info("✅ Bootstrap emergency predictions created!")
                except Exception as e:
                    logger.error(f"❌ Bootstrap prediction creation failed: {e}")
        else:
            logger.error("❌ ASYNC training failed")
            
    except Exception as e:
        logger.error(f"❌ ASYNC training error: {e}")
    finally:
        with training_lock:
            is_training = False

def scale_kubernetes_deployment(deployment_name, namespace, target_replicas):
    """Scale the Kubernetes deployment using in-cluster API (preferred), fallback to kubectl."""
    try:
        # Validate inputs
        if not deployment_name or not namespace:
            logger.error("❌ Invalid deployment name or namespace for scaling")
            return False

        if target_replicas < 1 or target_replicas > 20:  # Safety limits
            logger.error(f"❌ Target replicas {target_replicas} outside safe range (1-20)")
            return False

        # Preferred: Kubernetes Python client
        if K8S_AVAILABLE:
            try:
                # In-cluster config if running in k8s, else load default
                try:
                    k8s_config.load_incluster_config()
                    logger.info("🔐 Loaded in-cluster Kubernetes config")
                except Exception:
                    k8s_config.load_kube_config()
                    logger.info("💻 Loaded local kubeconfig")

                apps_v1 = k8s_client.AppsV1Api()
                body = {"spec": {"replicas": int(target_replicas)}}
                logger.info(f"🎯 Scaling via API: {namespace}/{deployment_name} -> {target_replicas}")
                apps_v1.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body=body
                )
                logger.info(f"✅ Successfully scaled {deployment_name} to {target_replicas} replicas (API)")
                current_actual_replicas.labels(service=SERVICE_LABEL).set(target_replicas)
                return True
            except Exception as api_err:
                logger.error(f"💥 Kubernetes API scaling failed: {api_err}. Falling back to kubectl...")

        # Fallback: kubectl command
        cmd = [
            "kubectl", "scale", "deployment", deployment_name,
            f"--replicas={target_replicas}",
            f"--namespace={namespace}"
        ]
        logger.info(f"🎯 Executing scaling command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            logger.info(f"✅ Successfully scaled {deployment_name} to {target_replicas} replicas (kubectl)")
            current_actual_replicas.labels(service=SERVICE_LABEL).set(target_replicas)
            return True
        else:
            logger.error(f"❌ kubectl scale failed ({result.returncode}): {result.stderr.strip()}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("⏱️ kubectl scale command timed out after 30 seconds")
        return False
    except FileNotFoundError:
        logger.error("🔎 kubectl not found - ensure kubectl is installed and in PATH")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during scaling: {e}")
        return False

def initialize_prometheus():
    """Initialize Prometheus client"""
    global prometheus_client
    try:
        prometheus_client = PrometheusConnect(
            url=config['prometheus_server'], 
            disable_ssl=True
        )
        logger.info("Prometheus client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Prometheus client: {e}")

def load_config():
    global config
    try:
        # Always use defaults from app.py - no persistent config loading
        logger.info("🔧 Using fresh configuration from Docker image (no persistent config loading)")
        logger.info(f"� CPU threshold: {config.get('cpu_threshold', 'unknown')}%")
        
        # Log the active cost optimization settings
        cost_opt = config['cost_optimization']
        logger.info(f"🎯 Active scaling thresholds: scale_up={cost_opt['scale_up_threshold']}%, scale_down={cost_opt['scale_down_threshold']}%, target_cpu={cost_opt['target_cpu_utilization']}%")
        
        # Save the merged config (this will preserve app.py defaults in persistent storage going forward)
        save_config()
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")

def save_config():
    """Save config to temporary location only (not persistent)"""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.debug(f"✅ Config saved to temporary location: {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"⚠️ Could not save config: {e}")

def load_data():
    global traffic_data, training_dataset
    try:
        # Load current traffic data
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            records = df.to_dict('records')
            # Keep only the most recent 240 points when loading
            traffic_data.clear()
            traffic_data.extend(records[-240:])
            logger.info(f"Loaded {len(traffic_data)} current data points from {DATA_FILE} (trimmed to 240)")
        
        # Load training dataset if it exists and collection is complete
        training_file = DATA_FILE.replace('.csv', '_training.csv')
        if os.path.exists(training_file) and data_collection_complete:
            training_df = pd.read_csv(training_file)
            training_dataset = training_df.to_dict('records')
            logger.info(f"Loaded {len(training_dataset)} training data points from {training_file}")
        logger.debug(f"Traffic buffer diagnostics after load: {get_traffic_buffer_diagnostics()}")
        ensure_traffic_buffer_integrity()
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Ensure traffic_data remains a deque
        try:
            traffic_data.clear()
        except Exception:
            traffic_data = deque(maxlen=240)
        training_dataset = []

def save_data():
    try:
        # Save current traffic data
        df = pd.DataFrame(traffic_data)
        df.to_csv(DATA_FILE, index=False)
        
        # Save training dataset separately if it exists
        if training_dataset:
            training_file = DATA_FILE.replace('.csv', '_training.csv')
            training_df = pd.DataFrame(training_dataset)
            training_df.to_csv(training_file, index=False)
            logger.info(f"Saved {len(training_dataset)} training data points to {training_file}")
        
        save_collection_status()  # Also save collection status when saving data
        logger.info(f"Saved {len(traffic_data)} current data points to {DATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def filter_stale_data(max_age_hours=24):
    """Filter out data points older than a specified age."""
    global traffic_data
    if not traffic_data:
        return

    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        original_count = len(traffic_data)
        
        # Ensure timestamp is a string before parsing
        filtered = [
            d for d in list(traffic_data)
            if 'timestamp' in d and isinstance(d['timestamp'], str) and 
               datetime.strptime(d['timestamp'], "%Y-%m-%d %H:%M:%S") > cutoff_time
        ]
        # Replace contents of the deque while preserving maxlen
        traffic_data.clear()
        # Keep only the most recent 240 points after filtering
        if filtered:
            traffic_data.extend(filtered[-240:])
        
        removed_count = original_count - len(traffic_data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} stale data points older than {max_age_hours} hours.")
        logger.debug(f"Traffic buffer diagnostics after stale filter: {get_traffic_buffer_diagnostics()}")
        ensure_traffic_buffer_integrity()
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing timestamps during stale data filtering: {e}. Data may be corrupt.")


def load_predictions_history():
    """Load predictions history for MSE calculation with file locking"""
    global predictions_history
    try:
        with file_lock:
            if os.path.exists(PREDICTIONS_FILE):
                with open(PREDICTIONS_FILE, 'r') as f:
                    loaded_history = json.load(f)
                    # Ensure all model keys exist (ensemble removed)
                    predictions_history = {'gru': [], 'holt_winters': []}
                    for model_name in predictions_history.keys():
                        if model_name in loaded_history:
                            predictions_history[model_name] = loaded_history[model_name]
                    logger.info("Loaded predictions history for MSE calculation")
            else:
                predictions_history = {'gru': [], 'holt_winters': []}
    except Exception as e:
        logger.error(f"Error loading predictions history: {e}")
    # Keep current predictions_history state on error

def save_predictions_history():
    """Save predictions history for MSE calculation with file locking"""
    try:
        with file_lock:
            with open(PREDICTIONS_FILE, 'w') as f:
                json.dump(predictions_history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving predictions history: {e}")

def load_collection_status():
    """Load data collection status from file"""
    global data_collection_complete, data_collection_start_time
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
                data_collection_complete = status.get('collection_complete', False)
                start_time_str = status.get('collection_start_time')
                if start_time_str:
                    data_collection_start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                logger.info(f"Loaded collection status: complete={data_collection_complete}, start_time={data_collection_start_time}")
    except Exception as e:
        logger.error(f"Error loading collection status: {e}")
        data_collection_complete = False
        data_collection_start_time = None

def save_collection_status():
    """Save data collection status to file"""
    try:
        status = {
            'collection_complete': data_collection_complete,
            'collection_start_time': data_collection_start_time.strftime("%Y-%m-%d %H:%M:%S") if data_collection_start_time else None,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
        logger.info(f"Saved collection status: complete={data_collection_complete}")
    except Exception as e:
        logger.error(f"Error saving collection status: {e}")

def create_advanced_features(data):
    """Create advanced features for better prediction accuracy."""
    if len(data) < 12:
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute
        
        # Rolling statistics (trend analysis)
        df['cpu_ma_5'] = df['cpu_utilization'].rolling(window=5, min_periods=1).mean()
        df['cpu_ma_10'] = df['cpu_utilization'].rolling(window=10, min_periods=1).mean()
        df['traffic_ma_5'] = df['traffic'].rolling(window=5, min_periods=1).mean()
        
        # Trend indicators
        df['cpu_trend'] = df['cpu_utilization'].diff().fillna(0)
        df['traffic_trend'] = df['traffic'].diff().fillna(0)
        
        # Load pressure indicators
        df['load_pressure'] = (df['cpu_utilization'] / df['replicas']).fillna(1)
        df['efficiency'] = (df['traffic'] / (df['replicas'] * df['cpu_utilization'] + 0.1)).fillna(1)
        
        # Lag features (previous values matter for scaling decisions)
        df['cpu_lag1'] = df['cpu_utilization'].shift(1).fillna(df['cpu_utilization'])
        df['traffic_lag1'] = df['traffic'].shift(1).fillna(df['traffic'])
        df['replicas_lag1'] = df['replicas'].shift(1).fillna(df['replicas'])
        
        # Volatility measures
        df['cpu_volatility'] = df['cpu_utilization'].rolling(window=5, min_periods=1).std().fillna(0)
        df['traffic_volatility'] = df['traffic'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Peak detection
        df['is_cpu_peak'] = (df['cpu_utilization'] > df['cpu_ma_10'] * 1.3).astype(int)
        df['is_traffic_peak'] = (df['traffic'] > df['traffic_ma_5'] * 1.5).astype(int)
        
        logger.info(f"✅ Advanced features created: {len(df.columns)} total features")
        return df
        
    except Exception as e:
        logger.error(f"Error creating advanced features: {e}")
        return None

def preprocess_advanced_data(enhanced_df, feature_columns, sequence_length=12):
    """Enhanced preprocessing for advanced features."""
    global scaler_X, scaler_y
    
    try:
        if len(enhanced_df) < sequence_length + 1:
            return None, None
        
        # Use enhanced features
        available_features = [col for col in feature_columns if col in enhanced_df.columns]
        if len(available_features) < 3:
            logger.warning(f"Too few features available: {available_features}")
            return None, None
            
        features = enhanced_df[available_features].values
        targets = enhanced_df['replicas'].values.reshape(-1, 1)
        
        # Handle missing values more intelligently
        features_df = pd.DataFrame(features, columns=available_features)
        features_clean = features_df.ffill().bfill().fillna(0).values
        targets_clean = pd.Series(targets.flatten()).ffill().bfill().fillna(1).values.reshape(-1, 1)
        
        # Robust scaling (more resistant to outliers)
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        
        features_scaled = scaler_X.fit_transform(features_clean)
        targets_scaled = scaler_y.fit_transform(targets_clean)
        
        # Create sequences
        X, y = [], []
        for i in range(len(enhanced_df) - sequence_length):
            X.append(features_scaled[i:i+sequence_length])
            y.append(targets_scaled[i+sequence_length])
        
        logger.info(f"Advanced preprocessing: {len(X)} sequences, {len(available_features)} features")
        return np.array(X), np.array(y)
        
    except Exception as e:
        logger.error(f"Error in advanced preprocessing: {e}")
        return None, None

def cleanup_old_predictions():
    """Remove predictions that are too old to be matched - throttled to prevent MSE disruption"""
    global last_cleanup_time
    current_time = datetime.now()
    
    # Only run cleanup every 5 minutes to prevent MSE disruption
    if last_cleanup_time and (current_time - last_cleanup_time).total_seconds() < 300:
        return
    
    last_cleanup_time = current_time
    max_age_hours = config['mse_config']['max_prediction_age_hours']
    cutoff_time = current_time - timedelta(hours=max_age_hours)
    
    total_removed = 0
    for model_name in predictions_history:
        try:
            original_count = len(predictions_history[model_name])
            predictions_history[model_name] = [
                p for p in predictions_history[model_name]
                if datetime.strptime(p['timestamp'], "%Y-%m-%d %H:%M:%S") > cutoff_time
            ]
            removed_count = original_count - len(predictions_history[model_name])
            total_removed += removed_count
            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} old predictions for {model_name}")
        except Exception as e:
            logger.warning(f"Cleanup failed for {model_name}: {e}")
    
    if total_removed > 0:
        logger.info(f"Cleanup completed: removed {total_removed} old predictions")

def add_prediction_to_history(model_name, predicted_replicas, timestamp):
    """Record a prediction for later matching against future actuals.
    Note: Do NOT immediately match using current actuals; we only match once
    actual data at or after the prediction timestamp is available.
    """
    global predictions_history
    
    # Get current state for context
    current_cpu = 0
    current_traffic = 0
    current_replicas = 1
    if traffic_data:
        latest_data = traffic_data[-1]
        current_cpu = latest_data['cpu_utilization']
        current_traffic = latest_data['traffic']
        current_replicas = latest_data['replicas']
    
    # Estimate predicted load for load-based MSE preference
    # Prefer throughput-per-replica * predicted_replicas to reflect capacity-based forecast
    predicted_load = None
    try:
        if traffic_data:
            # Use most recent non-zero replicas to avoid division by zero
            recent = traffic_data[-1]
            reps = max(1, int(recent.get('replicas', 1)))
            curr_tr = float(recent.get('traffic', 0.0))
            tpr = curr_tr / float(reps) if reps > 0 else 0.0  # throughput per replica (req/s)
            predicted_load = float(predicted_replicas) * tpr
        if predicted_load is None or predicted_load == 0.0:
            # Fallback to recent average if we have no meaningful throughput
            recent_tr = [float(d.get('traffic', 0.0)) for d in list(traffic_data)[-10:]] if traffic_data else []
            predicted_load = float(sum(recent_tr) / max(1, len(recent_tr))) if recent_tr else 0.0
    except Exception:
        # Final fallback
        predicted_load = 0.0

    prediction_entry = {
        'timestamp': timestamp,
        'predicted_replicas': predicted_replicas,
        'actual_replicas': None,  # Will be filled when we get actual data
        # Prefer load-based matching for MSE when available
        'predicted_load': predicted_load,
        'actual_load': None,
        'predicted_cpu': None,
        'actual_cpu': None,
        'context_cpu': current_cpu,  # CPU at time of prediction
        'context_traffic': current_traffic,  # Traffic at time of prediction
        'matched': False,
        'match_timestamp': None
    }
    
    # IMPORTANT: Do not perform any immediate matching here. We want to evaluate
    # forecast accuracy on future observations only.
    
    predictions_history[model_name].append(prediction_entry)

    # Export predicted load for visualization
    try:
        predicted_load_gauge.labels(service=SERVICE_LABEL, model=model_name).set(float(predicted_load or 0.0))
    except Exception:
        pass
    
    # Keep only recent predictions
    mse_window = config['mse_config']['mse_window_size']
    if len(predictions_history[model_name]) > mse_window * 3:  # Keep extra for matching
        predictions_history[model_name] = predictions_history[model_name][-mse_window * 2:]
    
    # Clean up old predictions periodically
    cleanup_old_predictions()
    
    # Save periodically
    if len(predictions_history[model_name]) % 5 == 0:
        save_predictions_history()

def match_new_data_with_predictions(current_replicas, current_cpu, timestamp):
    """Immediately match new data point with any waiting predictions"""
    global predictions_history
    
    try:
        current_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        match_count = 0
        
        for model_name in ['gru', 'holt_winters']:
            unmatched = [p for p in predictions_history[model_name] if not p.get('matched', False)]
            
            for prediction in unmatched:
                try:
                    pred_time = datetime.strptime(prediction['timestamp'], "%Y-%m-%d %H:%M:%S")
                    delta = (current_time - pred_time).total_seconds()
                    
                    # Allow timing skew and a broader tolerance window for robust matching
                    match_tolerance = max(
                        config['mse_config']['prediction_match_tolerance_minutes'] * 60,
                        int(config.get('collection_interval', 60)) * 2,
                        10 * 60
                    )
                    skew_slack = max(10, int(config.get('collection_interval', 60)))
                    if -skew_slack <= delta <= match_tolerance:
                        prediction['actual_replicas'] = current_replicas
                        prediction['actual_cpu'] = current_cpu
                        # Try to also fill actual_load using current timestamp
                        try:
                            # current_traffic is not passed in; derive from traffic_data latest if timestamps align
                            if traffic_data and traffic_data[-1].get('timestamp') == timestamp:
                                prediction['actual_load'] = traffic_data[-1].get('traffic')
                        except Exception:
                            pass
                        prediction['matched'] = True
                        prediction['match_timestamp'] = timestamp
                        prediction['match_time_diff'] = delta
                        match_count += 1
                        
                        logger.info(f"⚡ Quick match {model_name}: pred={prediction['predicted_replicas']}, actual={current_replicas}, diff={delta:.0f}s")
                        break  # Only match one prediction per model per data point
                        
                except Exception as e:
                    logger.debug(f"Error in quick matching for {model_name}: {e}")
        
        if match_count > 0:
            logger.info(f"Quick matched {match_count} predictions with new data")
            # Immediately recalculate MSE when we get new matches
            try:
                update_mse_metrics()
                logger.info("MSE recalculated after quick matching")
            except Exception as e:
                logger.debug(f"Failed to recalculate MSE after matching: {e}")
            
    except Exception as e:
        logger.debug(f"Error in match_new_data_with_predictions: {e}")

def create_emergency_gru_predictions():
    """Emergency system to create synthetic GRU predictions for MSE testing"""
    global predictions_history, traffic_data
    
    try:
        if len(traffic_data) < 10:
            return
        
        logger.warning("🆘 Creating emergency synthetic GRU predictions for MSE calculation")
        
        # Create realistic predictions based on recent data
        recent_replicas = [d['replicas'] for d in list(traffic_data)[-10:]]
        avg_replicas = sum(recent_replicas) / len(recent_replicas)
        
        # Create 5 synthetic predictions with slight variations
        for i in range(5):
            # Use data from 5-10 minutes ago and predict current values
            historical_data = traffic_data[-(10-i)]
            current_data = traffic_data[-1]
            
            # Create a realistic prediction (with small random variation)
            variation = [-1, 0, 0, 0, 1][i]  # Mostly accurate predictions
            predicted_replicas = max(1, min(10, current_data['replicas'] + variation))
            
            # Use historical timestamp for prediction, current for matching
            pred_timestamp = historical_data['timestamp']
            
            emergency_prediction = {
                'timestamp': pred_timestamp,
                'predicted_replicas': predicted_replicas,
                'actual_replicas': current_data['replicas'],
                'predicted_cpu': None,
                'actual_cpu': current_data['cpu_utilization'],
                'context_cpu': historical_data['cpu_utilization'],
                'context_traffic': historical_data['traffic'],
                'matched': True,  # Pre-matched
                'match_timestamp': current_data['timestamp'],
                'match_time_diff': 300,  # 5 minutes
                'emergency': True  # Flag as emergency prediction
            }
            
            predictions_history['gru'].append(emergency_prediction)
            
        logger.warning(f"🆘 Created {len([p for p in predictions_history['gru'] if p.get('emergency')])} emergency GRU predictions")
        
        # Immediately calculate MSE
        update_mse_metrics()
        
    except Exception as e:
        logger.error(f"Emergency prediction creation failed: {e}")

def update_predictions_with_actual_values():
    """Enhanced prediction-to-actual matching with improved timing logic"""
    global predictions_history, traffic_data
    
    if not traffic_data:
        logger.debug("No traffic data available for prediction matching")
        return

    match_count = 0
    now = datetime.now()
    debug_mode = config['mse_config']['debug_mse_matching']
    
    # Pre-convert all traffic data timestamps for efficiency
    traffic_data_with_dt = []
    for point in traffic_data:
        try:
            if isinstance(point['timestamp'], str):
                dt_obj = datetime.strptime(point['timestamp'], "%Y-%m-%d %H:%M:%S")
                traffic_data_with_dt.append({
                    **point,
                    'timestamp_dt': dt_obj
                })
            else:
                traffic_data_with_dt.append(point)
        except Exception as e:
            logger.warning(f"Invalid timestamp in traffic data: {point.get('timestamp', 'unknown')}")
    
    if debug_mode and len(traffic_data_with_dt) > 0:
        latest_data_time = max(td['timestamp_dt'] for td in traffic_data_with_dt)
        logger.debug(f"Latest traffic data timestamp: {latest_data_time}")

    for model_name in ['gru', 'holt_winters']:
        model_matches = 0
        unmatched_predictions = [p for p in predictions_history[model_name] if not p.get('matched', False)]
        
        if debug_mode and unmatched_predictions:
            logger.debug(f"{model_name}: Processing {len(unmatched_predictions)} unmatched predictions")
        
        for prediction in unmatched_predictions:
            try:
                pred_time = datetime.strptime(prediction['timestamp'], "%Y-%m-%d %H:%M:%S")
                
                # Only try to match predictions that should have data available
                # (i.e., prediction time has passed or is very close)
                time_since_prediction = (now - pred_time).total_seconds()
                
                if time_since_prediction < -180:  # Allow a bit more future skew before skipping
                    if debug_mode:
                        logger.debug(f"{model_name}: Prediction {prediction['timestamp']} is {-time_since_prediction:.0f}s in future, skipping")
                    continue
                
                # Find the best matching data point
                best_match = None
                min_time_diff = float('inf')
                match_tolerance = max(
                    config['mse_config']['prediction_match_tolerance_minutes'] * 60,
                    int(config.get('collection_interval', 60)) * 2
                )

                for data_point in traffic_data_with_dt:
                    data_time = data_point['timestamp_dt']
                    delta = (data_time - pred_time).total_seconds()
                    
                    # Accept future-or-equal matches; allow small negative delta for skew
                    skew_slack = max(10, int(config.get('collection_interval', 60)))
                    if -skew_slack <= delta <= match_tolerance and abs(delta) < min_time_diff:
                        min_time_diff = delta
                        best_match = data_point
                
                if best_match:
                    prediction['actual_replicas'] = best_match['replicas']
                    prediction['actual_cpu'] = best_match['cpu_utilization']
                    prediction['actual_load'] = best_match.get('traffic')
                    prediction['matched'] = True
                    prediction['match_timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
                    prediction['match_time_diff'] = min_time_diff
                    match_count += 1
                    model_matches += 1
                    
                    if debug_mode:
                        logger.info(f"✓ Matched {model_name} prediction: time={prediction['timestamp']}, "
                                  f"predicted={prediction['predicted_replicas']}, actual={best_match['replicas']}, "
                                  f"time_diff={min_time_diff:.0f}s")
                else:
                    # Backfill: if the prediction is older than tolerance and still unmatched,
                    # pick the closest datapoint within +/- match_tolerance to avoid starving MSE
                    if time_since_prediction > match_tolerance:
                        closest = None
                        closest_abs = float('inf')
                        for data_point in traffic_data_with_dt:
                            data_time = data_point['timestamp_dt']
                            delta_any = (data_time - pred_time).total_seconds()
                            if abs(delta_any) <= match_tolerance and abs(delta_any) < closest_abs:
                                closest_abs = abs(delta_any)
                                closest = data_point
                        if closest is not None:
                            prediction['actual_replicas'] = closest['replicas']
                            prediction['actual_cpu'] = closest['cpu_utilization']
                            prediction['actual_load'] = closest.get('traffic')
                            prediction['matched'] = True
                            prediction['match_timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
                            prediction['match_time_diff'] = (closest['timestamp_dt'] - pred_time).total_seconds()
                            match_count += 1
                            model_matches += 1
                            if debug_mode:
                                logger.info(f"↩ Backfilled {model_name} prediction: time={prediction['timestamp']}, "
                                            f"pred={prediction['predicted_replicas']}, actual={closest['replicas']}, "
                                            f"time_diff={prediction['match_time_diff']:.0f}s")
                # If still not matched (no best_match and no backfill), log why
                if not prediction.get('matched', False):
                    # Log why no match was found (only for older predictions)
                    if debug_mode and time_since_prediction > 60:
                        closest_data = min(
                            traffic_data_with_dt,
                            key=lambda x: abs((x['timestamp_dt'] - pred_time).total_seconds()),
                            default=None
                        )
                        if closest_data:
                            closest_diff = abs((closest_data['timestamp_dt'] - pred_time).total_seconds())
                            logger.debug(
                                f"✗ No match for {model_name} prediction {prediction['timestamp']}: "
                                f"closest data at {closest_data['timestamp']} (diff: {closest_diff:.0f}s)"
                            )

            except Exception as e:
                logger.error(f"Error matching {model_name} prediction: {e}")
        
        if debug_mode and model_matches > 0:
            total_matched = len([p for p in predictions_history[model_name] if p.get('matched', False)])
            logger.info(f"{model_name}: +{model_matches} new matches, {total_matched} total matched")

    if debug_mode and match_count > 0:
        logger.info(f"Prediction matching completed: {match_count} new matches found")

    if match_count > 0:
        logger.info(f"Matched {match_count} predictions with historical data.")

def calculate_mse_robust(model_name):
    """Enhanced MSE calculation preferring load-based error; fallback to replicas."""
    global predictions_history
    
    if model_name not in predictions_history:
        logger.warning(f"Model {model_name} not found in predictions history")
        return float('inf')
    
    predictions = predictions_history[model_name]
    matched_predictions = [
        p for p in predictions
        if p.get('matched', False)
        and (p.get('actual_replicas') is not None or p.get('actual_load') is not None)
        and not p.get('emergency', False)
    ]
    
    min_samples = max(1, config['mse_config']['min_samples_for_mse'])  # Ensure at least 1
    debug_mode = config['mse_config']['debug_mse_matching']
    
    total_predictions = len(predictions)
    matched_count = len(matched_predictions)
    
    # Always log status for debugging
    logger.info(f"{model_name} MSE Status: {matched_count}/{total_predictions} predictions matched")
    
    if debug_mode and total_predictions > 0:
        # Show recent prediction details
        recent_predictions = predictions[-3:] if predictions else []
        for i, p in enumerate(recent_predictions):
            match_status = "✓" if p.get('matched', False) else "✗"
            logger.debug(f"  Recent #{i+1}: {match_status} pred={p.get('predicted_replicas', 'N/A')}, "
                        f"actual={p.get('actual_replicas', 'N/A')}, time={p.get('timestamp', 'N/A')}")
    
    if matched_count < min_samples:
        if debug_mode:
            logger.info(f"{model_name} MSE: Need {min_samples} matches, have {matched_count} - waiting for more data")
        return float('inf')
    
    # Use a rolling window for stability; for small datasets, use all
    window_size = max(3, int(config['mse_config']['mse_window_size']))
    recent_predictions = matched_predictions if matched_count <= window_size else matched_predictions[-window_size:]
    
    try:
        squared_errors = []
        basis = None

        def _nearest_actual_load(ts_str):
            try:
                target_ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None
            tol = timedelta(minutes=max(1, int(config['mse_config'].get('prediction_match_tolerance_minutes', 5))))
            closest_val = None
            closest_dt = None
            for d in list(traffic_data)[-200:]:
                try:
                    dts = datetime.strptime(d.get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                if abs(dts - target_ts) <= tol:
                    if closest_dt is None or abs(dts - target_ts) < abs(closest_dt - target_ts):
                        closest_dt = dts
                        closest_val = d.get('traffic')
            return closest_val

        # Prefer load-based error
        for p in recent_predictions:
            p_load = p.get('predicted_load')
            a_load = p.get('actual_load')
            if a_load is None and p.get('match_timestamp'):
                a_load = _nearest_actual_load(p['match_timestamp'])
                if a_load is not None:
                    p['actual_load'] = a_load
            if p_load is not None and a_load is not None:
                err = float(p_load) - float(a_load)
                squared_errors.append(err * err)
                basis = 'load'

        # Fallback to replicas-based error if needed
        if not squared_errors:
            for p in recent_predictions:
                pred_val = p.get('predicted_replicas')
                actual_val = p.get('actual_replicas')
                if pred_val is not None and actual_val is not None:
                    error = (float(pred_val) - float(actual_val)) ** 2
                    squared_errors.append(error)
                    basis = 'replicas'

        if not squared_errors:
            logger.warning(f"{model_name} MSE: No valid data pairs (load or replicas)")
            return float('inf')

        mse_value = float(np.mean(squared_errors))
        logger.info(f"🎯 {model_name} MSE ({basis or 'unknown'}): {mse_value:.4f} (from {len(squared_errors)} samples)")
        logger.debug(f"{model_name} MSE window: {len(recent_predictions)} samples (window_size={window_size})")
        return mse_value
        
    except Exception as e:
        logger.error(f"Error calculating MSE for {model_name}: {e}")
        return float('inf')

def update_mse_metrics():
    """Enhanced MSE metric updates with better Prometheus integration and error resilience"""
    global gru_mse, holt_winters_mse, last_mse_calculation
    
    current_time = datetime.now()
    
    try:
        # Update predictions with actual values first (with timeout protection)
        update_predictions_with_actual_values()
        
        # Calculate MSE for all models (with fallback to previous values)
        previous_gru_mse = gru_mse
        previous_hw_mse = holt_winters_mse
        
        try:
            new_gru_mse = calculate_mse_robust('gru')
            if new_gru_mse != float('inf'):
                gru_mse = new_gru_mse
            # Keep previous value if calculation fails
        except Exception as e:
            logger.warning(f"GRU MSE calculation failed, keeping previous value: {e}")
        
        try:
            new_hw_mse = calculate_mse_robust('holt_winters')  
            if new_hw_mse != float('inf'):
                holt_winters_mse = new_hw_mse
            # Keep previous value if calculation fails
        except Exception as e:
            logger.warning(f"Holt-Winters MSE calculation failed, keeping previous value: {e}")
            
        # Ensemble removed
            
    except Exception as e:
        logger.error(f"MSE update process failed: {e}")
        # Don't update MSE values if the whole process fails
    
    # Update Prometheus metrics with stability protection
    try:
        if gru_mse != float('inf') and gru_mse >= 0:  # Allow MSE = 0.0 (perfect predictions)
            prediction_mse.labels(service=SERVICE_LABEL, model='gru').observe(gru_mse)
            current_mse.labels(service=SERVICE_LABEL, model='gru').set(gru_mse)
        elif gru_mse == float('inf'):
            current_mse.labels(service=SERVICE_LABEL, model='gru').set(-1)  # Signal insufficient data
        
        if holt_winters_mse != float('inf') and holt_winters_mse >= 0:  # Allow MSE = 0.0 (perfect predictions)
            prediction_mse.labels(service=SERVICE_LABEL, model='holt_winters').observe(holt_winters_mse)
            current_mse.labels(service=SERVICE_LABEL, model='holt_winters').set(holt_winters_mse)
        elif holt_winters_mse == float('inf'):
            current_mse.labels(service=SERVICE_LABEL, model='holt_winters').set(-1)  # Signal insufficient data
            
        # Ensemble metrics removed
            
    except Exception as e:
        logger.error(f"Failed to update Prometheus metrics: {e}")
    
    last_mse_calculation = current_time
    
    # Enhanced logging with anomaly detection
    gru_status = f"{gru_mse:.3f}" if gru_mse != float('inf') else "insufficient_data"
    hw_status = f"{holt_winters_mse:.3f}" if holt_winters_mse != float('inf') else "insufficient_data"
    # Detect and log MSE anomalies (sudden drops to 0 or -1)
    if previous_gru_mse != float('inf') and previous_gru_mse > 5 and gru_mse == float('inf'):
        logger.warning(f"🚨 GRU MSE dropped from {previous_gru_mse:.3f} to insufficient_data - possible calculation issue")
    
    if previous_hw_mse != float('inf') and previous_hw_mse > 5 and holt_winters_mse == float('inf'):
        logger.warning(f"🚨 Holt-Winters MSE dropped from {previous_hw_mse:.3f} to insufficient_data - possible calculation issue")
    
    logger.info(f"MSE Update - GRU: {gru_status}, Holt-Winters: {hw_status}")

def calculate_enhanced_mse(model_name):
    """Calculate multiple accuracy metrics beyond just MSE."""
    try:
        predictions = predictions_history[model_name]
        matched_predictions = [p for p in predictions if p.get('matched', False) and p['actual_replicas'] is not None]
        
        if len(matched_predictions) < 2:
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf'),
                'directional_accuracy': 0,
                'samples': len(matched_predictions)
            }
        
        # Extract predictions and actuals
        preds = [p['predicted_replicas'] for p in matched_predictions]
        actuals = [p['actual_replicas'] for p in matched_predictions]
        
        # Calculate multiple metrics
        mse = np.mean([(p - a) ** 2 for p, a in zip(preds, actuals)])
        mae = np.mean([abs(p - a) for p, a in zip(preds, actuals)])
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean([abs(p - a) / max(a, 1) * 100 for p, a in zip(preds, actuals)])
        
        # Directional accuracy (did we predict the right direction of change?)
        if len(matched_predictions) >= 3:
            correct_directions = 0
            total_directions = 0
            
            for i in range(1, len(matched_predictions)):
                pred_direction = preds[i] - preds[i-1]
                actual_direction = actuals[i] - actuals[i-1]
                
                if (pred_direction > 0 and actual_direction > 0) or \
                   (pred_direction < 0 and actual_direction < 0) or \
                   (pred_direction == 0 and actual_direction == 0):
                    correct_directions += 1
                total_directions += 1
            
            directional_accuracy = correct_directions / total_directions if total_directions > 0 else 0
        else:
            directional_accuracy = 0
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'samples': len(matched_predictions)
        }
        
    except Exception as e:
        logger.error(f"Error calculating enhanced metrics for {model_name}: {e}")
        return {
            'mse': float('inf'),
            'mae': float('inf'),
            'mape': float('inf'),
            'directional_accuracy': 0,
            'samples': 0
        }

# Ensemble prediction removed

def select_best_model_with_minheap():
    """MinHeap-based model selection using MSE as the primary criterion"""
    global gru_mse, holt_winters_mse, is_model_trained, gru_model
    
    # Create a min heap with (MSE, model_name, model_ready_status)
    model_heap = []
    
    # Add GRU to heap if available
    if is_model_trained and gru_model is not None:
        gru_mse_value = gru_mse if gru_mse != float('inf') else float('inf')
        heapq.heappush(model_heap, (gru_mse_value, 'gru', True))
        logger.debug(f"Added GRU to minheap: MSE={gru_mse_value}")
    else:
        # Add GRU with infinite MSE if not ready
        heapq.heappush(model_heap, (float('inf'), 'gru', False))
        logger.debug("Added GRU to minheap: MSE=inf (not ready)")
    
    # Add Holt-Winters to heap (always available)
    hw_mse_value = holt_winters_mse if holt_winters_mse != float('inf') else float('inf')
    heapq.heappush(model_heap, (hw_mse_value, 'holt_winters', True))
    logger.debug(f"Added Holt-Winters to minheap: MSE={hw_mse_value}")
    
    # Ensemble removed from minheap
    
    # Select models based on minheap (lowest MSE first)
    selected_models = []
    mse_threshold = config['mse_config']['mse_threshold_difference']
    
    while model_heap and len(selected_models) < 2:
        mse_value, model_name, is_ready = heapq.heappop(model_heap)
        
        if is_ready and mse_value != float('inf'):
            selected_models.append((mse_value, model_name))
            logger.info(f"🏆 MinHeap selected: {model_name} with MSE={mse_value:.4f}")
        elif is_ready and not selected_models:  # Fallback if no model has valid MSE
            selected_models.append((mse_value, model_name))
            logger.warning(f"⚠️ MinHeap fallback: {model_name} with MSE={mse_value}")
    
    if not selected_models:
        # Ultimate fallback
        model_selection.labels(service=SERVICE_LABEL, model='holt_winters', reason='minheap_ultimate_fallback').inc()
        logger.error("❌ MinHeap selection failed, using ultimate fallback")
        return 'holt_winters'
    
    # Use the best model (lowest MSE)
    best_mse, best_model = selected_models[0]
    
    # Check if MSE is disabled
    if not config['mse_config']['enabled']:
        if is_model_trained and gru_model is not None:
            model_selection.labels(service=SERVICE_LABEL, model='gru', reason='mse_disabled_prefer_gru').inc()
            return 'gru'
        else:
            model_selection.labels(service=SERVICE_LABEL, model='holt_winters', reason='mse_disabled_fallback').inc()
            return 'holt_winters'
    
    # Log minheap decision with simplified reason
    if best_mse < 1.0:
        reason = "low_mse_selected"
    elif best_mse < 5.0:
        reason = "medium_mse_selected"
    else:
        reason = "high_mse_selected"
    model_selection.labels(service=SERVICE_LABEL, model=best_model, reason=reason).inc()
    
    logger.info(f"🎯 MinHeap final selection: {best_model} (MSE: {best_mse:.4f})")
    return best_model

# Ensemble removed: no ensemble MSE calculation

def select_best_model():
    """Enhanced model selection using MinHeap system for MSE-based decisions"""
    return select_best_model_with_minheap()

def load_model_components():
    """Load GRU model and scalers with enhanced validation"""
    global gru_model, scaler_X, scaler_y, is_model_trained
    
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_FILE):
            logger.info("No existing GRU model file found")
            return False
        
        # Load model with error handling and compatibility fallback
        try:
            gru_model = tf.keras.models.load_model(MODEL_FILE)
            logger.info(f"GRU model loaded successfully from {MODEL_FILE}")
        except Exception as e:
            logger.error(f"Failed to load GRU model (likely TensorFlow compatibility issue): {e}")
            logger.info("🔧 Attempting to fix by removing incompatible model files...")
            
            # Remove incompatible model files to force retraining
            try:
                if os.path.exists(MODEL_FILE):
                    if os.path.isdir(MODEL_FILE):
                        import shutil
                        shutil.rmtree(MODEL_FILE)
                    else:
                        os.remove(MODEL_FILE)
                    logger.info(f"Removed incompatible model file: {MODEL_FILE}")
                
                if os.path.exists(SCALER_X_FILE):
                    os.remove(SCALER_X_FILE)
                    logger.info(f"Removed incompatible scaler file: {SCALER_X_FILE}")
                    
                if os.path.exists(SCALER_Y_FILE):
                    os.remove(SCALER_Y_FILE)
                    logger.info(f"Removed incompatible scaler file: {SCALER_Y_FILE}")
                
                logger.info("✅ Incompatible model files removed. Model will retrain with current TensorFlow version.")
                return False  # Return False to trigger retraining
                
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up incompatible model files: {cleanup_error}")
                return False
        
        # Load scalers with validation
        scalers_loaded = 0
        if os.path.exists(SCALER_X_FILE):
            try:
                scaler_X = joblib.load(SCALER_X_FILE)
                logger.info(f"Feature scaler loaded successfully from {SCALER_X_FILE}")
                scalers_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load feature scaler: {e}")
                gru_model = None
                return False
        else:
            logger.warning(f"Feature scaler file not found: {SCALER_X_FILE}")
            gru_model = None
            return False
        
        if os.path.exists(SCALER_Y_FILE):
            try:
                scaler_y = joblib.load(SCALER_Y_FILE)
                logger.info(f"Target scaler loaded successfully from {SCALER_Y_FILE}")
                scalers_loaded += 1
            except Exception as e:
                logger.error(f"Failed to load target scaler: {e}")
                gru_model = None
                return False
        else:
            logger.warning(f"Target scaler file not found: {SCALER_Y_FILE}")
            gru_model = None
            return False
        
        # Validate model compatibility
        try:
            # Test if model can make predictions
            input_shape = gru_model.input_shape
            test_input = np.random.random((1, input_shape[1], input_shape[2]))
            test_output = gru_model.predict(test_input, verbose=0)
            
            if test_output is None or len(test_output) == 0:
                raise ValueError("Model produces no output")
                
            logger.info(f"Model validation successful. Input shape: {input_shape}, Output shape: {test_output.shape}")
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            gru_model = None
            return False
        
        # Validate scaler compatibility
        try:
            if hasattr(scaler_X, 'n_features_in_') and scaler_X.n_features_in_ != input_shape[2]:
                logger.error(f"Scaler feature dimension mismatch: {scaler_X.n_features_in_} != {input_shape[2]}")
                gru_model = None
                return False
        except Exception as e:
            logger.warning(f"Could not validate scaler compatibility: {e}")
        
        # All components loaded successfully
        is_model_trained = True
        config['use_gru'] = True
        
        logger.info("🎯 ALL GRU COMPONENTS LOADED AND VALIDATED SUCCESSFULLY")
        logger.info(f"📊 Model input shape: {gru_model.input_shape}")
        logger.info(f"📊 Scaler features: {scaler_X.n_features_in_}")
        logger.info(f"📊 is_model_trained: {is_model_trained}")
        logger.info(f"📊 use_gru: {config['use_gru']}")
        
        # Create initial predictions for immediate MSE calculation if we have enough data
        if len(traffic_data) >= 12:
            try:
                logger.info("Creating initial GRU predictions for faster MSE calculation...")
                initial_predictions = predict_with_gru(1)
                if initial_predictions:
                    logger.info("Initial GRU predictions created successfully")
            except Exception as e:
                logger.warning(f"Failed to create initial predictions: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error loading model components: {e}")
        gru_model = None
        is_model_trained = False
        return False

def collect_mse_data_only():
    """Lightweight data collection only for MSE calculation after 4-hour training dataset is complete"""
    global is_collecting, predictions_history, last_mse_calculation
    
    logger.info("🎯 MSE-only mode: Collecting minimal data for prediction accuracy measurement")
    
    while is_collecting:
        try:
            if prometheus_client is None:
                initialize_prometheus()
                
            # Get only essential metrics for MSE calculation
            current_cpu = get_current_cpu_from_prometheus()
            
            replicas_query = f'kube_deployment_status_replicas{{deployment="{config["target_deployment"]}", namespace="{config["target_namespace"]}"}}'
            replicas_result = prometheus_client.custom_query(replicas_query)
            current_replicas = int(float(replicas_result[0]['value'][1])) if replicas_result else 1
            
            # Update actual replicas metric
            current_actual_replicas.labels(service=SERVICE_LABEL).set(current_replicas)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Always attempt to match predictions with actuals (even at low CPU) for accurate MSE
            try:
                match_new_data_with_predictions(current_replicas, current_cpu, timestamp)
            except Exception as e:
                logger.debug(f"MSE matching failed: {e}")
            
            # Generate fresh predictions for MSE tracking (using stored training dataset)
            if len(traffic_data) >= 12:  # Use stored training data
                try:
                    # Generate GRU prediction
                    if gru_model is not None and is_model_trained:
                        gru_pred = predict_with_gru(1)
                        if gru_pred:
                            logger.debug(f"MSE Mode - GRU prediction: {gru_pred}")
                    
                    # Generate Holt-Winters prediction  
                    hw_pred = predict_with_holtwinters(1)
                    if hw_pred:
                        logger.debug(f"MSE Mode - HW prediction: {hw_pred}")
                        
                except Exception as e:
                    logger.error(f"MSE mode prediction generation failed: {e}")
                        
            # Sleep longer in MSE mode (less frequent collection needed)
            time.sleep(config['collection_interval'] * 2)  # 2x normal interval
            
        except Exception as e:
            logger.error(f"Error in MSE collection mode: {e}")
            time.sleep(10)

def get_current_cpu_from_prometheus():
    """Get current CPU directly from Prometheus for status reporting"""
    try:
        if prometheus_client is None:
            initialize_prometheus()
        
        logger.info(f"🔍 DEBUG: Getting CPU for deployment='{config['target_deployment']}', namespace='{config['target_namespace']}'")
        
        # CPU queries using SAME method as Grafana dashboard for consistency
        # This matches Grafana: CPU Usage / CPU Limits * 100 = accurate percentage of allocated resources
        cpu_queries = [
            # Query 1: Sum of pod CPU usage / sum of pod CPU limits (1m window)
            {
                'query': f'sum(rate(container_cpu_usage_seconds_total{{namespace="{config["target_namespace"]}", pod=~"{config["target_deployment"]}-.*", container!="POD", image!=""}}[1m])) / sum(kube_pod_container_resource_limits{{namespace="{config["target_namespace"]}", pod=~"{config["target_deployment"]}-.*", resource="cpu"}}) * 100',
                'format': 'percentage',
                'description': 'CPU usage % of limits (sum over pods, 1m)'
            },
            # Query 2: Same as 1 with 5m window
            {
                'query': f'sum(rate(container_cpu_usage_seconds_total{{namespace="{config["target_namespace"]}", pod=~"{config["target_deployment"]}-.*", container!="POD", image!=""}}[5m])) / sum(kube_pod_container_resource_limits{{namespace="{config["target_namespace"]}", pod=~"{config["target_deployment"]}-.*", resource="cpu"}}) * 100',
                'format': 'percentage',
                'description': 'CPU usage % of limits (sum over pods, 5m)'
            },
            # Query 3: Use CPU requests if limits are unset
            {
                'query': f'sum(rate(container_cpu_usage_seconds_total{{namespace="{config["target_namespace"]}", pod=~"{config["target_deployment"]}-.*", container!="POD", image!=""}}[1m])) / sum(kube_pod_container_resource_requests{{namespace="{config["target_namespace"]}", pod=~"{config["target_deployment"]}-.*", resource="cpu"}}) * 100',
                'format': 'percentage',
                'description': 'CPU usage % of requests (1m)'
            },
            # Query 4: Namespace-agnostic fallback
            {
                'query': f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{config["target_deployment"]}-.*", container!="POD", image!=""}}[1m])) / sum(kube_pod_container_resource_limits{{pod=~"{config["target_deployment"]}-.*", resource="cpu"}}) * 100',
                'format': 'percentage',
                'description': 'CPU usage % of limits (no namespace filter)'
            },
            # Query 5: OLD METHOD (absolute cores) - kept as ultimate fallback
            {
                'query': f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{config["target_deployment"]}-.*", container!="POD", image!=""}}[5m])) * 100',
                'format': 'cores_converted',
                'description': 'OLD: CPU cores usage * 100 (fallback only)'
            },
            # Query 6: Node-level CPU as last resort
            {
                'query': f'100 - (avg(irate(node_cpu_seconds_total{{mode="idle"}}[5m])) * 100)',
                'format': 'percentage',
                'description': 'Node-level CPU usage percentage (last resort)'
            }
        ]
        
        for i, query_info in enumerate(cpu_queries, 1):
            try:
                query = query_info['query']
                expected_format = query_info['format']
                description = query_info['description']
                
                logger.info(f"🔍 Trying CPU query {i}: {description}")
                logger.info(f"🔍 Query {i} PromQL: {query}")
                
                cpu_result = prometheus_client.custom_query(query)
                logger.info(f"🔍 Query {i} raw result: {cpu_result}")
                
                if cpu_result and len(cpu_result) > 0:
                    cpu_value = float(cpu_result[0]['value'][1])
                    logger.info(f"✅ CPU query {i} returned: {cpu_value:.4f} (raw value)")
                    
                    # Process value based on expected format
                    logger.info(f"📊 Query {i} analysis: raw_value={cpu_value}, expected_format={expected_format}")
                    
                    if expected_format == 'percentage':
                        # For percentage queries (queries 1-4, 6), value should already be percentage
                        # If it's very small (< 1), it might be in decimal format, convert
                        if 0 <= cpu_value <= 1:
                            logger.warning(f"⚠️ CPU {cpu_value:.6f} appears to be decimal format, converting to percentage")
                            cpu_value = cpu_value * 100
                            logger.info(f"🔧 Converted CPU value: {cpu_value:.4f}%")
                        
                        # Accept percentage values
                        if cpu_value >= 0:
                            logger.info(f"🎯 Selected CPU query {i} ({description}): {cpu_value:.4f}%")
                            # Cache and return
                            globals()['_last_cpu_value'] = cpu_value
                            globals()['_last_cpu_time'] = datetime.now()
                            return cpu_value
                            
                    elif expected_format == 'cores_converted':
                        # For old-style cores * 100 queries (query 5), this is CPU cores used
                        # If value is high (>100), it's likely already been multiplied by 100
                        # If value is low (<10), it's likely raw cores, so it's a meaningful percentage
                        logger.info(f"🔧 Processing cores format: {cpu_value:.4f}")
                        if cpu_value >= 0:
                            logger.info(f"🎯 Selected CPU query {i} ({description}): {cpu_value:.4f}% (cores-based)")
                            globals()['_last_cpu_value'] = cpu_value
                            globals()['_last_cpu_time'] = datetime.now()
                            return cpu_value
                    
                    else:
                        # Unknown format, accept if reasonable
                        if cpu_value >= 0:
                            logger.info(f"🎯 Selected CPU query {i} ({description}): {cpu_value:.4f}% (unknown format)")
                            globals()['_last_cpu_value'] = cpu_value
                            globals()['_last_cpu_time'] = datetime.now()
                            return cpu_value
                else:
                    logger.warning(f"❌ CPU query {i} returned no results")
            except Exception as query_error:
                logger.error(f"💥 CPU query {i} failed: {query_error}")
        
        # If all specific queries fail, try to get any available CPU metrics
        logger.warning("🔍 All targeted queries failed, trying to discover available metrics...")
        try:
            # Try to list available metrics
            all_metrics = prometheus_client.custom_query('up')
            logger.info(f"🔍 Prometheus connection test result: {len(all_metrics) if all_metrics else 0} metrics found")
            
            # Try to discover all pods with CPU metrics
            all_pods_cpu = prometheus_client.custom_query('container_cpu_usage_seconds_total')
            logger.info(f"🔍 All CPU metrics available: {len(all_pods_cpu) if all_pods_cpu else 0}")
            
            if all_pods_cpu:
                # Show first few pod names for debugging
                pod_names = set()
                for metric in all_pods_cpu[:10]:  # Limit to first 10
                    if 'pod' in metric['metric']:
                        pod_names.add(metric['metric']['pod'])
                logger.info(f"🔍 Available pods with CPU metrics: {list(pod_names)[:5]}")
                
                # Check specifically for our target deployment
                target_pods = [m for m in all_pods_cpu if 'pod' in m['metric'] and config['target_deployment'] in m['metric']['pod']]
                logger.info(f"🔍 Target pods found: {len(target_pods)}")
                if target_pods:
                    for pod in target_pods[:3]:  # Show first 3 matching pods
                        logger.info(f"🔍 Target pod: {pod['metric'].get('pod', 'unknown')} = {pod['value'][1]}")
                
        except Exception as discovery_error:
            logger.error(f"💥 Metric discovery failed: {discovery_error}")
        
        # If all queries fail, return last known CPU if recent
        if _last_cpu_time and (datetime.now() - _last_cpu_time).total_seconds() < 120:
            logger.warning("Using cached CPU value due to query failures")
            return _last_cpu_value
        logger.error("💥 All CPU queries failed, returning 0.0")
        return 0.0
        
    except Exception as e:
        logger.error(f"💥 Error getting current CPU: {e}")
        # Return cached value if not too old to avoid blocking
        if _last_cpu_time and (datetime.now() - _last_cpu_time).total_seconds() < 120:
            return _last_cpu_value
        return 0.0

def collect_metrics_from_prometheus():
    """Collect real metrics from Prometheus continuously (no synthetic/MSE-only mode)."""
    global traffic_data, is_collecting, last_training_time, low_traffic_start_time, consecutive_low_cpu_count
    global data_collection_complete, data_collection_start_time, gru_model, is_model_trained
    
    # Continue normal collection even after 4-hour milestone (no mode switch)
    if data_collection_complete:
        logger.info("4-hour training dataset milestone reached. Continuing real data collection (no reset, no synthetic).")
    
    # Set collection start time if not already set
    if data_collection_start_time is None:
        data_collection_start_time = datetime.now()
        save_collection_status()
        logger.info(f"📅 Started 4-hour data collection period at {data_collection_start_time}")
        logger.info("🎓 AWS Academy Compatible: Collection based on calendar time, survives system restarts")
    else:
        elapsed_hours = (datetime.now() - data_collection_start_time).total_seconds() / 3600
        remaining_hours = max(0, 4 - elapsed_hours)
        logger.info(f"📅 Continuing data collection: {elapsed_hours:.1f}h elapsed, {remaining_hours:.1f}h remaining")
    
    iteration_count = 0
    max_iterations = 2000  # Safety limit: ~33 hours at 60s intervals
    
    while is_collecting and iteration_count < max_iterations:
        iteration_count += 1
        try:
            if prometheus_client is None:
                initialize_prometheus()
                
            # Get current CPU utilization using the improved detection function
            current_cpu = get_current_cpu_from_prometheus()
            
            # Track low CPU occurrences for better scale-down detection
            if current_cpu < config['cost_optimization']['scale_down_threshold']:
                consecutive_low_cpu_count += 1
            else:
                consecutive_low_cpu_count = 0
            
            # Memory usage
            memory_query = f'avg(container_memory_usage_bytes{{pod=~"{config["target_deployment"]}-.*"}}) / 1024 / 1024'
            memory_result = prometheus_client.custom_query(memory_query)
            current_memory = float(memory_result[0]['value'][1]) if memory_result else 0
            
            # Request rate (traffic)
            request_query = f'sum(rate(nginx_ingress_controller_requests{{service=~"{config["target_deployment"]}"}}[5m])) * 60'
            request_result = prometheus_client.custom_query(request_query)
            current_traffic = float(request_result[0]['value'][1]) if request_result else 0
            
            # Current replicas
            replicas_query = f'kube_deployment_status_replicas{{deployment="{config["target_deployment"]}", namespace="{config["target_namespace"]}"}}'
            replicas_result = prometheus_client.custom_query(replicas_query)
            current_replicas = int(float(replicas_result[0]['value'][1])) if replicas_result else 1
            
            # Update actual replicas metric
            current_actual_replicas.labels(service=SERVICE_LABEL).set(current_replicas)
            
            # Track idle time
            if current_traffic < config['cost_optimization']['zero_traffic_threshold'] and current_cpu < config['cpu_threshold']:
                if low_traffic_start_time is None:
                    low_traffic_start_time = datetime.now()
                    logger.info("Started tracking low traffic period")
            else:
                if low_traffic_start_time is not None:
                    logger.info(f"Low traffic period ended after {(datetime.now() - low_traffic_start_time).total_seconds() / 60:.1f} minutes")
                low_traffic_start_time = None
            
            # Update Prometheus metrics
            cpu_utilization.set(current_cpu)
            traffic_gauge.set(current_traffic)
            
            # Store the data - collect all points, but flag if below CPU threshold
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_collected = False
            # Always append to maintain a continuous series for forecasting and matching
            traffic_data.append({
                'timestamp': timestamp,
                'traffic': current_traffic,
                'cpu_utilization': current_cpu,
                'memory_usage': current_memory,
                'replicas': current_replicas,
                'idle': current_cpu < config['cpu_threshold']
            })
            data_collected = True
            if current_cpu >= config['cpu_threshold']:
                logger.debug(f"Data collected: CPU={current_cpu:.1f}% (above threshold {config['cpu_threshold']}%)")
            else:
                logger.debug(f"Data collected (idle): CPU={current_cpu:.1f}% (below threshold {config['cpu_threshold']}%)")
            
            # Keep only most recent points within retention window and limit to 240 points
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)

            original_size = len(traffic_data)
            filtered = [d for d in list(traffic_data) if 
                        datetime.strptime(d['timestamp'], "%Y-%m-%d %H:%M:%S") > cutoff_time]
            traffic_data.clear()
            if filtered:
                # Keep only the last 240 entries to enforce fixed-size buffer
                traffic_data.extend(filtered[-240:])

            ensure_traffic_buffer_integrity()

            if original_size != len(traffic_data):
                logger.debug(f"Cleaned traffic data: {original_size} -> {len(traffic_data)} points")
            
            # Only do prediction matching and generation if we collected data
            if data_collected:
                # Try to match any unmatched predictions with the new data point
                try:
                    match_new_data_with_predictions(current_replicas, current_cpu, timestamp)
                except Exception as e:
                    logger.debug(f"Failed to match new data with predictions: {e}")
                
                # Generate predictions from both models for MSE tracking (EVERY data point after minimum)
                if len(traffic_data) >= 12:
                    try:
                        # ALWAYS make Holt-Winters prediction
                        logger.info("🔄 Generating background Holt-Winters prediction for MSE tracking...")
                        hw_pred = predict_with_holtwinters(1)
                        
                        if hw_pred:
                            logger.info(f"✅ HW prediction: {hw_pred}")
                        else:
                            logger.error("❌ HW prediction failed!")
                        
                        # ALWAYS make GRU prediction if available (single call)
                        if gru_model is not None and is_model_trained:
                            logger.info("🔄 Generating background GRU prediction for MSE tracking...")
                            gru_pred = predict_with_gru(1)
                                
                            if gru_pred:
                                logger.info(f"✅ GRU prediction: {gru_pred}")
                            else:
                                logger.error("❌ GRU prediction failed!")
                        else:
                            logger.info(f"❌ GRU not available: model={gru_model is not None}, trained={is_model_trained}")
                            
                    except Exception as e:
                        logger.error(f"❌ Background prediction generation failed: {e}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                
                # Emergency MSE system: create synthetic predictions if GRU has never made any
                if len(predictions_history['gru']) == 0 and len(traffic_data) > 20 and is_model_trained:
                    try:
                        logger.warning("🆘 EMERGENCY: Creating synthetic GRU predictions for MSE testing")
                        create_emergency_gru_predictions()
                    except Exception as e:
                        logger.error(f"Emergency prediction creation failed: {e}")
                        
                # Periodically save data
                if len(traffic_data) % 10 == 0:
                    save_data()
            else:
                logger.debug(f"⏸️  Monitoring only - CPU {current_cpu:.1f}% below threshold {config['cpu_threshold']}%")
            
            # Enhanced GRU model training logic
            minutes_elapsed = (current_time - start_time).total_seconds() / 60
            gru_config = config["models"]["gru"]
            
            # Check training conditions with better logging
            should_train = False
            training_reason = ""
            
            # Initial training conditions - REDUCED REQUIREMENTS FOR AUTO-TRAINING
            if not is_model_trained:
                if minutes_elapsed >= config['training_threshold_minutes']:
                    # 🚀 REDUCED: was +10,20 now +2,15 to match manual training success
                    min_data_required = max(int(gru_config["look_back"]) + 2, 15)
                    if len(traffic_data) >= min_data_required:
                        # Use consistent CPU threshold for training
                        if current_cpu >= config['cpu_threshold']:
                            should_train = True
                            training_reason = f"initial_training (elapsed: {minutes_elapsed:.1f}min, data: {len(traffic_data)}, cpu: {current_cpu:.1f}%)"
                        else:
                            logger.debug(f"Waiting for CPU threshold: {current_cpu:.1f}% < {config['cpu_threshold']}%")
                    else:
                        logger.debug(f"Waiting for more data: {len(traffic_data)} < {min_data_required}")
                else:
                    logger.debug(f"Waiting for time threshold: {minutes_elapsed:.1f} < {config['training_threshold_minutes']} minutes")
            
            # Retrain conditions
            elif is_model_trained and current_cpu >= config['cpu_threshold']:
                if last_training_time is None or (current_time - last_training_time).total_seconds() > 3600:
                    # 🚀 REDUCED: was +10,20 now +2,15 for consistent requirements
                    min_data_required = max(int(gru_config["look_back"]) + 2, 15)
                    if len(traffic_data) >= min_data_required:
                        should_train = True
                        training_reason = f"retrain (last: {last_training_time}, data: {len(traffic_data)})"
            
            # Execute training if conditions are met - ASYNC to prevent Flask blocking
            if should_train and not is_training:
                logger.info(f"🚀 Starting ASYNC GRU training: {training_reason}")
                try:
                    # Create thread-safe copy of traffic data for training
                    traffic_data_copy = list(traffic_data)
                    was_first_training = not is_model_trained
                    
                    # Start async training in separate thread
                    global training_thread
                    training_thread = threading.Thread(
                        target=async_model_training, 
                        args=(training_reason, traffic_data_copy, len(traffic_data_copy) >= 30)
                    )
                    training_thread.daemon = True
                    training_thread.start()
                    
                    # Config will be updated after successful training
                    
                    logger.info("🔄 ASYNC training started - Flask metrics remain unblocked!")
                    
                except Exception as e:
                    logger.error(f"❌ ASYNC training startup error: {e}")
            elif should_train and is_training:
                logger.debug("⏳ Training conditions met but training already in progress")
            
            # Check if 4 hours of actual runtime has accumulated (reduced for testing)
            if data_collection_start_time:
                # Calculate total runtime across all sessions by counting data points
                # Each data point represents ~1 minute of runtime (collection_interval = 60s)
                runtime_hours = len(traffic_data) / 60  # Approximate runtime hours based on data points
                
                if runtime_hours >= 4 and not data_collection_complete:
                    # Mark milestone and snapshot a training dataset, but DO NOT reset or stop collection
                    data_collection_complete = True
                    global training_dataset
                    training_dataset = list(traffic_data)
                    save_collection_status()
                    save_data()
                    logger.info(f"✅ 4-hour runtime milestone reached. Snapshot training dataset size: {len(training_dataset)}. Continuing collection.")
            
            # Sleep for the collection interval
            time.sleep(config['collection_interval'])
            
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            time.sleep(5)  # Sleep for a short time on error
    
    # Safety check: log if we hit the iteration limit
    if iteration_count >= max_iterations:
        logger.warning(f"🚨 Data collection stopped due to safety limit ({max_iterations} iterations)")

def preprocess_data(data, sequence_length=100):
    """Preprocess data for GRU model with enhanced error handling."""
    if len(data) < sequence_length + 1:
        logger.warning(f"Not enough data for preprocessing: {len(data)} < {sequence_length + 1}")
        return None, None
    
    try:
        df = pd.DataFrame(data)
        X, y = [], []
        
        # Ensure required columns exist
        required_features = ['traffic', 'cpu_utilization', 'replicas']
        for feature in required_features:
            if feature not in df.columns:
                logger.error(f"Missing required feature: {feature}")
                return None, None
        
        # Use multiple features
        features = ['traffic', 'cpu_utilization']
        if 'memory_usage' in df.columns and df['memory_usage'].notna().any():
            features.append('memory_usage')
        
        # Check for valid data
        feature_data = df[features].values
        replica_data = df['replicas'].values.reshape(-1, 1)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(feature_data)) or np.any(np.isinf(feature_data)):
            logger.warning("NaN or infinite values found in feature data, cleaning...")
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=1000.0, neginf=0.0)
        
        if np.any(np.isnan(replica_data)) or np.any(np.isinf(replica_data)):
            logger.warning("NaN or infinite values found in replica data, cleaning...")
            replica_data = np.nan_to_num(replica_data, nan=1.0, posinf=10.0, neginf=1.0)
        
        # Ensure replica data is valid (between 1 and 10)
        replica_data = np.clip(replica_data, 1, 10)
        
        # Fit and transform with error handling
        try:
            feature_scaled = scaler_X.fit_transform(feature_data)
            replica_scaled = scaler_y.fit_transform(replica_data)
        except Exception as e:
            logger.error(f"Error during scaling: {e}")
            return None, None
        
        # Create sequences
        for i in range(len(data) - sequence_length):
            X.append(feature_scaled[i:i+sequence_length])
            y.append(replica_scaled[i+sequence_length])
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No sequences created during preprocessing")
            return None, None
            
        logger.info(f"Preprocessed data: {len(X)} sequences, {len(features)} features")
        return np.array(X), np.array(y)
        
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return None, None

def build_gru_model_with_data(data_copy):
    """Thread-safe GRU model training with provided data copy."""
    global gru_model, last_training_time, is_model_trained, scaler_X, scaler_y
    
    # Temporarily replace global traffic_data with our copy for preprocessing
    original_traffic_data = list(traffic_data)
    # Replace contents of the global deque with the provided data copy (trim to 240)
    traffic_data.clear()
    if data_copy:
        traffic_data.extend(list(data_copy)[-240:])
    
    try:
        # Use the existing build_gru_model function which is already complete
        result = build_gru_model()
        logger.info(f"✅ Thread-safe GRU training result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Thread-safe GRU training failed: {e}")
        return False
    finally:
        # Always restore original traffic data
        traffic_data.clear()
        if original_traffic_data:
            traffic_data.extend(original_traffic_data[-240:])

def build_gru_model():
    """Build and train GRU model with enhanced error handling and validation."""
    global gru_model, last_training_time, is_model_trained, scaler_X, scaler_y
    
    start_time = time.time()
    gru_config = config["models"]["gru"]
    
    try:
        logger.info(f"Starting GRU model training with {len(traffic_data)} data points")
        
        # Check minimum data requirements (reduced)
        min_data_points = max(int(gru_config["look_back"]) + 5, 15)
        if len(traffic_data) < min_data_points:
            logger.warning(f"Insufficient data for GRU training: {len(traffic_data)} < {min_data_points}")
            elapsed_ms = (time.time() - start_time) * 1000
            training_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)
            return False
        
        # Preprocess data
        X, y = preprocess_data(traffic_data, sequence_length=int(gru_config["look_back"]))
        
        if X is None or y is None:
            logger.error("Failed to preprocess data for GRU model")
            elapsed_ms = (time.time() - start_time) * 1000
            training_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)
            return False
            
        if len(X) < 10:
            logger.warning(f"Not enough sequences for GRU training: {len(X)} < 10")
            elapsed_ms = (time.time() - start_time) * 1000
            training_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)
            return False
        
        logger.info(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
        
        # Build model architecture
        input_shape = (X.shape[1], X.shape[2])
        logger.info(f"Building GRU model with input shape: {input_shape}")
        
        # Clear any existing model
        if gru_model is not None:
            del gru_model
            tf.keras.backend.clear_session()
        
        model = Sequential([
            GRU(32, return_sequences=True, input_shape=input_shape, name='gru_1'),
            Dropout(0.2, name='dropout_1'),
            GRU(16, return_sequences=False, name='gru_2'),
            Dropout(0.2, name='dropout_2'),
            Dense(8, activation='relu', name='dense_1'),
            Dense(1, name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model compiled, starting training...")
        
        # Train model with validation
        batch_size = min(int(gru_config["batch_size"]),
                         len(X) // 4)
        epochs = min(int(gru_config["epochs"]),
                     50)  # Limit epochs to prevent overfitting
        
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            shuffle=True
        )
        
        # Validate training results
        final_loss = history.history['loss'][-1]
        logger.info(f"Training completed. Final loss: {final_loss:.4f}")
        
        if final_loss > 10.0:  # High loss indicates poor training
            logger.warning(f"High training loss detected: {final_loss:.4f}")
        
        # Test prediction to ensure model works
        try:
            test_pred = model.predict(X[:1], verbose=0)
            test_pred_scaled = scaler_y.inverse_transform(test_pred)
            logger.info(f"Test prediction successful: {test_pred_scaled[0][0]:.2f}")
        except Exception as e:
            logger.error(f"Test prediction failed: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            training_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)
            return False
        
        # Save model and scalers
        try:
            model.save(MODEL_FILE)
            joblib.dump(scaler_X, SCALER_X_FILE)
            joblib.dump(scaler_y, SCALER_Y_FILE)
            logger.info(f"Model and scalers saved successfully to {MODEL_FILE}, {SCALER_X_FILE}, {SCALER_Y_FILE}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            training_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)
            return False
        
        # Update global variables
        gru_model = model
        is_model_trained = True
        last_training_time = datetime.now()
        
        # Record training time
        elapsed_ms = (time.time() - start_time) * 1000
        training_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)
        
        logger.info(f"🎯 GRU MODEL TRAINED SUCCESSFULLY in {elapsed_ms:.2f}ms with {len(X)} samples")
        logger.info(f"📊 Final model state: input_shape={gru_model.input_shape}, is_trained={is_model_trained}")
        
        # Create initial predictions for immediate MSE calculation
        try:
            logger.info("🚀 Creating initial GRU predictions for faster MSE calculation...")
            initial_predictions = predict_with_gru(1)
            if initial_predictions:
                logger.info(f"✅ Initial GRU predictions created successfully: {initial_predictions}")
            else:
                logger.error("❌ Initial GRU predictions failed!")
        except Exception as e:
            logger.error(f"❌ Failed to create initial predictions after training: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error building GRU model: {e}")
        elapsed_ms = (time.time() - start_time) * 1000
        training_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)
        return False



def predict_with_holtwinters(steps=None):
    """Predict using Holt-Winters with original hyperparameters."""
    start_time_func = time.time()
    hw_config = config["models"]["holt_winters"]

    if steps is None:
        steps = int(hw_config["look_forward"])

    try:
        # Always use recent sliding window of traffic_data (paper approach)
        prediction_data = list(traffic_data)

        if len(prediction_data) < 5:  # Reduced minimum for faster testing
            logger.info(f"Not enough data for Holt-Winters prediction: {len(prediction_data)} < 5")
            return None

        # Use replicas history (last 5 minutes)
        replicas = [d['replicas'] for d in prediction_data[-60:]]

        # If replicas series is constant or insufficiently variable, fall back to CPU-based scaling heuristic
        if len(replicas) >= 3 and (max(replicas) - min(replicas) == 0):
            try:
                current = prediction_data[-1]
                current_replicas = int(current.get('replicas', 1))
                current_cpu = float(current.get('cpu_utilization', 0))
                target_cpu = float(config.get('cost_optimization', {}).get('target_cpu_utilization', 75))
                # Compute recommended replicas to achieve target CPU
                rec = max(1, int(np.ceil((current_cpu / max(target_cpu, 1e-6)) * current_replicas / 100.0 * 100)))
                # Clip to bounds
                rec = max(config['cost_optimization'].get('min_replicas', 1), min(config['cost_optimization'].get('max_replicas', 10), rec))
                forecast = [rec] * steps
                # Record for next interval to track MSE
                current_time = datetime.now()
                future_timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
                # Avoid duplicate records for the exact same future timestamp
                try:
                    last_hw = predictions_history.get('holt_winters', [])[-1] if predictions_history.get('holt_winters') else None
                    if not last_hw or last_hw.get('timestamp') != future_timestamp:
                        add_prediction_to_history('holt_winters', forecast[0], future_timestamp)
                    else:
                        logger.debug("Skipping duplicate HW prediction record for same future timestamp")
                except Exception:
                    add_prediction_to_history('holt_winters', forecast[0], future_timestamp)
                elapsed_ms = (time.time() - start_time_func) * 1000
                prediction_time.labels(service=SERVICE_LABEL, model="holt_winters").set(elapsed_ms)
                logger.info(f"Holt-Winters CPU-fallback forecast: {forecast} (cpu={current_cpu:.1f}%, replicas={current_replicas}, target={target_cpu}%)")
                return forecast
            except Exception as e:
                logger.debug(f"HW CPU-fallback failed, continuing with time-series model: {e}")

        # Fit Holt-Winters with adaptive seasonality based on data availability
        seasonal_periods = int(hw_config["slen"])

        if len(replicas) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                replicas,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                damped_trend=True
            ).fit(
                smoothing_level=float(hw_config["alpha"]),
                smoothing_trend=float(hw_config["beta"]),
                smoothing_seasonal=float(hw_config["gamma"])
            )
        else:
            # Use non-seasonal model for small datasets
            logger.info(
                f"Using non-seasonal Holt-Winters (data points: {len(replicas)}, need {2 * seasonal_periods} for seasonal)"
            )
            model = ExponentialSmoothing(
                replicas,
                trend='add',
                damped_trend=True
            ).fit(
                smoothing_level=float(hw_config["alpha"]),
                smoothing_trend=float(hw_config["beta"])
            )

        forecast = model.forecast(steps).tolist()

        # Round and clip predictions
        forecast = [max(config['cost_optimization'].get('min_replicas', 1), min(config['cost_optimization'].get('max_replicas', 10), round(p))) for p in forecast]

        # Record prediction for the NEXT interval only (future timestamp) to avoid self-matching
        current_time = datetime.now()
        future_timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
        # Avoid duplicate records for the exact same future timestamp
        try:
            last_hw = predictions_history.get('holt_winters', [])[-1] if predictions_history.get('holt_winters') else None
            if not last_hw or last_hw.get('timestamp') != future_timestamp:
                add_prediction_to_history('holt_winters', forecast[0], future_timestamp)
            else:
                logger.debug("Skipping duplicate HW prediction record for same future timestamp")
        except Exception:
            add_prediction_to_history('holt_winters', forecast[0], future_timestamp)

        elapsed_ms = (time.time() - start_time_func) * 1000
        prediction_time.labels(service=SERVICE_LABEL, model="holt_winters").set(elapsed_ms)

        logger.info(f"Holt-Winters forecast generated: {forecast}")
        return forecast

    except Exception as e:
        logger.error(f"Error in Holt-Winters prediction: {e}")
        return None

def predict_with_gru(steps=None):
    """Enhanced GRU prediction with better error handling and validation."""
    global gru_model, scaler_X, scaler_y
    start_time_func = time.time()

    gru_config = config["models"]["gru"]

    if steps is None:
        steps = int(gru_config["look_forward"])

    # Validate model availability
    if gru_model is None or not is_model_trained:
        logger.error(f"🚨 GRU PREDICTION BLOCKED: model={gru_model is not None}, trained={is_model_trained}")
        return None

    # Use training dataset if collection is complete, otherwise use current data
    source_data = training_dataset if data_collection_complete and training_dataset else traffic_data
    prediction_data = list(source_data)

    logger.info(
        f"🔄 GRU prediction starting: model_loaded={gru_model is not None}, data_points={len(prediction_data)}, using_training_data={data_collection_complete}"
    )
    look_back = int(gru_config["look_back"])

    if len(prediction_data) < look_back:
        logger.warning(f"Not enough data for GRU prediction: {len(prediction_data)} < {look_back}")
        return None

    try:
        # Prepare recent data with validation
        recent_data = prediction_data[-look_back:]
        df = pd.DataFrame(recent_data)

        # Ensure required columns exist
        required_features = ['traffic', 'cpu_utilization']
        for feature in required_features:
            if feature not in df.columns:
                logger.error(f"Missing required feature for prediction: {feature}")
                return None

        # Use same features as training
        features = ['traffic', 'cpu_utilization']
        if 'memory_usage' in df.columns and df['memory_usage'].notna().any():
            features.append('memory_usage')

        # Extract and validate feature data
        X = df[features].values

        # Check for NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("NaN or infinite values found in prediction data, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=0.0)

        # Validate scaler compatibility
        if X.shape[1] != scaler_X.n_features_in_:
            logger.error(f"🚨 SCALER MISMATCH: got {X.shape[1]} features, expected {scaler_X.n_features_in_}")
            logger.error(f"Available features: {features}")
            logger.error(f"Data shape: {X.shape}")
            return None

        logger.info(f"✅ Scaler validation passed: {X.shape[1]} features match expected {scaler_X.n_features_in_}")

        # Scale and reshape
        try:
            X_scaled = scaler_X.transform(X)
            X_seq = X_scaled.reshape(1, look_back, len(features))
        except Exception as e:
            logger.error(f"Error during data scaling for prediction: {e}")
            return None

        # Validate input shape
        expected_shape = gru_model.input_shape
        if X_seq.shape[1:] != expected_shape[1:]:
            logger.error(f"🚨 MODEL SHAPE MISMATCH: input={X_seq.shape[1:]}, expected={expected_shape[1:]}")
            logger.error(f"Full input shape: {X_seq.shape}, model expects: {expected_shape}")
            return None

        logger.info(f"✅ Model shape validation passed: {X_seq.shape} matches {expected_shape}")

        # Initialize prediction variables
        predicted_replicas = 2  # Default fallback value
        raw_prediction = 0.0

        # Make prediction
        try:
            y_pred_scaled = gru_model.predict(X_seq, verbose=0)

            # Validate prediction output
            if y_pred_scaled is None or len(y_pred_scaled) == 0:
                logger.error("Empty prediction from GRU model")
                return None

            # Inverse transform
            y_pred = scaler_y.inverse_transform(y_pred_scaled)

            # Validate and sanitize prediction
            raw_prediction = float(y_pred[0][0])
            if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                logger.warning(f"Invalid prediction value: {raw_prediction}, using fallback")
                predicted_replicas = 2  # Safe fallback
            else:
                predicted_replicas = max(1, min(10, round(raw_prediction)))

        except Exception as e:
            logger.error(f"Error during GRU model prediction: {e}")
            return None

        # Record prediction for NEXT interval only (future timestamp) to avoid self-matching
        current_time = datetime.now()
        future_timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('gru', predicted_replicas, future_timestamp)

        # For multi-step prediction, use the single prediction
        predictions = [predicted_replicas] * steps

        # Record prediction time
        elapsed_ms = (time.time() - start_time_func) * 1000
        prediction_time.labels(service=SERVICE_LABEL, model="gru").set(elapsed_ms)

        logger.info(f"🎯 GRU forecast SUCCESS: {predictions} (raw: {raw_prediction:.3f})")
        return predictions

    except Exception as e:
        logger.error(f"🚨 UNEXPECTED GRU ERROR: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return None

def optimize_holtwinters_parameters(data, test_size=0.3):
    """Automatically optimize Holt-Winters parameters for minimum MSE."""
    try:
        if len(data) < 24:
            return None
        
        replicas_series = pd.Series([d['replicas'] for d in data])
        
        # Split data for parameter optimization
        train_size = int(len(replicas_series) * (1 - test_size))
        train_data = replicas_series[:train_size]
        test_data = replicas_series[train_size:]
        
        if len(train_data) < 12 or len(test_data) < 3:
            return None
        
        best_params = None
        best_mse = float('inf')
        
        # Grid search for optimal parameters (reduced for speed)
        alpha_values = [0.3, 0.5, 0.7]
        beta_values = [0.1, 0.3]
        gamma_values = [0.3, 0.5, 0.7]
        
        logger.info("Optimizing Holt-Winters parameters...")
        
        for alpha in alpha_values:
            for beta in beta_values:
                for gamma in gamma_values:
                    try:
                        # Fit model with current parameters
                        model = ExponentialSmoothing(
                            train_data,
                            seasonal_periods=min(12, len(train_data) // 2),
                            trend='add',
                            seasonal='add',
                            damped_trend=True
                        ).fit(
                            smoothing_level=alpha,
                            smoothing_trend=beta,
                            smoothing_seasonal=gamma,
                            optimized=False
                        )
                        
                        # Make predictions on test set
                        forecast = model.forecast(len(test_data))
                        mse = np.mean((forecast - test_data) ** 2)
                        
                        if mse < best_mse:
                            best_mse = mse
                            best_params = {
                                'alpha': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'mse': mse
                            }
                    
                    except Exception:
                        continue
        
        if best_params:
            logger.info(f"✅ Optimized HW parameters: α={best_params['alpha']}, β={best_params['beta']}, γ={best_params['gamma']}, MSE={best_params['mse']:.3f}")
            return best_params
        else:
            logger.warning("Parameter optimization failed, using defaults")
            return None
            
    except Exception as e:
        logger.error(f"Error optimizing Holt-Winters parameters: {e}")
        return None





    

    


def make_scaling_decision_with_minheap(predictions):
    """MinHeap-based scaling decision using MSE-ranked model predictions"""
    global last_scaling_time, last_scale_up_time, last_scale_down_time, low_traffic_start_time, consecutive_low_cpu_count
    
    if not traffic_data:
        return "maintain", 1
    
    current_time = time.time()
    current_metrics = traffic_data[-1]
    current_replicas = current_metrics['replicas']
    current_cpu = current_metrics['cpu_utilization']
    current_traffic = current_metrics['traffic']
    
    cost_config = config.get('cost_optimization', {})
    
    # Create decision heap: (priority_score, decision, replicas, reason)
    decision_heap = []
    
    # Basic cost optimization decisions (highest priority for extreme cases)
    if cost_config.get('enabled', True):
        scale_up_threshold = cost_config.get('scale_up_threshold', 80)
        scale_down_threshold = cost_config.get('scale_down_threshold', 20)
        min_replicas = cost_config.get('min_replicas', 1)
        max_replicas_limit = cost_config.get('max_replicas', 10)
        scale_down_delay = cost_config.get('scale_down_delay_minutes', 2) * 60
        idle_scale_minutes = cost_config.get('idle_scale_down_minutes', 5)
        zero_traffic_threshold = cost_config.get('zero_traffic_threshold', 0.1)
        target_cpu = cost_config.get('target_cpu_utilization', 75)
        
        # Priority 1: Emergency scale down for idle periods (highest priority)
        if low_traffic_start_time is not None:
            idle_duration_minutes = (datetime.now() - low_traffic_start_time).total_seconds() / 60
            if idle_duration_minutes >= idle_scale_minutes and current_replicas > min_replicas:
                heapq.heappush(decision_heap, (1.0, "scale_down", min_replicas, f"idle_{idle_duration_minutes:.1f}min"))
        
        # Priority 2: Sustained low CPU (high priority)
        if consecutive_low_cpu_count >= 3 and current_replicas > min_replicas:
            if current_time - last_scale_down_time >= scale_down_delay:
                recommended = max(min_replicas, current_replicas - 1)
                heapq.heappush(decision_heap, (2.0, "scale_down", recommended, f"sustained_low_cpu_{consecutive_low_cpu_count}"))
        
        # Priority 1.5: Emergency very high CPU scale up (outranks predictive decisions)
        if current_cpu >= 90:
            # Immediate proportional step without cooldown to avoid overload
            overload = max(0.0, current_cpu - target_cpu)
            step = max(2, int(np.ceil((overload / max(1.0, target_cpu)) * current_replicas)))
            recommended = min(max_replicas_limit, current_replicas + step)
            heapq.heappush(decision_heap, (0.5, "scale_up", recommended, f"emergency_high_cpu_{current_cpu:.1f}%_step{step}"))

        # Priority 3: High CPU requires scale up (reactive)
        if current_cpu > scale_up_threshold:
            # Short cooldown for responsiveness
            if current_time - last_scale_up_time >= 5:
                recommended = None
                # Allow predictions to guide target replicas
                if isinstance(predictions, dict) and len(predictions) > 0:
                    try:
                        predicted_replicas = max(int(round(v)) for v in predictions.values() if isinstance(v, (int, float)))
                        recommended = min(max_replicas_limit, max(predicted_replicas, current_replicas))
                    except Exception:
                        recommended = None
                elif isinstance(predictions, (list, tuple)) and len(predictions) > 0 and isinstance(predictions[0], (int, float)):
                    predicted_replicas = int(round(predictions[0]))
                    recommended = min(max_replicas_limit, max(predicted_replicas, current_replicas))

                if recommended is None:
                    # Proportional reactive step based on how far above threshold we are
                    inc = max(1, int(np.ceil((current_cpu - scale_up_threshold) / 5)))
                    recommended = min(max_replicas_limit, current_replicas + inc)

                if recommended > current_replicas:
                    heapq.heappush(decision_heap, (3.0, "scale_up", recommended, f"high_cpu_{current_cpu:.1f}%"))
        
        # Priority 4: Very low activity emergency scale down
        if current_cpu < 5 and current_traffic < zero_traffic_threshold and current_replicas > min_replicas:
            if current_time - last_scale_down_time >= scale_down_delay:
                heapq.heappush(decision_heap, (4.0, "scale_down", min_replicas, f"very_low_activity"))
    
    # Priority 2.5-4.5: MSE-based prediction decisions (HIGHER priority than reactive scaling)
    if predictions and len(predictions) > 0:
        # Get all model predictions with their MSE rankings
        model_predictions = get_ranked_model_predictions()
        
        for rank, (mse, model_name, pred_value) in enumerate(model_predictions):
            if pred_value is None:
                continue
                
            predicted_replicas = int(round(pred_value))
            priority = 2.5 + (rank * 0.1)  # Predictions get higher priority than reactive scaling
            
            # Predictive scale up - be MORE proactive to avoid success rate drops
            if predicted_replicas > current_replicas and current_time - last_scale_up_time >= 5:
                # Trust predictions early - scale up when CPU > 30% to stay ahead of load
                if current_cpu > 30 or model_name in ['holt_winters']:  # Much more aggressive
                    heapq.heappush(decision_heap, (priority, "scale_up", predicted_replicas, f"predictive_{model_name}_mse_{mse:.3f}"))
            
            # Predictive scale down - be efficient  
            elif predicted_replicas < current_replicas and current_replicas > cost_config.get('min_replicas', 1):
                if current_time - last_scale_down_time >= 30:  # Faster predictive scale down
                    # Trust predictions for scale down when CPU is not extremely high
                    if current_cpu < 85:
                        heapq.heappush(decision_heap, (priority + 0.5, "scale_down", predicted_replicas, f"predictive_{model_name}_mse_{mse:.3f}"))
    
    # Priority 11: Maintain current state (lowest priority)
    heapq.heappush(decision_heap, (11.0, "maintain", current_replicas, "default_maintain"))
    
    # Select the highest priority decision (lowest priority score)
    if decision_heap:
        priority, decision, replicas, reason = heapq.heappop(decision_heap)
        
        # Update timing based on decision
        if decision == "scale_up":
            last_scale_up_time = current_time
        elif decision == "scale_down":
            last_scale_down_time = current_time
        
        # Final safety check: Enforce absolute bounds
        max_replicas_absolute = cost_config.get('max_replicas', 10)
        min_replicas_absolute = cost_config.get('min_replicas', 1)
        replicas = max(min_replicas_absolute, min(max_replicas_absolute, int(replicas)))
        
        logger.info(f"🎯 MinHeap scaling decision: {decision} to {replicas} replicas (priority: {priority:.1f}, reason: {reason}, bounds: {min_replicas_absolute}-{max_replicas_absolute})")
        
        return decision, int(replicas)
    
    # Fallback
    return "maintain", int(current_replicas)

def get_ranked_model_predictions():
    """Get predictions from all models ranked by their MSE (best first)"""
    model_predictions = []
    
    # Get current predictions from each model
    try:
        # GRU prediction
        if gru_model is not None and is_model_trained:
            gru_pred = predict_with_gru(1)
            if gru_pred and len(gru_pred) > 0:
                model_predictions.append((gru_mse, 'gru', gru_pred[0]))
    except Exception as e:
        logger.debug(f"Failed to get GRU prediction for ranking: {e}")
    
    try:
        # Holt-Winters prediction
        hw_pred = predict_with_holtwinters(1)
        if hw_pred and len(hw_pred) > 0:
            model_predictions.append((holt_winters_mse, 'holt_winters', hw_pred[0]))
    except Exception as e:
        logger.debug(f"Failed to get HW prediction for ranking: {e}")
    
    # Ensemble removed from ranking
    
    # Sort by MSE (lower is better) and filter out infinite MSE
    valid_predictions = [(mse, name, pred) for mse, name, pred in model_predictions if mse != float('inf')]
    invalid_predictions = [(mse, name, pred) for mse, name, pred in model_predictions if mse == float('inf')]
    
    # Sort valid predictions by MSE, then add invalid ones at the end
    valid_predictions.sort(key=lambda x: x[0])
    
    ranked_predictions = valid_predictions + invalid_predictions
    
    logger.debug(f"Ranked model predictions: {[(name, mse, pred) for mse, name, pred in ranked_predictions]}")
    
    return ranked_predictions

def make_scaling_decision(predictions):
    """Enhanced scaling decision using MinHeap system for MSE-based ranking"""
    return make_scaling_decision_with_minheap(predictions)

def compare_predictions_with_heap(gru_predictions, hw_predictions):
    """Compare predictions from GRU and Holt-Winters models using a min heap."""
    if gru_predictions is None:
        return "Holt-Winters", hw_predictions
    if hw_predictions is None:
        return "GRU", gru_predictions
    
    prediction_heap = []
    
    for pred in gru_predictions:
        heapq.heappush(prediction_heap, (-pred, "GRU"))
    
    for pred in hw_predictions:
        heapq.heappush(prediction_heap, (-pred, "Holt-Winters"))
    
    if prediction_heap:
        highest_neg_pred, highest_method = heapq.heappop(prediction_heap)
        highest_pred = -highest_neg_pred
        
        logger.info(f"Heap analysis: Highest prediction {highest_pred} from {highest_method}")
        
        if highest_method == "GRU":
            return "GRU", gru_predictions
        else:
            return "Holt-Winters", hw_predictions
    
    return "Holt-Winters", hw_predictions

def continuous_mse_updater():
    """Dedicated thread to continuously update MSE calculations every 30 seconds"""
    global is_mse_updating, last_mse_calculation
    
    logger.info("🔄 Started continuous MSE updater thread")
    
    while is_mse_updating:
        try:
            current_time = datetime.now()
            
            # Update MSE every 30 seconds
            if last_mse_calculation is None or (current_time - last_mse_calculation).total_seconds() >= 30:
                logger.debug("🔄 Running automatic MSE calculation...")
                update_mse_metrics()
                last_mse_calculation = current_time
                logger.debug("✅ Automatic MSE calculation completed")
            
            # Sleep for 10 seconds, check every 10s but update every 30s
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in continuous MSE updater: {e}")
            time.sleep(30)  # Wait longer on error
    
    logger.info("🛑 Continuous MSE updater thread stopped")

def start_mse_updater():
    """Start the continuous MSE updater thread."""
    global is_mse_updating, mse_update_thread
    
    if not is_mse_updating:
        is_mse_updating = True
        mse_update_thread = threading.Thread(target=continuous_mse_updater)
        mse_update_thread.daemon = True
        mse_update_thread.start()
        logger.info("Started continuous MSE updater")
        return True
    return False

def stop_mse_updater():
    """Stop the continuous MSE updater thread."""
    global is_mse_updating
    
    if is_mse_updating:
        is_mse_updating = False
        logger.info("Stopped continuous MSE updater")

def start_collection():
    """Start the metrics collection thread."""
    global is_collecting, collection_thread, data_collection_complete
    
    ensure_traffic_buffer_integrity()

    if data_collection_complete:
        logger.info("Data collection already complete. Using stored 4-hour dataset.")
        return False
    
    if not is_collecting:
        is_collecting = True
        collection_thread = threading.Thread(target=collect_metrics_from_prometheus)
        collection_thread.daemon = True
        collection_thread.start()
        logger.info("Started metrics collection")
        return True
    return False

def stop_collection():
    """Stop the metrics collection thread."""
    global is_collecting
    
    if is_collecting:
        is_collecting = False
        if collection_thread:
            collection_thread.join(timeout=5)
        save_data()
        save_predictions_history()
        logger.info("Stopped metrics collection")
        return True
    return False

# API endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/status', methods=['GET'])
def status():
    """Enhanced status endpoint with detailed GRU debugging information."""
    global gru_model
    
    minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60
    idle_minutes = 0
    if low_traffic_start_time:
        idle_minutes = (datetime.now() - low_traffic_start_time).total_seconds() / 60
    
    # Enhanced GRU status information
    gru_config = config["models"]["gru"]
    # Align required data threshold with training logic used elsewhere (look_back + 2, min 15)
    min_data_required = max(int(gru_config["look_back"]) + 2, 15)
    try:
        current_cpu = get_current_cpu_from_prometheus()  # Get real-time CPU for status
    except Exception:
        # Don’t let status fail — use cached or 0
        current_cpu = globals().get('_last_cpu_value', 0.0)
    
    # Training readiness check
    training_ready = {
        'time_threshold_met': minutes_elapsed >= config['training_threshold_minutes'],
        'data_sufficient': len(traffic_data) >= min_data_required,
        'cpu_threshold_met': current_cpu >= config['cpu_threshold'],
        'data_points': len(traffic_data),
        'required_data': min_data_required,
        'current_cpu': current_cpu,
        'cpu_threshold': config['cpu_threshold'],
        'minutes_elapsed': minutes_elapsed,
        'training_threshold_minutes': config['training_threshold_minutes']
    }
    
    training_ready['all_conditions_met'] = (
        training_ready['time_threshold_met'] and 
        training_ready['data_sufficient'] and 
        training_ready['cpu_threshold_met']
    )
    
    # Model file status
    model_files = {
        'gru_model_exists': os.path.exists(MODEL_FILE),
        'scaler_x_exists': os.path.exists(SCALER_X_FILE),
        'scaler_y_exists': os.path.exists(SCALER_Y_FILE),
        'model_loaded_in_memory': gru_model is not None
    }
    
    # Data collection status - runtime based for AWS Academy compatibility
    collection_status = {
        'collection_complete': data_collection_complete,
        'collection_start_time': data_collection_start_time.isoformat() if data_collection_start_time else None,
        'is_collecting': is_collecting,
        'runtime_hours_collected': (len(traffic_data) / 60),  # Each data point ≈ 1 minute
        'remaining_runtime_hours': max(0, 4 - (len(traffic_data) / 60)),
        'progress_percentage': min(100, (len(traffic_data) / 240) * 100),  # 240 = 4hrs * 60min
        'data_points_collected': len(traffic_data),
        'target_data_points': 240,  # 4 hours * 60 minutes
        'sessions_info': {
            'aws_academy_sessions_needed': 1,  # 4 hours / 4 hours per session
            'hours_per_session': 4,
            'estimated_sessions_completed': min(1, int((len(traffic_data) / 60) / 4))
        },
        'note': 'Collection based on 4 hours of actual runtime for memory-efficient training dataset'
    }
    
    if not data_collection_complete and len(traffic_data) > 0:
        collection_status['current_session_runtime'] = (datetime.now() - start_time).total_seconds() / 3600

    return jsonify({
        'data_points': len(traffic_data),
        'model_trained': is_model_trained,
        'using_gru': config['use_gru'],
        'uptime_minutes': minutes_elapsed,
        'time_until_gru': max(0, config['training_threshold_minutes'] - minutes_elapsed) if not is_model_trained else 0,
        'last_training': last_training_time.isoformat() if last_training_time else None,
        'cost_optimization': config.get('cost_optimization', {}),
        'idle_minutes': idle_minutes,
        'consecutive_low_cpu_count': consecutive_low_cpu_count,
        'current_cpu': current_cpu,
        'current_traffic': traffic_data[-1]['traffic'] if traffic_data else 0,
        'current_replicas': traffic_data[-1]['replicas'] if traffic_data else 1,
        'traffic_buffer': get_traffic_buffer_diagnostics(),
        'last_scaling_decision': last_scaling_decision_info,
        'data_collection_status': collection_status,
        'gru_debug': {
            'training_ready': training_ready,
            'model_files': model_files,
            'can_predict': gru_model is not None and is_model_trained,
            'training_config': gru_config
        },
        'mse_stats': {
            'gru_mse': gru_mse if gru_mse != float('inf') else None,
            'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
            'best_model': select_best_model(),
            'last_mse_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None,
            'prediction_samples': {
                # Exclude emergency/synthetic entries from counts
                'gru_matched': len([p for p in predictions_history['gru'] if p.get('matched', False) and not p.get('emergency', False)]),
                'gru_total': len([p for p in predictions_history['gru'] if not p.get('emergency', False)]),
                'holt_winters_matched': len([p for p in predictions_history['holt_winters'] if p.get('matched', False) and not p.get('emergency', False)]),
                'holt_winters_total': len([p for p in predictions_history['holt_winters'] if not p.get('emergency', False)])
            }
        }
    })

@app.route('/data', methods=['GET'])
def get_metrics_data():
    """Return collected metrics data."""
    recent_data = list(traffic_data)[-100:] if traffic_data else []
    
    return jsonify({
        "data_points": len(traffic_data),
        "recent_metrics": recent_data,
        "buffer_maxlen": traffic_data.maxlen or 240,
        "buffer_within_bounds": len(traffic_data) <= (traffic_data.maxlen or 240),
        "using_gru": config['use_gru'],
        "mse_enabled": config['mse_config']['enabled']
    })

@app.route('/predict', methods=['GET'])
def get_prediction():
    """Return prediction and scaling recommendation with enhanced MSE tracking."""
    global gru_model, is_model_trained
    
    gru_config = config["models"]["gru"]
    hw_config = config["models"]["holt_winters"]
    
    steps = request.args.get('steps', default=int(hw_config["look_forward"]), type=int)
    
    # ALWAYS make predictions from both models for MSE tracking
    gru_predictions = None
    hw_predictions = None
    
    # Make GRU prediction if available
    if gru_model is not None and is_model_trained:
        try:
            start_time_func = time.time()
            gru_predictions = predict_with_gru(steps)
            prediction_latency.labels(service=SERVICE_LABEL, method="gru").observe(time.time() - start_time_func)
            prediction_requests.labels(service=SERVICE_LABEL, method="gru").inc()
            logger.info("GRU prediction made for MSE tracking")
        except Exception as e:
            logger.error(f"GRU prediction failed: {e}")
    
    # Always make Holt-Winters prediction
    try:
        start_time_func = time.time()
        hw_predictions = predict_with_holtwinters(steps)
        prediction_latency.labels(service=SERVICE_LABEL, method="holt_winters").observe(time.time() - start_time_func)
        prediction_requests.labels(service=SERVICE_LABEL, method="holt_winters").inc()
        logger.info("Holt-Winters prediction made for MSE tracking")
    except Exception as e:
        logger.error(f"Holt-Winters prediction failed: {e}")
    
    # Select the best model using MinHeap system for the actual scaling decision
    best_model = select_best_model()
    
    # Get ranked predictions for MinHeap-based decision making
    ranked_predictions = get_ranked_model_predictions()
    
    # Use the best model's predictions for scaling (lowest MSE first)
    if ranked_predictions:
        best_mse, best_method_name, best_prediction = ranked_predictions[0]
        predictions = [best_prediction]
        method = best_method_name.upper()
        logger.info(f"🏆 MinHeap selected {method} for scaling: {predictions} (MSE: {best_mse:.4f})")
    elif best_model == 'gru' and gru_predictions is not None:
        predictions = gru_predictions
        method = "GRU"
        logger.info(f"Fallback to GRU predictions for scaling: {predictions}")
    elif hw_predictions is not None:
        predictions = hw_predictions
        method = "Holt-Winters"
        logger.info(f"Fallback to Holt-Winters predictions for scaling: {predictions}")
    else:
        # Ultimate fallback
        predictions = [2]  # Safe default
        method = "Fallback"
        logger.warning("No predictions available, using ultimate fallback")
    
    decision, replicas = make_scaling_decision(predictions)
    scaling_decisions.labels(service=SERVICE_LABEL, decision=decision).inc()
    recommended_replicas.set(replicas)
    
    # Enhanced MSE status
    gru_status = f"{gru_mse:.3f}" if gru_mse != float('inf') else "insufficient_data"
    hw_status = f"{holt_winters_mse:.3f}" if holt_winters_mse != float('inf') else "insufficient_data"
    
    return jsonify({
        "target_deployment": config.get('target_deployment', 'unknown'),
        "method": method,
        "predictions": predictions,
        "scaling_decision": decision,
        "recommended_replicas": replicas,
        "model_selection_reason": f"MSE-based selection: GRU={gru_status}, HW={hw_status}",
        "mse_debug": {
            "gru_matched_predictions": len([p for p in predictions_history['gru'] if p.get('matched', False)]),
            "hw_matched_predictions": len([p for p in predictions_history['holt_winters'] if p.get('matched', False)])
        }
    })

@app.route('/predict_combined', methods=['GET'])
def predict_combined():
    """Combined prediction endpoint with enhanced MSE-based model selection."""
    global gru_model, is_model_trained
    
    try:
        minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        if not traffic_data:
            # No data yet, return minimum replicas
            return jsonify({
                'target_deployment': config.get('target_deployment', 'unknown'),
                'method_used': 'fallback',
                'predictions': [1],
                'recommended_replicas': 1,
                'scaling_decision': 'maintain',
                'current_replicas': 1,
                'time_until_gru': max(0, config['training_threshold_minutes'] - minutes_elapsed),
                'is_model_trained': is_model_trained,
                'mse_stats': {
                    'gru_mse': None,
                    'holt_winters_mse': None,
                    'best_model': 'holt_winters'
                },
                'error': 'No data available'
            })
        
        current_metrics = traffic_data[-1]
        current_replicas = current_metrics['replicas']
        current_cpu = current_metrics['cpu_utilization']
        current_traffic = current_metrics['traffic']
        
        # ALWAYS make all predictions for MSE tracking
        gru_predictions = None
        hw_predictions = None
        ensemble_predictions = None
        
        # Make GRU prediction if available
        if gru_model is not None and is_model_trained:
            try:
                gru_predictions = predict_with_gru()
                logger.info(f"GRU prediction made in combined endpoint: {gru_predictions}")
            except Exception as e:
                logger.error(f"GRU prediction failed in combined endpoint: {e}")
                gru_predictions = None
        
        # Always make Holt-Winters prediction
        try:
            hw_predictions = predict_with_holtwinters()
            logger.info(f"Holt-Winters prediction made in combined endpoint: {hw_predictions}")
        except Exception as e:
            logger.error(f"Holt-Winters prediction failed in combined endpoint: {e}")
            hw_predictions = None
            
        # Ensemble removed
        
        # Select method for scaling decision
        if config['mse_config']['enabled']:
            best_model = select_best_model()
            method = best_model
            
            if best_model == 'gru' and gru_predictions is not None:
                predictions = gru_predictions
            else:
                predictions = hw_predictions if hw_predictions else [2]
        else:
            # Fallback to original logic
            if is_model_trained and gru_model is not None and gru_predictions is not None:
                method = 'gru'
                predictions = gru_predictions
            else:
                method = 'holt_winters'
                predictions = hw_predictions if hw_predictions else [2]
        
        # Reactive scaling as fallback with improved logic
        if not predictions:
            method = 'reactive'
            cost_config = config.get('cost_optimization', {})
            
            if cost_config.get('enabled', True):
                scale_up_threshold = cost_config.get('scale_up_threshold', 80)
                scale_down_threshold = cost_config.get('scale_down_threshold', 20)
                min_replicas = cost_config.get('min_replicas', 1)
                max_replicas_limit = cost_config.get('max_replicas', 10)
                zero_traffic_threshold = cost_config.get('zero_traffic_threshold', 0.1)
                
                # Check for zero/minimal load
                if current_cpu < 5 and current_traffic < zero_traffic_threshold:
                    predictions = [min_replicas]
                elif current_cpu < scale_down_threshold and current_replicas > min_replicas:
                    predictions = [max(min_replicas, current_replicas - 1)]
                elif current_cpu > scale_up_threshold:
                    predictions = [min(max_replicas_limit, current_replicas + 1)]
                else:
                    predictions = [current_replicas]
            else:
                # Original reactive logic
                if current_cpu > 70:
                    predictions = [min(10, current_replicas + 2)]
                elif current_cpu > 50:
                    predictions = [min(10, current_replicas + 1)]
                elif current_cpu < 30 and current_replicas > 1:
                    predictions = [max(1, current_replicas - 1)]
                else:
                    predictions = [current_replicas]
        
        # Make scaling decision with improved logic
        decision, recommended_replicas_value = make_scaling_decision(predictions)
        
        # Update Prometheus metrics for ALL models (not just the selected one)
        # This ensures Grafana always shows the latest predictions from each model
        if gru_predictions is not None:
            gru_value = int(gru_predictions[0]) if len(gru_predictions) > 0 else 0
            predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='gru').set(gru_value)
            logger.debug(f"Updated GRU metric in combined endpoint: {gru_value}")
        else:
            predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='gru').set(0)
            
        if hw_predictions is not None:
            hw_value = int(hw_predictions[0]) if len(hw_predictions) > 0 else 0
            predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='holt_winters').set(hw_value)
            logger.info(f"✅ Updated Holt-Winters metric in combined endpoint: {hw_value}")
        else:
            predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='holt_winters').set(0)
            logger.info("❌ Holt-Winters predictions is None, setting metric to 0")
            
        # Ensemble metrics removed in combined endpoint
        
        prediction_requests.labels(service=SERVICE_LABEL, method=method).inc()
        scaling_decisions.labels(service=SERVICE_LABEL, decision=decision).inc()
        
        # Enhanced MSE status reporting
        gru_status = f"{gru_mse:.3f}" if gru_mse != float('inf') else "insufficient_data"
        hw_status = f"{holt_winters_mse:.3f}" if holt_winters_mse != float('inf') else "insufficient_data"
        
        return jsonify({
            'target_deployment': config.get('target_deployment', 'unknown'),
            'method_used': method,
            'predictions': predictions if predictions else [recommended_replicas_value],
            'recommended_replicas': int(recommended_replicas_value),
            'current_replicas': int(current_replicas),
            'current_cpu': round(current_cpu, 2),
            'current_traffic': round(current_traffic, 2),
            'scaling_decision': decision,
            'time_until_gru': max(0, config['training_threshold_minutes'] - minutes_elapsed) if not is_model_trained else 0,
            'is_model_trained': is_model_trained,
            'cost_optimization': config.get('cost_optimization', {}),
            'idle_minutes': (datetime.now() - low_traffic_start_time).total_seconds() / 60 if low_traffic_start_time else 0,
            'consecutive_low_cpu_count': consecutive_low_cpu_count,
            'mse_stats': {
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
                'best_model': select_best_model() if config['mse_config']['enabled'] else method,
                'mse_enabled': config['mse_config']['enabled'],
                'gru_status': gru_status,
                'hw_status': hw_status,
                'prediction_matching': {
                    'gru_matched': len([p for p in predictions_history['gru'] if p.get('matched', False)]),
                    'gru_total': len(predictions_history['gru']),
                    'hw_matched': len([p for p in predictions_history['holt_winters'] if p.get('matched', False)]),
                    'hw_total': len(predictions_history['holt_winters'])
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in predict_combined: {e}")
        return jsonify({
            'target_deployment': config.get('target_deployment', 'unknown'),
            'method_used': 'fallback',
            'predictions': [1],
            'recommended_replicas': 1,
            'scaling_decision': 'maintain',
            'current_replicas': 1,
            'time_until_gru': 0,
            'is_model_trained': False,
            'error': str(e),
            'mse_stats': {
                'gru_mse': None,
                'holt_winters_mse': None,
                'best_model': None,
                'mse_enabled': config['mse_config']['enabled']
            }
        })

@app.route('/config', methods=['GET', 'PUT'])
def handle_config():
    """Get or update configuration."""
    global config
    
    if request.method == 'GET':
        return jsonify(config)
    
    elif request.method == 'PUT':
        try:
            new_config = request.get_json()
            if new_config:
                config.update(new_config)
                save_config()
            return jsonify({"status": "success", "config": config})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/collection', methods=['POST'])
def handle_collection():
    """Start or stop metrics collection."""
    action = request.args.get('action', '')
    
    if action == 'start':
        success = start_collection()
        return jsonify({"status": "success" if success else "already_running"})
    
    elif action == 'stop':
        success = stop_collection()
        return jsonify({"status": "success" if success else "not_running"})
    
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400

@app.route('/rebuild-model', methods=['POST'])
def rebuild_model():
    """Force rebuilding of the GRU model."""
    global gru_model
    
    try:
        if build_gru_model():
            config['use_gru'] = True
            save_config()
            return jsonify({"status": "success", "message": "GRU model rebuilt"})
        else:
            return jsonify({"status": "error", "message": "Not enough data to build model"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/mse', methods=['GET'])
def get_mse_data():
    """Enhanced MSE data endpoint with detailed debugging information."""
    gru_status = f"{gru_mse:.3f}" if gru_mse != float('inf') else "insufficient_data"
    hw_status = f"{holt_winters_mse:.3f}" if holt_winters_mse != float('inf') else "insufficient_data"
    
    return jsonify({
        'current_mse': {
            'gru': gru_mse if gru_mse != float('inf') else None,
            'holt_winters': holt_winters_mse if holt_winters_mse != float('inf') else None,
            'gru_status': gru_status,
            'hw_status': hw_status
        },
        'best_model': select_best_model(),
        'prediction_history_summary': {
            'gru_total': len(predictions_history['gru']),
            'gru_matched': len([p for p in predictions_history['gru'] if p.get('matched', False)]),
            'hw_total': len(predictions_history['holt_winters']),
            'hw_matched': len([p for p in predictions_history['holt_winters'] if p.get('matched', False)])
        },
        'recent_predictions': {
            'gru': predictions_history['gru'][-5:],  # Last 5 predictions with match status
            'holt_winters': predictions_history['holt_winters'][-5:]
        },
        'config': config['mse_config'],
        'last_mse_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None
    })

# Initialization function
def initialize():
    """Enhanced initialization with better GRU model status logging."""
    # Ensure only one initializer runs
    with _init_lock:
        if getattr(app, '_initialized', False):
            return
        logger.info("Starting predictive scaler initialization...")
    
    # Load configuration and data
    load_config()
    load_data()
    load_predictions_history()
    load_collection_status()  # Load data collection status
    
    # Load model components with status reporting
    model_loaded = load_model_components()
    if model_loaded:
        logger.info("GRU model loaded successfully from existing files")
    else:
        logger.info("No existing GRU model found - will train when conditions are met")
    
    # Initialize Prometheus
    initialize_prometheus()
    
    # Initialize metrics with zero values
    # Do not observe 0 into histograms at startup to avoid misleading flat lines
    current_mse.labels(service=SERVICE_LABEL, model="gru").set(-1.0)  # Start with -1 (insufficient data)
    current_mse.labels(service=SERVICE_LABEL, model="holt_winters").set(-1.0)
    training_time.labels(service=SERVICE_LABEL, model="gru").set(0.0)
    training_time.labels(service=SERVICE_LABEL, model="holt_winters").set(0.0)
    prediction_time.labels(service=SERVICE_LABEL, model="gru").set(0.0)
    prediction_time.labels(service=SERVICE_LABEL, model="holt_winters").set(0.0)
    predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='gru').set(0)
    predictive_scaler_recommended_replicas.labels(service=SERVICE_LABEL, method='holt_winters').set(0)
    model_selection.labels(service=SERVICE_LABEL, model='gru', reason='initialization').inc(0)
    model_selection.labels(service=SERVICE_LABEL, model='holt_winters', reason='initialization').inc(0)
    
    # Initialize HTTP metrics counter
    http_requests.labels(service=SERVICE_LABEL, app='predictive-scaler', status_code='200', path='/').inc(0)
    
    # Log current GRU status
    gru_config = config["models"]["gru"]
    min_data_required = max(int(gru_config["look_back"]) + 10, 20)
    
    logger.info(f"GRU Status: trained={is_model_trained}, use_gru={config['use_gru']}")
    logger.info(f"GRU Training requirements: {config['training_threshold_minutes']} minutes, {min_data_required} data points, CPU > {config['cpu_threshold']}%")
    logger.info(f"Current data points: {len(traffic_data)}")
    
    # Start continuous MSE updater
    start_mse_updater()
    
    # Start collection
    start_collection()
    
    # Start the two-coroutine prediction system (essential for metrics updates)
    if not hasattr(app, '_coroutines_started'):
        logger.info("🚀 Starting two-coroutine system for predictive scaling...")
        start_two_coroutine_system()
        app._coroutines_started = True
        logger.info("✅ Two-coroutine system started successfully")
    
    # Schedule GRU model training asynchronously if we have enough data but no working model
    if not hasattr(app, '_gru_retrain_attempted'):
        if not is_model_trained and len(traffic_data) >= 110:
            logger.info("🧠 Sufficient data available, scheduling asynchronous GRU training...")
            try:
                # Start GRU training in background thread to avoid blocking Flask
                training_thread = threading.Thread(
                    target=async_model_training,
                    args=("initialization", list(traffic_data), True)
                )
                training_thread.daemon = True
                training_thread.start()
                logger.info("✅ GRU training scheduled in background thread")
            except Exception as e:
                logger.error(f"Failed to schedule GRU training during initialization: {e}")
        else:
            logger.info(f"🔄 GRU training not needed: trained={is_model_trained}, data_points={len(traffic_data)}")
        app._gru_retrain_attempted = True
    
    logger.info("Predictive scaler initialized successfully")
    # Mark as initialized at the very end
    setattr(app, '_initialized', True)

def _ensure_initialized_async():
    """Start initialization in a background thread if not already initialized."""
    if getattr(app, '_initialized', False):
        return
    with _init_lock:
        if getattr(app, '_initialized', False):
            return
        t = threading.Thread(target=initialize, name="initializer")
        t.daemon = True
        t.start()

@app.before_request
def before_request():
    request.start_time = time.time()
    # Do not trigger heavy init on health checks to avoid probe timeouts
    if request.path == '/health':
        return
    # Trigger async initialize once on first non-health request
    _ensure_initialized_async()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        app_name = os.environ.get('APP_NAME', 'predictive-scaler')
        
        http_duration.labels(service=SERVICE_LABEL, app=app_name).observe(duration)
        http_requests.labels(
            service=SERVICE_LABEL,
            app=app_name,
            status_code=str(response.status_code), 
            path=request.path
        ).inc()
    
    return response

@app.route('/debug/train_gru', methods=['POST'])
def debug_train_gru():
    """Debug endpoint to force GRU training regardless of conditions."""
    global gru_model, is_model_trained
    
    try:
        logger.info("Debug: Forcing GRU training...")
        
        # Check data availability
        if len(traffic_data) < 20:
            return jsonify({
                'success': False,
                'error': f'Insufficient data: {len(traffic_data)} < 20',
                'data_points': len(traffic_data)
            }), 400
        
        # Force training
        success = build_gru_model()
        
        if success:
            config['use_gru'] = True
            return jsonify({
                'success': True,
                'message': 'GRU model trained successfully',
                'model_trained': is_model_trained,
                'using_gru': config['use_gru'],
                'data_points': len(traffic_data)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training failed',
                'model_trained': is_model_trained,
                'data_points': len(traffic_data)
            }), 500
            
    except Exception as e:
        logger.error(f"Debug training error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug/gru_status', methods=['GET'])
def debug_gru_status():
    """Debug endpoint for detailed GRU status information."""
    global gru_model, is_model_trained
    
    minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60
    gru_config = config["models"]["gru"]
    min_data_required = max(int(gru_config["look_back"]) + 10, 20)
    current_cpu = traffic_data[-1]['cpu_utilization'] if traffic_data else 0
    
    return jsonify({
        'gru_model_loaded': gru_model is not None,
        'is_model_trained': is_model_trained,
        'use_gru_config': config['use_gru'],
        'data_points': len(traffic_data),
        'min_data_required': min_data_required,
        'minutes_elapsed': minutes_elapsed,
        'training_threshold_minutes': config['training_threshold_minutes'],
        'current_cpu': current_cpu,
        'cpu_threshold': config['cpu_threshold'],
        'model_files': {
            'model_exists': os.path.exists(MODEL_FILE),
            'scaler_x_exists': os.path.exists('scaler_X.pkl'),
            'scaler_y_exists': os.path.exists('scaler_y.pkl')
        },
        'conditions_met': {
            'time_ok': minutes_elapsed >= config['training_threshold_minutes'],
            'data_ok': len(traffic_data) >= min_data_required,
            'cpu_ok': current_cpu >= config['cpu_threshold']
        },
        'predictions_history': {
            'gru_total': len(predictions_history['gru']),
            'gru_matched': len([p for p in predictions_history['gru'] if p.get('matched', False)]),
            'holt_winters_total': len(predictions_history['holt_winters']),
            'holt_winters_matched': len([p for p in predictions_history['holt_winters'] if p.get('matched', False)]),
            'recent_gru': predictions_history['gru'][-3:] if predictions_history['gru'] else [],
            'recent_holt_winters': predictions_history['holt_winters'][-3:] if predictions_history['holt_winters'] else []
        },
        'mse_values': {
            'gru_mse': gru_mse if gru_mse != float('inf') else None,
            'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
            'last_mse_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None
        }
    })

@app.route('/debug/force_mse_update', methods=['POST'])
def debug_force_mse_update():
    """Debug endpoint to force MSE calculation and matching."""
    try:
        logger.info("Debug: Forcing MSE update and prediction matching...")
        
        # Force prediction matching
        update_predictions_with_actual_values()
        
        # Force MSE calculation
        update_mse_metrics()
        
        # Get current status
        gru_matched = len([p for p in predictions_history['gru'] if p.get('matched', False)])
        hw_matched = len([p for p in predictions_history['holt_winters'] if p.get('matched', False)])
        
        return jsonify({
            'success': True,
            'message': 'MSE update forced successfully',
            'results': {
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
                'gru_matched_predictions': gru_matched,
                'holt_winters_matched_predictions': hw_matched,
                'gru_total_predictions': len(predictions_history['gru']),
                'holt_winters_total_predictions': len(predictions_history['holt_winters']),
                'last_mse_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None
            }
        })
        
    except Exception as e:
        logger.error(f"Debug MSE update error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug/predictions', methods=['GET'])
def debug_predictions():
    """Debug endpoint to show all predictions and their matching status."""
    global gru_model, is_model_trained
    
    try:
        result = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(traffic_data),
            'latest_data': list(traffic_data)[-3:] if traffic_data else [],
            'predictions': {}
        }
        
        for model_name in ['gru', 'holt_winters']:
            model_predictions = predictions_history[model_name]
            result['predictions'][model_name] = {
                'total': len(model_predictions),
                'matched': len([p for p in model_predictions if p.get('matched', False)]),
                'unmatched': len([p for p in model_predictions if not p.get('matched', False)]),
                'recent_predictions': model_predictions[-5:] if model_predictions else [],
                'mse': gru_mse if model_name == 'gru' and gru_mse != float('inf') else 
                       holt_winters_mse if model_name == 'holt_winters' and holt_winters_mse != float('inf') else None
            }
        
        result['model_status'] = {
            'gru_trained': is_model_trained,
            'gru_model_loaded': gru_model is not None,
            'use_gru': config['use_gru'],
            'best_model': select_best_model()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/debug/force_predictions', methods=['POST'])
def debug_force_predictions():
    """Debug endpoint to force both models to make predictions."""
    global gru_model, is_model_trained
    
    try:
        if len(traffic_data) < 5:
            return jsonify({
                'success': False,
                'error': f'Not enough data: {len(traffic_data)} < 5'
            }), 400
        
        results = {}
        
        # Force Holt-Winters prediction
        try:
            hw_pred = predict_with_holtwinters(1)
            results['holt_winters'] = {
                'success': True,
                'prediction': hw_pred,
                'predictions_count': len(predictions_history['holt_winters'])
            }
        except Exception as e:
            results['holt_winters'] = {
                'success': False,
                'error': str(e)
            }
        
        # Force GRU prediction if available
        if gru_model is not None and is_model_trained:
            try:
                gru_pred = predict_with_gru(1)
                results['gru'] = {
                    'success': True,
                    'prediction': gru_pred,
                    'predictions_count': len(predictions_history['gru'])
                }
            except Exception as e:
                results['gru'] = {
                    'success': False,
                    'error': str(e)
                }
        else:
            results['gru'] = {
                'success': False,
                'error': 'GRU model not available or not trained'
            }
        
        # Force MSE update
        try:
            update_mse_metrics()
            results['mse_update'] = 'success'
        except Exception as e:
            results['mse_update'] = f'failed: {e}'
        
        return jsonify({
            'success': True,
            'results': results,
            'current_mse': {
                'gru': gru_mse if gru_mse != float('inf') else None,
                'holt_winters': holt_winters_mse if holt_winters_mse != float('inf') else None
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug/gru_deep_check', methods=['GET'])
def debug_gru_deep_check():
    """Deep diagnostic check of GRU model state and prediction capability."""
    global gru_model, is_model_trained
    
    try:
        result = {
            'timestamp': datetime.now().isoformat(),
            'gru_model_state': {
                'model_loaded': gru_model is not None,
                'is_trained': is_model_trained,
                'use_gru_config': config['use_gru'],
                'model_input_shape': str(gru_model.input_shape) if gru_model else None,
                'model_output_shape': str(gru_model.output_shape) if gru_model else None
            },
            'scaler_state': {
                'scaler_x_loaded': scaler_X is not None,
                'scaler_y_loaded': scaler_y is not None,
                'scaler_x_features': getattr(scaler_X, 'n_features_in_', None),
                'scaler_x_scale': getattr(scaler_X, 'scale_', [None])[:3] if hasattr(scaler_X, 'scale_') else None
            },
            'data_state': {
                'total_data_points': len(traffic_data),
                'recent_data_sample': list(traffic_data)[-3:] if traffic_data else [],
                'data_features': list(traffic_data[-1].keys()) if traffic_data else []
            },
            'prediction_history': {
                'gru_total': len(predictions_history['gru']),
                'gru_matched': len([p for p in predictions_history['gru'] if p.get('matched', False)]),
                'gru_emergency': len([p for p in predictions_history['gru'] if p.get('emergency', False)]),
                'gru_recent': predictions_history['gru'][-3:] if predictions_history['gru'] else [],
                'holt_winters_total': len(predictions_history['holt_winters']),
                'holt_winters_matched': len([p for p in predictions_history['holt_winters'] if p.get('matched', False)])
            },
            'mse_state': {
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
                'last_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None
            }
        }
        
        # Test GRU prediction capability
        if gru_model and is_model_trained and len(traffic_data) >= 12:
            try:
                test_result = predict_with_gru(1)
                result['gru_test_prediction'] = {
                    'success': test_result is not None,
                    'result': test_result,
                    'error': None
                }
            except Exception as e:
                result['gru_test_prediction'] = {
                    'success': False,
                    'result': None,
                    'error': str(e)
                }
        else:
            result['gru_test_prediction'] = {
                'success': False,
                'result': None,
                'error': 'Cannot test - model not ready or insufficient data'
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/debug/force_emergency_predictions', methods=['POST'])
def debug_force_emergency():
    """Force creation of emergency synthetic predictions for testing."""
    global traffic_data
    try:
        if len(traffic_data) < 10:
            return jsonify({
                'success': False,
                'error': f'Not enough data: {len(traffic_data)} < 10'
            })
        
        # Force emergency predictions
        create_emergency_gru_predictions()
        
        # Force MSE calculation
        update_mse_metrics()
        
        return jsonify({
            'success': True,
            'message': 'Emergency predictions created',
            'results': {
                'gru_predictions_total': len(predictions_history['gru']),
                'gru_predictions_matched': len([p for p in predictions_history['gru'] if p.get('matched', False)]),
                'gru_emergency_count': len([p for p in predictions_history['gru'] if p.get('emergency', False)]),
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug/enhanced_mse', methods=['GET'])
def debug_enhanced_mse():
    """Debug endpoint for enhanced MSE metrics."""
    try:
        result = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name in ['gru', 'holt_winters']:
            if model_name in predictions_history:
                enhanced_metrics = calculate_enhanced_mse(model_name)
                result['models'][model_name] = enhanced_metrics
        
        if result['models']:
            valid_mses = [result['models'][m]['mse'] for m in result['models'] if result['models'][m]['mse'] != float('inf')]
            if valid_mses:
                result['comparison'] = {
                    'best_mse': min(valid_mses),
                    'best_model': min(result['models'].items(), key=lambda x: x[1]['mse'])[0]
                }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ensemble debug endpoint removed

@app.route('/debug/model_comparison', methods=['GET'])
def debug_model_comparison():
    """Compare all model performances with detailed metrics."""
    try:
        result = {
            'timestamp': datetime.now().isoformat(),
            'model_performances': {},
            'recommendations': {}
        }
        
        models = ['gru', 'holt_winters']
        
        best_mse = float('inf')
        best_model = None
        
        for model in models:
            if model in predictions_history:
                metrics = calculate_enhanced_mse(model)
                result['model_performances'][model] = metrics
                
                if metrics['mse'] < best_mse:
                    best_mse = metrics['mse']
                    best_model = model
        
        # Generate recommendations
        if best_model:
            result['recommendations']['best_overall'] = best_model
            result['recommendations']['mse_improvement'] = {}
            
            for model in models:
                if model != best_model and model in result['model_performances']:
                    current_mse = result['model_performances'][model]['mse']
                    if current_mse != float('inf') and best_mse != float('inf'):
                        improvement = ((current_mse - best_mse) / current_mse) * 100
                        result['recommendations']['mse_improvement'][model] = f"{improvement:.1f}% improvement available"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/minheap_status', methods=['GET'])
def debug_minheap_status():
    """Debug endpoint to inspect MinHeap model selection and decision making"""
    try:
        # Get current model rankings
        ranked_predictions = get_ranked_model_predictions()
        
        # Test MinHeap model selection
        selected_model = select_best_model_with_minheap()
        
        # Create a test scaling decision
        test_predictions = [ranked_predictions[0][2]] if ranked_predictions else [2]
        test_decision, test_replicas = make_scaling_decision_with_minheap(test_predictions)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'minheap_model_selection': {
                'selected_model': selected_model,
                'ranked_predictions': [
                    {
                        'model': name,
                        'mse': mse if mse != float('inf') else None,
                        'prediction': pred,
                        'rank': i + 1
                    }
                    for i, (mse, name, pred) in enumerate(ranked_predictions)
                ],
                'selection_algorithm': 'MinHeap (lowest MSE first)'
            },
            'minheap_scaling_decision': {
                'decision': test_decision,
                'recommended_replicas': test_replicas,
                'input_predictions': test_predictions,
                'algorithm': 'Priority-based MinHeap'
            },
            'current_mse_values': {
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None
            },
            'model_readiness': {
                'gru_ready': is_model_trained and gru_model is not None,
                'holt_winters_ready': True
            },
            'current_metrics': traffic_data[-1] if traffic_data else None,
            'config': {
                'mse_enabled': config['mse_config']['enabled'],
                'mse_threshold_difference': config['mse_config']['mse_threshold_difference'],
                'min_samples_for_mse': config['mse_config']['min_samples_for_mse']
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'details': str(e)}), 500

@app.route('/debug/mse_stability', methods=['GET'])
def debug_mse_stability():
    """Debug endpoint to monitor MSE stability and detect interruption patterns."""
    return jsonify({
        'mse_status': {
            'gru_mse': gru_mse if gru_mse != float('inf') else 'insufficient_data',
            'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else 'insufficient_data',
            'last_mse_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None,
            'last_cleanup_time': last_cleanup_time.isoformat() if last_cleanup_time else None
        },
        'prediction_counts': {
            'gru_total': len(predictions_history.get('gru', [])),
            'gru_matched': len([p for p in predictions_history.get('gru', []) if p.get('matched', False)]),
            'hw_total': len(predictions_history.get('holt_winters', [])),
            'hw_matched': len([p for p in predictions_history.get('holt_winters', []) if p.get('matched', False)])
        },
        'recent_gru_predictions': [
            {
                'timestamp': p.get('timestamp'),
                'predicted': p.get('predicted_replicas'),
                'actual': p.get('actual_replicas'),
                'matched': p.get('matched', False)
            }
            for p in predictions_history.get('gru', [])[-5:]
        ],
        'system_info': {
            'data_collection_complete': data_collection_complete,
            'current_data_points': len(traffic_data),
            'training_data_points': len(training_dataset),
            'collection_mode': 'MSE-only' if data_collection_complete else 'Training collection'
        },
        'training_status': {
            'is_training': is_training,
            'training_thread_alive': training_thread.is_alive() if training_thread else False,
            'last_training_time': last_training_time.isoformat() if last_training_time else None,
            'is_model_trained': is_model_trained
        },
        'troubleshooting': {
            'common_causes': [
                'RESOLVED: Model training blocking Flask metrics endpoint (now async)',
                'Cleanup operations removing predictions too aggressively',
                'File I/O blocking MSE calculations',
                'Memory pressure causing calculation failures',
                'Kubernetes resource limits causing temporary interruptions'
            ],
            'fixes_applied': [
                'CRITICAL: Moved model training to separate thread (prevents 1.5min blocking)',
                'Throttled cleanup to every 5 minutes',
                'Added error resilience to MSE calculations',
                'Keep previous MSE values on calculation failure',
                'Enhanced logging for anomaly detection',
                'TensorFlow thread limiting to reduce resource contention'
            ],
            'monitoring': {
                'check_training_status': 'Monitor is_training flag during MSE drops',
                'verify_metrics_continuity': 'Ensure /metrics endpoint responds during training',
                'thread_isolation': 'Training runs in daemon thread separate from Flask'
            }
        }
    })

@app.route('/debug/dataset_status', methods=['GET'])
def debug_dataset_status():
    """Debug endpoint to check dataset status and MSE issues."""
    return jsonify({
        'collection_status': {
            'collection_complete': data_collection_complete,
            'current_data_points': len(traffic_data),
            'training_data_points': len(training_dataset),
            'using_training_data': data_collection_complete and len(training_dataset) > 0
        },
        'mse_analysis': {
            'gru_predictions_total': len(predictions_history.get('gru', [])),
            'gru_predictions_matched': len([p for p in predictions_history.get('gru', []) if p.get('matched', False)]),
            'hw_predictions_total': len(predictions_history.get('holt_winters', [])),
            'hw_predictions_matched': len([p for p in predictions_history.get('holt_winters', []) if p.get('matched', False)]),
            'current_mse_values': {
                'gru': gru_mse if gru_mse != float('inf') else 'insufficient_data',
                'holt_winters': holt_winters_mse if holt_winters_mse != float('inf') else 'insufficient_data'
            }
        },
        'recent_predictions': {
            'gru_recent': predictions_history.get('gru', [])[-3:] if predictions_history.get('gru') else [],
            'hw_recent': predictions_history.get('holt_winters', [])[-3:] if predictions_history.get('holt_winters') else []
        },
        'suggestion': 'If MSE is poor, try /force_reload_model or check prediction-data mismatch'
    })

@app.route('/force_reload_model', methods=['POST'])
def force_reload_model():
    """Force reload GRU model and scalers from disk."""
    global gru_model, scaler_X, scaler_y, is_model_trained
    
    try:
        # Reset model components
        gru_model = None
        is_model_trained = False
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Try to reload from disk
        success = load_model_components()
        
        if success:
            logger.info("✅ Model components reloaded successfully!")
            return jsonify({
                "status": "success",
                "message": "Model and scalers reloaded successfully",
                "details": {
                    "model_loaded": gru_model is not None,
                    "model_trained": is_model_trained,
                    "files_found": {
                        "gru_model": os.path.exists(MODEL_FILE),
                        "scaler_x": os.path.exists(SCALER_X_FILE),
                        "scaler_y": os.path.exists(SCALER_Y_FILE)
                    }
                }
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to reload model components",
                "suggestion": "Check if model files exist and are valid"
            }), 400
            
    except Exception as e:
        logger.error(f"Error during model reload: {e}")
        return jsonify({
            "status": "error",
            "message": f"Model reload failed: {str(e)}"
        }), 500

@app.route('/reset_data', methods=['POST'])
def reset_data():
    """COMPLETE reset of all collected data and restart data collection from scratch."""
    global traffic_data, data_collection_complete, data_collection_start_time
    global predictions_history, gru_model, is_model_trained, scaler_X, scaler_y
    global is_collecting, collection_thread, training_dataset, start_time
    # Clear ALL memory variables that could cause data to "come back"
    global gru_mse, holt_winters_mse, last_mse_calculation, last_cleanup_time
    global low_traffic_start_time, consecutive_low_cpu_count, is_training
    global training_thread, last_training_time, last_gru_retraining, last_holt_winters_update
    global is_main_coroutine_running, is_update_coroutine_running, mse_update_thread
    global models_performance_history, using_synthetic_data
    
    try:
        logger.info("🔄 Starting COMPLETE data reset...")
        
        # Stop current collection if running
        if is_collecting:
            stop_collection()
            time.sleep(2)  # Wait for collection to stop
        
        # COMPLETELY clear ALL data variables and memory state
        try:
            traffic_data.clear()
        except Exception:
            traffic_data = deque(maxlen=240)
        ensure_traffic_buffer_integrity()
        training_dataset = []
        predictions_history = {'gru': [], 'holt_winters': []}
        
        # Reset collection status completely
        data_collection_complete = False
        data_collection_start_time = datetime.now()
        start_time = datetime.now()
        
        # Reset model components completely
        gru_model = None
        is_model_trained = False
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Clear ALL MSE and performance tracking variables
        gru_mse = float('inf')
        holt_winters_mse = float('inf')
        last_mse_calculation = None
        last_cleanup_time = None
        
        # Clear traffic monitoring variables
        low_traffic_start_time = None
        consecutive_low_cpu_count = 0
        
        # Clear training state management
        is_training = False
        training_thread = None
        last_training_time = None
        last_gru_retraining = None
        last_holt_winters_update = None
        
        # Clear coroutine system state
        is_main_coroutine_running = False
        is_update_coroutine_running = False
        mse_update_thread = None
        
        # Clear performance history
        models_performance_history = {'gru': [], 'holt_winters': []}
        using_synthetic_data = False
        
        # Clear config flags
        config['use_gru'] = False
        
        # DELETE ALL PERSISTENT FILES FIRST (before saving anything)
        files_to_delete = [DATA_FILE, STATUS_FILE, PREDICTIONS_FILE, MODEL_FILE, SCALER_X_FILE, SCALER_Y_FILE]
        # Also delete any config files that might reload data
        config_files_to_delete = [
            os.path.join(CONFIG_DIR, "config.json"),
            os.path.join(DATA_DIR, "config.json"),  # In case it's in data dir
        ]
        files_to_delete.extend(config_files_to_delete)
        
        deleted_files = []
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
                    logger.info(f"🗑️ Deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
        
        # ONLY AFTER files are deleted, save the fresh empty state
        save_collection_status()  # Save fresh status
        save_data()  # Save empty data
        save_predictions_history()  # Save empty predictions
        save_config()  # Save clean config
        
        # Start completely fresh data collection
        start_collection()
        
        logger.info("✅ COMPLETE data reset finished. Starting fresh 4-hour data collection.")
        
        return jsonify({
            "status": "success",
            "message": "COMPLETE data reset performed - all persistent data cleared",
            "details": {
                "deleted_files": deleted_files,
                "training_dataset_cleared": True,
                "collection_restarted": True,
                "fresh_start_time": data_collection_start_time.isoformat(),
                "estimated_completion": (data_collection_start_time + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        
    except Exception as e:
        logger.error(f"Error during data reset: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Data reset failed: {str(e)}"
        }), 500

@app.route('/debug/file_status', methods=['GET'])
def debug_file_status():
    """Debug endpoint to check what files exist and their sizes."""
    global traffic_data, training_dataset, data_collection_complete
    
    files_to_check = [DATA_FILE, STATUS_FILE, PREDICTIONS_FILE, MODEL_FILE, SCALER_X_FILE, SCALER_Y_FILE]
    config_files_to_check = [
        os.path.join(CONFIG_DIR, "config.json"),
        os.path.join(DATA_DIR, "config.json"),
    ]
    all_files = files_to_check + config_files_to_check
    
    file_status = {}
    for file_path in all_files:
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                file_status[os.path.basename(file_path)] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": file_path
                }
            else:
                file_status[os.path.basename(file_path)] = {"exists": False, "path": file_path}
        except Exception as e:
            file_status[os.path.basename(file_path)] = {"error": str(e), "path": file_path}
    
    # Also check in-memory data
    memory_status = {
        "traffic_data_points": len(traffic_data),
        "training_dataset_points": len(training_dataset),
        "data_collection_complete": data_collection_complete,
        "is_model_trained": is_model_trained,
        "gru_model_loaded": gru_model is not None
    }
    
    return jsonify({
        "file_status": file_status,
        "memory_status": memory_status,
        "data_dir": DATA_DIR,
        "config_dir": CONFIG_DIR
    })

@app.route('/debug/mse_updater_status', methods=['GET'])
def debug_mse_updater_status():
    """Check the status of the continuous MSE updater"""
    try:
        return jsonify({
            'mse_updater_running': is_mse_updating,
            'mse_thread_alive': mse_update_thread.is_alive() if mse_update_thread else False,
            'last_mse_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None,
            'seconds_since_last_mse': (datetime.now() - last_mse_calculation).total_seconds() if last_mse_calculation else None,
            'current_mse': {
                'gru': gru_mse if gru_mse != float('inf') else -1,
                'holt_winters': holt_winters_mse if holt_winters_mse != float('inf') else -1
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/force_mse_calculation', methods=['POST'])
def debug_force_mse_calculation():
    """Force MSE calculation and return detailed results"""
    try:
        logger.info("🔧 DEBUG: Forcing MSE calculation...")
        
        # Check prediction history
        gru_predictions = len(predictions_history.get('gru', []))
        hw_predictions = len(predictions_history.get('holt_winters', []))
        gru_matched = len([p for p in predictions_history.get('gru', []) if p.get('matched', False) and not p.get('emergency', False)])
        hw_matched = len([p for p in predictions_history.get('holt_winters', []) if p.get('matched', False) and not p.get('emergency', False)])
        
        # Force MSE calculation
        update_mse_metrics()
        
        result = {
            'forced_calculation': True,
            'timestamp': datetime.now().isoformat(),
            'predictions_before': {
                'gru_total': gru_predictions,
                'gru_matched': gru_matched,
                'hw_total': hw_predictions,
                'hw_matched': hw_matched
            },
            'mse_results': {
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
                'last_calculation': last_mse_calculation.isoformat() if last_mse_calculation else None
            },
            'debug_info': {
                'min_samples_required': config['mse_config']['min_samples_for_mse'],
                'mse_window_size': config['mse_config']['mse_window_size'],
                'mse_enabled': config['mse_config']['enabled']
            }
        }
        
        logger.info(f"🔧 DEBUG: MSE calculation completed - GRU: {result['mse_results']['gru_mse']}, HW: {result['mse_results']['holt_winters_mse']}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"🔧 DEBUG: Force MSE calculation failed: {e}")
        return jsonify({'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

@app.route('/debug/test_scaling', methods=['POST'])
def debug_test_scaling():
    """Test the Kubernetes scaling functionality"""
    try:
        # Get parameters from request or use defaults
        data = request.get_json() if request.is_json else {}
        target_replicas = data.get('replicas', 3)
        deployment_name = data.get('deployment', config.get('target_deployment', 'product-app-combined'))
        namespace = data.get('namespace', config.get('target_namespace', 'default'))
        
        logger.info(f"🔧 DEBUG: Testing scaling of {deployment_name} to {target_replicas} replicas")
        
        # Test the scaling function
        scaling_success = scale_kubernetes_deployment(
            deployment_name=deployment_name,
            namespace=namespace,
            target_replicas=target_replicas
        )
        
        return jsonify({
            'test_scaling_result': scaling_success,
            'deployment': deployment_name,
            'namespace': namespace,
            'target_replicas': target_replicas,
            'timestamp': datetime.now().isoformat(),
            'message': 'Scaling test completed successfully' if scaling_success else 'Scaling test failed'
        })
        
    except Exception as e:
        logger.error(f"🔧 DEBUG: Test scaling failed: {e}")
        return jsonify({'error': str(e), 'timestamp': datetime.now().isoformat()}), 500

@app.route('/debug/coroutine_status', methods=['GET'])
def debug_coroutine_status():
    """Check the status of the two-coroutine system"""
    try:
        return jsonify({
            'two_coroutine_system': {
                'main_coroutine_running': is_main_coroutine_running,
                'update_coroutine_running': is_update_coroutine_running,
                'main_thread_alive': main_coroutine_thread.is_alive() if main_coroutine_thread else False,
                'update_thread_alive': update_coroutine_thread.is_alive() if update_coroutine_thread else False
            },
            'configuration': {
                'prediction_interval': PREDICTION_INTERVAL,
                'update_interval': UPDATE_INTERVAL,
                'gru_retraining_interval': GRU_RETRAINING_INTERVAL,
                'holt_winters_update_interval': HOLT_WINTERS_UPDATE_INTERVAL
            },
            'retraining_status': {
                'last_gru_retraining': last_gru_retraining.isoformat() if last_gru_retraining else None,
                'last_holt_winters_update': last_holt_winters_update.isoformat() if last_holt_winters_update else None,
                'gru_needs_retraining': gru_needs_retraining,
                'seconds_since_gru_retrain': (datetime.now() - last_gru_retraining).total_seconds() if last_gru_retraining else None,
                'seconds_since_hw_update': (datetime.now() - last_holt_winters_update).total_seconds() if last_holt_winters_update else None
            },
            'data_status': {
                'traffic_data_points': len(traffic_data),
                'using_sliding_window': True,
                'window_size_hours': 24,
                'paper_architecture': True
            },
            'real_data_collection': {
                'synthetic_data_enabled': SYNTHETIC_DATA_ENABLED,
                'collection_approach': '24-hour realistic scenario',
                'real_data_count': sum(1 for p in traffic_data if not p.get('synthetic', False)),
                'data_collection_hours': round(len(traffic_data) / 60, 2) if traffic_data else 0,
                'gru_ready_threshold': max(int(config["models"]["gru"]["look_back"]) + 5, 105),
                'gru_ready': len(traffic_data) >= max(int(config["models"]["gru"]["look_back"]) + 5, 105)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/restart_coroutines', methods=['POST'])
def debug_restart_coroutines():
    """Restart the two-coroutine system"""
    try:
        logger.info("🔄 DEBUG: Restarting two-coroutine system...")
        
        # Stop current system
        stop_two_coroutine_system()
        time.sleep(2)  # Give threads time to stop
        
        # Start new system
        start_two_coroutine_system()
        
        return jsonify({
            'success': True,
            'message': 'Two-coroutine system restarted',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔧 DEBUG: Restart coroutines failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/mse_status', methods=['GET'])
def debug_mse_status():
    """Get current MSE calculation status for all models"""
    try:
        def get_model_mse_data(model_name):
            if model_name not in predictions_history:
                return {'matched': 0, 'total': 0, 'mse': None}
            
            predictions = predictions_history[model_name]
            matched_count = sum(1 for p in predictions if p.get('matched', False))
            total_count = len(predictions)
            
            # Calculate current MSE
            mse_value = calculate_mse_robust(model_name)
            mse_display = mse_value if mse_value != float('inf') else None
            
            return {
                'matched': matched_count,
                'total': total_count,
                'mse': mse_display
            }
        
        return jsonify({
            'mse_data': {
                'gru': get_model_mse_data('gru'),
                'holt_winters': get_model_mse_data('holt_winters'),
                
            },
            'predictions_count': {
                'gru': len(predictions_history.get('gru', [])),
                'holt_winters': len(predictions_history.get('holt_winters', [])),
                
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/hybrid_status', methods=['GET'])
def debug_hybrid_status():
    """Check hybrid dataset status and composition"""
    try:
        synthetic_count = sum(1 for p in traffic_data if p.get('synthetic', False))
        real_count = sum(1 for p in traffic_data if not p.get('synthetic', False))
        
        return jsonify({
            'hybrid_dataset': {
                'total_points': len(traffic_data),
                'synthetic_points': synthetic_count,
                'real_points': real_count,
                'synthetic_percentage': round((synthetic_count / len(traffic_data)) * 100, 2) if traffic_data else 0,
                'using_synthetic_data': using_synthetic_data
            },
            'configuration': {
                'synthetic_data_enabled': SYNTHETIC_DATA_ENABLED,
                'synthetic_data_file': SYNTHETIC_DATA_FILE,
                'fallback_hours': FALLBACK_SYNTHETIC_HOURS
            },
            'transition_status': {
                'transition_time': synthetic_to_real_transition_time.isoformat() if synthetic_to_real_transition_time else None,
                'ready_for_transition': real_count >= 120 and synthetic_to_real_transition_time and datetime.now() > synthetic_to_real_transition_time,
                'real_data_threshold': 120,
                'current_real_data': real_count
            },
            'data_ranges': {
                'traffic_range': [min(p['traffic'] for p in traffic_data), max(p['traffic'] for p in traffic_data)] if traffic_data else [0, 0],
                'cpu_range': [min(p['cpu_utilization'] for p in traffic_data), max(p['cpu_utilization'] for p in traffic_data)] if traffic_data else [0, 0],
                'replica_range': [min(p['replicas'] for p in traffic_data), max(p['replicas'] for p in traffic_data)] if traffic_data else [0, 0]
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug/regenerate_synthetic', methods=['POST'])
def debug_regenerate_synthetic():
    """Regenerate synthetic dataset with specified parameters"""
    try:
        data = request.get_json() or {}
        hours = data.get('hours', FALLBACK_SYNTHETIC_HOURS)
        
        logger.info(f"🔧 DEBUG: Regenerating synthetic dataset with {hours} hours...")
        
        # Since synthetic data is disabled, return info message
        return jsonify({
            'success': False,
            'message': 'Synthetic data generation is disabled - using real data collection only',
            'synthetic_data_enabled': SYNTHETIC_DATA_ENABLED,
            'current_data_points': len(traffic_data),
            'timestamp': datetime.now().isoformat()
        })
        
            
    except Exception as e:
        logger.error(f"🔧 DEBUG: Regenerate synthetic failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/force_transition', methods=['POST'])
def debug_force_transition():
    """Force transition from synthetic to real data"""
    try:
        logger.info("🔄 DEBUG: Forcing transition from synthetic to real data...")
        
        real_count = sum(1 for p in traffic_data if not p.get('synthetic', False))
        
        if real_count < 50:
            return jsonify({
                'error': f'Not enough real data for transition: {real_count} < 50',
                'real_data_count': real_count
            }), 400
        
        # Force transition by setting time to past
        global synthetic_to_real_transition_time
        synthetic_to_real_transition_time = datetime.now() - timedelta(minutes=1)
        
        # Since synthetic data is disabled, return info message
        return jsonify({
            'success': False,
            'message': 'Synthetic data is disabled - using real data collection only',
            'synthetic_data_enabled': SYNTHETIC_DATA_ENABLED,
            'real_data_count': len(traffic_data),
            'using_synthetic_data': False,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"🔧 DEBUG: Force transition failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/load_synthetic_file', methods=['POST'])
def debug_load_synthetic_file():
    """Load specific synthetic dataset file"""
    try:
        data = request.get_json() or {}
        filename = data.get('filename', SYNTHETIC_DATA_FILE)
        
        logger.info(f"🔧 DEBUG: Loading synthetic dataset from {filename}...")
        
        # Since synthetic data is disabled, return info message
        return jsonify({
            'success': False,
            'message': 'Synthetic data loading is disabled - using real data collection only',
            'synthetic_data_enabled': SYNTHETIC_DATA_ENABLED,
            'requested_filename': filename,
            'current_data_points': len(traffic_data),
            'timestamp': datetime.now().isoformat()
        })
        
            
    except Exception as e:
        logger.error(f"🔧 DEBUG: Load synthetic file failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/test_hw_metric', methods=['POST'])
def test_hw_metric():
    """Debug endpoint to directly test Holt-Winters metric export."""
    try:
        # Call basic Holt-Winters directly
        hw_pred = predict_with_holtwinters()
        logger.info(f"🔧 DEBUG: Direct HW prediction result: {hw_pred}")
        
        if hw_pred is not None and len(hw_pred) > 0:
            hw_value = int(hw_pred[0])
            predictive_scaler_recommended_replicas.labels(method='holt_winters').set(hw_value)
            logger.info(f"🔧 DEBUG: Set HW metric directly to: {hw_value}")
            
            return jsonify({
                'success': True,
                'hw_prediction': hw_pred,
                'hw_metric_set': hw_value,
                'message': f'Successfully set holtwinters metric to {hw_value}'
            })
        else:
            logger.warning("🔧 DEBUG: HW prediction returned None or empty")
            return jsonify({
                'success': False,
                'hw_prediction': hw_pred,
                'message': 'HW prediction returned None or empty'
            })
            
    except Exception as e:
        logger.error(f"🔧 DEBUG: Test HW metric failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# =====================================================================
# BASELINE DATASET MANAGEMENT ENDPOINTS
# =====================================================================

@app.route('/api/baseline/load', methods=['POST'])
def load_baseline_dataset():
    """
    Load a pre-collected baseline dataset into the autoscaler
    
    Body: {"scenario": "low"|"medium"|"high"}
    
    This endpoint:
    1. Clears all existing traffic data and predictions
    2. Loads the specified baseline dataset
    3. Resets all models to untrained state
    4. Prepares the system for testing with consistent baseline data
    """
    global traffic_data, training_dataset, predictions_history, gru_mse, holt_winters_mse
    global is_model_trained, gru_model, last_training_time
    
    try:
        from baseline_datasets import BaselineDatasetManager
        
        data = request.get_json()
        scenario = data.get('scenario', '').lower()
        
        if scenario not in ['low', 'medium', 'high']:
            return jsonify({
                'success': False,
                'error': f"Invalid scenario: {scenario}. Must be 'low', 'medium', or 'high'"
            }), 400
        
        logger.info(f"📊 Loading baseline dataset for scenario: {scenario}")
        
        # Load baseline data
        manager = BaselineDatasetManager()
        baseline_data = manager.load_baseline(scenario)
        
        # Clear existing data
        traffic_data.clear()
        training_dataset.clear()
        predictions_history = {'gru': [], 'holt_winters': []}
        
        # Reset MSE values
        gru_mse = float('inf')
        holt_winters_mse = float('inf')
        
        # Reset model state
        is_model_trained = False
        gru_model = None
        last_training_time = None
        
        # Load baseline traffic data into traffic_data deque
        for point in baseline_data['traffic_data']:
            traffic_data.append(point)
        
        # Also copy to training_dataset for model training
        training_dataset = list(baseline_data['traffic_data'])
        
        # Save the loaded data
        save_data()
        save_predictions_history()
        
        logger.info(f"✅ Baseline dataset loaded: {len(traffic_data)} points from scenario '{scenario}'")
        
        return jsonify({
            'success': True,
            'scenario': scenario,
            'data_points_loaded': len(traffic_data),
            'collected_at': baseline_data['collected_at'],
            'message': f"Baseline dataset '{scenario}' loaded successfully. All models reset.",
            'next_steps': [
                "Models are now untrained and ready for fresh training",
                "Start your load test to collect new predictions",
                "Or call /rebuild-model to train on baseline data"
            ]
        })
        
    except FileNotFoundError as e:
        logger.error(f"Baseline dataset not found: {e}")
        return jsonify({
            'success': False,
            'error': f"Baseline dataset for '{scenario}' not found. Please collect it first.",
            'hint': "Use baseline_datasets.py collect --scenario <scenario> --duration 3600"
        }), 404
    except Exception as e:
        logger.error(f"Failed to load baseline dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/baseline/list', methods=['GET'])
def list_baseline_datasets():
    """List all available baseline datasets"""
    try:
        from baseline_datasets import BaselineDatasetManager
        
        manager = BaselineDatasetManager()
        baselines = manager.list_available_baselines()
        
        return jsonify({
            'success': True,
            'baselines': baselines,
            'count': len(baselines)
        })
        
    except Exception as e:
        logger.error(f"Failed to list baselines: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/baseline/validate/<scenario>', methods=['GET'])
def validate_baseline_dataset(scenario):
    """Validate a baseline dataset"""
    try:
        from baseline_datasets import BaselineDatasetManager
        
        manager = BaselineDatasetManager()
        validation = manager.validate_baseline(scenario)
        
        return jsonify({
            'success': True,
            'validation': validation
        })
        
    except Exception as e:
        logger.error(f"Failed to validate baseline: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/baseline/clear', methods=['POST'])
def clear_all_data():
    """
    Clear all traffic data and reset autoscaler to clean state
    
    This is called BEFORE starting baseline collection to ensure no old data interferes.
    """
    global traffic_data, training_dataset, predictions_history, gru_mse, holt_winters_mse
    global is_model_trained, gru_model, last_training_time, data_collection_complete
    
    try:
        logger.info("🗑️ Clearing all data and resetting autoscaler...")
        
        # Clear all data structures
        traffic_data.clear()
        training_dataset.clear()
        predictions_history = {'gru': [], 'holt_winters': []}
        
        # Reset MSE values
        gru_mse = float('inf')
        holt_winters_mse = float('inf')
        
        # Reset model state
        is_model_trained = False
        gru_model = None
        last_training_time = None
        data_collection_complete = False
        
        # Save empty state to disk
        save_data()
        save_predictions_history()
        
        logger.info("✅ All data cleared and autoscaler reset to clean state")
        
        return jsonify({
            'success': True,
            'message': 'All data cleared successfully. Ready for fresh baseline collection.',
            'cleared': {
                'traffic_data': True,
                'training_dataset': True,
                'predictions_history': True,
                'model_state': True,
                'mse_values': True
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Failed to validate baseline: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/baseline/delete/<scenario>', methods=['DELETE'])
def delete_baseline_dataset(scenario):
    """Delete a baseline dataset"""
    try:
        from baseline_datasets import BaselineDatasetManager
        
        manager = BaselineDatasetManager()
        manager.delete_baseline(scenario)
        
        return jsonify({
            'success': True,
            'message': f"Baseline dataset '{scenario}' deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Failed to delete baseline: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/baseline/save', methods=['POST'])
def save_current_as_baseline():
    """
    Save current traffic_data as a baseline dataset
    
    Body: {"scenario": "low"|"medium"|"high"}
    
    This is called after collecting data during a load test to save it as a baseline.
    """
    global traffic_data
    
    try:
        from baseline_datasets import BaselineDatasetManager
        
        data = request.get_json()
        scenario = data.get('scenario', '').lower()
        
        if scenario not in ['low', 'medium', 'high']:
            return jsonify({
                'success': False,
                'error': f"Invalid scenario: {scenario}. Must be 'low', 'medium', or 'high'"
            }), 400
        
        if len(traffic_data) < 100:
            return jsonify({
                'success': False,
                'error': 'Insufficient data to save as baseline',
                'current_points': len(traffic_data),
                'required_points': 100
            }), 400
        
        logger.info(f"💾 Saving current data as baseline for scenario: {scenario}")
        
        # Convert deque to list
        traffic_data_list = list(traffic_data)
        
        # Save as baseline
        manager = BaselineDatasetManager()
        metadata = {
            'source': 'autoscaler',
            'saved_from_running_system': True
        }
        
        filepath = manager.save_baseline(scenario, traffic_data_list, metadata)
        
        return jsonify({
            'success': True,
            'scenario': scenario,
            'data_points_saved': len(traffic_data_list),
            'filepath': filepath,
            'message': f"Current data saved as baseline dataset '{scenario}'"
        })
        
    except Exception as e:
        logger.error(f"Failed to save baseline: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# =====================================================================
# MODEL VARIANTS ENDPOINTS
# =====================================================================

@app.route('/api/models/list', methods=['GET'])
def list_available_models():
    """List all available model types"""
    from model_variants import MODEL_TYPES
    
    return jsonify({
        'success': True,
        'models': MODEL_TYPES,
        'count': len(MODEL_TYPES)
    })

@app.route('/api/models/train/<model_name>', methods=['POST'])
def train_specific_model(model_name):
    """Train a specific model variant on current data"""
    global traffic_data, training_dataset
    
    try:
        from model_variants import initialize_model_registry
        
        if len(traffic_data) < 200:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training',
                'current_points': len(traffic_data),
                'required_points': 200
            }), 400
        
        # Initialize registry if not already done
        registry = initialize_model_registry()
        
        # Get the model
        model = registry.get_model(model_name)
        if model is None:
            return jsonify({
                'success': False,
                'error': f"Model '{model_name}' not found",
                'available_models': registry.list_models()
            }), 404
        
        # Convert traffic_data to list format for training
        data_list = list(traffic_data) if training_dataset else list(traffic_data)
        
        # Train the model
        logger.info(f"Training {model_name} with {len(data_list)} data points...")
        result = model.train(data_list)
        
        # Save the model
        if result.get('success'):
            registry.save_all_models()
        
        return jsonify({
            'success': result.get('success', False),
            'model_name': model_name,
            'training_result': result
        })
        
    except Exception as e:
        logger.error(f"Failed to train {model_name}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models/train_all', methods=['POST'])
def train_all_models():
    """Train all model variants on current data"""
    global traffic_data, training_dataset
    
    try:
        from model_variants import initialize_model_registry
        
        if len(traffic_data) < 200:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training',
                'current_points': len(traffic_data),
                'required_points': 200
            }), 400
        
        # Initialize registry
        registry = initialize_model_registry()
        
        # Convert data
        data_list = list(training_dataset) if training_dataset else list(traffic_data)
        
        # Train all models
        logger.info(f"Training all models with {len(data_list)} data points...")
        results = registry.train_all_models(data_list)
        
        # Save all models
        registry.save_all_models()
        
        # Count successes
        successful = sum(1 for r in results.values() if r.get('success'))
        
        return jsonify({
            'success': True,
            'models_trained': successful,
            'total_models': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Failed to train all models: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reset_data', methods=['POST'])
def reset_data():
    """Reset all predictive models and training data"""
    global traffic_data, training_dataset, gru_model, hw_model_hpa, hw_model_combined
    global gru_scaler, gru_predictions, mse_buffer, current_mse, gru_ready_at
    global prediction_thread_running, update_thread_running
    
    try:
        logger.info("🧹 Resetting all predictive models and data...")
        
        # Reset data structures
        traffic_data.clear()
        if training_dataset:
            training_dataset.clear()
        
        # Reset models
        gru_model = None
        hw_model_hpa = None
        hw_model_combined = None
        gru_scaler = None
        
        # Reset prediction buffers
        gru_predictions.clear()
        mse_buffer.clear()
        current_mse = float('inf')
        gru_ready_at = None
        
        # Reset model registry if available
        try:
            from model_variants import initialize_model_registry
            registry = initialize_model_registry()
            registry.reset_all_models()
            logger.info("✅ Model registry reset")
        except Exception as e:
            logger.warning(f"⚠️ Could not reset model registry: {e}")
        
        # Clear any saved model files
        import glob
        for model_file in glob.glob("/app/models/*.pkl") + glob.glob("/app/models/*.h5"):
            try:
                os.remove(model_file)
                logger.info(f"🗑️ Removed model file: {model_file}")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {model_file}: {e}")
        
        return jsonify({
            'success': True,
            'message': 'All predictive models and training data reset successfully',
            'reset_components': [
                'traffic_data',
                'training_dataset', 
                'gru_model',
                'hw_models',
                'prediction_buffers',
                'model_registry',
                'saved_model_files'
            ]
        })
        
    except Exception as e:
        logger.error(f"❌ Failed to reset predictive models: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize paper-based two-coroutine architecture
    logger.info("🚀 Starting predictive autoscaler with Paper-based Two-Coroutine Architecture...")
    logger.info("📄 Implementation: 'A Time Series-Based Approach to Elastic Kubernetes Scaling'")
    logger.info("🔄 Architecture: Main Coroutine (predictions) + Update Coroutine (model training)")
    
    # Log configuration for debugging
    logger.info(f"🎯 Target deployment: {config['target_deployment']}")
    logger.info(f"🎯 Target namespace: {config['target_namespace']}")  
    logger.info(f"🎯 Prometheus server: {config['prometheus_server']}")
    logger.info(f"🎯 CPU threshold: {config['cpu_threshold']}%")
    
    load_data()
    filter_stale_data()
    initialize()
    
    # Real Data Collection Approach (24-hour realistic scenario)
    logger.info("🌐 Real Data Collection: 24-Hour Realistic Scenario")
    logger.info("📊 Models will train as real data is collected")
    logger.info("⏰ GRU training will begin after 105+ data points (~1.75 hours)")
    logger.info("🎯 Full model performance available after 24 hours of operation")
    
    # Check if we have any existing data
    if len(traffic_data) > 0:
        logger.info(f"📈 Found {len(traffic_data)} existing data points")
        
        # Try to load existing models first
        logger.info("🔍 Checking for existing trained models...")
        model_loaded = load_model_components()
        if model_loaded:
            logger.info("✅ Pre-trained GRU model loaded successfully")
        else:
            logger.info("❌ No pre-trained model found or failed to load")
        
        # Try to train models if we have enough data
        gru_config = config["models"]["gru"]  # Define gru_config
        min_hw_data = 10
        min_gru_data = max(int(gru_config["look_back"]) + 5, 105)
        
        if len(traffic_data) >= min_hw_data:
            logger.info("🧠 Attempting Holt-Winters training with existing data...")
            try:
                hw_pred = predict_with_holtwinters(steps=1)
                if hw_pred and len(hw_pred) > 0:
                    logger.info("✅ Holt-Winters ready with existing data")
            except Exception as e:
                logger.info(f"⏰ Holt-Winters not ready yet: {e}")
        
        if len(traffic_data) >= min_gru_data:
            logger.info("🧠 Attempting GRU training with existing data...")
            try:
                # If model loading failed, try to build a new one
                if not model_loaded:
                    logger.info("🔄 Building new GRU model since loading failed...")
                    success = build_gru_model()
                    if success:
                        logger.info("✅ GRU ready with existing data")
                else:
                    logger.info("✅ Using loaded GRU model")
            except Exception as e:
                logger.info(f"⏰ GRU not ready yet: {e}")
        elif not model_loaded:
            logger.info("⏰ Not enough data for GRU training yet, will train when sufficient data is available")
    else:
        logger.info("🆕 Starting fresh - no existing data found")
        logger.info("📊 Beginning real-time data collection for 24-hour realistic scenario")
    
    # Start the paper-based two-coroutine system
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   📊 Prediction Interval: {PREDICTION_INTERVAL}s (Main Coroutine)")
    logger.info(f"   🔄 Update Interval: {UPDATE_INTERVAL}s (Update Coroutine)")
    logger.info(f"   🧠 GRU Retraining: {GRU_RETRAINING_INTERVAL}s")
    logger.info(f"   📈 Holt-Winters Updates: {HOLT_WINTERS_UPDATE_INTERVAL}s")
    
    start_two_coroutine_system()
    
    # Keep legacy MSE updater for compatibility (will be phased out)
    start_mse_updater()
    
    try:
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        stop_two_coroutine_system()
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        stop_two_coroutine_system()
