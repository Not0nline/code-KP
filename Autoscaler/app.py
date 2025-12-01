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
import gc
from collections import deque
from prometheus_client import Counter, Gauge, Summary, Histogram
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_api_client import PrometheusConnect
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import GRU, LSTM, Dense, Dropout
import heapq
import subprocess
import warnings
import yaml
import re

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lstm_model import LSTMPredictor
    LSTM_MODEL_AVAILABLE = True
except ImportError:
    LSTM_MODEL_AVAILABLE = False
    LSTMPredictor = None

try:
    from tree_models import LightGBMPredictor, XGBoostPredictor
    TREE_MODELS_AVAILABLE = True
except ImportError:
    TREE_MODELS_AVAILABLE = False
    LightGBMPredictor = None
    XGBoostPredictor = None

try:
    from statuscale_model import StatuScalePredictor
    STATUSCALE_AVAILABLE = True
except ImportError:
    STATUSCALE_AVAILABLE = False
    StatuScalePredictor = None

try:
    from advanced_models import (ARIMAPredictor, CNNPredictor, AutoencoderPredictor, 
                                ProphetPredictor, EnsemblePredictor, get_available_advanced_models)
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    ARIMAPredictor = None
    CNNPredictor = None
    AutoencoderPredictor = None
    ProphetPredictor = None
    EnsemblePredictor = None

try:
    from enhanced_metrics import (get_metrics_collector, record_model_training, 
                                 record_model_prediction, record_scaling_decision,
                                 update_model_rankings, get_system_metrics,
                                 export_system_metrics, generate_system_report)
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    ENHANCED_METRICS_AVAILABLE = False

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

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


# Define Prometheus metrics
SERVICE_LABEL = os.getenv('APP_NAME', 'predictive-scaler')

# Ensure a consistent 'service' label on custom metrics
def _labels_with_service(**labels):
        labels = dict(labels)
        labels['service'] = SERVICE_LABEL
        return labels

prediction_requests = Counter('prediction_requests_total', 'Number of prediction requests', 
                                                         labelnames=['method'])
scaling_decisions = Counter('scaling_decisions_total', 'Number of scaling decisions', 
                                                    labelnames=['decision'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency in seconds',
                                                         labelnames=['method'])
cpu_utilization = Gauge('current_cpu_utilization', 'Current CPU utilization')
traffic_gauge = Gauge('current_traffic', 'Current traffic level')
recommended_replicas = Gauge('recommended_replicas', 'Number of recommended replicas')
prediction_accuracy = Summary('prediction_accuracy', 'Accuracy of predictions vs actual',
                                                        labelnames=['method'])
http_requests = Counter('http_requests_total', 'Total HTTP requests', 
                                             labelnames=['service','app', 'status_code', 'path'])
http_duration = Histogram('http_request_duration_seconds', 'HTTP request duration',
                                                 labelnames=['service','app'])

# Changed MSE from Gauge to Histogram for better graphing
prediction_mse = Histogram('predictive_scaler_mse', 'Prediction Mean Squared Error', 
                          labelnames=['model'], 
                          buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

training_time = Gauge('predictive_scaler_training_time_ms', 'Model training time in ms',
                     labelnames=['model'])
prediction_time = Gauge('predictive_scaler_prediction_time_ms', 'Prediction time in ms',
                      labelnames=['model'])
predictive_scaler_recommended_replicas = Gauge('predictive_scaler_recommended_replicas', 
                                             'Recommended number of replicas', 
                                             ['method'])
predicted_load_gauge = Gauge('predictive_scaler_predicted_load',
                             'Predicted request load (req/s)',
                             labelnames=['model'])
current_actual_replicas = Gauge('predictive_scaler_actual_replicas', 
                              'Current actual number of replicas')

# Enhanced MSE metrics with -1 for invalid states
current_mse = Gauge('current_model_mse', 'Current MSE for each model (-1 = invalid/insufficient data)', 
                   labelnames=['model'])
model_selection = Counter('model_selection_total', 'Number of times each model was selected',
                         labelnames=['model', 'reason'])

# Best Model Winner Tracking - True/False metric for each model
model_ever_best = Gauge('model_ever_best', 'Binary metric: 1 if model was ever the best (lowest MSE), 0 otherwise',
                       labelnames=['model'])
model_best_count = Counter('model_best_total', 'Total number of times each model achieved best performance',
                          labelnames=['model'])
model_wins_streak = Gauge('model_current_win_streak', 'Current consecutive wins for each model',
                         labelnames=['model'])

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

# Enhanced metrics and model files
METRICS_EXPORT_FILE = os.path.join(DATA_DIR, "metrics_export.csv")
PERFORMANCE_REPORT_FILE = os.path.join(DATA_DIR, "performance_report.json")
ADVANCED_MODEL_DIR = os.path.join(DATA_DIR, "advanced_models")

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

# DATA MANAGEMENT CONFIGURATION
MAX_DATA_POINTS = 2000  # Maintain exactly 2000 points for fair model comparison
DATA_CYCLING_ENABLED = True  # When true: add 1, remove oldest to maintain 2000

# Default configuration with your original hyperparameters
config = {
    "use_gru": False,
    "collection_interval": 60,  # seconds
    "cpu_threshold": 5.0,  # Skip collection below this CPU % (avoid idle periods)
    "training_threshold_minutes": 3,  # Switch to GRU after this (reduced from 5)
    "prometheus_server": os.getenv('PROMETHEUS_SERVER', 
        os.getenv('PROMETHEUS_URL', 
            'http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090')),
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
        },
        "lstm": {
            "look_back": 100,
            "train_size": 2000,
            "batch_size": 10,
            "epochs": 200,
            "units": 50,
            "dropout": 0.2,
            "needTrain": True
        },
        "lightgbm": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "train_size": 2000,  # Use all data points for fair comparison
            "n_lags": 24,
            "look_back": 24,  # For lag features
            "needTrain": True
        },
        "xgboost": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "train_size": 2000,  # Use all data points for fair comparison
            "n_lags": 24,
            "look_back": 24,  # For lag features
            "needTrain": True
        },
        "statuscale": {
            "trend_window": 10,
            "spike_threshold": 2.0,
            "train_size": 2000,  # Rule-based but uses full history for trend analysis
            "needTrain": False  # Rule-based, no training needed
        },
        # ADVANCED MODELS (5 additional models)
        "arima": {
            "order": (1, 1, 1),  # (p, d, q) - auto-selected during training
            "seasonal_order": (1, 1, 1, 12),
            "auto_arima": True,
            "train_size": 2000,
            "needTrain": True
        },
        "cnn": {
            "sequence_length": 20,
            "filters": 16,
            "kernel_size": 2,
            "dense_units": 20,
            "epochs": 10,
            "batch_size": 16,
            "train_size": 2000,
            "needTrain": True
        },
        "autoencoder": {
            "sequence_length": 20,
            "encoding_dim": 16,
            "latent_dim": 8,
            "epochs": 10,
            "batch_size": 16,
            "train_size": 2000,
            "needTrain": True
        },
        "prophet": {
            "seasonality_mode": "multiplicative",
            "daily_seasonality": True,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "train_size": 2000,
            "needTrain": True
        },
        "ensemble": {
            "base_models": ["gru", "lstm", "holt_winters", "arima", "prophet"],
            "weighting_method": "performance",  # or "equal"
            "min_models": 2,  # Minimum models needed to create ensemble
            "needTrain": True
        }
    }
}

# Tracking variables
last_scaling_time = 0
last_scale_up_time = 0
last_scale_down_time = 0
# Use a fixed-size ring buffer (deque) to enforce exactly 2000 most-recent points
# This ensures fair comparison: ALL models use the SAME 2000 data points
# When a new point is added and we're at 2000, the oldest is automatically removed
traffic_data = deque(maxlen=MAX_DATA_POINTS)
training_dataset = []  # Will be set to list(traffic_data) once we have 2000 points
is_collecting = False
collection_thread = None
data_collection_complete = False  # Flag to track if initial 4-hour data collection is complete
data_collection_start_time = None  # Track when data collection started

# Model instances - original models
gru_model = None
lstm_model = None
lightgbm_model = None
xgboost_model = None
statuscale_model = None  # Rule-based, not actually a model object

# Advanced model instances from research paper
arima_model = None
cnn_model = None
autoencoder_model = None
prophet_model = None
ensemble_model = None

# Enhanced metrics collector instance
metrics_collector = None

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Model training status per model (will be set to True after training)
model_training_status = {
    # BASIC MODELS (6 models)
    'gru': False,        # Will be trained with existing implementation
    'holt_winters': True, # Rule-based, always ready
    'lstm': False,       # Will be trained with fixed implementation  
    'lightgbm': False,   # Will be trained with fixed implementation
    'xgboost': False,    # Will be trained with fixed implementation
    'statuscale': False, # Will be initialized (rule-based)
    # ADVANCED MODELS (5 models)
    'arima': False,      # Will be trained from advanced_models.py
    'cnn': False,        # Will be trained from advanced_models.py
    'autoencoder': False,# Will be trained from advanced_models.py
    'prophet': False,    # Will be trained from advanced_models.py
    'ensemble': False    # Will be created after other models are trained
}

# Model performance tracking - tracks which models have ever been "best" (ALL 11 MODELS)
model_performance_tracker = {
    # BASIC MODELS (6 models)
    'gru': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'holt_winters': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'lstm': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'lightgbm': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'xgboost': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'statuscale': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    # ADVANCED MODELS (5 models)
    'arima': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'cnn': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'autoencoder': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'prophet': {'ever_best': False, 'total_wins': 0, 'current_streak': 0},
    'ensemble': {'ever_best': False, 'total_wins': 0, 'current_streak': 0}
}

is_model_trained = False  # Legacy - for backward compatibility
last_training_time = None
start_time = datetime.now()
prometheus_client = None
_last_cpu_value = 0.0
_last_cpu_time = None

# Model configuration (loaded from YAML/ConfigMap or defaults)
model_config_loaded = False
enabled_models = {} # Will be populated dynamically by the function below

def load_dynamic_model_config():
    """Loads model configuration from YAML file and overwrites the enabled_models dict."""
    global enabled_models, model_config_loaded
    if model_config_loaded:
        return # Already loaded
   
    MODEL_CONFIG_PATH = '/app/model-config.yaml'
    try:
        if os.path.exists(MODEL_CONFIG_PATH):
            with open(MODEL_CONFIG_PATH, 'r') as f:
                config_from_file = yaml.safe_load(f)
                if config_from_file and 'models' in config_from_file:
                    enabled_models = config_from_file['models']
                    logger.info(f"Successfully loaded model configuration from {MODEL_CONFIG_PATH}.")
                    logger.info(f"Enabled models: {[m for m, e in enabled_models.items() if e]}")
                    model_config_loaded = True
                    return
    except Exception as e:
        logger.error(f"Error loading model-config.yaml, will use default: {e}")

    # Fallback to default if file not found or is invalid
    logger.warning(f"Falling back to default model configuration.")
    enabled_models = {
        'gru': True, 'holt_winters': True, 'lstm': True, 'lightgbm': True,
        'xgboost': True, 'statuscale': True, 'arima': True, 'cnn': True,
        'autoencoder': True, 'prophet': True, 'ensemble': True
    }
    model_config_loaded = True

# Call the function to load the config at startup
load_dynamic_model_config()
 
# Initialization guards
_init_lock = threading.Lock()

# Traffic monitoring variables
low_traffic_start_time = None  # Track when low traffic period started
consecutive_low_cpu_count = 0  # Track consecutive low CPU measurements

# Enhanced MSE tracking variables with MinHeap system (ALL 11 MODELS)
# The MinHeap system uses MSE (Mean Squared Error) to rank models and make scaling decisions
# Lower MSE = Better model performance = Higher priority in the heap
predictions_history = {
    # BASIC MODELS (6 models)
    'gru': [],
    'holt_winters': [],
    'lstm': [],
    'lightgbm': [],
    'xgboost': [],
    'statuscale': [],
    # ADVANCED MODELS (5 models)
    'arima': [],
    'cnn': [],
    'autoencoder': [],
    'prophet': [],
    'ensemble': []
}

# MSE values for all models (ALL 11 MODELS)
model_mse_values = {
    # BASIC MODELS (6 models)
    'gru': float('inf'),
    'holt_winters': float('inf'),
    'lstm': float('inf'),
    'lightgbm': float('inf'),
    'xgboost': float('inf'),
    'statuscale': float('inf'),
    # ADVANCED MODELS (5 models)
    'arima': float('inf'),
    'cnn': float('inf'),
    'autoencoder': float('inf'),
    'prophet': float('inf'),
    'ensemble': float('inf')
}

# Legacy variables for backward compatibility
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

_last_traffic_total = 0
_last_traffic_timestamp = None

@app.route('/predict', methods=['GET'])
def predict():
    """Placeholder prediction endpoint to satisfy load tester."""
    # In a real scenario, this would trigger a prediction from the best model.
    # For now, return a static response to avoid 404 errors.
    return jsonify({
        "prediction": 1, 
        "model_used": "placeholder",
        "message": "This is a placeholder response."
    })

@app.route('/reset_data', methods=['POST'])
def reset_data():
    """Reset all in-memory data structures for a clean test run."""
    global traffic_data, training_dataset, predictions_history, model_mse_values
    global model_training_status, model_performance_tracker
    global gru_model, lstm_model, lightgbm_model, xgboost_model, statuscale_model
    global arima_model, cnn_model, autoencoder_model, prophet_model, ensemble_model
    global _last_traffic_total, _last_traffic_timestamp

    reset_components = []

    # Reset data collections
    traffic_data.clear()
    training_dataset = []
    _last_traffic_total = 0
    _last_traffic_timestamp = None
    reset_components.append("traffic_data")

    # Reset model predictions history
    for model_name in predictions_history:
        predictions_history[model_name] = []
    reset_components.append("predictions_history")

    # Reset MSE values
    for model_name in model_mse_values:
        model_mse_values[model_name] = float('inf')
    reset_components.append("model_mse_values")

    # Reset training status (except for rule-based models)
    for model_name in model_training_status:
        if model_name != 'holt_winters':
            model_training_status[model_name] = False
    reset_components.append("model_training_status")
    
    # Reset performance tracker
    for model_name in model_performance_tracker:
        model_performance_tracker[model_name] = {'ever_best': False, 'total_wins': 0, 'current_streak': 0}
    reset_components.append("model_performance_tracker")

    # Reset model objects
    gru_model, lstm_model, lightgbm_model, xgboost_model, statuscale_model = None, None, None, None, None
    arima_model, cnn_model, autoencoder_model, prophet_model, ensemble_model = None, None, None, None, None
    reset_components.append("model_objects")
    
    gc.collect() # Force garbage collection
    
    logger.info("Successfully reset application state for new test run.")
    return jsonify({
        "success": True,
        "message": "Application state cleared.",
        "reset_components": reset_components
    })

@app.route('/debug/model_comparison', methods=['GET'])
def get_model_comparison():
    """Return a snapshot of all current model MSEs and performance stats."""
    return jsonify({
        "model_mse_values": model_mse_values,
        "model_performance_tracker": model_performance_tracker
    })

@app.route('/api/baseline/load', methods=['POST'])
def load_baseline_data():
    global traffic_data, training_dataset, data_collection_complete
    try:
        from baseline_datasets import BaselineDatasetManager
        data = request.get_json()
        scenario = data.get('scenario', '').lower()
        if scenario not in ['low', 'medium', 'high']:
            return jsonify({'success': False, 'error': f"Invalid scenario: {scenario}"}), 400
        
        logger.info(f"Loading baseline dataset for scenario: {scenario}")
        manager = BaselineDatasetManager()
        baseline_data = manager.load_baseline(scenario)
        
        traffic_data.clear()
        traffic_data.extend(baseline_data['traffic_data'])
        training_dataset = list(traffic_data)
        data_collection_complete = True
        logger.info(f"Data loaded. Training dataset has {len(training_dataset)} points.")
        
        return jsonify({
            'success': True,
            'scenario': scenario,
            'data_points_loaded': len(training_dataset)
        })
    except Exception as e:
        logger.error(f"Failed during baseline load: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models/train_all', methods=['POST'])
def train_all_models_api():
    global training_dataset
    try:
        from model_variants import initialize_model_registry
        if len(training_dataset) < 200:
            return jsonify({
                'success': False,
                'error': 'Insufficient data for training',
                'current_points': len(training_dataset),
                'required_points': 200
            }), 400
        
        registry = initialize_model_registry(config.get('models'), enabled_models)
        logger.info(f"Training all models with {len(training_dataset)} data points...")
        results = registry.train_all_models(training_dataset)
        registry.save_all_models()
        
        for model_name, result in results.items():
            success = False
            if isinstance(result, bool):
                success = result
            elif isinstance(result, dict):
                success = result.get('success', False)
        
                if success:
                    model_training_status[model_name] = True
                    logger.info(f"Marking {model_name} as READY for predictions.")

        successful = sum(1 for r in results.values() if (isinstance(r, bool) and r) or (isinstance(r, dict) and r.get('success')))
        
        return jsonify({
            'success': True,
            'operation': 'train_all_models',
            'models_trained': successful,
            'total_models': len(results),
            'results': results
        })
    except Exception as e:
        logger.error(f"Failed during model training: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

model_objects = {}


def initialize_models():
    """Instantiate all enabled models and store them."""
    global model_objects
    logger.info("Initializing all predictive models...")

    model_constructors = {
        'lstm': LSTMPredictor,
        'lightgbm': LightGBMPredictor,
        'xgboost': XGBoostPredictor,
        'statuscale': StatuScalePredictor,
        'arima': ARIMAPredictor,
        'cnn': CNNPredictor,
        'autoencoder': AutoencoderPredictor,
        'prophet': ProphetPredictor,
    }

    for name, is_enabled in enabled_models.items():
        if is_enabled and name in model_constructors:
            try:
                model_class = model_constructors[name]
                model_objects[name] = model_class(config['models'].get(name, {}))
                logger.info(f"{name} model initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize model {name}: {e}")
    
    logger.info(f"Initialized {len(model_objects)} models.")


def get_best_model():
    """Get the model with the lowest MSE."""
    valid_mses = {model: mse for model, mse in model_mse_values.items() if mse != float('inf')}
    if not valid_mses:
        return "holt_winters"  # Default fallback
    
    best_model_name = min(valid_mses, key=valid_mses.get)
    return best_model_name

def get_current_replicas(api_instance, namespace, deployment_name):
    """Get current replica count from Kubernetes."""
    try:
        deployment = api_instance.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        return deployment.status.replicas or deployment.spec.replicas
    except k8s_client.ApiException as e:
        logger.error(f"Could not read deployment {deployment_name}: {e}")
        return None

def scale_deployment(api_instance, namespace, deployment_name, replicas):
    """Scale a deployment to a specific number of replicas."""
    global last_scaling_decision_info
    try:
        logger.info(f"Scaling deployment {deployment_name} to {replicas} replicas.")
        api_instance.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body={"spec": {"replicas": replicas}}
        )
        last_scaling_decision_info = {
            'decision': 'scale', 'replicas': replicas,
            'reason': 'Applying model prediction', 'time': datetime.now().isoformat(),
        }
        return True
    except k8s_client.ApiException as e:
        logger.error(f"Could not scale deployment {deployment_name}: {e}")
        last_scaling_decision_info = {
            'decision': 'fail', 'replicas': replicas,
            'reason': str(e), 'time': datetime.now().isoformat(),
        }
        return False

def main_loop():
    """Main coroutine for prediction and scaling."""
    logger.info("--- Entering main_loop ---")
    global last_scaling_time, traffic_data, _last_traffic_total, _last_traffic_timestamp
    
    logger.info("Starting main prediction and scaling loop.")
    
    if K8S_AVAILABLE:
        try:
            k8s_config.load_incluster_config()
            k8s_apps_v1 = k8s_client.AppsV1Api()
            logger.info("Kubernetes client loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Kubernetes in-cluster config: {e}", exc_info=True)
            k8s_apps_v1 = None
    else:
        k8s_apps_v1 = None
        logger.warning("Kubernetes library not available. Scaling will be skipped.")

    try:
        prom = PrometheusConnect(url=config["prometheus_server"], disable_ssl=True)
        logger.info(f"Prometheus client connected to {config['prometheus_server']}.")
    except Exception as e:
        logger.error(f"Failed to connect to Prometheus: {e}", exc_info=True)
        prom = None

    while True:
        try:
            if not k8s_apps_v1:
                logger.error("FATAL: Kubernetes client not available. Main loop cannot proceed. Sleeping for 60s.")
                time.sleep(PREDICTION_INTERVAL)
                continue

            deployment_name = config["target_deployment"]
            namespace = config["target_namespace"]
            
            current_replicas = get_current_replicas(k8s_apps_v1, namespace, deployment_name) or 1
            current_actual_replicas.set(current_replicas)

            target_app_label = os.getenv("TARGET_APP_LABEL", "product-app-combined")
            
            # CPU and Memory queries still use Prometheus
            cpu_query = f'avg(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", pod=~"{target_app_label}-.*", container="product-app"}}[1m])) / avg(kube_pod_container_resource_limits{{namespace="{namespace}", pod=~"{target_app_label}-.*", resource="cpu"}}) * 100'
            memory_query = f'avg(container_memory_working_set_bytes{{namespace="{namespace}", pod=~"{target_app_label}-.*", container="product-app"}}) / avg(kube_pod_container_resource_limits{{namespace="{namespace}", pod=~"{target_app_label}-.*", resource="memory"}}) * 100'

            current_cpu = 0.0
            current_memory = 0.0
            if prom:
                try:
                    current_cpu = float(prom.custom_query(query=cpu_query)[0]['value'][1]) if prom.custom_query(query=cpu_query) else 0.0
                    current_memory = float(prom.custom_query(query=memory_query)[0]['value'][1]) if prom.custom_query(query=memory_query) else 0.0
                except Exception as e:
                    logger.warning(f"Could not query Prometheus for CPU/memory: {e}")

            # --- Prometheus Bypass for Traffic ---
            current_traffic = 0.0
            try:
                import requests
                import re
                metrics_url = "http://load-tester-metrics-service.default.svc.cluster.local:9105/metrics"
                response = requests.get(metrics_url, timeout=5)
                if response.status_code == 200:
                    content = response.text
                    matches = re.findall(r'^load_tester_requests_total{.*?}\s+([0-9\.]+)', content, re.MULTILINE)
                    current_total = sum(float(m) for m in matches)
                    now = time.time()

                    if _last_traffic_timestamp is not None and current_total >= _last_traffic_total:
                        time_delta = now - _last_traffic_timestamp
                        value_delta = current_total - _last_traffic_total
                        logger.info(f"Traffic calculation: current_total={current_total}, last_total={_last_traffic_total}, value_delta={value_delta}, time_delta={time_delta:.2f}s")
                        if time_delta > 1: # Avoid division by zero or stale data
                            current_traffic = value_delta / time_delta
                    else:
                        logger.info(f"Traffic calculation: Seeding initial values. current_total={current_total}")
                    
                    _last_traffic_total = current_total
                    _last_traffic_timestamp = now
                else:
                    logger.warning(f"Failed to fetch traffic metrics directly. Status: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching traffic metrics directly: {e}", exc_info=True)
                _last_traffic_total = 0
                _last_traffic_timestamp = None
            # --- End Bypass ---

            cpu_utilization.set(current_cpu)
            traffic_gauge.set(current_traffic)

            new_data_point = {
                "timestamp": datetime.now().isoformat(), "cpu_utilization": current_cpu,
                "traffic": current_traffic, "replicas": current_replicas, "memory_usage": current_memory
            }
            traffic_data.append(new_data_point)
            logger.info(f"Collected data: {new_data_point}")

            if len(traffic_data) > 10:
                # SHADOW MODE: Run all enabled models for metrics
                for model_name, model_obj in model_objects.items():
                    if model_training_status.get(model_name, False):
                        try:
                            start_p = time.time()
                            pred = model_obj.predict(list(traffic_data))
                            dur_ms = (time.time() - start_p) * 1000
                            
                            if pred is not None and np.isfinite(pred):
                                pred_replicas = int(round(pred))
                                predictions_history[model_name].append({
                                    "timestamp": datetime.now(),
                                    "prediction": pred_replicas,
                                    "actual": None
                                })
                                # Update metrics for every model
                                predictive_scaler_recommended_replicas.labels(method=model_name).set(pred_replicas)
                                if ENHANCED_METRICS_AVAILABLE:
                                    record_model_prediction(model_name, dur_ms, pred_replicas)
                        except Exception as e:
                            logger.warning(f"Shadow prediction failed for {model_name}: {e}")

                # Also run Holt-Winters if not already in model_objects (it's special cased in current app.py)
                if 'holt_winters' not in model_objects: 
                     # (This logic was in the original 'elif' block, moving it here for shadow execution)
                     try:
                        start_p = time.time()
                        history = [d['traffic'] for d in traffic_data if 'traffic' in d]
                        if len(history) > config['models']['holt_winters']['look_backward']:
                            model_hw = ExponentialSmoothing(history, seasonal_periods=config['models']['holt_winters']['slen'], trend='add', seasonal='add', initialization_method='estimated').fit()
                            prediction = model_hw.forecast(steps=1)[0]
                            predicted_load = max(0, prediction)
                            pred_replicas = int(round(predicted_load / 50)) or 1
                            
                            dur_ms = (time.time() - start_p) * 1000
                            predictions_history['holt_winters'].append({
                                    "timestamp": datetime.now(),
                                    "prediction": pred_replicas,
                                    "actual": None
                            })
                            predictive_scaler_recommended_replicas.labels(method='holt_winters').set(pred_replicas)
                            if ENHANCED_METRICS_AVAILABLE:
                                record_model_prediction('holt_winters', dur_ms, pred_replicas)
                     except Exception as e:
                        logger.warning(f"Shadow prediction failed for holt_winters: {e}")


                best_model_name = get_best_model()
                logger.info(f"Best model selected: {best_model_name}")
                
                # Retrieve the prediction we just made (or make it if logic dictates)
                # For simplicity/safety with existing logic, we'll trust the history we just appended to
                if predictions_history[best_model_name]:
                    predicted_replicas = predictions_history[best_model_name][-1]['prediction']
                else:
                    predicted_replicas = current_replicas

                predicted_replicas = max(config['cost_optimization']['min_replicas'], min(predicted_replicas, config['cost_optimization']['max_replicas']))
                logger.info(f"Prediction by {best_model_name}: {predicted_replicas} replicas.")

                cooldown_seconds = config['cost_optimization']['scale_down_delay_minutes'] * 60
                if time.time() - last_scaling_time > cooldown_seconds and predicted_replicas != current_replicas:
                    scale_deployment(k8s_apps_v1, namespace, deployment_name, predicted_replicas)
                    last_scaling_time = time.time()
                    if ENHANCED_METRICS_AVAILABLE:
                        record_scaling_decision(best_model_name, 
                                                'scale_up' if predicted_replicas > current_replicas else 'scale_down', 
                                                True, current_replicas, predicted_replicas)
                else:
                    logger.info(f"Skipping scaling decision due to cooldown or no change needed.")

            else:
                logger.info(f"Not enough data for prediction. Have {len(traffic_data)}, need > 10.")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        time.sleep(PREDICTION_INTERVAL)

def update_loop():
    """Secondary coroutine for model updates and MSE calculations."""
    logger.info("Starting model update and MSE calculation loop.")
    while True:
        try:
            time.sleep(UPDATE_INTERVAL)
            logger.info("Update loop iteration started.")
            
            with file_lock: # Assuming file_lock is defined globally
                live_data = list(traffic_data)

            for model_name in predictions_history:
                history = predictions_history[model_name]
                if not history:
                    continue

                squared_errors = []
                for pred_item in history:
                    if pred_item['actual'] is not None:
                        squared_errors.append((pred_item['prediction'] - pred_item['actual'])**2)
                        continue

                    # Match prediction with actual data
                    pred_time = pred_item['timestamp']
                    for data_point in reversed(live_data):
                        actual_time = datetime.fromisoformat(data_point['timestamp'])
                        if actual_time > pred_time:
                            pred_item['actual'] = data_point['traffic']
                            squared_errors.append((pred_item['prediction'] - data_point['traffic'])**2)
                            break
                
                if squared_errors:
                    mse = np.mean(squared_errors)
                    model_mse_values[model_name] = mse
                    current_mse.labels(model=model_name).set(mse)
                    logger.info(f"Updated MSE for {model_name}: {mse}")

        except Exception as e:
            logger.error(f"Error in update loop: {e}", exc_info=True)

# Workaround: Save metrics to a file periodically
def save_metrics_periodically():
    from prometheus_client import generate_latest, REGISTRY
    METRICS_FILE = os.path.join(DATA_DIR, "metrics_export.txt")
    while True:
        try:
            with open(METRICS_FILE, "w") as f:
                f.write(generate_latest(REGISTRY).decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to save metrics to file: {e}")
        time.sleep(15)

# Start the metrics saving thread in a robust way for Gunicorn
def on_starting(server):
    logger.info("Starting background threads.")
    
    initialize_models()

    metrics_thread = threading.Thread(target=save_metrics_periodically, daemon=True)
    metrics_thread.start()

    global main_coroutine_thread, update_coroutine_thread
    global is_main_coroutine_running, is_update_coroutine_running

    if not is_main_coroutine_running:
        main_coroutine_thread = threading.Thread(target=main_loop, daemon=True)
        main_coroutine_thread.start()
        is_main_coroutine_running = True
        logger.info("Main application thread started.")

    if not is_update_coroutine_running:
        update_coroutine_thread = threading.Thread(target=update_loop, daemon=True)
        update_coroutine_thread.start()
        is_update_coroutine_running = True
        logger.info("Model update thread started.")


# If not running with Gunicorn, start the thread directly for development
if __name__ != '__main__':
    logging.getLogger(__name__).info("Gunicorn worker process starting, __name__ is '%s'. Kicking off background threads.", __name__)
    on_starting(None)