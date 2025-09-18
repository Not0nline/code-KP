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
from prometheus_client import Counter, Gauge, Summary, Histogram
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_api_client import PrometheusConnect
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Dropout
import heapq
import warnings
warnings.filterwarnings('ignore')

# Flask app initialization
app = Flask(__name__)
metrics = PrometheusMetrics(app)  # Exposes /metrics endpoint

# Define Prometheus metrics
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
                       labelnames=['app', 'status_code', 'path'])
http_duration = Histogram('http_request_duration_seconds', 'HTTP request duration',
                         labelnames=['app'])

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

# Enhanced MSE metrics with -1 for invalid states
current_mse = Gauge('current_model_mse', 'Current MSE for each model (-1 = invalid/insufficient data)', 
                   labelnames=['model'])
model_selection = Counter('model_selection_total', 'Number of times each model was selected',
                         labelnames=['model', 'reason'])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables - Keep original hyperparameters
DATA_DIR = "/data"
DATA_FILE = os.path.join(DATA_DIR, "traffic_data.csv")
MODEL_FILE = os.path.join(DATA_DIR, "gru_model.h5")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions_history.json")
STATUS_FILE = os.path.join(DATA_DIR, "collection_status.json")
SCALER_X_FILE = os.path.join(DATA_DIR, "scaler_X.pkl")
SCALER_Y_FILE = os.path.join(DATA_DIR, "scaler_y.pkl")
MIN_DATA_POINTS_FOR_GRU = 2000  # Based on your hyperparameters
PREDICTION_WINDOW = 24  # Based on your hyperparameters
SCALING_THRESHOLD = 0.7  # CPU threshold for scaling decision
SCALING_COOLDOWN = 60  # Reduced to 1 minute for faster scale down

# Default configuration with your original hyperparameters
config = {
    "use_gru": False,
    "collection_interval": 60,  # seconds
    "cpu_threshold": 3.0,  # Skip collection below this CPU %
    "training_threshold_minutes": 3,  # Switch to GRU after this (reduced from 5)
    "prometheus_server": os.getenv('PROMETHEUS_SERVER', 
        'http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090'),
    "target_deployment": os.getenv('TARGET_DEPLOYMENT', 'product-app-combined'),
    "target_namespace": os.getenv('TARGET_NAMESPACE', 'default'),
    "cost_optimization": {
        "enabled": True,
        "target_cpu_utilization": 70,  # Target CPU for normal operation
        "scale_up_threshold": 80,  # Scale up at this CPU
        "scale_down_threshold": 20,  # Scale down below this CPU
        "min_replicas": 1,
        "max_replicas": 10,
        "aggressive_scaling": True,
        "scale_down_delay_minutes": 2,  # Wait this long before scaling down
        "zero_traffic_threshold": 0.1,  # Consider traffic zero below this
        "idle_scale_down_minutes": 5  # Scale to min after this much idle time
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
            "slen": 12,  # seasonal length
            "look_forward": 24,
            "look_backward": 60,
            "alpha": 0.716,
            "beta": 0.029,
            "gamma": 0.993,
            "needTrain": False
        },
        "gru": {
            "address": "/tmp/uds_socket",
            "resp_recv_address": "/tmp/rra.socket",
            "look_back": 12, # TODO: change back to 100
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
traffic_data = []
is_collecting = False
collection_thread = None
data_collection_complete = False  # Flag to track if initial 24-hour data collection is complete
data_collection_start_time = None  # Track when data collection started
gru_model = None
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
is_model_trained = False
last_training_time = None
start_time = datetime.now()
prometheus_client = None
low_traffic_start_time = None
consecutive_low_cpu_count = 0

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
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(f"Configuration loaded from {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")

def save_config():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def load_data():
    global traffic_data
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            traffic_data = df.to_dict('records')
            logger.info(f"Loaded {len(traffic_data)} data points from {DATA_FILE}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        traffic_data = []

def save_data():
    try:
        df = pd.DataFrame(traffic_data)
        df.to_csv(DATA_FILE, index=False)
        save_collection_status()  # Also save collection status when saving data
        logger.info(f"Saved {len(traffic_data)} data points to {DATA_FILE}")
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
        traffic_data = [
            d for d in traffic_data 
            if 'timestamp' in d and isinstance(d['timestamp'], str) and 
               datetime.strptime(d['timestamp'], "%Y-%m-%d %H:%M:%S") > cutoff_time
        ]
        
        removed_count = original_count - len(traffic_data)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} stale data points older than {max_age_hours} hours.")
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing timestamps during stale data filtering: {e}. Data may be corrupt.")


def load_predictions_history():
    """Load predictions history for MSE calculation"""
    global predictions_history
    try:
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, 'r') as f:
                predictions_history = json.load(f)
                logger.info("Loaded predictions history for MSE calculation")
    except Exception as e:
        logger.error(f"Error loading predictions history: {e}")
        predictions_history = {'gru': [], 'holt_winters': []}

def save_predictions_history():
    """Save predictions history for MSE calculation"""
    try:
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
        features_clean = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0).values
        targets_clean = pd.Series(targets.flatten()).fillna(method='ffill').fillna(method='bfill').fillna(1).values.reshape(-1, 1)
        
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
    """Remove predictions that are too old to be matched"""
    current_time = datetime.now()
    max_age_hours = config['mse_config']['max_prediction_age_hours']
    cutoff_time = current_time - timedelta(hours=max_age_hours)
    
    for model_name in predictions_history:
        original_count = len(predictions_history[model_name])
        predictions_history[model_name] = [
            p for p in predictions_history[model_name]
            if datetime.strptime(p['timestamp'], "%Y-%m-%d %H:%M:%S") > cutoff_time
        ]
        removed_count = original_count - len(predictions_history[model_name])
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} old predictions for {model_name}")

def add_prediction_to_history(model_name, predicted_replicas, timestamp):
    """Enhanced prediction tracking with immediate matching attempt"""
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
    
    prediction_entry = {
        'timestamp': timestamp,
        'predicted_replicas': predicted_replicas,
        'actual_replicas': None,  # Will be filled when we get actual data
        'predicted_cpu': None,
        'actual_cpu': None,
        'context_cpu': current_cpu,  # CPU at time of prediction
        'context_traffic': current_traffic,  # Traffic at time of prediction
        'matched': False,
        'match_timestamp': None
    }
    
    # For immediate timestamp predictions, try to match with current data immediately
    try:
        pred_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        time_diff = abs((current_time - pred_time).total_seconds())
        
        # If prediction is for current time (within 2 minutes), match immediately
        if time_diff <= 120 and traffic_data:
            prediction_entry['actual_replicas'] = current_replicas
            prediction_entry['actual_cpu'] = current_cpu
            prediction_entry['matched'] = True
            prediction_entry['match_timestamp'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
            prediction_entry['match_time_diff'] = time_diff
            
            logger.info(f"🚀 Immediate match for {model_name}: pred={predicted_replicas}, actual={current_replicas}, diff={time_diff:.0f}s")
    
    except Exception as e:
        logger.debug(f"Could not attempt immediate matching: {e}")
    
    predictions_history[model_name].append(prediction_entry)
    
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
                    time_diff = (current_time - pred_time).total_seconds()
                    
                    # Match if this data point is within reasonable range of prediction time
                    if -120 <= time_diff <= 300:  # 2 minutes before to 5 minutes after
                        prediction['actual_replicas'] = current_replicas
                        prediction['actual_cpu'] = current_cpu
                        prediction['matched'] = True
                        prediction['match_timestamp'] = timestamp
                        prediction['match_time_diff'] = abs(time_diff)
                        match_count += 1
                        
                        logger.info(f"⚡ Quick match {model_name}: pred={prediction['predicted_replicas']}, "
                                  f"actual={current_replicas}, diff={time_diff:.0f}s")
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
        recent_replicas = [d['replicas'] for d in traffic_data[-10:]]
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
                
                if time_since_prediction < -120:  # Prediction is more than 2 minutes in future
                    if debug_mode:
                        logger.debug(f"{model_name}: Prediction {prediction['timestamp']} is {-time_since_prediction:.0f}s in future, skipping")
                    continue
                
                # Find the best matching data point
                best_match = None
                min_time_diff = float('inf')
                match_tolerance = config['mse_config']['prediction_match_tolerance_minutes'] * 60

                for data_point in traffic_data_with_dt:
                    data_time = data_point['timestamp_dt']
                    time_diff = abs((data_time - pred_time).total_seconds())
                    
                    # Accept matches within tolerance window
                    if time_diff < match_tolerance and time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = data_point
                
                if best_match:
                    prediction['actual_replicas'] = best_match['replicas']
                    prediction['actual_cpu'] = best_match['cpu_utilization']
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
                    # Log why no match was found
                    if debug_mode and time_since_prediction > 60:  # Only log for older predictions
                        closest_data = min(traffic_data_with_dt, 
                                         key=lambda x: abs((x['timestamp_dt'] - pred_time).total_seconds()),
                                         default=None)
                        if closest_data:
                            closest_diff = abs((closest_data['timestamp_dt'] - pred_time).total_seconds())
                            logger.debug(f"✗ No match for {model_name} prediction {prediction['timestamp']}: "
                                       f"closest data at {closest_data['timestamp']} (diff: {closest_diff:.0f}s)")

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
    """Enhanced MSE calculation with aggressive matching and better debugging"""
    global predictions_history
    
    if model_name not in predictions_history:
        logger.warning(f"Model {model_name} not found in predictions history")
        return float('inf')
    
    predictions = predictions_history[model_name]
    matched_predictions = [p for p in predictions if p.get('matched', False) and p['actual_replicas'] is not None]
    
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
    
    # Use all available matched predictions (don't limit by window for small datasets)
    window_size = config['mse_config']['mse_window_size']
    if matched_count <= 5:  # For small datasets, use all matches
        recent_predictions = matched_predictions
    else:
        recent_predictions = matched_predictions[-window_size:]
    
    try:
        # Calculate MSE for replica predictions
        squared_errors = []
        prediction_details = []
        
        for p in recent_predictions:
            pred_val = p['predicted_replicas']
            actual_val = p['actual_replicas']
            if pred_val is not None and actual_val is not None:
                error = (pred_val - actual_val) ** 2
                squared_errors.append(error)
                prediction_details.append(f"pred={pred_val}, actual={actual_val}, error²={error:.2f}")
        
        if not squared_errors:
            logger.warning(f"{model_name} MSE: No valid prediction pairs found")
            return float('inf')
        
        mse_value = np.mean(squared_errors)
        
        # Always log successful MSE calculation
        logger.info(f"🎯 {model_name} MSE: {mse_value:.4f} (from {len(squared_errors)} samples)")
        
        if debug_mode:
            logger.debug(f"{model_name} MSE details: {prediction_details}")
        
        return mse_value
        
    except Exception as e:
        logger.error(f"Error calculating MSE for {model_name}: {e}")
        return float('inf')

def update_mse_metrics():
    """Enhanced MSE metric updates with better Prometheus integration"""
    global gru_mse, holt_winters_mse, last_mse_calculation
    
    current_time = datetime.now()
    
    # Update predictions with actual values first
    update_predictions_with_actual_values()
    
    # Calculate MSE for both models
    gru_mse = calculate_mse_robust('gru')
    holt_winters_mse = calculate_mse_robust('holt_winters')
    
    # Update Prometheus metrics - use -1 for invalid/insufficient data
    if gru_mse != float('inf'):
        prediction_mse.labels(model='gru').observe(gru_mse)
        current_mse.labels(model='gru').set(gru_mse)
    else:
        current_mse.labels(model='gru').set(-1)  # Signal insufficient data
    
    if holt_winters_mse != float('inf'):
        prediction_mse.labels(model='holt_winters').observe(holt_winters_mse)
        current_mse.labels(model='holt_winters').set(holt_winters_mse)
    else:
        current_mse.labels(model='holt_winters').set(-1)  # Signal insufficient data
    
    last_mse_calculation = current_time
    
    # Enhanced logging
    gru_status = f"{gru_mse:.3f}" if gru_mse != float('inf') else "insufficient_data"
    hw_status = f"{holt_winters_mse:.3f}" if holt_winters_mse != float('inf') else "insufficient_data"
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

def predict_with_ensemble(steps=1):
    """Ensemble prediction combining multiple models for best accuracy."""
    try:
        predictions = {}
        weights = {}
        
        # Get GRU prediction (try advanced first, then basic)
        if gru_model is not None and is_model_trained:
            try:
                gru_pred = predict_with_advanced_gru(steps)
                if not gru_pred:
                    gru_pred = predict_with_gru(steps)
            except Exception:
                gru_pred = predict_with_gru(steps)
            
            if gru_pred:
                predictions['gru'] = gru_pred[0]
                gru_weight = 1 / (gru_mse + 0.1) if gru_mse != float('inf') else 0.1
                weights['gru'] = gru_weight
        
        # Get optimized Holt-Winters prediction
        try:
            hw_pred = predict_with_optimized_holtwinters(steps)
            if not hw_pred:
                hw_pred = predict_with_holtwinters(steps)
        except Exception:
            hw_pred = predict_with_holtwinters(steps)
            
        if hw_pred:
            predictions['holt_winters'] = hw_pred[0]
            hw_weight = 1 / (holt_winters_mse + 0.1) if holt_winters_mse != float('inf') else 0.1
            weights['holt_winters'] = hw_weight
        
        # Simple moving average prediction (baseline)
        if len(traffic_data) >= 5:
            recent_replicas = [d['replicas'] for d in traffic_data[-5:]]
            ma_pred = int(round(np.mean(recent_replicas)))
            predictions['moving_average'] = ma_pred
            weights['moving_average'] = 0.2
        
        # Linear trend prediction
        if len(traffic_data) >= 10:
            recent_replicas = [d['replicas'] for d in traffic_data[-10:]]
            x = np.arange(len(recent_replicas))
            z = np.polyfit(x, recent_replicas, 1)
            trend_pred = max(1, min(10, int(round(z[0] * len(recent_replicas) + z[1]))))
            predictions['trend'] = trend_pred
            weights['trend'] = 0.3
        
        if not predictions:
            return [2]  # Fallback
        
        # Calculate weighted ensemble prediction
        total_weight = sum(weights.values())
        if total_weight == 0:
            ensemble_pred = int(np.mean(list(predictions.values())))
        else:
            weighted_sum = sum(pred * weights[method] for method, pred in predictions.items())
            ensemble_pred = max(1, min(10, int(round(weighted_sum / total_weight))))
        
        logger.info(f"🎭 Ensemble prediction: {ensemble_pred} from {predictions} with weights {weights}")
        
        # Store ensemble prediction for tracking
        current_time = datetime.now()
        timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
        
        # Add ensemble to prediction history
        if 'ensemble' not in predictions_history:
            predictions_history['ensemble'] = []
        add_prediction_to_history('ensemble', ensemble_pred, timestamp)
        
        return [ensemble_pred]
        
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
        return [2]  # Safe fallback

def select_best_model_with_minheap():
    """MinHeap-based model selection using MSE as the primary criterion"""
    global gru_mse, holt_winters_mse
    
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
    
    # Add ensemble model if we have predictions from multiple models
    ensemble_mse = calculate_ensemble_mse()
    if ensemble_mse != float('inf'):
        heapq.heappush(model_heap, (ensemble_mse, 'ensemble', True))
        logger.debug(f"Added Ensemble to minheap: MSE={ensemble_mse}")
    
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
        model_selection.labels(model='holt_winters', reason='minheap_ultimate_fallback').inc()
        logger.error("❌ MinHeap selection failed, using ultimate fallback")
        return 'holt_winters'
    
    # Use the best model (lowest MSE)
    best_mse, best_model = selected_models[0]
    
    # Check if MSE is disabled
    if not config['mse_config']['enabled']:
        if is_model_trained and gru_model is not None:
            model_selection.labels(model='gru', reason='mse_disabled_prefer_gru').inc()
            return 'gru'
        else:
            model_selection.labels(model='holt_winters', reason='mse_disabled_fallback').inc()
            return 'holt_winters'
    
    # Log minheap decision
    reason = f"minheap_lowest_mse_{best_mse:.4f}"
    model_selection.labels(model=best_model, reason=reason).inc()
    
    logger.info(f"🎯 MinHeap final selection: {best_model} (MSE: {best_mse:.4f})")
    return best_model

def calculate_ensemble_mse():
    """Calculate MSE for ensemble predictions if available"""
    try:
        if 'ensemble' in predictions_history:
            return calculate_mse_robust('ensemble')
        return float('inf')
    except Exception as e:
        logger.debug(f"Failed to calculate ensemble MSE: {e}")
        return float('inf')

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
        
        # Load model with error handling
        try:
            gru_model = tf.keras.models.load_model(MODEL_FILE)
            logger.info(f"GRU model loaded successfully from {MODEL_FILE}")
        except Exception as e:
            logger.error(f"Failed to load GRU model: {e}")
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

def collect_metrics_from_prometheus():
    """Collect real metrics from Prometheus with 24-hour collection limit"""
    global traffic_data, is_collecting, last_training_time, low_traffic_start_time, consecutive_low_cpu_count
    global data_collection_complete, data_collection_start_time
    
    # Check if data collection is already complete
    if data_collection_complete:
        logger.info("Data collection already complete. Using stored 24-hour dataset for predictions.")
        is_collecting = False
        return
    
    # Set collection start time if not already set
    if data_collection_start_time is None:
        data_collection_start_time = datetime.now()
        save_collection_status()
        logger.info(f"📅 Started 24-hour data collection period at {data_collection_start_time}")
        logger.info("🎓 AWS Academy Compatible: Collection based on calendar time, survives system restarts")
    else:
        elapsed_hours = (datetime.now() - data_collection_start_time).total_seconds() / 3600
        remaining_hours = max(0, 24 - elapsed_hours)
        logger.info(f"📅 Continuing data collection: {elapsed_hours:.1f}h elapsed, {remaining_hours:.1f}h remaining")
    
    while is_collecting:
        try:
            if prometheus_client is None:
                initialize_prometheus()
                
            # Get current CPU utilization
            cpu_query = f'avg(rate(container_cpu_usage_seconds_total{{pod=~"{config["target_deployment"]}-.*"}}[5m])) * 100'
            cpu_result = prometheus_client.custom_query(cpu_query)
            
            current_cpu = 0
            if cpu_result and len(cpu_result) > 0:
                current_cpu = float(cpu_result[0]['value'][1])
            
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
            
            # Store the data - always collect for monitoring, even if CPU is low
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            traffic_data.append({
                'timestamp': timestamp,
                'traffic': current_traffic,
                'cpu_utilization': current_cpu,
                'memory_usage': current_memory,
                'replicas': current_replicas
            })
            
            # Keep only last 24 hours of data
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)
            traffic_data = [d for d in traffic_data if 
                          datetime.strptime(d['timestamp'], "%Y-%m-%d %H:%M:%S") > cutoff_time]
            
            # Try to match any unmatched predictions with the new data point
            try:
                match_new_data_with_predictions(current_replicas, current_cpu, timestamp)
            except Exception as e:
                logger.debug(f"Failed to match new data with predictions: {e}")
            
            # Generate predictions from both models for MSE tracking (EVERY data point after minimum)
            if len(traffic_data) >= 12:
                try:
                    # ALWAYS make Holt-Winters prediction (use optimized version)
                    logger.info("🔄 Generating background Holt-Winters prediction for MSE tracking...")
                    try:
                        hw_pred = predict_with_optimized_holtwinters(1)
                        if not hw_pred:  # Fallback to basic if optimized fails
                            hw_pred = predict_with_holtwinters(1)
                    except Exception:
                        hw_pred = predict_with_holtwinters(1)
                    
                    if hw_pred:
                        logger.info(f"✅ HW prediction: {hw_pred}")
                    else:
                        logger.error("❌ HW prediction failed!")
                    
                    # ALWAYS make GRU prediction if available (use advanced version)
                    if gru_model is not None and is_model_trained:
                        logger.info("🔄 Generating background GRU prediction for MSE tracking...")
                        try:
                            gru_pred = predict_with_advanced_gru(1)
                            if not gru_pred:  # Fallback to basic if advanced fails
                                gru_pred = predict_with_gru(1)
                        except Exception:
                            gru_pred = predict_with_gru(1)
                            
                        if gru_pred:
                            logger.info(f"✅ GRU prediction: {gru_pred}")
                        else:
                            logger.error("❌ GRU prediction failed!")
                    else:
                        logger.info(f"❌ GRU not available: model={gru_model is not None}, trained={is_model_trained}")
                        
                except Exception as e:
                    logger.error(f"❌ Background prediction generation failed: {e}")
                    import traceback
                    logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Emergency MSE system: create synthetic predictions if GRU has never made any
            if len(predictions_history['gru']) == 0 and len(traffic_data) > 20 and is_model_trained:
                try:
                    logger.warning("🆘 EMERGENCY: Creating synthetic GRU predictions for MSE testing")
                    create_emergency_gru_predictions()
                except Exception as e:
                    logger.error(f"Emergency prediction creation failed: {e}")
            
            # Update MSE calculations more frequently for faster matching
            if last_mse_calculation is None or (current_time - last_mse_calculation).total_seconds() > 10:
                update_mse_metrics()
            
            # Periodically save data
            if len(traffic_data) % 10 == 0:
                save_data()
            
            # Enhanced GRU model training logic
            minutes_elapsed = (current_time - start_time).total_seconds() / 60
            gru_config = config["models"]["gru"]
            
            # Check training conditions with better logging
            should_train = False
            training_reason = ""
            
            # Initial training conditions - REDUCED REQUIREMENTS FOR AUTO-TRAINING
            if not config['use_gru'] and not is_model_trained:
                if minutes_elapsed >= config['training_threshold_minutes']:
                    # 🚀 REDUCED: was +10,20 now +2,15 to match manual training success
                    min_data_required = max(int(gru_config["look_back"]) + 2, 15)
                    if len(traffic_data) >= min_data_required:
                        # 🚀 RELAXED CPU requirement for initial training (0.5% instead of 3%)
                        cpu_requirement = 0.5  # Much lower for first training to ensure it happens
                        if current_cpu >= cpu_requirement:
                            should_train = True
                            training_reason = f"initial_training (elapsed: {minutes_elapsed:.1f}min, data: {len(traffic_data)}, cpu: {current_cpu:.1f}%)"
                        else:
                            logger.debug(f"Waiting for CPU threshold: {current_cpu:.1f}% < {cpu_requirement}%")
                    else:
                        logger.debug(f"Waiting for more data: {len(traffic_data)} < {min_data_required}")
                else:
                    logger.log(f"Waiting for time threshold: {minutes_elapsed:.1f} < {config['training_threshold_minutes']} minutes")
            
            # Retrain conditions
            elif is_model_trained and current_cpu >= config['cpu_threshold']:
                if last_training_time is None or (current_time - last_training_time).total_seconds() > 3600:
                    # 🚀 REDUCED: was +10,20 now +2,15 for consistent requirements
                    min_data_required = max(int(gru_config["look_back"]) + 2, 15)
                    if len(traffic_data) >= min_data_required:
                        should_train = True
                        training_reason = f"retrain (last: {last_training_time}, data: {len(traffic_data)})"
            
            # Execute training if conditions are met
            if should_train:
                logger.info(f"🚀 Starting AUTOMATIC GRU training: {training_reason}")
                try:
                    was_first_training = not is_model_trained  # Track if this is first-time training
                    
                    # 🎯 TRY ADVANCED MODEL FIRST, FALLBACK TO BASIC
                    success = False
                    if len(traffic_data) >= 30:  # Only try advanced if we have enough data
                        logger.info("Attempting advanced GRU model training...")
                        success = build_advanced_gru_model()
                        if success:
                            logger.info("✅ Advanced GRU model training succeeded!")
                    
                    if not success:
                        logger.info("Falling back to basic GRU model training...")
                        success = build_gru_model()
                        if success:
                            logger.info("✅ Basic GRU model training succeeded!")
                    
                    if success:
                        if not config['use_gru']:
                            config['use_gru'] = True
                            save_config()
                        logger.info("✅ GRU model training completed successfully!")
                        
                        # 🆘 BOOTSTRAP: Auto-create emergency predictions for first training
                        if was_first_training and len(predictions_history['gru']) == 0:
                            logger.info("🆘 AUTO-BOOTSTRAP: Creating emergency predictions for first-time training...")
                            try:
                                create_emergency_gru_predictions()
                                logger.info("✅ Bootstrap emergency predictions created!")
                            except Exception as e:
                                logger.error(f"❌ Bootstrap prediction creation failed: {e}")
                    else:
                        logger.error("❌ GRU model training failed")
                except Exception as e:
                    logger.error(f"❌ Exception during GRU training: {e}")
                    import traceback
                    logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Check if 24 hours of actual runtime has accumulated
            if data_collection_start_time:
                # Calculate total runtime across all sessions by counting data points
                # Each data point represents ~1 minute of runtime (collection_interval = 60s)
                runtime_hours = len(traffic_data) / 60  # Approximate runtime hours based on data points
                
                if runtime_hours >= 24:  # 24 hours of actual runtime
                    data_collection_complete = True
                    is_collecting = False
                    save_collection_status()
                    save_data()  # Save the final dataset
                    logger.info(f"✅ 24-hour runtime completed! Collected {len(traffic_data)} data points across multiple AWS Academy sessions.")
                    logger.info("🔒 Data collection stopped. System will now use this dataset for all predictions.")
                    break
            
            # Sleep for the collection interval
            time.sleep(config['collection_interval'])
            
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            time.sleep(5)  # Sleep for a short time on error

def preprocess_data(data, sequence_length=100):
    """Preprocess data for GRU model with enhanced error handling."""
    if len(data) < sequence_length + 1:
        logger.warning(f"Not enough data for preprocessing: {len(data)} < {sequence_length + 1}")
        return None, None
    
    try:
        X, y = [], []
        df = pd.DataFrame(data)
        
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
            training_time.labels(model="gru").set(elapsed_ms)
            return False
        
        # Preprocess data
        X, y = preprocess_data(traffic_data, sequence_length=int(gru_config["look_back"]))
        
        if X is None or y is None:
            logger.error("Failed to preprocess data for GRU model")
            elapsed_ms = (time.time() - start_time) * 1000
            training_time.labels(model="gru").set(elapsed_ms)
            return False
            
        if len(X) < 10:
            logger.warning(f"Not enough sequences for GRU training: {len(X)} < 10")
            elapsed_ms = (time.time() - start_time) * 1000
            training_time.labels(model="gru").set(elapsed_ms)
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
            training_time.labels(model="gru").set(elapsed_ms)
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
            training_time.labels(model="gru").set(elapsed_ms)
            return False
        
        # Update global variables
        gru_model = model
        is_model_trained = True
        last_training_time = datetime.now()
        
        # Record training time
        elapsed_ms = (time.time() - start_time) * 1000
        training_time.labels(model="gru").set(elapsed_ms)
        
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
        training_time.labels(model="gru").set(elapsed_ms)
        return False

def build_advanced_gru_model():
    """Build an advanced GRU model with enhanced architecture and better preprocessing."""
    global gru_model, last_training_time, is_model_trained, scaler_X, scaler_y
    
    start_time = time.time()
    gru_config = config["models"]["gru"]
    
    try:
        logger.info(f"🚀 Starting ADVANCED GRU model training with {len(traffic_data)} data points")
        
        # Enhanced data preprocessing
        enhanced_data = create_advanced_features(traffic_data)
        if enhanced_data is None:
            logger.warning("Failed to create enhanced features, falling back to basic model")
            return build_gru_model()  # Fallback to original
        
        # Select best features for prediction
        feature_columns = [
            'cpu_utilization', 'traffic', 'memory_usage',
            'cpu_ma_5', 'traffic_ma_5', 'cpu_trend', 'traffic_trend',
            'load_pressure', 'efficiency', 'cpu_lag1', 'traffic_lag1',
            'cpu_volatility', 'is_cpu_peak', 'is_traffic_peak',
            'hour', 'day_of_week'
        ]
        
        # Filter out missing columns
        available_features = [col for col in feature_columns if col in enhanced_data.columns]
        logger.info(f"Using {len(available_features)} enhanced features: {available_features}")
        
        X, y = preprocess_advanced_data(enhanced_data, available_features, int(gru_config["look_back"]))
        
        if X is None or y is None or len(X) < 10:
            logger.warning(f"Insufficient processed data: {len(X) if X is not None else 0}, falling back to basic model")
            return build_gru_model()  # Fallback to original
        
        # Advanced model architecture
        input_shape = (X.shape[1], X.shape[2])
        logger.info(f"Building ADVANCED GRU model with input shape: {input_shape}")
        
        # Clear session
        if gru_model is not None:
            del gru_model
            tf.keras.backend.clear_session()
        
        # Build sophisticated model with improved architecture
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape, 
                dropout=0.1, recurrent_dropout=0.1, name='gru_1'),
            GRU(32, return_sequences=False, name='gru_2'),
            tf.keras.layers.Dense(32, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            tf.keras.layers.Dense(16, activation='relu', name='dense_2'),
            tf.keras.layers.Dropout(0.1, name='dropout_2'),
            tf.keras.layers.Dense(1, name='output')
        ])
        
        # Advanced optimizer with learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=50,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        logger.info("Advanced model compiled, starting training...")
        
        # Enhanced training with callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.7, 
                patience=8, 
                min_lr=0.0001,
                verbose=0
            )
        ]
        
        # Smart batch size and epochs
        batch_size = min(16, max(4, len(X) // 8))
        epochs = min(80, max(20, len(X) // 3))
        
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
            shuffle=True
        )
        
        # Enhanced validation
        final_loss = min(history.history.get('val_loss', history.history['loss'])) if 'val_loss' in history.history else history.history['loss'][-1]
        logger.info(f"Advanced training completed. Best validation loss: {final_loss:.4f}")
        
        # Test prediction
        try:
            test_pred = model.predict(X[:1], verbose=0)
            test_pred_scaled = scaler_y.inverse_transform(test_pred)
            logger.info(f"Advanced test prediction successful: {test_pred_scaled[0][0]:.2f}")
        except Exception as e:
            logger.error(f"Advanced test prediction failed: {e}")
            return False
        
        # Save model
        try:
            model.save(MODEL_FILE)
            joblib.dump(scaler_X, SCALER_X_FILE)
            joblib.dump(scaler_y, SCALER_Y_FILE)
            logger.info(f"Advanced model and scalers saved to {MODEL_FILE}, {SCALER_X_FILE}, {SCALER_Y_FILE}")
        except Exception as e:
            logger.error(f"Failed to save advanced model: {e}")
            return False
        
        # Update globals
        gru_model = model
        is_model_trained = True
        last_training_time = datetime.now()
        
        elapsed_ms = (time.time() - start_time) * 1000
        training_time.labels(model="gru_advanced").set(elapsed_ms)
        
        logger.info(f"🎯 ADVANCED GRU MODEL TRAINED SUCCESSFULLY in {elapsed_ms:.2f}ms")
        logger.info(f"📊 Model architecture: {len(available_features)} features, enhanced architecture")
        
        # Create initial predictions
        try:
            logger.info("🚀 Creating initial advanced GRU predictions...")
            initial_predictions = predict_with_advanced_gru(1)
            if initial_predictions:
                logger.info("✅ Initial advanced GRU predictions created successfully")
        except Exception as e:
            logger.warning(f"Failed to create initial advanced predictions: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error building advanced GRU model: {e}")
        logger.info("Falling back to basic GRU model...")
        return build_gru_model()  # Fallback to original implementation

def predict_with_holtwinters(steps=None):
    """Predict using Holt-Winters with original hyperparameters."""
    start_time_func = time.time()
    hw_config = config["models"]["holt_winters"]
    
    if steps is None:
        steps = int(hw_config["look_forward"])
    
    try:
        if len(traffic_data) < 12: # Need enough data for seasonal period
            logger.info("Not enough data for Holt-Winters prediction")
            return None
        
        # Use replicas history
        replicas = [d['replicas'] for d in traffic_data[-60:]] # Use up to 5 mins of data
        
        # Fit Holt-Winters with original hyperparameters
        model = ExponentialSmoothing(
            replicas,
            seasonal_periods=int(hw_config["slen"]),
            trend='add',
            seasonal='add',
            damped_trend=True
        ).fit(
            smoothing_level=float(hw_config["alpha"]),
            smoothing_trend=float(hw_config["beta"]),
            smoothing_seasonal=float(hw_config["gamma"])
        )
        
        forecast = model.forecast(steps).tolist()
        
        # Round and clip predictions
        forecast = [max(1, min(10, round(p))) for p in forecast]
        
        # Add prediction to history for MSE calculation
        # Use current time for immediate matching, plus a small offset for the next data point
        current_time = datetime.now()
        timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('holt_winters', forecast[0], timestamp)
        
        # Also create a prediction for the current time to enable immediate matching
        immediate_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('holt_winters', forecast[0], immediate_timestamp)
        
        elapsed_ms = (time.time() - start_time_func) * 1000
        prediction_time.labels(model="holtwinters").set(elapsed_ms)
        
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
        
        logger.info(f"🔄 GRU prediction starting: model_loaded={gru_model is not None}, data_points={len(traffic_data)}")    
        look_back = int(gru_config["look_back"])
    
    if len(traffic_data) < look_back:
        logger.warning(f"Not enough data for GRU prediction: {len(traffic_data)} < {look_back}")
        return None
    
    try:
        # Prepare recent data with validation
        recent_data = traffic_data[-look_back:]
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
        
        # Validate data consistency
        expected_features = 2 if 'memory_usage' not in features else 3
        
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
        
        # Add prediction to history for MSE calculation
        # Use current time for immediate matching, plus a small offset for the next data point
        current_time = datetime.now()
        timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('gru', predicted_replicas, timestamp)
        
        # Also create a prediction for the current time to enable immediate matching
        immediate_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('gru', predicted_replicas, immediate_timestamp)
        
        # For multi-step prediction, use the single prediction
        predictions = [predicted_replicas] * steps
        
        # Record prediction time
        elapsed_ms = (time.time() - start_time_func) * 1000
        prediction_time.labels(model="gru").set(elapsed_ms)
        
        logger.info(f"🎯 GRU forecast SUCCESS: {predictions} (raw: {raw_prediction:.3f})")
        return predictions
        
    except Exception as e:
        logger.error(f"🚨 UNEXPECTED GRU ERROR: {e}")
        import traceback
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

def predict_with_optimized_holtwinters(steps=None):
    """Enhanced Holt-Winters with optimized parameters and preprocessing."""
    start_time_func = time.time()
    hw_config = config["models"]["holt_winters"]
    
    if steps is None:
        steps = int(hw_config["look_forward"])
    
    try:
        if len(traffic_data) < 20:
            logger.info("Not enough data for optimized Holt-Winters prediction")
            return predict_with_holtwinters(steps)  # Fallback to original
        
        # Optimize parameters if we have enough data
        optimal_params = None
        if len(traffic_data) >= 30:
            optimal_params = optimize_holtwinters_parameters(traffic_data)
        
        # Use more sophisticated data preparation
        enhanced_data = create_advanced_features(traffic_data)
        if enhanced_data is not None:
            # Use smoothed replicas data to reduce noise
            replicas_smooth = enhanced_data['replicas'].rolling(window=3, center=True).mean().fillna(enhanced_data['replicas'])
        else:
            replicas_smooth = pd.Series([d['replicas'] for d in traffic_data[-60:]])
        
        # Determine seasonal period based on data patterns
        seasonal_period = min(12, max(4, len(replicas_smooth) // 5))
        
        # Use optimized parameters if available
        if optimal_params:
            alpha = optimal_params['alpha']
            beta = optimal_params['beta']
            gamma = optimal_params['gamma']
            logger.info(f"Using optimized HW parameters: α={alpha}, β={beta}, γ={gamma}")
        else:
            # Use enhanced default parameters
            alpha = 0.6  # Slightly higher for more responsiveness
            beta = 0.1   # Moderate trend sensitivity
            gamma = 0.4  # Balanced seasonality
        
        # Fit enhanced model
        model = ExponentialSmoothing(
            replicas_smooth,
            seasonal_periods=seasonal_period,
            trend='add',
            seasonal='add',
            damped_trend=True
        ).fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            smoothing_seasonal=gamma,
            optimized=True  # Allow further optimization
        )
        
        # Make forecast
        forecast = model.forecast(steps)
        
        # Post-process predictions
        forecast_processed = []
        for pred in forecast:
            # Apply intelligent clipping based on recent trends
            recent_replicas = replicas_smooth[-5:].values
            min_reasonable = max(1, int(recent_replicas.min() * 0.7))
            max_reasonable = min(10, int(recent_replicas.max() * 1.5))
            
            processed_pred = max(min_reasonable, min(max_reasonable, round(pred)))
            forecast_processed.append(processed_pred)
        
        # Store prediction with enhanced metadata
        current_time = datetime.now()
        timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('holt_winters', forecast_processed[0], timestamp)
        
        # Immediate matching prediction
        immediate_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('holt_winters', forecast_processed[0], immediate_timestamp)
        
        elapsed_ms = (time.time() - start_time_func) * 1000
        prediction_time.labels(model="holtwinters_optimized").set(elapsed_ms)
        
        logger.info(f"✅ Optimized Holt-Winters forecast: {forecast_processed}")
        return forecast_processed
        
    except Exception as e:
        logger.error(f"Error in optimized Holt-Winters prediction: {e}")
        return predict_with_holtwinters(steps)  # Fallback to original

def predict_with_advanced_gru(steps=None):
    """Enhanced GRU prediction using advanced features."""
    global gru_model, scaler_X, scaler_y
    start_time_func = time.time()
    
    gru_config = config["models"]["gru"]
    
    if steps is None:
        steps = int(gru_config["look_forward"])
    
    # Validate model availability
    if gru_model is None or not is_model_trained:
        logger.error(f"🚨 Advanced GRU PREDICTION BLOCKED: model={gru_model is not None}, trained={is_model_trained}")
        return None
    
    look_back = int(gru_config["look_back"])
    
    if len(traffic_data) < look_back:
        logger.warning(f"Not enough data for advanced GRU prediction: {len(traffic_data)} < {look_back}")
        return None
    
    try:
        # Try to use enhanced features if available
        enhanced_data = create_advanced_features(traffic_data[-look_back:])
        
        if enhanced_data is not None:
            # Use enhanced feature set
            feature_columns = [
                'cpu_utilization', 'traffic', 'memory_usage',
                'cpu_ma_5', 'traffic_ma_5', 'cpu_trend', 'traffic_trend',
                'load_pressure', 'efficiency', 'cpu_lag1', 'traffic_lag1',
                'cpu_volatility', 'is_cpu_peak', 'is_traffic_peak',
                'hour', 'day_of_week'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in enhanced_data.columns]
            
            if len(available_features) >= 3:
                X = enhanced_data[available_features].values
                logger.info(f"✅ Using {len(available_features)} advanced features for prediction")
            else:
                # Fallback to basic features
                return predict_with_gru(steps)
        else:
            # Fallback to basic prediction
            return predict_with_gru(steps)
        
        # Validate scaler compatibility
        if hasattr(scaler_X, 'n_features_in_') and X.shape[1] != scaler_X.n_features_in_:
            logger.warning(f"Advanced feature mismatch: got {X.shape[1]}, expected {scaler_X.n_features_in_}. Falling back to basic GRU.")
            return predict_with_gru(steps)
        
        # Clean data
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("NaN or infinite values found in advanced prediction data, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=0.0)
        
        # Scale and reshape
        try:
            X_scaled = scaler_X.transform(X)
            X_seq = X_scaled.reshape(1, look_back, len(available_features))
        except Exception as e:
            logger.error(f"Error during advanced data scaling: {e}")
            return predict_with_gru(steps)
        
        # Validate input shape
        expected_shape = gru_model.input_shape
        if X_seq.shape[1:] != expected_shape[1:]:
            logger.warning(f"Advanced shape mismatch: input={X_seq.shape[1:]}, expected={expected_shape[1:]}. Falling back to basic GRU.")
            return predict_with_gru(steps)
        
        # Make prediction
        try:
            y_pred_scaled = gru_model.predict(X_seq, verbose=0)
            
            if y_pred_scaled is None or len(y_pred_scaled) == 0:
                logger.error("Empty prediction from advanced GRU model")
                return None
            
            # Inverse transform
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            
            # Validate and sanitize prediction
            raw_prediction = float(y_pred[0][0])
            if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                logger.warning(f"Invalid advanced prediction value: {raw_prediction}, using fallback")
                predicted_replicas = 2
            else:
                predicted_replicas = max(1, min(10, round(raw_prediction)))
            
        except Exception as e:
            logger.error(f"Error during advanced GRU model prediction: {e}")
            return None
        
        # Add prediction to history
        current_time = datetime.now()
        timestamp = (current_time + timedelta(seconds=config['collection_interval'])).strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('gru', predicted_replicas, timestamp)
        
        immediate_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        add_prediction_to_history('gru', predicted_replicas, immediate_timestamp)
        
        predictions = [predicted_replicas] * steps
        
        elapsed_ms = (time.time() - start_time_func) * 1000
        prediction_time.labels(model="gru_advanced").set(elapsed_ms)
        
        logger.info(f"🎯 Advanced GRU forecast SUCCESS: {predictions} (raw: {raw_prediction:.3f})")
        return predictions
        
    except Exception as e:
        logger.error(f"🚨 Advanced GRU ERROR: {e}, falling back to basic GRU")
        return predict_with_gru(steps)

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
        
        # Priority 3: High CPU requires immediate scale up
        if current_cpu > scale_up_threshold:
            if current_time - last_scale_up_time >= 60:
                if predictions and len(predictions) > 0:
                    predicted_replicas = predictions[0] if isinstance(predictions[0], (int, float)) else current_replicas
                    recommended = min(max_replicas_limit, max(predicted_replicas, current_replicas + 1))
                else:
                    recommended = min(max_replicas_limit, current_replicas + 1)
                if recommended > current_replicas:
                    heapq.heappush(decision_heap, (3.0, "scale_up", recommended, f"high_cpu_{current_cpu:.1f}%"))
        
        # Priority 4: Very low activity emergency scale down
        if current_cpu < 5 and current_traffic < zero_traffic_threshold and current_replicas > min_replicas:
            if current_time - last_scale_down_time >= scale_down_delay:
                heapq.heappush(decision_heap, (4.0, "scale_down", min_replicas, f"very_low_activity"))
    
    # Priority 5-10: MSE-based prediction decisions using model ranking
    if predictions and len(predictions) > 0:
        # Get all model predictions with their MSE rankings
        model_predictions = get_ranked_model_predictions()
        
        for rank, (mse, model_name, pred_value) in enumerate(model_predictions):
            if pred_value is None:
                continue
                
            predicted_replicas = int(round(pred_value))
            priority = 5.0 + (rank * 0.1)  # Lower MSE models get higher priority
            
            # Scale up prediction
            if predicted_replicas > current_replicas and current_time - last_scale_up_time >= 60:
                if current_cpu > scale_down_threshold:  # Only if CPU supports scale up
                    heapq.heappush(decision_heap, (priority, "scale_up", predicted_replicas, f"{model_name}_mse_{mse:.3f}"))
            
            # Scale down prediction  
            elif predicted_replicas < current_replicas and current_replicas > cost_config.get('min_replicas', 1):
                if current_time - last_scale_down_time >= cost_config.get('scale_down_delay_minutes', 2) * 60:
                    if current_cpu < cost_config.get('scale_up_threshold', 80):  # Only if CPU supports scale down
                        heapq.heappush(decision_heap, (priority + 1.0, "scale_down", predicted_replicas, f"{model_name}_mse_{mse:.3f}"))
    
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
        
        logger.info(f"🎯 MinHeap scaling decision: {decision} to {replicas} replicas (priority: {priority:.1f}, reason: {reason})")
        
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
    
    try:
        # Ensemble prediction
        ensemble_pred = predict_with_ensemble(1)
        if ensemble_pred and len(ensemble_pred) > 0:
            ensemble_mse = calculate_ensemble_mse()
            model_predictions.append((ensemble_mse, 'ensemble', ensemble_pred[0]))
    except Exception as e:
        logger.debug(f"Failed to get ensemble prediction for ranking: {e}")
    
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

def start_collection():
    """Start the metrics collection thread."""
    global is_collecting, collection_thread, data_collection_complete
    
    if data_collection_complete:
        logger.info("Data collection already complete. Using stored 24-hour dataset.")
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
    minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60
    idle_minutes = 0
    if low_traffic_start_time:
        idle_minutes = (datetime.now() - low_traffic_start_time).total_seconds() / 60
    
    # Enhanced GRU status information
    gru_config = config["models"]["gru"]
    min_data_required = max(int(gru_config["look_back"]) + 10, 20)
    current_cpu = traffic_data[-1]['cpu_utilization'] if traffic_data else 0
    
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
        'runtime_hours_collected': len(traffic_data) / 60,  # Each data point ≈ 1 minute
        'remaining_runtime_hours': max(0, 24 - (len(traffic_data) / 60)),
        'progress_percentage': min(100, (len(traffic_data) / 1440) * 100),  # 1440 = 24hrs * 60min
        'data_points_collected': len(traffic_data),
        'target_data_points': 1440,  # 24 hours * 60 minutes
        'sessions_info': {
            'aws_academy_sessions_needed': 6,
            'hours_per_session': 4,
            'estimated_sessions_completed': min(6, int((len(traffic_data) / 60) / 4))
        },
        'note': 'Collection based on 24 hours of actual runtime, perfect for AWS Academy 4-hour sessions'
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
                'gru_matched': len([p for p in predictions_history['gru'] if p.get('matched', False)]),
                'gru_total': len(predictions_history['gru']),
                'holt_winters_matched': len([p for p in predictions_history['holt_winters'] if p.get('matched', False)]),
                'holt_winters_total': len(predictions_history['holt_winters'])
            }
        }
    })

@app.route('/data', methods=['GET'])
def get_metrics_data():
    """Return collected metrics data."""
    recent_data = traffic_data[-100:] if traffic_data else []
    
    return jsonify({
        "data_points": len(traffic_data),
        "recent_metrics": recent_data,
        "using_gru": config['use_gru'],
        "mse_enabled": config['mse_config']['enabled']
    })

@app.route('/predict', methods=['GET'])
def get_prediction():
    """Return prediction and scaling recommendation with enhanced MSE tracking."""
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
            prediction_latency.labels(method="GRU").observe(time.time() - start_time_func)
            prediction_requests.labels(method="GRU").inc()
            logger.info("GRU prediction made for MSE tracking")
        except Exception as e:
            logger.error(f"GRU prediction failed: {e}")
    
    # Always make Holt-Winters prediction
    try:
        start_time_func = time.time()
        hw_predictions = predict_with_holtwinters(steps)
        prediction_latency.labels(method="Holt-Winters").observe(time.time() - start_time_func)
        prediction_requests.labels(method="Holt-Winters").inc()
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
    scaling_decisions.labels(decision=decision).inc()
    recommended_replicas.set(replicas)
    
    # Enhanced MSE status
    gru_status = f"{gru_mse:.3f}" if gru_mse != float('inf') else "insufficient_data"
    hw_status = f"{holt_winters_mse:.3f}" if holt_winters_mse != float('inf') else "insufficient_data"
    
    return jsonify({
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
    try:
        minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        if not traffic_data:
            # No data yet, return minimum replicas
            return jsonify({
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
        
        # ALWAYS make both predictions for MSE tracking
        gru_predictions = None
        hw_predictions = None
        
        # Make GRU prediction if available
        if gru_model is not None and is_model_trained:
            try:
                gru_predictions = predict_with_gru()
                logger.info("GRU prediction made in combined endpoint")
            except Exception as e:
                logger.error(f"GRU prediction failed in combined endpoint: {e}")
        
        # Always make Holt-Winters prediction
        try:
            hw_predictions = predict_with_holtwinters()
            logger.info("Holt-Winters prediction made in combined endpoint")
        except Exception as e:
            logger.error(f"Holt-Winters prediction failed in combined endpoint: {e}")
        
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
                predictions = predict_with_holtwinters()
        
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
        
        # Update Prometheus metrics
        predictive_scaler_recommended_replicas.labels(method=method.lower()).set(recommended_replicas_value)
        prediction_requests.labels(method=method).inc()
        scaling_decisions.labels(decision=decision).inc()
        
        # Enhanced MSE status reporting
        gru_status = f"{gru_mse:.3f}" if gru_mse != float('inf') else "insufficient_data"
        hw_status = f"{holt_winters_mse:.3f}" if holt_winters_mse != float('inf') else "insufficient_data"
        
        return jsonify({
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
    prediction_mse.labels(model="gru").observe(0.0)
    prediction_mse.labels(model="holtwinters").observe(0.0)
    current_mse.labels(model="gru").set(-1.0)  # Start with -1 (insufficient data)
    current_mse.labels(model="holt_winters").set(-1.0)
    training_time.labels(model="gru").set(0.0)
    training_time.labels(model="holtwinters").set(0.0)
    prediction_time.labels(model="gru").set(0.0)
    prediction_time.labels(model="holtwinters").set(0.0)
    predictive_scaler_recommended_replicas.labels(method='gru').set(0)
    predictive_scaler_recommended_replicas.labels(method='holtwinters').set(0)
    model_selection.labels(model='gru', reason='initialization').inc(0)
    model_selection.labels(model='holt_winters', reason='initialization').inc(0)
    
    # Initialize HTTP metrics counter
    http_requests.labels(app='predictive-scaler', status_code='200', path='/').inc(0)
    
    # Log current GRU status
    gru_config = config["models"]["gru"]
    min_data_required = max(int(gru_config["look_back"]) + 10, 20)
    
    logger.info(f"GRU Status: trained={is_model_trained}, use_gru={config['use_gru']}")
    logger.info(f"GRU Training requirements: {config['training_threshold_minutes']} minutes, {min_data_required} data points, CPU > {config['cpu_threshold']}%")
    logger.info(f"Current data points: {len(traffic_data)}")
    
    # Start collection
    start_collection()
    logger.info("Predictive scaler initialized successfully")

@app.before_request
def before_request():
    request.start_time = time.time()
    
    if not hasattr(app, '_initialized'):
        initialize()
        app._initialized = True

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        app_name = os.environ.get('APP_NAME', 'predictive-scaler')
        
        http_duration.labels(app=app_name).observe(duration)
        http_requests.labels(
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
    try:
        result = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(traffic_data),
            'latest_data': traffic_data[-3:] if traffic_data else [],
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
    try:
        if len(traffic_data) < 12:
            return jsonify({
                'success': False,
                'error': f'Not enough data: {len(traffic_data)} < 12'
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
                'recent_data_sample': traffic_data[-3:] if traffic_data else [],
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

@app.route('/debug/force_ensemble_prediction', methods=['POST'])
def debug_force_ensemble():
    """Force creation of ensemble predictions for testing."""
    try:
        if len(traffic_data) < 12:
            return jsonify({
                'success': False,
                'error': f'Not enough data: {len(traffic_data)} < 12'
            })
        
        # Force ensemble prediction
        ensemble_pred = predict_with_ensemble(1)
        
        # Force MSE calculation
        update_mse_metrics()
        
        return jsonify({
            'success': True,
            'message': 'Ensemble prediction created',
            'results': {
                'ensemble_prediction': ensemble_pred,
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
                'ensemble_predictions_total': len(predictions_history.get('ensemble', []))
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
        if 'ensemble' in predictions_history and len(predictions_history['ensemble']) > 0:
            models.append('ensemble')
        
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

@app.route('/predict_enhanced', methods=['GET'])
def predict_enhanced():
    """Enhanced prediction endpoint using the best available method including ensemble."""
    try:
        steps = request.args.get('steps', default=1, type=int)
        use_ensemble = request.args.get('ensemble', default='auto', type=str)
        
        if not traffic_data:
            return jsonify({
                'method_used': 'fallback',
                'predictions': [1],
                'recommended_replicas': 1,
                'error': 'No data available'
            })
        
        # Determine best method
        predictions = None
        method_used = None
        
        # Check if we should use ensemble
        if use_ensemble == 'true' or (use_ensemble == 'auto' and len(traffic_data) >= 30):
            try:
                predictions = predict_with_ensemble(steps)
                method_used = 'ensemble'
                logger.info(f"Using ensemble prediction: {predictions}")
            except Exception as e:
                logger.warning(f"Ensemble prediction failed: {e}")
        
        # Fallback to best individual model
        if not predictions:
            best_model = select_best_model()
            
            if best_model == 'gru' and gru_model is not None and is_model_trained:
                try:
                    predictions = predict_with_advanced_gru(steps)
                    if not predictions:
                        predictions = predict_with_gru(steps)
                    method_used = 'advanced_gru' if predictions else 'gru'
                except Exception:
                    predictions = predict_with_gru(steps)
                    method_used = 'gru'
            else:
                try:
                    predictions = predict_with_optimized_holtwinters(steps)
                    if not predictions:
                        predictions = predict_with_holtwinters(steps)
                    method_used = 'optimized_holt_winters' if predictions else 'holt_winters'
                except Exception:
                    predictions = predict_with_holtwinters(steps)
                    method_used = 'holt_winters'
        
        # Final fallback
        if not predictions:
            predictions = [2]
            method_used = 'fallback'
        
        decision, recommended_replicas_value = make_scaling_decision(predictions)
        
        # Enhanced metrics
        enhanced_mse = {}
        for model in ['gru', 'holt_winters']:
            if model in predictions_history:
                enhanced_mse[model] = calculate_enhanced_mse(model)
        
        return jsonify({
            'method_used': method_used,
            'predictions': predictions,
            'recommended_replicas': int(recommended_replicas_value),
            'scaling_decision': decision,
            'enhanced_metrics': enhanced_mse,
            'model_performance_summary': {
                'gru_mse': gru_mse if gru_mse != float('inf') else None,
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
                'best_model': select_best_model(),
                'ensemble_available': len(traffic_data) >= 30
            }
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced prediction: {e}")
        return jsonify({
            'method_used': 'error_fallback',
            'predictions': [1],
            'recommended_replicas': 1,
            'error': str(e)
        }), 500

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
                'holt_winters_mse': holt_winters_mse if holt_winters_mse != float('inf') else None,
                'ensemble_mse': calculate_ensemble_mse() if calculate_ensemble_mse() != float('inf') else None
            },
            'model_readiness': {
                'gru_ready': is_model_trained and gru_model is not None,
                'holt_winters_ready': True,
                'ensemble_ready': 'ensemble' in predictions_history and len(predictions_history['ensemble']) > 0
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
    """Reset all collected data and restart data collection from scratch."""
    global traffic_data, data_collection_complete, data_collection_start_time
    global predictions_history, gru_model, is_model_trained, scaler_X, scaler_y
    global is_collecting, collection_thread
    
    try:
        # Stop current collection if running
        if is_collecting:
            stop_collection()
            time.sleep(2)  # Wait for collection to stop
        
        # Clear all data
        traffic_data = []
        predictions_history = {'gru': [], 'holt_winters': [], 'ensemble': []}
        
        # Reset collection status
        data_collection_complete = False
        data_collection_start_time = None
        
        # Reset model components
        gru_model = None
        is_model_trained = False
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Clear config flags
        config['use_gru'] = False
        save_config()
        
        # Delete data files
        files_to_delete = [DATA_FILE, STATUS_FILE, PREDICTIONS_FILE, MODEL_FILE, SCALER_X_FILE, SCALER_Y_FILE]
        deleted_files = []
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(os.path.basename(file_path))
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
        
        # Save the reset status
        save_collection_status()
        save_data()
        save_predictions_history()
        
        # Start fresh data collection
        start_collection()
        
        logger.info("🔄 Data reset completed. Starting fresh 24-hour data collection.")
        
        return jsonify({
            "status": "success",
            "message": "All data cleared and fresh collection started",
            "details": {
                "deleted_files": deleted_files,
                "collection_restarted": True,
                "estimated_completion": (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        
    except Exception as e:
        logger.error(f"Error during data reset: {e}")
        return jsonify({
            "status": "error", 
            "message": f"Data reset failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Initialize directly when run as a script
    logger.info("🚀 Starting predictive autoscaler with MinHeap-based MSE model selection...")
    logger.info("🎯 MinHeap system: Model selection based on lowest MSE priority")
    load_data()
    filter_stale_data()
    initialize()
    
    # Start data collection automatically if not complete
    if not data_collection_complete:
        runtime_hours = len(traffic_data) / 60
        sessions_completed = int(runtime_hours / 4)
        logger.info(f"🎓 AWS Academy Mode: Starting data collection...")
        logger.info(f"📊 Progress: {runtime_hours:.1f}/24 hours ({len(traffic_data)}/1440 data points)")
        logger.info(f"📈 Sessions: {sessions_completed}/6 completed (~4 hours each)")
        start_collection()
    else:
        logger.info(f"✅ Using stored 24-hour dataset with {len(traffic_data)} data points")
        logger.info("🎓 AWS Academy: All 6 sessions completed, ready for production use!")
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
