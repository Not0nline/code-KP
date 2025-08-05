# autoscaler.py
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
from sklearn.preprocessing import MinMaxScaler
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
prediction_mse = Gauge('predictive_scaler_mse', 'Prediction Mean Squared Error', 
                     labelnames=['model'])
training_time = Gauge('predictive_scaler_training_time_ms', 'Model training time in ms',
                     labelnames=['model'])
prediction_time = Gauge('predictive_scaler_prediction_time_ms', 'Prediction time in ms',
                      labelnames=['model'])
predictive_scaler_recommended_replicas = Gauge('predictive_scaler_recommended_replicas', 
                                             'Recommended number of replicas', 
                                             ['method'])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
DATA_FILE = "traffic_data.csv"
MODEL_FILE = "gru_model.h5"
CONFIG_FILE = "config.json"
MIN_DATA_POINTS_FOR_GRU = 2000  # Based on your hyperparameters
PREDICTION_WINDOW = 24  # Based on your hyperparameters
SCALING_THRESHOLD = 0.7  # CPU threshold for scaling decision
SCALING_COOLDOWN = 300  # 5 minutes cooldown between scaling actions

# Default configuration with your hyperparameters
config = {
    "use_gru": False,
    "collection_interval": 60,  # seconds
    "cpu_threshold": 3.0,  # Skip collection below this CPU %
    "training_threshold_minutes": 5,  # Switch to GRU after this
    "prometheus_server": os.getenv('PROMETHEUS_SERVER', 
        'http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090'),
    "target_deployment": os.getenv('TARGET_DEPLOYMENT', 'product-app-combined'),
    "target_namespace": os.getenv('TARGET_NAMESPACE', 'default'),
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
traffic_data = []
is_collecting = False
collection_thread = None
gru_model = None
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
is_model_trained = False
last_training_time = None
start_time = datetime.now()
prometheus_client = None

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
        logger.info(f"Saved {len(traffic_data)} data points to {DATA_FILE}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def load_model_components():
    """Load GRU model and scalers if available"""
    global gru_model, scaler_X, scaler_y, is_model_trained
    
    if os.path.exists(MODEL_FILE):
        try:
            gru_model = tf.keras.models.load_model(MODEL_FILE)
            logger.info("GRU model loaded")
            
            if os.path.exists('scaler_X.pkl'):
                scaler_X = joblib.load('scaler_X.pkl')
            if os.path.exists('scaler_y.pkl'):
                scaler_y = joblib.load('scaler_y.pkl')
                
            is_model_trained = True
            config['use_gru'] = True
            return True
        except Exception as e:
            logger.error(f"Error loading GRU model: {e}")
    return False

def collect_metrics_from_prometheus():
    """Collect real metrics from Prometheus"""
    global traffic_data, is_collecting, last_training_time
    
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
            
            # Skip data collection if CPU is below threshold
            if current_cpu < config['cpu_threshold']:
                logger.info(f"CPU utilization {current_cpu:.2f}% is below {config['cpu_threshold']}% threshold. Skipping data collection.")
                time.sleep(config['collection_interval'])
                continue
            
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
            
            # Update Prometheus metrics
            cpu_utilization.set(current_cpu)  # Normalize to 0-1
            traffic_gauge.set(current_traffic)
            
            # Store the data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            traffic_data.append({
                'timestamp': timestamp,
                'traffic': current_traffic,
                'cpu_utilization': current_cpu,  # Normalize to 0-1
                'memory_usage': current_memory,
                'replicas': current_replicas
            })
            
            # Keep only last 24 hours of data
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)
            traffic_data = [d for d in traffic_data if 
                          datetime.strptime(d['timestamp'], "%Y-%m-%d %H:%M:%S") > cutoff_time]
            
            # Periodically save data
            if len(traffic_data) % 10 == 0:
                save_data()
            
            # Check if we should train/retrain the model
            minutes_elapsed = (current_time - start_time).total_seconds() / 60
            gru_config = config["models"]["gru"]
            train_size_needed = int(gru_config["train_size"])
            
            # Initial training after threshold
            if not config['use_gru'] and minutes_elapsed >= config['training_threshold_minutes'] and len(traffic_data) >= 12:
                logger.info(f"Training threshold reached ({minutes_elapsed:.1f} minutes), attempting to train GRU model")
                if build_gru_model():
                    config['use_gru'] = True
                    save_config()
            
            # Retrain every hour if model is trained
            elif is_model_trained and (last_training_time is None or 
                 (current_time - last_training_time).total_seconds() > 3600):
                logger.info("Retraining model with new data...")
                build_gru_model()
            
            # Sleep for the collection interval
            time.sleep(config['collection_interval'])
            
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            time.sleep(5)  # Sleep for a short time on error

def preprocess_data(data, sequence_length=100):
    """Preprocess data for GRU model."""
    if len(data) < sequence_length + 1:
        return None, None
    
    X, y = [], []
    df = pd.DataFrame(data)
    
    # Use multiple features
    features = ['traffic', 'cpu_utilization']
    if 'memory_usage' in df.columns:
        features.append('memory_usage')
    
    # Prepare data for scaling
    feature_data = df[features].values
    replica_data = df['replicas'].values.reshape(-1, 1)
    
    # Fit and transform
    feature_scaled = scaler_X.fit_transform(feature_data)
    replica_scaled = scaler_y.fit_transform(replica_data)
    
    # Create sequences
    for i in range(len(data) - sequence_length):
        X.append(feature_scaled[i:i+sequence_length])
        y.append(replica_scaled[i+sequence_length])
    
    return np.array(X), np.array(y)

def build_gru_model():
    """Build and train GRU model with your hyperparameters."""
    global gru_model, last_training_time
    
    start_time = time.time()
    gru_config = config["models"]["gru"]
    
    X, y = preprocess_data(traffic_data, sequence_length=int(gru_config["look_back"]))
    
    if X is None or y is None or len(X) < 10:
        logger.warning("Not enough data to build GRU model")
        elapsed_ms = (time.time() - start_time) * 1000
        training_time.labels(model="gru").set(elapsed_ms)
        return False
    
    input_shape = (X.shape[1], X.shape[2])
    
    model = Sequential([
        GRU(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(16, return_sequences=False),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    model.fit(
        X, y,
        epochs=min(int(gru_config["epochs"]), 50),  # Limit epochs for faster training
        batch_size=int(gru_config["batch_size"]),
        validation_split=0.2,
        verbose=0
    )
    
    # Save model and scalers
    model.save(MODEL_FILE)
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    
    gru_model = model
    last_training_time = datetime.now()
    
    # Record training time
    elapsed_ms = (time.time() - start_time) * 1000
    training_time.labels(model="gru").set(elapsed_ms)
    
    logger.info(f"GRU model trained successfully in {elapsed_ms:.2f}ms")
    return True

def predict_with_holtwinters(steps=None):
    """Predict using Holt-Winters with improved error handling."""
    start_time = time.time()
    hw_config = config["models"]["holt_winters"]
    
    if steps is None:
        steps = int(hw_config["look_forward"])
    
    try:
        if len(traffic_data) < 3:
            logger.info("Not enough data for Holt-Winters prediction")
            return None
        
        # Use replicas history
        replicas = [d['replicas'] for d in traffic_data[-12:]]  # Last hour
        
        if len(set(replicas)) == 1:  # All values are the same
            return [replicas[-1]] * steps
        
        # Fit Holt-Winters
        model = ExponentialSmoothing(
            replicas,
            seasonal_periods=12,
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
        
        elapsed_ms = (time.time() - start_time) * 1000
        prediction_time.labels(model="holtwinters").set(elapsed_ms)
        
        logger.info(f"Holt-Winters forecast generated: {forecast}")
        return forecast
        
    except Exception as e:
        logger.error(f"Error in Holt-Winters prediction: {e}")
        return None

def predict_with_gru(steps=None):
    """Predict using GRU model with your hyperparameters."""
    global gru_model
    start_time = time.time()
    
    gru_config = config["models"]["gru"]
    
    if steps is None:
        steps = int(gru_config["look_forward"])
    
    if gru_model is None:
        logger.warning("GRU model not available")
        return None
    
    look_back = int(gru_config["look_back"])
    
    if len(traffic_data) < look_back:
        logger.warning("Not enough data for GRU prediction")
        return None
    
    try:
        # Prepare recent data
        df = pd.DataFrame(traffic_data[-look_back:])
        
        # Use same features as training
        features = ['traffic', 'cpu_utilization']
        if 'memory_usage' in df.columns:
            features.append('memory_usage')
        
        X = df[features].values
        
        # Scale and reshape
        X_scaled = scaler_X.transform(X)
        X_seq = X_scaled.reshape(1, look_back, len(features))
        
        # Predict
        y_pred_scaled = gru_model.predict(X_seq, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Round and clip
        predicted_replicas = max(1, min(10, round(y_pred[0][0])))
        
        # For multi-step prediction, use the single prediction
        # In practice, you might want to implement recursive prediction
        predictions = [predicted_replicas] * steps
        
        elapsed_ms = (time.time() - start_time) * 1000
        prediction_time.labels(model="gru").set(elapsed_ms)
        
        logger.info(f"GRU forecast generated: {predictions}")
        return predictions
        
    except Exception as e:
        logger.error(f"Error in GRU prediction: {e}")
        return None

def make_scaling_decision(predictions):
    """Determine if scaling is needed based on predictions."""
    global last_scaling_time
    
    if predictions is None or len(predictions) == 0:
        return "maintain", 0
    
    current_time = time.time()
    if current_time - last_scaling_time < SCALING_COOLDOWN:
        return "cooldown", 0
    
    # Use the predictions directly as they're already replica counts
    max_replicas = max(predictions)
    current_replicas = traffic_data[-1]['replicas'] if traffic_data else 1
    
    if max_replicas > current_replicas:
        last_scaling_time = current_time
        return "scale_up", int(max_replicas)
    elif max_replicas < current_replicas:
        last_scaling_time = current_time
        return "scale_down", int(max_replicas)
    else:
        return "maintain", int(current_replicas)

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
    global is_collecting, collection_thread
    
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
    """Status endpoint with detailed information."""
    minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60
    return jsonify({
        'data_points': len(traffic_data),
        'model_trained': is_model_trained,
        'using_gru': config['use_gru'],
        'uptime_minutes': minutes_elapsed,
        'time_until_gru': max(0, config['training_threshold_minutes'] - minutes_elapsed) if not is_model_trained else 0,
        'last_training': last_training_time.isoformat() if last_training_time else None
    })

@app.route('/data', methods=['GET'])
def get_metrics_data():
    """Return collected metrics data."""
    recent_data = traffic_data[-100:] if traffic_data else []
    
    return jsonify({
        "data_points": len(traffic_data),
        "recent_metrics": recent_data,
        "using_gru": config['use_gru']
    })

@app.route('/predict', methods=['GET'])
def get_prediction():
    """Return prediction and scaling recommendation with metrics tracking."""
    gru_config = config["models"]["gru"]
    hw_config = config["models"]["holt_winters"]
    
    steps = request.args.get('steps', default=int(hw_config["look_forward"]), type=int)
    
    if config['use_gru'] and gru_model is not None:
        start_time = time.time()
        predictions = predict_with_gru(steps)
        method = "GRU"
        prediction_latency.labels(method=method).observe(time.time() - start_time)
    else:
        start_time = time.time()
        predictions = predict_with_holtwinters(steps)
        method = "Holt-Winters"
        prediction_latency.labels(method=method).observe(time.time() - start_time)
    
    prediction_requests.labels(method=method).inc()
    
    decision, replicas = make_scaling_decision(predictions)
    scaling_decisions.labels(decision=decision).inc()
    recommended_replicas.set(replicas)
    
    return jsonify({
        "method": method,
        "predictions": predictions,
        "scaling_decision": decision,
        "recommended_replicas": replicas
    })

@app.route('/predict_combined', methods=['GET'])
def predict_combined():
    """Combined prediction endpoint matching the new autoscaler interface."""
    try:
        minutes_elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        if not traffic_data:
            return jsonify({
                'method_used': 'fallback',
                'predictions': [],
                'recommended_replicas': 1,
                'scaling_decision': 'maintain',
                'current_replicas': 1,
                'time_until_gru': max(0, config['training_threshold_minutes'] - minutes_elapsed),
                'is_model_trained': is_model_trained,
                'error': 'No data available'
            })
        
        current_metrics = traffic_data[-1]
        current_replicas = current_metrics['replicas']
        
        # Determine which method to use
        if is_model_trained and gru_model is not None:
            method = 'gru'
            predictions = predict_with_gru()
        elif minutes_elapsed < config['training_threshold_minutes']:
            method = 'holt_winters_initial'
            predictions = predict_with_holtwinters()
        else:
            method = 'holt_winters_fallback'
            predictions = predict_with_holtwinters()
        
        if not predictions:
            # Simple reactive scaling as last resort
            method = 'reactive'
            if current_metrics['cpu_utilization'] > 70:
                predictions = [min(10, current_replicas + 2)]
            elif current_metrics['cpu_utilization'] > 50:
                predictions = [min(10, current_replicas + 1)]
            elif current_metrics['cpu_utilization'] < 30 and current_replicas > 1:
                predictions = [max(1, current_replicas - 1)]
            else:
                predictions = [current_replicas]
        
        recommended_replicas_value = predictions[0] if predictions else current_replicas
        
        # Determine scaling decision
        if recommended_replicas_value > current_replicas:
            scaling_decision = 'scale_up'
        elif recommended_replicas_value < current_replicas:
            scaling_decision = 'scale_down'
        else:
            scaling_decision = 'maintain'
        
        # Update Prometheus metrics
        predictive_scaler_recommended_replicas.labels(method=method.lower()).set(recommended_replicas_value)
        prediction_requests.labels(method=method).inc()
        scaling_decisions.labels(decision=scaling_decision).inc()
        
        return jsonify({
            'method_used': method,
            'predictions': predictions,
            'recommended_replicas': int(recommended_replicas_value),
            'current_replicas': int(current_replicas),
            'scaling_decision': scaling_decision,
            'time_until_gru': max(0, config['training_threshold_minutes'] - minutes_elapsed) if not is_model_trained else 0,
            'is_model_trained': is_model_trained
        })
        
    except Exception as e:
        logger.error(f"Error in predict_combined: {e}")
        return jsonify({
            'method_used': 'fallback',
            'predictions': [],
            'recommended_replicas': 1,
            'scaling_decision': 'maintain',
            'current_replicas': 1,
            'time_until_gru': 0,
            'is_model_trained': False,
            'error': str(e)
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

# Initialization function
def initialize():
    load_config()
    load_data()
    load_model_components()
    initialize_prometheus()
    
    # Initialize metrics with zero values
    prediction_mse.labels(model="gru").set(0.0)
    prediction_mse.labels(model="holtwinters").set(0.0)
    training_time.labels(model="gru").set(0.0)
    training_time.labels(model="holtwinters").set(0.0)
    prediction_time.labels(model="gru").set(0.0)
    prediction_time.labels(model="holtwinters").set(0.0)
    predictive_scaler_recommended_replicas.labels(method='gru').set(0)
    predictive_scaler_recommended_replicas.labels(method='holtwinters').set(0)
    
    # Initialize HTTP metrics counter
    http_requests.labels(app='predictive-scaler', status_code='200', path='/').inc(0)
    
    start_collection()

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

if __name__ == '__main__':
    # Initialize directly when run as a script
    initialize()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))