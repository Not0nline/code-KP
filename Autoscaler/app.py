# app.py
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
from prometheus_client import Counter, Gauge, Summary, Histogram
from prometheus_flask_exporter import PrometheusMetrics
import threading
import heapq


# Add after the Flask app initialization
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
MODEL_FILE = "gru_model"
CONFIG_FILE = "config.json"
MIN_DATA_POINTS_FOR_GRU = 2000  # Based on your hyperparameters
PREDICTION_WINDOW = 24  # Based on your hyperparameters
SCALING_THRESHOLD = 0.7  # CPU threshold for scaling decision
SCALING_COOLDOWN = 300  # 5 minutes cooldown between scaling actions

# Default configuration with your hyperparameters
config = {
    "use_gru": False,
    "collection_interval": 60,  # seconds
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

def preprocess_data(data, sequence_length=100):  # Using your look_back parameter
    """Preprocess data for GRU model."""
    if len(data) < sequence_length + 1:
        return None, None
    
    X, y = [], []
    df = pd.DataFrame(data)
    
    # Normalize the data
    traffic_values = df['traffic'].values
    cpu_values = df['cpu_utilization'].values
    
    max_traffic = np.max(traffic_values)
    max_cpu = np.max(cpu_values)
    
    if max_traffic == 0:
        normalized_traffic = traffic_values
    else:
        normalized_traffic = traffic_values / max_traffic
    
    if max_cpu == 0:
        normalized_cpu = cpu_values
    else:
        normalized_cpu = cpu_values / max_cpu
    
    # Create sequences
    for i in range(len(data) - sequence_length):
        X.append(np.column_stack((normalized_traffic[i:i+sequence_length], 
                                  normalized_cpu[i:i+sequence_length])))
        y.append(np.column_stack((normalized_traffic[i+1:i+sequence_length+1], 
                                  normalized_cpu[i+1:i+sequence_length+1])))
    
    return np.array(X), np.array(y)

def build_gru_model():
    """Build and return a GRU model with your hyperparameters."""
    start_time = time.time()
    hw_config = config["models"]["holt_winters"]
    gru_config = config["models"]["gru"]
    
    X, y = preprocess_data(traffic_data, sequence_length=int(gru_config["look_back"]))
    
    if X is None or y is None:
        logger.warning("Not enough data to build GRU model")
        elapsed_ms = (time.time() - start_time) * 1000
        training_time.labels(model="gru").set(elapsed_ms)
        return None
    
    input_shape = X.shape[1:]
    
    model = tf.keras.Sequential()
    
    # Add GRU layers based on n_layers parameter
    n_layers = int(gru_config["n_layers"])
    
    # First layer
    model.add(tf.keras.layers.GRU(
        64, activation='relu', 
        return_sequences=(n_layers > 1),  # Return sequences if we have more layers
        input_shape=input_shape
    ))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Middle layers if any
    for i in range(1, n_layers):
        return_seq = i < n_layers - 1  # Return sequences except for the last layer
        model.add(tf.keras.layers.GRU(32, activation='relu', return_sequences=return_seq))
        model.add(tf.keras.layers.Dropout(0.2))
    
    # Output layer
    model.add(tf.keras.layers.Dense(y.shape[2]))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    model.fit(
        X, y,
        epochs=int(gru_config["epochs"]),
        batch_size=int(gru_config["batch_size"]),
        validation_split=0.2,
        verbose=1
    )
    
    # Record training time
    elapsed_ms = (time.time() - start_time) * 1000
    training_time.labels(model="gru").set(elapsed_ms)
    
    logger.info(f"GRU model trained successfully in {elapsed_ms:.2f}ms")
    model.save(MODEL_FILE)
    return model

def load_gru_model():
    """Load the trained GRU model if available."""
    global gru_model
    if os.path.exists(MODEL_FILE):
        try:
            gru_model = tf.keras.models.load_model(MODEL_FILE)
            logger.info("GRU model loaded")
            return gru_model
        except Exception as e:
            logger.error(f"Error loading GRU model: {e}")
    return None

def predict_with_holtwinters(steps=None):
    """Predict using Holt-Winters with improved error handling."""
    start_time = time.time()
    hw_config = config["models"]["holt_winters"]
    
    if steps is None:
        steps = int(hw_config["look_forward"])
    
    try:
        if len(traffic_data) < 24:
            logger.info("Not enough data for prediction, using fallback")
            return None  # Return None to indicate no prediction
        
        # Real forecast with real data
        ts = pd.Series([point['cpu_utilization'] for point in traffic_data[-60:]])
        model = ExponentialSmoothing(
            ts,
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
        elapsed_ms = (time.time() - start_time) * 1000
        prediction_time.labels(model="holtwinters").set(elapsed_ms)
        logger.info(f"Holt-Winters forecast generated: {forecast}")
        return forecast
        
    except Exception as e:
        logger.error(f"Error in Holt-Winters prediction: {e}")
        return None  # Return None to indicate prediction failure
    
def predict_with_gru(steps=None):
    """Predict using GRU model with your hyperparameters."""
    global gru_model
    start_time = time.time()
    
    gru_config = config["models"]["gru"]
    
    if steps is None:
        steps = int(gru_config["look_forward"])
    
    if gru_model is None:
        logger.warning("GRU model not available")
        elapsed_ms = (time.time() - start_time) * 1000
        prediction_time.labels(model="gru").set(elapsed_ms)
        return None
    
    look_back = int(gru_config["look_back"])
    
    if len(traffic_data) < look_back:  # Need at least one sequence
        logger.warning("Not enough data for GRU prediction")
        elapsed_ms = (time.time() - start_time) * 1000
        prediction_time.labels(model="gru").set(elapsed_ms)
        return None
    
    try:
        # Prepare input data (last look_back points)
        df = pd.DataFrame(traffic_data[-look_back:])
        
        traffic_values = df['traffic'].values
        cpu_values = df['cpu_utilization'].values
        
        # Normalize
        all_traffic = pd.DataFrame(traffic_data)['traffic'].values
        all_cpu = pd.DataFrame(traffic_data)['cpu_utilization'].values
        
        max_traffic = np.max(all_traffic)
        max_cpu = np.max(all_cpu)
        
        if max_traffic == 0:
            normalized_traffic = traffic_values
        else:
            normalized_traffic = traffic_values / max_traffic
        
        if max_cpu == 0:
            normalized_cpu = cpu_values
        else:
            normalized_cpu = cpu_values / max_cpu
        
        # Create input sequence
        X_input = np.column_stack((normalized_traffic, normalized_cpu))
        X_input = np.expand_dims(X_input, axis=0)
        
        # Predict one step at a time for multi-step prediction
        predictions = []
        current_input = X_input.copy()
        
        for _ in range(steps):
            pred = gru_model.predict(current_input, verbose=0)
            predictions.append(pred[0, -1, 1])  # Get the CPU prediction only
            
            # Update input for next prediction
            new_input = np.roll(current_input[0], -1, axis=0)
            new_input[-1] = pred[0, -1]
            current_input = np.expand_dims(new_input, axis=0)
        
        # Denormalize predictions
        if max_cpu > 0:
            predictions = [p * max_cpu for p in predictions]
        
        # Calculate MSE if we have enough historical data
        if len(traffic_data) > steps + look_back:
            # Get the last 'steps' actual values
            actual_values = pd.DataFrame(traffic_data[-steps:])['cpu_utilization'].values
            
            # Make a prediction for the same period using historical data
            historical_data = pd.DataFrame(traffic_data[-(steps+look_back):-steps])
            historical_traffic = historical_data['traffic'].values
            historical_cpu = historical_data['cpu_utilization'].values
            
            if max_traffic == 0:
                historical_normalized_traffic = historical_traffic
            else:
                historical_normalized_traffic = historical_traffic / max_traffic
            
            if max_cpu == 0:
                historical_normalized_cpu = historical_cpu
            else:
                historical_normalized_cpu = historical_cpu / max_cpu
                
            historical_X = np.column_stack((historical_normalized_traffic, historical_normalized_cpu))
            historical_X = np.expand_dims(historical_X, axis=0)
            
            historical_predictions = []
            current_historical_input = historical_X.copy()
            
            for _ in range(steps):
                pred = gru_model.predict(current_historical_input, verbose=0)
                historical_predictions.append(pred[0, -1, 1] * max_cpu if max_cpu > 0 else pred[0, -1, 1])
                
                new_input = np.roll(current_historical_input[0], -1, axis=0)
                new_input[-1] = pred[0, -1]
                current_historical_input = np.expand_dims(new_input, axis=0)
            
            # Calculate MSE
            mse = np.mean((np.array(historical_predictions) - actual_values) ** 2)
            prediction_mse.labels(model="gru").set(mse)
            logger.info(f"GRU MSE: {mse}")
        
        logger.info(f"GRU forecast generated for {steps} steps")
        
        # Record prediction time
        elapsed_ms = (time.time() - start_time) * 1000
        prediction_time.labels(model="gru").set(elapsed_ms)
        logger.info(f"GRU prediction time: {elapsed_ms:.2f}ms")
        
        return predictions
    except Exception as e:
        logger.error(f"Error in GRU prediction: {e}")
        
        # Record prediction time even on error
        elapsed_ms = (time.time() - start_time) * 1000
        prediction_time.labels(model="gru").set(elapsed_ms)
        
        return None


def make_scaling_decision(predictions):
    """Determine if scaling is needed based on predictions."""
    global last_scaling_time
    
    if predictions is None or len(predictions) == 0:
        return "maintain", 0
    
    current_time = time.time()
    if current_time - last_scaling_time < SCALING_COOLDOWN:
        return "cooldown", 0
    
    # Calculate max predicted utilization
    max_prediction = max(predictions)
    
    # Determine number of replicas needed
    if max_prediction > SCALING_THRESHOLD:
        # Simple formula: each replica handles about 70% CPU utilization
        replicas_needed = round(max_prediction / SCALING_THRESHOLD)
        last_scaling_time = current_time
        return "scale", replicas_needed
    elif max_prediction < SCALING_THRESHOLD * 0.5:  # Only scale down if much lower
        last_scaling_time = current_time
        return "scale", max(1, round(max_prediction / SCALING_THRESHOLD))
    else:
        return "maintain", 0

def collect_metrics():
    """Simulate or collect real metrics on traffic and CPU."""
    global traffic_data, is_collecting
    
    while is_collecting:
        try:
            # In a real implementation, you would collect metrics from Kubernetes API
            # For simulation, we'll generate some data with daily patterns
            current_time = datetime.now()
            hour_of_day = current_time.hour
            
            # Simulated traffic pattern with higher values during business hours
            base_traffic = 100
            if 9 <= hour_of_day <= 17:  # Business hours
                traffic = base_traffic + np.random.normal(100, 20)
            else:
                traffic = base_traffic + np.random.normal(30, 10)
            
            # Simulated CPU utilization (correlated with traffic)
            cpu_util_value = 0.3 + (traffic / 300) + np.random.normal(0, 0.05)
            cpu_util_value = max(0, min(cpu_util_value, 1.0))  # Clamp between 0 and 1

            # Update Prometheus metrics
            cpu_utilization.set(cpu_util_value)  # Use different variable names
            traffic_gauge.set(traffic)

            # Store the data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            traffic_data.append({
                'timestamp': timestamp,
                'traffic': traffic,
                'cpu_utilization': cpu_util_value
            })
            
            # Periodically save data
            if len(traffic_data) % 10 == 0:
                save_data()
                
            # Check if we have enough data to train GRU model
            gru_config = config["models"]["gru"]
            train_size_needed = int(gru_config["train_size"])
            
            if not config['use_gru'] and len(traffic_data) >= train_size_needed:
                logger.info(f"Sufficient data collected ({len(traffic_data)} points), training GRU model")
                global gru_model
                gru_model = build_gru_model()
                if gru_model is not None:
                    config['use_gru'] = True
                    save_config()
            
            # Sleep for the collection interval
            time.sleep(config['collection_interval'])
            
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")
            time.sleep(5)  # Sleep for a short time on error

def start_collection():
    """Start the metrics collection thread."""
    global is_collecting, collection_thread
    
    if not is_collecting:
        is_collecting = True
        collection_thread = threading.Thread(target=collect_metrics)
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
    return jsonify({"status": "ok"})

# Change this route to a different path
@app.route('/data', methods=['GET'])
def get_metrics_data():
    """Return collected metrics data."""
    global traffic_data
    
    # Limit to last 100 records for API response
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
    
    # Get predictions
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
    
    # Make scaling decision
    decision, replicas = make_scaling_decision(predictions)
    scaling_decisions.labels(decision=decision).inc()
    recommended_replicas.set(replicas)
    
    # For future accuracy tracking, you might want to store these predictions
    # to compare with actual values later
    
    return jsonify({
        "method": method,
        "predictions": predictions,
        "scaling_decision": decision,
        "recommended_replicas": replicas
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
        gru_model = build_gru_model()
        if gru_model is not None:
            config['use_gru'] = True
            save_config()
            return jsonify({"status": "success", "message": "GRU model rebuilt"})
        else:
            return jsonify({"status": "error", "message": "Not enough data to build model"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500    

@app.route('/predict_combined', methods=['GET'])
def predict_combined():
    """
    Combined prediction endpoint that uses both GRU and Holt-Winters models,
    compares their predictions, and returns the final scaling decision.
    """
    try:
        # Load configuration
        gru_config = config["models"]["gru"]
        hw_config = config["models"]["holt_winters"]
        
        steps = request.args.get('steps', default=int(hw_config["look_forward"]), type=int)
        
        # Get predictions from both models with proper error handling
        gru_predictions = None
        hw_predictions = None
        
        # Holt-Winters prediction (more reliable as fallback)
        start_time_hw = time.time()
        try:
            hw_predictions = predict_with_holtwinters(steps)
            prediction_latency.labels(method="Holt-Winters").observe(time.time() - start_time_hw)
            prediction_requests.labels(method="Holt-Winters").inc()
            logger.info(f"Holt-Winters prediction successful: {hw_predictions}")
        except Exception as e:
            logger.error(f"Holt-Winters prediction failed: {e}")
            prediction_time.labels(model="holtwinters").set((time.time() - start_time_hw) * 1000)
            
        # Try GRU prediction if possible
        if config['use_gru'] and gru_model is not None:
            start_time_gru = time.time()
            try:
                gru_predictions = predict_with_gru(steps)
                prediction_latency.labels(method="GRU").observe(time.time() - start_time_gru)
                prediction_requests.labels(method="GRU").inc()
                logger.info(f"GRU prediction successful: {gru_predictions}")
            except Exception as e:
                logger.error(f"GRU prediction failed: {e}")
                prediction_time.labels(model="gru").set((time.time() - start_time_gru) * 1000)
        
        # Handle cases where one or both predictions are not available
        if hw_predictions is None and gru_predictions is None:
            logger.info("Not enough data for predictions, using fallback scaling")
            return jsonify({
                "method": "fallback",
                "predictions": [],
                "scaling_decision": "maintain",
                "recommended_replicas": 2  # Maintain a conservative number of replicas
            })
        
        # If only one prediction method is available, use it
        if gru_predictions is None:
            method = "Holt-Winters"
            selected_predictions = hw_predictions
        elif hw_predictions is None:
            method = "GRU"
            selected_predictions = gru_predictions
        else:
            # Both predictions available, compare them
            # Prefer the higher prediction for more conservative scaling
            gru_max = max(gru_predictions)
            hw_max = max(hw_predictions)
            
            if gru_max >= hw_max:
                method = "GRU"
                selected_predictions = gru_predictions
            else:
                method = "Holt-Winters"
                selected_predictions = hw_predictions
            
            logger.info(f"Comparison: GRU max={gru_max}, HW max={hw_max}, selected {method}")
        
        # Make scaling decision based on the highest prediction
        max_prediction = max(selected_predictions)
        recommended_replicas = max(1, round(max_prediction / SCALING_THRESHOLD))
        
        # Limit to reasonable range (1-10 replicas)
        recommended_replicas = min(10, max(1, recommended_replicas))
        
        # Update Prometheus metrics
        predictive_scaler_recommended_replicas.labels(method=method.lower()).set(recommended_replicas)
        
        return jsonify({
            "method": method,
            "predictions": selected_predictions,
            "scaling_decision": "scale",
            "recommended_replicas": recommended_replicas,
            "comparison": {
                "gru_available": gru_predictions is not None,
                "hw_available": hw_predictions is not None,
                "gru_max": max(gru_predictions) if gru_predictions else None,
                "hw_max": max(hw_predictions) if hw_predictions else None
            }
        })
    
    except Exception as e:
        logger.error(f"Error in predict_combined: {e}")
        # Return a safe fallback that won't crash the system
        return jsonify({
            "method": "error_fallback",
            "predictions": [1.0] * steps,
            "scaling_decision": "maintain", 
            "recommended_replicas": 1,
            "error": str(e)
        }), 200  # Return 200 instead of 500 to avoid breaking the controller
    

def compare_predictions_with_heap(gru_predictions, hw_predictions):
    """
    Compare predictions from GRU and Holt-Winters models using a min heap.
    
    This function inserts all predicted values from both models into a min heap
    (using negation to get max heap behavior), then determines which model
    provided the highest prediction value.
    
    Args:
        gru_predictions (list): Predictions from the GRU model
        hw_predictions (list): Predictions from the Holt-Winters model
        
    Returns:
        tuple: (selected_method, selected_predictions)
    """
    # Handle cases where one prediction method is unavailable
    if gru_predictions is None:
        return "Holt-Winters", hw_predictions
    if hw_predictions is None:
        return "GRU", gru_predictions
    
    # Create a min heap
    prediction_heap = []
    
    # Insert all individual predictions into the heap
    # Using negative values to create a max heap behavior
    for pred in gru_predictions:
        heapq.heappush(prediction_heap, (-pred, "GRU"))
    
    for pred in hw_predictions:
        heapq.heappush(prediction_heap, (-pred, "Holt-Winters"))
    
    # Get the highest prediction (top of our max heap)
    if prediction_heap:
        highest_neg_pred, highest_method = heapq.heappop(prediction_heap)
        highest_pred = -highest_neg_pred  # Convert back to positive
        
        logger.info(f"Heap analysis: Highest prediction {highest_pred} from {highest_method}")
        
        # Select the model that gave the highest prediction (most conservative for scaling)
        if highest_method == "GRU":
            return "GRU", gru_predictions
        else:
            return "Holt-Winters", hw_predictions
    
    # Fallback (should never happen if both predictions are valid)
    return "Holt-Winters", hw_predictions

# Initialization function
# app.py for predictive-scaler (only showing the changes)

# Initialization function
def initialize():
    load_config()
    load_data()
    load_gru_model()
    
    # Initialize metrics with zero values
    prediction_mse.labels(model="gru").set(0.0)
    prediction_mse.labels(model="holtwinters").set(0.0)
    training_time.labels(model="gru").set(0.0)
    training_time.labels(model="holtwinters").set(0.0)
    prediction_time.labels(model="gru").set(0.0)
    prediction_time.labels(model="holtwinters").set(0.0)
    predictive_scaler_recommended_replicas.labels(method='gru').set(0)
    predictive_scaler_recommended_replicas.labels(method='holtwinters').set(0)
    
    # Initialize HTTP metrics counter to avoid "no data" issues
    http_requests.labels(app='predictive-scaler', status_code='200', path='/').inc(0)
    
    start_collection()

# Flask application factory pattern for better compatibility with WSGI servers
@app.before_request
def before_request():
    # Start timing HTTP requests
    request.start_time = time.time()
    
    # Only initialize once
    if not hasattr(app, '_initialized'):
        initialize()
        app._initialized = True

@app.after_request
def after_request(response):
    # Record request duration
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        app_name = os.environ.get('APP_NAME', 'predictive-scaler')

        http_duration.labels(app=app_name).observe(duration)

        # Include the 'path' label when updating the 'http_requests' metric
        http_requests.labels(app=app_name, status_code=str(response.status_code), path=request.path).inc()

    return response

if __name__ == '__main__':
    # Initialize directly when run as a script
    initialize()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))