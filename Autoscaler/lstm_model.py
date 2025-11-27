"""
LSTM Model Implementation for Predictive Autoscaling
Long Short-Term Memory Neural Network for time-series prediction
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging
import time

logger = logging.getLogger(__name__)

class LSTMPredictor:
    def __init__(self, config):
        """
        Initialize LSTM predictor
        
        Args:
            config: Dictionary with hyperparameters
                - look_back: int, input sequence length
                - look_forward: int, prediction steps ahead
                - train_size: int, minimum data points for training
                - batch_size: int
                - epochs: int
                - n_layers: int, number of LSTM layers
                - units: int, hidden units per layer
                - dropout: float, dropout rate
                - learning_rate: float
        """
        self.config = config
        self.model_type = 'lstm'
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.is_trained = False
        
    def build_model(self):
        """Build LSTM architecture"""
        look_back = self.config.get('look_back', 100)
        n_features = 6  # hour + day_of_week + cpu + memory + traffic + replicas
        n_layers = self.config.get('n_layers', 2)
        units = self.config.get('units', 50)
        dropout = self.config.get('dropout', 0.2)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        model = Sequential()
        
        # First LSTM layer
        if n_layers > 1:
            model.add(LSTM(units=units, return_sequences=True, 
                          input_shape=(look_back, n_features)))
        else:
            model.add(LSTM(units=units, return_sequences=False,
                          input_shape=(look_back, n_features)))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, n_layers):
            return_seq = (i < n_layers - 1)  # Last layer doesn't return sequences
            model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def preprocess_data(self, data):
        """Preprocess data for LSTM"""
        look_back = self.config.get('look_back', 100)
        
        if len(data) < look_back + 1:
            return None, None
        
        try:
            df = pd.DataFrame(data)
            
            # Create time features
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Select features
            features = ['hour', 'day_of_week', 'cpu_utilization', 
                       'memory_usage', 'traffic', 'replicas']
            
            X_data = df[features].ffill().bfill().fillna(0).values  # Use ffill()/bfill() instead of deprecated method
            y_data = df['replicas'].values.reshape(-1, 1)
            
            # Scale
            X_scaled = self.scaler_X.fit_transform(X_data)
            y_scaled = self.scaler_y.fit_transform(y_data)
            
            # Create sequences
            X, y = [], []
            for i in range(len(X_scaled) - look_back):
                X.append(X_scaled[i:i + look_back])
                y.append(y_scaled[i + look_back])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"LSTM preprocessing error: {e}")
            return None, None
    
    def train(self, data):
        """Train LSTM model"""
        start_time = time.time()
        
        try:
            X, y = self.preprocess_data(data)
            
            if X is None or len(X) < 10:
                logger.warning("LSTM: Insufficient data for training")
                return False
            
            # Build model if not exists
            if self.model is None:
                self.model = self.build_model()
            
            # Train
            batch_size = self.config.get('batch_size', 10)
            epochs = self.config.get('epochs', 200)
            
            self.model.fit(
                X, y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_split=0.1
            )
            
            self.is_trained = True
            training_time = (time.time() - start_time) * 1000
            
            logger.info(f"LSTM trained successfully in {training_time:.2f}ms with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"LSTM training error: {e}")
            return False
    
    def predict(self, data, steps=24):
        """Make prediction"""
        if not self.is_trained or self.model is None:
            logger.warning("LSTM: Model not trained")
            return None
        
        try:
            look_back = self.config.get('look_back', 100)
            
            if len(data) < look_back:
                logger.warning(f"LSTM: Need {look_back} data points, got {len(data)}")
                return None
            
            # Prepare input
            X, _ = self.preprocess_data(data)
            
            if X is None or len(X) == 0:
                return None
            
            # Use last sequence
            last_sequence = X[-1:, :, :]
            
            # Predict
            prediction_scaled = self.model.predict(last_sequence, verbose=0)
            prediction = self.scaler_y.inverse_transform(prediction_scaled)
            
            predicted_replicas = max(1, int(round(prediction[0][0])))
            
            logger.debug(f"LSTM prediction: {predicted_replicas} replicas")
            return predicted_replicas
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return None
    
    def save(self, filepath):
        """Save model"""
        if self.model:
            self.model.save(filepath)
            logger.info(f"LSTM model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        try:
            self.model = load_model(filepath)
            self.is_trained = True
            logger.info(f"LSTM model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"LSTM load error: {e}")
            return False
