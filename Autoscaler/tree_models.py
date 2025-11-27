"""
Tree-based Models (LightGBM & XGBoost) for Predictive Autoscaling
Gradient Boosting implementations for time-series prediction
"""

import numpy as np
import pandas as pd
import logging
import time
import joblib

logger = logging.getLogger(__name__)

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


class LightGBMPredictor:
    def __init__(self, config):
        self.config = config
        self.model_type = 'lightgbm'
        self.model = None
        self.is_trained = False
        
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not installed! Install with: pip install lightgbm")
    
    def create_lag_features(self, df, n_lags):
        """Create lag features for tree models"""
        for i in range(1, n_lags + 1):
            df[f'cpu_lag_{i}'] = df['cpu_utilization'].shift(i)
            df[f'traffic_lag_{i}'] = df['traffic'].shift(i)
            df[f'replicas_lag_{i}'] = df['replicas'].shift(i)
        return df.bfill()  # Use bfill() instead of deprecated fillna(method='bfill')
    
    def preprocess_data(self, data):
        """Preprocess data for LightGBM"""
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['minute'] = df['timestamp'].dt.minute
            
            # Lag features
            look_back = self.config.get('look_back', 24)
            df = self.create_lag_features(df, look_back)
            
            # Rolling statistics
            df['cpu_ma_5'] = df['cpu_utilization'].rolling(5, min_periods=1).mean()
            df['traffic_ma_5'] = df['traffic'].rolling(5, min_periods=1).mean()
            
            # Drop NaN rows
            df = df.dropna()
            
            if len(df) < 10:
                return None, None
            
            # Features and target (exclude non-numeric and non-feature columns)
            exclude_cols = ['timestamp', 'replicas', 'synthetic', 'label', 'idle']
            feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
            X = df[feature_cols].values
            y = df['replicas'].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"LightGBM preprocessing error: {e}")
            return None, None
    
    def train(self, data):
        """Train LightGBM model"""
        if not LIGHTGBM_AVAILABLE:
            return False
        
        start_time = time.time()
        
        try:
            X, y = self.preprocess_data(data)
            
            if X is None or len(X) < 10:
                logger.warning("LightGBM: Insufficient data")
                return False
            
            # LightGBM parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': self.config.get('num_leaves', 31),
                'learning_rate': self.config.get('learning_rate', 0.05),
                'max_depth': self.config.get('max_depth', 7),
                'verbose': -1
            }
            
            n_estimators = self.config.get('n_estimators', 100)
            
            train_data = lgb.Dataset(X, label=y)
            self.model = lgb.train(params, train_data, num_boost_round=n_estimators)
            
            self.is_trained = True
            training_time = (time.time() - start_time) * 1000
            
            logger.info(f"LightGBM trained in {training_time:.2f}ms with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"LightGBM training error: {e}")
            return False
    
    def predict(self, data, steps=24):
        """Make prediction"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            X, _ = self.preprocess_data(data)
            
            if X is None or len(X) == 0:
                return None
            
            # Use last row for prediction
            prediction = self.model.predict(X[-1:])
            predicted_replicas = max(1, int(round(prediction[0])))
            
            logger.debug(f"LightGBM prediction: {predicted_replicas}")
            return predicted_replicas
            
        except Exception as e:
            logger.error(f"LightGBM prediction error: {e}")
            return None
    
    def save(self, filepath):
        if self.model:
            joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        try:
            self.model = joblib.load(filepath)
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"LightGBM load error: {e}")
            return False


class XGBoostPredictor:
    def __init__(self, config):
        self.config = config
        self.model_type = 'xgboost'
        self.model = None
        self.is_trained = False
        
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not installed! Install with: pip install xgboost")
    
    def create_lag_features(self, df, n_lags):
        """Create lag features"""
        for i in range(1, n_lags + 1):
            df[f'cpu_lag_{i}'] = df['cpu_utilization'].shift(i)
            df[f'traffic_lag_{i}'] = df['traffic'].shift(i)
            df[f'replicas_lag_{i}'] = df['replicas'].shift(i)
        return df.bfill()  # Use bfill() instead of deprecated fillna(method='bfill')
    
    def preprocess_data(self, data):
        """Preprocess data for XGBoost"""
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['minute'] = df['timestamp'].dt.minute
            
            # Lag features
            look_back = self.config.get('look_back', 24)
            df = self.create_lag_features(df, look_back)
            
            # Rolling statistics
            df['cpu_ma_5'] = df['cpu_utilization'].rolling(5, min_periods=1).mean()
            df['traffic_ma_5'] = df['traffic'].rolling(5, min_periods=1).mean()
            
            df = df.dropna()
            
            if len(df) < 10:
                return None, None
            
            # Features and target (exclude non-numeric and non-feature columns)
            exclude_cols = ['timestamp', 'replicas', 'synthetic', 'label', 'idle']
            feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
            X = df[feature_cols].values
            y = df['replicas'].values
            
            return X, y
            
        except Exception as e:
            logger.error(f"XGBoost preprocessing error: {e}")
            return None, None
    
    def train(self, data):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            return False
        
        start_time = time.time()
        
        try:
            X, y = self.preprocess_data(data)
            
            if X is None or len(X) < 10:
                logger.warning("XGBoost: Insufficient data")
                return False
            
            # XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'max_depth': self.config.get('max_depth', 7),
                'learning_rate': self.config.get('learning_rate', 0.05),
                'subsample': self.config.get('subsample', 0.8),
                'colsample_bytree': self.config.get('colsample_bytree', 0.8),
                'verbosity': 0
            }
            
            n_estimators = self.config.get('n_estimators', 100)
            
            dtrain = xgb.DMatrix(X, label=y)
            self.model = xgb.train(params, dtrain, num_boost_round=n_estimators)
            
            self.is_trained = True
            training_time = (time.time() - start_time) * 1000
            
            logger.info(f"XGBoost trained in {training_time:.2f}ms with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"XGBoost training error: {e}")
            return False
    
    def predict(self, data, steps=24):
        """Make prediction"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            X, _ = self.preprocess_data(data)
            
            if X is None or len(X) == 0:
                return None
            
            dtest = xgb.DMatrix(X[-1:])
            prediction = self.model.predict(dtest)
            predicted_replicas = max(1, int(round(prediction[0])))
            
            logger.debug(f"XGBoost prediction: {predicted_replicas}")
            return predicted_replicas
            
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return None
    
    def save(self, filepath):
        if self.model:
            self.model.save_model(filepath)
    
    def load(self, filepath):
        try:
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"XGBoost load error: {e}")
            return False
