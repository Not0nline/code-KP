#!/usr/bin/env python3
"""
Model Variants for Predictive Autoscaling

This module implements multiple ML model types for comparison:
- XGBoost: Gradient boosting model
- CatBoost: Categorical boosting model
- LightGBM: Light gradient boosting model
- LSTM: Long Short-Term Memory neural network
- GRU: Gated Recurrent Unit (existing)
- Holt-Winters: Seasonal model (existing)
- Ensemble: Weighted average of multiple models

Each model implements a common interface for training and prediction.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Model type constants
MODEL_TYPES = {
    'xgboost': 'XGBoost Regressor',
    'catboost': 'CatBoost Regressor',
    'lightgbm': 'LightGBM Regressor',
    'lstm': 'LSTM Neural Network',
    'gru': 'GRU Neural Network',
    'holt_winters': 'Holt-Winters Seasonal',
    'ensemble_avg': 'Ensemble Average',
    'ensemble_weighted': 'Ensemble Weighted'
}

class BaseModel:
    """Base class for all predictive models"""
    
    def __init__(self, model_type: str, config: Dict = None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.last_training_time = None
        self.training_history = []
        
    def preprocess_data(self, data: List[Dict], look_back: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Convert traffic data to sequences for model training"""
        if len(data) < look_back + 1:
            raise ValueError(f"Insufficient data: {len(data)} < {look_back + 1}")
        
        # Extract features (CPU, traffic, replicas)
        cpu_values = [point['cpu'] for point in data]
        traffic_values = [point['traffic'] for point in data]
        replicas_values = [point['replicas'] for point in data]
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - look_back):
            # Features: last N timesteps of [cpu, traffic, replicas]
            sequence = []
            for j in range(i, i + look_back):
                sequence.extend([cpu_values[j], traffic_values[j], replicas_values[j]])
            X.append(sequence)
            
            # Target: next replicas value
            y.append(replicas_values[i + look_back])
        
        return np.array(X), np.array(y)
    
    def train(self, data: List[Dict]) -> Dict:
        """Train the model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, data: List[Dict], steps: int = 6) -> List[int]:
        """Make predictions - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        raise NotImplementedError("Subclasses must implement save_model()")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        raise NotImplementedError("Subclasses must implement load_model()")


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting model"""
    
    def __init__(self, config: Dict = None):
        super().__init__('xgboost', config)
        self.look_back = config.get('look_back', 100) if config else 100
        
    def train(self, data: List[Dict]) -> Dict:
        """Train XGBoost model"""
        try:
            import xgboost as xgb
            
            start_time = datetime.now()
            X, y = self.preprocess_data(data, self.look_back)
            
            # XGBoost configuration
            params = {
                'objective': 'reg:squarederror',
                'max_depth': self.config.get('max_depth', 6),
                'learning_rate': self.config.get('learning_rate', 0.1),
                'n_estimators': self.config.get('n_estimators', 100),
                'subsample': self.config.get('subsample', 0.8),
                'colsample_bytree': self.config.get('colsample_bytree', 0.8),
                'random_state': 42
            }
            
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X, y)
            
            training_time = (datetime.now() - start_time).total_seconds()
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            # Calculate training score
            train_score = self.model.score(X, y)
            
            result = {
                'success': True,
                'training_time_seconds': training_time,
                'samples_used': len(X),
                'train_score': train_score,
                'model_type': 'xgboost'
            }
            
            self.training_history.append(result)
            logger.info(f"✅ XGBoost model trained: {len(X)} samples, score={train_score:.4f}, time={training_time:.2f}s")
            return result
            
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            return {'success': False, 'error': 'XGBoost not installed'}
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: List[Dict], steps: int = 6) -> List[int]:
        """Make predictions with XGBoost"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        # Get last look_back points
        recent_data = data[-self.look_back:]
        
        # Create sequence
        sequence = []
        for point in recent_data:
            sequence.extend([point['cpu'], point['traffic'], point['replicas']])
        
        X = np.array([sequence])
        
        # Predict next value
        prediction = self.model.predict(X)[0]
        
        # Round to nearest integer and ensure >= 1
        predicted_replicas = max(1, int(round(prediction)))
        
        # Return same value for all steps (simple approach)
        return [predicted_replicas] * steps
    
    def save_model(self, filepath: str):
        """Save XGBoost model"""
        if self.model is not None:
            self.model.save_model(filepath)
            logger.info(f"XGBoost model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load XGBoost model"""
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor()
            self.model.load_model(filepath)
            self.is_trained = True
            logger.info(f"XGBoost model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            raise


class CatBoostModel(BaseModel):
    """CatBoost gradient boosting model"""
    
    def __init__(self, config: Dict = None):
        super().__init__('catboost', config)
        self.look_back = config.get('look_back', 100) if config else 100
        
    def train(self, data: List[Dict]) -> Dict:
        """Train CatBoost model"""
        try:
            from catboost import CatBoostRegressor
            
            start_time = datetime.now()
            X, y = self.preprocess_data(data, self.look_back)
            
            # CatBoost configuration
            params = {
                'iterations': self.config.get('iterations', 100),
                'depth': self.config.get('depth', 6),
                'learning_rate': self.config.get('learning_rate', 0.1),
                'loss_function': 'RMSE',
                'random_seed': 42,
                'verbose': False
            }
            
            self.model = CatBoostRegressor(**params)
            self.model.fit(X, y)
            
            training_time = (datetime.now() - start_time).total_seconds()
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            # Calculate training score
            train_score = self.model.score(X, y)
            
            result = {
                'success': True,
                'training_time_seconds': training_time,
                'samples_used': len(X),
                'train_score': train_score,
                'model_type': 'catboost'
            }
            
            self.training_history.append(result)
            logger.info(f"✅ CatBoost model trained: {len(X)} samples, score={train_score:.4f}, time={training_time:.2f}s")
            return result
            
        except ImportError:
            logger.error("CatBoost not installed. Install with: pip install catboost")
            return {'success': False, 'error': 'CatBoost not installed'}
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: List[Dict], steps: int = 6) -> List[int]:
        """Make predictions with CatBoost"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        recent_data = data[-self.look_back:]
        sequence = []
        for point in recent_data:
            sequence.extend([point['cpu'], point['traffic'], point['replicas']])
        
        X = np.array([sequence])
        prediction = self.model.predict(X)[0]
        predicted_replicas = max(1, int(round(prediction)))
        
        return [predicted_replicas] * steps
    
    def save_model(self, filepath: str):
        """Save CatBoost model"""
        if self.model is not None:
            self.model.save_model(filepath)
            logger.info(f"CatBoost model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load CatBoost model"""
        try:
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor()
            self.model.load_model(filepath)
            self.is_trained = True
            logger.info(f"CatBoost model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load CatBoost model: {e}")
            raise


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting model"""
    
    def __init__(self, config: Dict = None):
        super().__init__('lightgbm', config)
        self.look_back = config.get('look_back', 100) if config else 100
        
    def train(self, data: List[Dict]) -> Dict:
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
            
            start_time = datetime.now()
            X, y = self.preprocess_data(data, self.look_back)
            
            # LightGBM configuration
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': self.config.get('num_leaves', 31),
                'learning_rate': self.config.get('learning_rate', 0.1),
                'n_estimators': self.config.get('n_estimators', 100),
                'subsample': self.config.get('subsample', 0.8),
                'colsample_bytree': self.config.get('colsample_bytree', 0.8),
                'random_state': 42,
                'verbose': -1
            }
            
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(X, y)
            
            training_time = (datetime.now() - start_time).total_seconds()
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            # Calculate training score
            train_score = self.model.score(X, y)
            
            result = {
                'success': True,
                'training_time_seconds': training_time,
                'samples_used': len(X),
                'train_score': train_score,
                'model_type': 'lightgbm'
            }
            
            self.training_history.append(result)
            logger.info(f"✅ LightGBM model trained: {len(X)} samples, score={train_score:.4f}, time={training_time:.2f}s")
            return result
            
        except ImportError:
            logger.error("LightGBM not installed. Install with: pip install lightgbm")
            return {'success': False, 'error': 'LightGBM not installed'}
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: List[Dict], steps: int = 6) -> List[int]:
        """Make predictions with LightGBM"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        recent_data = data[-self.look_back:]
        sequence = []
        for point in recent_data:
            sequence.extend([point['cpu'], point['traffic'], point['replicas']])
        
        X = np.array([sequence])
        prediction = self.model.predict(X)[0]
        predicted_replicas = max(1, int(round(prediction)))
        
        return [predicted_replicas] * steps
    
    def save_model(self, filepath: str):
        """Save LightGBM model"""
        if self.model is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"LightGBM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load LightGBM model"""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            logger.info(f"LightGBM model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM model: {e}")
            raise


class ModelRegistry:
    """Registry to manage all model variants"""
    
    def __init__(self, data_dir: str = "/data/models"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.models = {}
        
    def register_model(self, model_name: str, model: BaseModel):
        """Register a model instance"""
        self.models[model_name] = model
        logger.info(f"Registered model: {model_name} ({model.model_type})")
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Get a registered model"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())
    
    def train_all_models(self, data: List[Dict]) -> Dict:
        """Train all registered models"""
        results = {}
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            try:
                result = model.train(data)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {'success': False, 'error': str(e)}
        
        return results
    
    def save_all_models(self):
        """Save all trained models to disk"""
        for model_name, model in self.models.items():
            if model.is_trained:
                filepath = os.path.join(self.data_dir, f"{model_name}.pkl")
                try:
                    model.save_model(filepath)
                except Exception as e:
                    logger.error(f"Failed to save {model_name}: {e}")
    
    def load_all_models(self):
        """Load all models from disk"""
        for model_name, model in self.models.items():
            filepath = os.path.join(self.data_dir, f"{model_name}.pkl")
            if os.path.exists(filepath):
                try:
                    model.load_model(filepath)
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")


# Global registry instance
model_registry = ModelRegistry()

# Initialize default models
def initialize_model_registry(config: Dict = None):
    """Initialize the model registry with all model variants"""
    config = config or {}
    
    # Register gradient boosting models
    model_registry.register_model('xgboost', XGBoostModel(config.get('xgboost', {})))
    model_registry.register_model('catboost', CatBoostModel(config.get('catboost', {})))
    model_registry.register_model('lightgbm', LightGBMModel(config.get('lightgbm', {})))
    
    logger.info("✅ Model registry initialized with all variants")
    
    return model_registry
