#!/usr/bin/env python3
"""
Model Variants for Predictive Autoscaling
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Custom model imports - use try-except to handle missing dependencies gracefully
try:
    from advanced_models import ARIMAPredictor, CNNPredictor, AutoencoderPredictor, ProphetPredictor
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

try:
    from tree_models import LightGBMPredictor, XGBoostPredictor
    TREE_AVAILABLE = True
except ImportError:
    TREE_AVAILABLE = False
    
try:
    from lstm_model import LSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


logger = logging.getLogger(__name__)

# Model type constants
MODEL_TYPES = {
    'xgboost': 'XGBoost Regressor',
    'lightgbm': 'LightGBM Regressor',
    'lstm': 'LSTM Neural Network',
    'gru': 'GRU Neural Network',
    'arima': 'ARIMA Model',
    'cnn': 'CNN Model',
    'autoencoder': 'Autoencoder Model',
    'prophet': 'Prophet Model',
    'holt_winters': 'Holt-Winters Seasonal',
}

class BaseModel:
    """Base class for all predictive models"""
    
    def __init__(self, model_type: str, config: Dict = None):
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.is_trained = False
        
    def train(self, data: List[Dict]) -> Dict:
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, data: List[Dict], steps: int = 6) -> Optional[int]:
        raise NotImplementedError("Subclasses must implement predict()")
    
    def save_model(self, filepath: str):
        raise NotImplementedError("Subclasses must implement save_model()")
    
    def load_model(self, filepath: str):
        raise NotImplementedError("Subclasses must implement load_model()")


class ModelRegistry:
    """Registry to manage all model variants"""
    
    def __init__(self, data_dir: str = "/data/models"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.models: Dict[str, BaseModel] = {}
        
    def register_model(self, model_name: str, model: BaseModel):
        """Register a model instance"""
        self.models[model_name] = model
        logger.info(f"Registered model: {model_name} ({model.model_type})")
    
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

# Global registry instance
model_registry = ModelRegistry()

def initialize_model_registry(config: Dict = None, enabled_models: Dict = None):
    """Initialize the model registry with all specified and available model variants"""
    global model_registry
    config = config or {}
    enabled_models = enabled_models or {}

    # Define all possible models and their dependencies
    all_possible_models = {
        'xgboost': (XGBoostPredictor, TREE_AVAILABLE),
        'lightgbm': (LightGBMPredictor, TREE_AVAILABLE),
        'lstm': (LSTMPredictor, LSTM_AVAILABLE),
        'arima': (ARIMAPredictor, ADVANCED_AVAILABLE),
        'cnn': (CNNPredictor, ADVANCED_AVAILABLE),
        'autoencoder': (AutoencoderPredictor, ADVANCED_AVAILABLE),
        'prophet': (ProphetPredictor, ADVANCED_AVAILABLE),
        # GRU and Holt-Winters are often special-cased in app.py, but can be registered if needed
    }

    # Clear previous registrations
    model_registry.models = {}

    for name, (model_class, is_available) in all_possible_models.items():
        if enabled_models.get(name, False):
            if is_available:
                model_registry.register_model(name, model_class(config.get(name, {})))
            else:
                logger.warning(f"Model '{name}' is enabled but its dependencies are not available. Skipping.")
    
    logger.info(f"✅ Model registry initialized with {len(model_registry.models)} models.")
    
    return model_registry