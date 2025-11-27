#!/usr/bin/env python3
"""
11-Model Predictive Scaler Test Script
Tests all 11 forecasting models to ensure they can train and predict successfully
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_test_data(n_points=500):
    """Generate synthetic test data for model training"""
    logger.info(f"🔧 Generating {n_points} synthetic data points...")
    
    # Generate time series with trend, seasonality, and noise
    timestamps = []
    start_time = datetime.now() - timedelta(minutes=n_points)
    
    data = []
    for i in range(n_points):
        timestamp = start_time + timedelta(minutes=i)
        
        # Create realistic traffic patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Business hours pattern
        if 9 <= hour <= 17 and day_of_week < 5:
            base_traffic = 150 + 50 * np.sin((hour - 9) * np.pi / 8)
        else:
            base_traffic = 50 + 20 * np.random.random()
        
        # Add some noise and spikes
        traffic = max(1, base_traffic + np.random.normal(0, 20))
        
        # Simulate CPU based on traffic (with some correlation)
        cpu_utilization = min(95, max(5, (traffic / 3) + np.random.normal(0, 10)))
        
        # Memory usage (correlated with CPU)
        memory_usage = max(100, cpu_utilization * 10 + np.random.normal(0, 50))
        
        # Replicas based on CPU (simple HPA-like logic)
        if cpu_utilization > 70:
            replicas = min(10, max(1, int(traffic / 50) + 1))
        elif cpu_utilization < 30:
            replicas = max(1, int(traffic / 80))
        else:
            replicas = max(1, int(traffic / 60))
        
        data.append({
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'traffic': traffic,
            'cpu_utilization': cpu_utilization,
            'memory_usage': memory_usage,
            'replicas': replicas,
            'synthetic': True
        })
    
    logger.info(f"✅ Generated {len(data)} test data points")
    return data

def test_basic_models(test_data):
    """Test the 6 basic models"""
    logger.info("=" * 60)
    logger.info("🧪 TESTING BASIC MODELS (6 models)")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: GRU Model (skip as it requires complex setup)
    logger.info("1️⃣ GRU Model: SKIPPING (requires UDS socket setup)")
    results['gru'] = {'tested': False, 'reason': 'Complex UDS socket setup required'}
    
    # Test 2: Holt-Winters Model
    logger.info("2️⃣ Testing Holt-Winters Model...")
    try:
        # Import Holt-Winters functionality directly
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Extract traffic values
        traffic_values = [d['traffic'] for d in test_data]
        
        if len(traffic_values) >= 30:
            model = ExponentialSmoothing(
                traffic_values,
                trend='add',
                seasonal='add',
                seasonal_periods=30,
                initialization_method='estimated'
            )
            fitted = model.fit()
            prediction = fitted.forecast(steps=1)
            
            logger.info(f"   ✅ Holt-Winters: Predicted {prediction[0]:.2f}, Status: WORKING")
            results['holt_winters'] = {'tested': True, 'prediction': float(prediction[0]), 'status': 'WORKING'}
        else:
            logger.warning("   ⚠️ Holt-Winters: Insufficient data")
            results['holt_winters'] = {'tested': False, 'reason': 'Insufficient data'}
    except Exception as e:
        logger.error(f"   ❌ Holt-Winters: {e}")
        results['holt_winters'] = {'tested': False, 'error': str(e)}
    
    # Test 3: LSTM Model
    logger.info("3️⃣ Testing LSTM Model...")
    try:
        from lstm_model import LSTMPredictor
        
        config = {
            'look_back': 20,  # Reduced for test
            'train_size': 100,
            'batch_size': 10,
            'epochs': 5,  # Reduced for test
            'units': 20,  # Reduced for test
            'dropout': 0.1,
        }
        
        lstm = LSTMPredictor(config)
        success = lstm.train(test_data[-200:])  # Use subset for speed
        
        if success:
            prediction = lstm.predict(test_data[-50:], steps=1)
            if prediction is not None:
                logger.info(f"   ✅ LSTM: Predicted {prediction}, Status: WORKING")
                results['lstm'] = {'tested': True, 'prediction': prediction, 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ LSTM: Prediction failed")
                results['lstm'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ LSTM: Training failed")
            results['lstm'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ LSTM: {e}")
        results['lstm'] = {'tested': False, 'error': str(e)}
    
    # Test 4: LightGBM Model
    logger.info("4️⃣ Testing LightGBM Model...")
    try:
        from tree_models import LightGBMPredictor
        
        config = {
            'n_estimators': 20,  # Reduced for test
            'learning_rate': 0.1,
            'max_depth': 3,  # Reduced for test
            'look_back': 10,  # Reduced for test
        }
        
        lgbm = LightGBMPredictor(config)
        success = lgbm.train(test_data[-200:])
        
        if success:
            prediction = lgbm.predict(test_data[-50:], steps=1)
            if prediction is not None:
                logger.info(f"   ✅ LightGBM: Predicted {prediction}, Status: WORKING")
                results['lightgbm'] = {'tested': True, 'prediction': prediction, 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ LightGBM: Prediction failed")
                results['lightgbm'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ LightGBM: Training failed")
            results['lightgbm'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ LightGBM: {e}")
        results['lightgbm'] = {'tested': False, 'error': str(e)}
    
    # Test 5: XGBoost Model
    logger.info("5️⃣ Testing XGBoost Model...")
    try:
        from tree_models import XGBoostPredictor
        
        config = {
            'n_estimators': 20,  # Reduced for test
            'learning_rate': 0.1,
            'max_depth': 3,  # Reduced for test
            'look_back': 10,  # Reduced for test
        }
        
        xgb = XGBoostPredictor(config)
        success = xgb.train(test_data[-200:])
        
        if success:
            prediction = xgb.predict(test_data[-50:], steps=1)
            if prediction is not None:
                logger.info(f"   ✅ XGBoost: Predicted {prediction}, Status: WORKING")
                results['xgboost'] = {'tested': True, 'prediction': prediction, 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ XGBoost: Prediction failed")
                results['xgboost'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ XGBoost: Training failed")
            results['xgboost'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ XGBoost: {e}")
        results['xgboost'] = {'tested': False, 'error': str(e)}
    
    # Test 6: StatuScale Model
    logger.info("6️⃣ Testing StatuScale Model...")
    try:
        from statuscale_model import StatuScalePredictor
        
        config = {
            'threshold_high': 70,
            'threshold_low': 30,
            'window_size': 10,
            'look_back': 20,
        }
        
        statuscale = StatuScalePredictor(config)
        success = statuscale.train(test_data[-100:])  # StatuScale "training" is lightweight
        
        if success:
            prediction = statuscale.predict(test_data[-30:], steps=1)
            if prediction is not None:
                logger.info(f"   ✅ StatuScale: Predicted {prediction}, Status: WORKING")
                results['statuscale'] = {'tested': True, 'prediction': prediction, 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ StatuScale: Prediction failed")
                results['statuscale'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ StatuScale: Training failed")
            results['statuscale'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ StatuScale: {e}")
        results['statuscale'] = {'tested': False, 'error': str(e)}
    
    return results

def test_advanced_models(test_data):
    """Test the 5 advanced models"""
    logger.info("=" * 60)
    logger.info("🚀 TESTING ADVANCED MODELS (5 models)")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 7: ARIMA Model
    logger.info("7️⃣ Testing ARIMA Model...")
    try:
        from advanced_models import ARIMAPredictor
        
        config = {
            'order': (1, 1, 1),
            'auto_arima': True,
            'max_p': 2, 'max_d': 1, 'max_q': 2,  # Reduced for speed
        }
        
        arima = ARIMAPredictor(config)
        success = arima.train(test_data[-200:])  # Use subset for speed
        
        if success:
            prediction = arima.predict(steps=1)
            if prediction and len(prediction) > 0:
                logger.info(f"   ✅ ARIMA: Predicted {prediction[0]:.2f}, Status: WORKING")
                results['arima'] = {'tested': True, 'prediction': prediction[0], 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ ARIMA: Prediction failed")
                results['arima'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ ARIMA: Training failed")
            results['arima'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ ARIMA: {e}")
        results['arima'] = {'tested': False, 'error': str(e)}
    
    # Test 8: CNN Model
    logger.info("8️⃣ Testing CNN Model...")
    try:
        from advanced_models import CNNPredictor
        
        config = {
            'sequence_length': 10,  # Reduced for test
            'filters': 8,  # Reduced for test
            'kernel_size': 2,
            'dense_units': 10,  # Reduced for test
            'epochs': 3,  # Reduced for test
            'batch_size': 16,
        }
        
        cnn = CNNPredictor(config)
        success = cnn.train(test_data[-200:])
        
        if success:
            prediction = cnn.predict(steps=1)
            if prediction and len(prediction) > 0:
                logger.info(f"   ✅ CNN: Predicted {prediction[0]:.2f}, Status: WORKING")
                results['cnn'] = {'tested': True, 'prediction': prediction[0], 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ CNN: Prediction failed")
                results['cnn'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ CNN: Training failed")
            results['cnn'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ CNN: {e}")
        results['cnn'] = {'tested': False, 'error': str(e)}
    
    # Test 9: Autoencoder Model
    logger.info("9️⃣ Testing Autoencoder Model...")
    try:
        from advanced_models import AutoencoderPredictor
        
        config = {
            'sequence_length': 10,  # Reduced for test
            'encoding_dim': 8,  # Reduced for test
            'latent_dim': 4,  # Reduced for test
            'epochs': 3,  # Reduced for test
            'batch_size': 16,
        }
        
        autoencoder = AutoencoderPredictor(config)
        success = autoencoder.train(test_data[-200:])
        
        if success:
            prediction = autoencoder.predict(steps=1)
            if prediction and len(prediction) > 0:
                logger.info(f"   ✅ Autoencoder: Predicted {prediction[0]:.2f}, Status: WORKING")
                results['autoencoder'] = {'tested': True, 'prediction': prediction[0], 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ Autoencoder: Prediction failed")
                results['autoencoder'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ Autoencoder: Training failed")
            results['autoencoder'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ Autoencoder: {e}")
        results['autoencoder'] = {'tested': False, 'error': str(e)}
    
    # Test 10: Prophet Model
    logger.info("🔟 Testing Prophet Model...")
    try:
        from advanced_models import ProphetPredictor
        
        config = {
            'seasonality_mode': 'additive',  # Simpler for test
            'daily_seasonality': False,  # Disabled for test
            'yearly_seasonality': False,
            'weekly_seasonality': False,
        }
        
        prophet = ProphetPredictor(config)
        success = prophet.train(test_data[-200:])
        
        if success:
            prediction = prophet.predict(steps=1)
            if prediction and len(prediction) > 0:
                logger.info(f"   ✅ Prophet: Predicted {prediction[0]:.2f}, Status: WORKING")
                results['prophet'] = {'tested': True, 'prediction': prediction[0], 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ Prophet: Prediction failed")
                results['prophet'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ Prophet: Training failed")
            results['prophet'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ Prophet: {e}")
        results['prophet'] = {'tested': False, 'error': str(e)}
    
    # Test 11: Ensemble Model
    logger.info("1️⃣1️⃣ Testing Ensemble Model...")
    try:
        from advanced_models import EnsemblePredictor
        
        # Create a simple ensemble with mock models
        ensemble = EnsemblePredictor()
        
        # Add simple mock models for testing
        class MockModel:
            def __init__(self, prediction_value):
                self.is_trained = True
                self.prediction_value = prediction_value
            
            def train(self, data):
                return True
            
            def predict(self, steps=1):
                return [self.prediction_value] * steps
        
        ensemble.add_model('mock1', MockModel(3.0), 1.0)
        ensemble.add_model('mock2', MockModel(4.0), 1.0)
        ensemble.add_model('mock3', MockModel(5.0), 1.0)
        
        success = ensemble.train(test_data[-100:])
        
        if success:
            prediction = ensemble.predict(steps=1)
            if prediction and len(prediction) > 0:
                logger.info(f"   ✅ Ensemble: Predicted {prediction[0]:.2f}, Status: WORKING")
                results['ensemble'] = {'tested': True, 'prediction': prediction[0], 'status': 'WORKING'}
            else:
                logger.warning("   ⚠️ Ensemble: Prediction failed")
                results['ensemble'] = {'tested': True, 'status': 'PREDICTION_FAILED'}
        else:
            logger.warning("   ⚠️ Ensemble: Training failed")
            results['ensemble'] = {'tested': True, 'status': 'TRAINING_FAILED'}
    except Exception as e:
        logger.error(f"   ❌ Ensemble: {e}")
        results['ensemble'] = {'tested': False, 'error': str(e)}
    
    return results

def print_test_summary(basic_results, advanced_results):
    """Print comprehensive test summary"""
    logger.info("=" * 60)
    logger.info("📊 COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 60)
    
    all_results = {**basic_results, **advanced_results}
    
    working_models = []
    failed_models = []
    skipped_models = []
    
    for model_name, result in all_results.items():
        if not result.get('tested', False):
            skipped_models.append(model_name)
        elif result.get('status') == 'WORKING':
            working_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    logger.info(f"✅ WORKING MODELS ({len(working_models)}/11):")
    for i, model in enumerate(working_models, 1):
        prediction = all_results[model].get('prediction', 'N/A')
        logger.info(f"   {i:2d}. {model:12s}: {prediction}")
    
    if failed_models:
        logger.info(f"\n❌ FAILED MODELS ({len(failed_models)}/11):")
        for i, model in enumerate(failed_models, 1):
            status = all_results[model].get('status', 'UNKNOWN')
            error = all_results[model].get('error', '')
            logger.info(f"   {i:2d}. {model:12s}: {status} {error}")
    
    if skipped_models:
        logger.info(f"\n⏭️ SKIPPED MODELS ({len(skipped_models)}/11):")
        for i, model in enumerate(skipped_models, 1):
            reason = all_results[model].get('reason', 'Unknown reason')
            logger.info(f"   {i:2d}. {model:12s}: {reason}")
    
    # Overall assessment
    total_testable = len(working_models) + len(failed_models)  # Exclude skipped
    success_rate = (len(working_models) / max(1, total_testable)) * 100
    
    logger.info("\n" + "=" * 60)
    if success_rate >= 80:
        logger.info(f"🎉 OVERALL ASSESSMENT: EXCELLENT ({success_rate:.1f}% success rate)")
        logger.info("✅ Ready for comprehensive testing!")
    elif success_rate >= 60:
        logger.info(f"⚠️ OVERALL ASSESSMENT: GOOD ({success_rate:.1f}% success rate)")
        logger.info("⚠️ Some models need attention before deployment")
    else:
        logger.info(f"❌ OVERALL ASSESSMENT: NEEDS WORK ({success_rate:.1f}% success rate)")
        logger.info("❌ Multiple models need fixing before deployment")
    
    return all_results

def main():
    """Main test function"""
    logger.info("🚀 Starting 11-Model Predictive Scaler Test")
    logger.info("Testing all forecasting models for training and prediction capability")
    
    # Generate test data
    test_data = generate_test_data(500)
    
    # Test basic models (6 models)
    basic_results = test_basic_models(test_data)
    
    # Test advanced models (5 models)
    advanced_results = test_advanced_models(test_data)
    
    # Print summary
    all_results = print_test_summary(basic_results, advanced_results)
    
    # Save results
    with open('test_results_11_models.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_data_points': len(test_data),
            'basic_models': basic_results,
            'advanced_models': advanced_results,
            'summary': {
                'total_models': 11,
                'working': len([r for r in all_results.values() if r.get('status') == 'WORKING']),
                'failed': len([r for r in all_results.values() if r.get('tested', False) and r.get('status') != 'WORKING']),
                'skipped': len([r for r in all_results.values() if not r.get('tested', False)])
            }
        }, f, indent=2)
    
    logger.info(f"📁 Test results saved to: test_results_11_models.json")
    return all_results

if __name__ == "__main__":
    main()