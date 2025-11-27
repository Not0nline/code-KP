#!/usr/bin/env python3
"""
Test Script for Advanced Models and Enhanced Metrics
Validates the implementation of research paper-based forecasting models
"""

import sys
import os
import time
import numpy as np
import requests
import json
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_DATA_POINTS = 200

def generate_test_data(points=200):
    """Generate synthetic time series data for testing"""
    np.random.seed(42)
    t = np.arange(points)
    
    # Create realistic Kubernetes workload pattern
    # - Daily seasonality (24-hour cycle scaled to data points)
    # - Weekly seasonality (7-day cycle)
    # - Random noise
    # - Occasional spikes
    
    daily_cycle = 2 * np.sin(2 * np.pi * t / (points / 7))  # 7 cycles over data
    weekly_cycle = 1 * np.sin(2 * np.pi * t / points)  # 1 full cycle
    noise = np.random.normal(0, 0.3, points)
    
    # Add occasional spikes (traffic bursts)
    spikes = np.zeros(points)
    spike_indices = np.random.choice(points, size=5, replace=False)
    spikes[spike_indices] = np.random.uniform(2, 4, 5)
    
    # Combine components and ensure positive values (replica counts)
    base_load = 3.0  # Base replica count
    data = base_load + daily_cycle + weekly_cycle + noise + spikes
    data = np.maximum(data, 1.0)  # Ensure minimum 1 replica
    
    return data.tolist()

def test_basic_connectivity():
    """Test basic API connectivity"""
    print("🔌 Testing basic connectivity...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Basic connectivity: OK")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_enhanced_metrics():
    """Test enhanced metrics system"""
    print("\n📊 Testing enhanced metrics system...")
    
    # Test metrics summary
    try:
        response = requests.get(f"{BASE_URL}/api/metrics/summary", timeout=10)
        if response.status_code == 200:
            summary = response.json()
            print("✅ Metrics summary: Available")
            
            if 'system' in summary and 'performance' in summary:
                print(f"   📈 Registered models: {summary['system']['registered_models']}")
                print(f"   🎯 Total predictions: {summary['system']['total_predictions']}")
            else:
                print("   ℹ️ Basic metrics (enhanced not available)")
                
        else:
            print(f"⚠️ Metrics summary failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Metrics test error: {e}")
    
    # Test metrics export
    try:
        response = requests.post(f"{BASE_URL}/api/metrics/export", timeout=10)
        if response.status_code == 200:
            print("✅ Metrics export: Available")
        else:
            print("⚠️ Metrics export: Not available")
            
    except Exception as e:
        print(f"⚠️ Metrics export error: {e}")

def test_advanced_models_status():
    """Test advanced models availability"""
    print("\n🧠 Testing advanced models status...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/models/advanced/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print("✅ Advanced models status: Available")
            
            if 'advanced_models' in status:
                models = status['advanced_models']
                for model_name, model_info in models.items():
                    if isinstance(model_info, dict):
                        available = model_info.get('available', False)
                        enabled = model_info.get('enabled', False) 
                        trained = model_info.get('trained', False)
                        
                        status_icon = "✅" if available else "❌"
                        print(f"   {status_icon} {model_name}: available={available}, enabled={enabled}, trained={trained}")
            else:
                print("   ⚠️ Advanced models not available")
                
        else:
            print(f"❌ Advanced models status failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Advanced models test error: {e}")

def test_model_training(model_name):
    """Test training of specific advanced model"""
    print(f"\n🔧 Testing {model_name} model training...")
    
    try:
        response = requests.post(f"{BASE_URL}/api/models/advanced/train/{model_name}", timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                training_time = result.get('training_time_ms', 0)
                print(f"✅ {model_name} training: SUCCESS ({training_time:.2f}ms)")
                return True
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ {model_name} training failed: {error}")
                return False
        else:
            print(f"❌ {model_name} training request failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name} training error: {e}")
        return False

def test_model_prediction(model_name):
    """Test prediction with specific advanced model"""
    print(f"\n🎯 Testing {model_name} model prediction...")
    
    try:
        payload = {"steps": 3}
        response = requests.post(
            f"{BASE_URL}/api/models/advanced/predict/{model_name}",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                predictions = result.get('predictions', [])
                prediction_time = result.get('prediction_time_ms', 0)
                print(f"✅ {model_name} prediction: SUCCESS")
                print(f"   📊 Predictions: {predictions}")
                print(f"   ⏱️ Time: {prediction_time:.2f}ms")
                return True
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ {model_name} prediction failed: {error}")
                return False
        else:
            print(f"❌ {model_name} prediction request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name} prediction error: {e}")
        return False

def test_existing_models():
    """Test existing model endpoints"""
    print("\n🔄 Testing existing models...")
    
    # Test basic prediction endpoint
    try:
        response = requests.get(f"{BASE_URL}/predict", timeout=10)
        if response.status_code == 200:
            print("✅ Basic prediction endpoint: OK")
        else:
            print(f"⚠️ Basic prediction endpoint: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Basic prediction error: {e}")
    
    # Test model status
    try:
        response = requests.get(f"{BASE_URL}/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print("✅ Status endpoint: OK")
            
            # Check model information
            if 'models' in status:
                models = status['models']
                print(f"   📊 Available models: {list(models.keys())}")
        else:
            print(f"⚠️ Status endpoint: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Status test error: {e}")

def run_performance_benchmark():
    """Run performance benchmark of all available models"""
    print("\n🏁 Running performance benchmark...")
    
    # Models to test (in order of complexity)
    models_to_test = ['arima', 'cnn', 'autoencoder', 'prophet', 'ensemble']
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🔬 Benchmarking {model_name}...")
        
        # Try training
        start_time = time.time()
        training_success = test_model_training(model_name)
        training_time = (time.time() - start_time) * 1000
        
        # Try prediction if training succeeded
        prediction_success = False
        prediction_time = 0
        
        if training_success:
            time.sleep(1)  # Brief pause between training and prediction
            start_time = time.time()
            prediction_success = test_model_prediction(model_name)
            prediction_time = (time.time() - start_time) * 1000
        
        results[model_name] = {
            'training_success': training_success,
            'training_time_ms': training_time,
            'prediction_success': prediction_success,
            'prediction_time_ms': prediction_time
        }
    
    # Print benchmark results
    print("\n📊 BENCHMARK RESULTS:")
    print("=" * 60)
    print(f"{'Model':<15} {'Train':<8} {'Predict':<8} {'Train Time':<12} {'Pred Time':<12}")
    print("-" * 60)
    
    for model_name, result in results.items():
        train_status = "✅ OK" if result['training_success'] else "❌ FAIL"
        pred_status = "✅ OK" if result['prediction_success'] else "❌ FAIL"
        train_time = f"{result['training_time_ms']:.0f}ms"
        pred_time = f"{result['prediction_time_ms']:.0f}ms" if result['prediction_success'] else "N/A"
        
        print(f"{model_name:<15} {train_status:<8} {pred_status:<8} {train_time:<12} {pred_time:<12}")
    
    return results

def generate_test_report(benchmark_results):
    """Generate comprehensive test report"""
    print("\n📋 COMPREHENSIVE TEST REPORT")
    print("=" * 70)
    
    total_models = len(benchmark_results)
    successful_training = sum(1 for r in benchmark_results.values() if r['training_success'])
    successful_predictions = sum(1 for r in benchmark_results.values() if r['prediction_success'])
    
    print(f"📊 Summary:")
    print(f"   Total models tested: {total_models}")
    print(f"   Successful training: {successful_training}/{total_models} ({successful_training/total_models*100:.1f}%)")
    print(f"   Successful predictions: {successful_predictions}/{total_models} ({successful_predictions/total_models*100:.1f}%)")
    
    print(f"\n🎯 Recommendations:")
    
    if successful_training == 0:
        print("   ❌ No models trained successfully - Check dependencies and data")
    elif successful_training < total_models:
        failed_models = [name for name, result in benchmark_results.items() if not result['training_success']]
        print(f"   ⚠️ Some models failed training: {failed_models}")
        print("   💡 Check logs for specific error messages")
    else:
        print("   ✅ All models trained successfully!")
    
    if successful_predictions > 0:
        # Find fastest model
        pred_times = {name: r['prediction_time_ms'] for name, r in benchmark_results.items() 
                     if r['prediction_success'] and r['prediction_time_ms'] > 0}
        if pred_times:
            fastest_model = min(pred_times.keys(), key=lambda x: pred_times[x])
            print(f"   🚀 Fastest prediction: {fastest_model} ({pred_times[fastest_model]:.0f}ms)")
    
    print(f"\n🔧 Next Steps:")
    print("   1. Enable successful models in model-config.yaml")
    print("   2. Monitor prediction accuracy over time")
    print("   3. Adjust model parameters based on workload characteristics")
    print("   4. Set up alerts for model performance degradation")

def main():
    """Main test execution"""
    print("🧪 Advanced Models and Enhanced Metrics Test Suite")
    print("=" * 60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Testing URL: {BASE_URL}")
    
    # Basic connectivity test
    if not test_basic_connectivity():
        print("❌ Cannot proceed - basic connectivity failed")
        sys.exit(1)
    
    # Test existing functionality
    test_existing_models()
    
    # Test enhanced metrics
    test_enhanced_metrics()
    
    # Test advanced models status
    test_advanced_models_status()
    
    # Run performance benchmark
    benchmark_results = run_performance_benchmark()
    
    # Generate report
    generate_test_report(benchmark_results)
    
    print(f"\n✅ Test suite completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 Check the autoscaler logs for detailed information about any failures")

if __name__ == "__main__":
    main()