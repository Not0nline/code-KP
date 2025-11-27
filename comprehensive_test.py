import requests
import json
import time

def comprehensive_system_test():
    """Comprehensive test of all working models and system integration"""
    
    scaler_ip = "10.43.82.244"
    
    print("=" * 60)
    print("COMPREHENSIVE ADVANCED MODELS SYSTEM TEST")
    print("=" * 60)
    
    # 1. Check all advanced models status
    print("\n1. ADVANCED MODELS STATUS:")
    try:
        response = requests.get(f"http://{scaler_ip}:5000/api/models/advanced/status")
        data = response.json()
        
        working_models = []
        for model_name, info in data['advanced_models'].items():
            status = "✅ ENABLED" if info.get('enabled') else "❌ DISABLED"
            trained = "✅ TRAINED" if info.get('trained') else "❌ NOT TRAINED"
            available = "✅ AVAILABLE" if info.get('available') else "❌ NOT AVAILABLE"
            
            print(f"  {model_name.upper()}: {status} | {trained} | {available}")
            
            # Add training details if available
            if info.get('trained') and 'metrics' in info:
                metrics = info['metrics']
                if 'training_time_ms' in metrics:
                    train_time = metrics['training_time_ms'] / 1000
                    print(f"    Training Time: {train_time:.2f}s")
                if 'prediction_time_ms' in metrics:
                    pred_time = metrics['prediction_time_ms']
                    print(f"    Last Prediction Time: {pred_time:.2f}ms")
            
            if info.get('trained') and info.get('available'):
                working_models.append(model_name)
        
        print(f"\n  📊 Working Advanced Models: {len(working_models)}")
        print(f"  🎯 Models Ready for Integration: {', '.join(working_models)}")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False
    
    # 2. Test predictions for all working advanced models
    print(f"\n2. PREDICTION TESTS ({len(working_models)} models):")
    successful_predictions = []
    
    for model in working_models:
        print(f"\n   Testing {model.upper()}:")
        try:
            start_time = time.time()
            response = requests.post(
                f"http://{scaler_ip}:5000/api/models/advanced/predict/{model}",
                json={},
                timeout=15
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('predictions', [0])[0]
                pred_time = result.get('prediction_time_ms', 0)
                
                print(f"     ✅ SUCCESS: {prediction:.6f}")
                print(f"     ⏱️ Prediction time: {pred_time:.2f}ms")
                print(f"     🌐 Total time: {elapsed:.2f}s")
                
                successful_predictions.append({
                    'model': model,
                    'prediction': prediction,
                    'prediction_time_ms': pred_time,
                    'total_time_s': elapsed
                })
            else:
                print(f"     ❌ FAILED ({response.status_code}): {response.text[:50]}")
                
        except Exception as e:
            print(f"     ❌ ERROR: {str(e)[:50]}")
    
    # 3. System Integration Analysis
    print(f"\n3. SYSTEM INTEGRATION ANALYSIS:")
    print(f"   📊 Successfully Tested Models: {len(successful_predictions)}")
    print(f"   🎯 Ready for MinHeap Integration: {'✅ Yes' if len(successful_predictions) >= 2 else '❌ Need more models'}")
    
    if successful_predictions:
        avg_pred_time = sum(p['prediction_time_ms'] for p in successful_predictions) / len(successful_predictions)
        print(f"   ⏱️ Average Prediction Time: {avg_pred_time:.2f}ms")
        
        predictions = [p['prediction'] for p in successful_predictions]
        avg_prediction = sum(predictions) / len(predictions)
        print(f"   📈 Average Prediction Value: {avg_prediction:.6f}")
        
        # Check prediction consistency
        if all(0.5 <= p <= 2.0 for p in predictions):
            print(f"   ✅ Predictions are in reasonable range (0.5-2.0)")
        else:
            print(f"   ⚠️ Some predictions outside reasonable range")
    
    # 4. Test multiple predictions for stability
    print(f"\n4. STABILITY TEST (5 consecutive predictions):")
    if successful_predictions:
        test_model = successful_predictions[0]['model']
        print(f"   Testing {test_model.upper()} stability:")
        
        stability_results = []
        for i in range(5):
            try:
                response = requests.post(
                    f"http://{scaler_ip}:5000/api/models/advanced/predict/{test_model}",
                    json={},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get('predictions', [0])[0]
                    stability_results.append(prediction)
                    print(f"     Test {i+1}: {prediction:.6f}")
                else:
                    print(f"     Test {i+1}: ❌ Failed")
                    
            except Exception as e:
                print(f"     Test {i+1}: ❌ Error")
            
            time.sleep(1)
        
        if len(stability_results) >= 4:
            variance = max(stability_results) - min(stability_results)
            print(f"   📊 Variance: {variance:.6f}")
            print(f"   🎯 Stability: {'✅ Good' if variance < 0.1 else '⚠️ High variance'}")
    
    # 5. Final Summary and Recommendations
    print(f"\n" + "="*60)
    print("FINAL SYSTEM STATUS")
    print("="*60)
    
    working_count = len(successful_predictions)
    print(f"✅ Working Advanced Models: {working_count}/5")
    
    if working_count >= 3:
        print("🎯 RECOMMENDATION: ✅ SYSTEM READY FOR PRODUCTION")
        print("   - Enable working models in main configuration")
        print("   - Integrate with MinHeap selection")
        print("   - Start MSE tracking")
        print("   - Enable advanced model scaling decisions")
    elif working_count >= 2:
        print("🎯 RECOMMENDATION: ⚡ SYSTEM READY FOR TESTING")
        print("   - Enable working models for testing")
        print("   - Continue optimizing remaining models")
        print("   - Monitor performance metrics")
    else:
        print("🎯 RECOMMENDATION: ⚠️ SYSTEM NEEDS MORE WORK")
        print("   - Fix remaining model issues")
        print("   - Focus on stability improvements")
    
    print(f"\n📋 Working Models Summary:")
    for pred in successful_predictions:
        print(f"   • {pred['model'].upper()}: {pred['prediction']:.4f} ({pred['prediction_time_ms']:.1f}ms)")
    
    return working_count >= 2

if __name__ == "__main__":
    system_ready = comprehensive_system_test()
    
    if system_ready:
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Enable working models in configuration")
        print(f"   2. Test ensemble model creation")
        print(f"   3. Validate MinHeap integration")
        print(f"   4. Monitor system performance")
    else:
        print(f"\n🔧 NEXT STEPS:")
        print(f"   1. Debug remaining model issues")
        print(f"   2. Optimize training parameters")
        print(f"   3. Check system dependencies")