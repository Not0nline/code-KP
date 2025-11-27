#!/usr/bin/env python3
"""
Direct test of advanced model endpoints with error details
"""
import requests
import json
import sys

def test_advanced_endpoint(base_url, model_name):
    """Test a specific advanced model endpoint"""
    url = f"{base_url}/api/models/advanced/predict/{model_name}"
    
    headers = {'Content-Type': 'application/json'}
    data = {'steps': 1}
    
    print(f"\n🧪 Testing {model_name.upper()} endpoint:")
    print(f"   URL: {url}")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Success: {result}")
        else:
            try:
                error = response.json()
                print(f"   ❌ Error: {error}")
            except:
                print(f"   ❌ Raw Error: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Network Error: {e}")

def test_status_endpoint(base_url):
    """Test the status endpoint"""
    url = f"{base_url}/status"
    
    print(f"\n📊 Testing STATUS endpoint:")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Models Status: {result.get('models', 'N/A')}")
            print(f"   ✅ Advanced Models: {result.get('advanced_models', 'N/A')}")
        else:
            print(f"   ❌ Error: {response.text}")
                
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Network Error: {e}")

if __name__ == "__main__":
    # Get service IP
    if len(sys.argv) > 1:
        base_url = f"http://{sys.argv[1]}:5000"
    else:
        base_url = "http://predictive-scaler:5000"
    
    print(f"🔬 Testing Advanced Models API")
    print(f"🌐 Base URL: {base_url}")
    
    # Test status first
    test_status_endpoint(base_url)
    
    # Test each advanced model
    advanced_models = ['arima', 'cnn', 'autoencoder', 'prophet', 'ensemble']
    
    for model in advanced_models:
        test_advanced_endpoint(base_url, model)
    
    print(f"\n🏁 Testing complete!")