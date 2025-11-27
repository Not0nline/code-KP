import requests
import json
import sys
import os

# Add /app to path if needed
sys.path.append('/app')

print("Loading baseline (medium)...")
try:
    resp = requests.post('http://localhost:5000/api/baseline/load', json={'scenario': 'medium'})
    print(f"LOAD_RESPONSE: {resp.text}")
except Exception as e:
    print(f"LOAD_FAILED: {e}")

print("Triggering training...")
try:
    resp = requests.post('http://localhost:5000/api/models/train_all')
    print(f"TRAIN_RESPONSE: {resp.text}")
except Exception as e:
    print(f"TRAIN_FAILED: {e}")
