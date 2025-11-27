import requests
import json
import sys
import os

# Add /app to path if needed
sys.path.append('/app')

try:
    from app import enabled_models
    print(f"ENABLED_MODELS_IN_APP: {enabled_models}")
except ImportError as e:
    print(f"Could not import enabled_models from app: {e}")

try:
    print("Triggering training...")
    resp = requests.post('http://localhost:5000/api/models/train_all')
    print(f"TRAIN_RESPONSE: {resp.text}")
except Exception as e:
    print(f"REQUEST_FAILED: {e}")
