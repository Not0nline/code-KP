#!/bin/bash
# Load baseline and train all 6 models

echo "🔄 Loading baseline data (low scenario)..."
kubectl run curl-helper --image=curlimages/curl:latest --rm -i --restart=Never -- sh -c 'curl -s -X POST http://predictive-scaler.default.svc.cluster.local:5000/api/baseline/load -H "Content-Type: application/json" -d "{\"scenario\": \"low\"}"'

echo ""
echo "🧠 Training all models..."
kubectl run curl-helper --image=curlimages/curl:latest --rm -i --restart=Never -- sh -c 'curl -s -X POST http://predictive-scaler.default.svc.cluster.local:5000/api/models/train_all'

echo ""
echo "✅ Done! Checking model status..."
kubectl run curl-helper --image=curlimages/curl:latest --rm -i --restart=Never -- sh -c 'curl -s http://predictive-scaler.default.svc.cluster.local:5000/debug/model_comparison'
