# Load Testing Guide

## 🚀 Quick Start

### 1. Deploy Latest Images
```bash
# SSH to EC2 server
ssh -i key_ta_2.pem ec2-user@ec2-52-54-42-13.compute-1.amazonaws.com

# Update deployments with latest images (with reset functionality)
kubectl set image deployment/product-app-hpa product-app=4dri4l/product-app:v20251027-1952
kubectl set image deployment/product-app-combined product-app=4dri4l/product-app:v20251027-1952
kubectl set image deployment/predictive-scaler predictive-scaler=4dri4l/predictive-scaler:v20251027-2015
kubectl set image deployment/load-tester load-tester=4dri4l/stress-test:v20251027-2015

# Apply updated HPA configuration
kubectl apply -f Product-App/HPA/product-app-hpa-hpa.yaml

# Wait for rollouts
kubectl rollout status deployment/product-app-hpa
kubectl rollout status deployment/predictive-scaler
kubectl rollout status deployment/load-tester
```

### 2. Start Medium Test
```bash
# Get load tester pod
kubectl get pods -l app=load-tester

# Access load tester
kubectl exec -it <load-tester-pod> -- /bin/bash

# Start test with medium scenario
python load_test.py --orchestrator \
  --iterations 20 \
  --duration 1800 \
  --scenarios medium \
  --cooldown 300 \
  --batch-name "medium_test_$(date +%Y%m%d_%H%M%S)"
```

### 3. Monitor Progress
```bash
# Check test status via API
curl http://load-tester-service.default.svc.cluster.local:8080/status

# Monitor pods and scaling
kubectl get pods -w
kubectl get hpa -w

# Check logs
kubectl logs -f deployment/load-tester
kubectl logs -f deployment/predictive-scaler
```

## 🧹 Clean Old Storage

### Remove Failed Test Data
```bash
# SSH to EC2 and find old test storage
sudo find /var/lib/rancher/k3s/storage -name "*batch_*" -type d

# Remove old/failed test data
sudo rm -rf /var/lib/rancher/k3s/storage/pvc-*/**/batch_20251027_*

# Reset applications
curl -X POST http://product-app-hpa.default.svc.cluster.local:5000/reset_data
curl -X POST http://product-app-combined.default.svc.cluster.local:5000/reset_data
curl -X POST http://predictive-scaler.default.svc.cluster.local:5000/reset_data
```

## 📊 Expected Results

- **Error Rate**: <5% (down from 45%)
- **Automatic Cleanup**: Databases reset between iterations
- **Consistent Performance**: No memory accumulation
- **Fair Scaling Comparison**: Both HPA and Combined use similar thresholds