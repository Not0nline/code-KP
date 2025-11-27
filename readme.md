# 11-Model Predictive Kubernetes Autoscaling Research

This repository contains a comprehensive implementation for comparing traditional HPA against an advanced 11-model predictive autoscaling system for Kubernetes. The research evaluates performance, cost efficiency, and quality-of-service tradeoffs across multiple forecasting approaches.

## Research Overview

- **Total Models**: 11 predictive models spanning 4 algorithmic paradigms
- **Test Protocol**: 360 total tests (30 HPA baseline + 330 predictive model tests)  
- **Infrastructure**: AWS K3s cluster with Prometheus monitoring
- **Docker Images**: `4dri4l/predictive-scaler:v3.0-11models` with complete model suite

## 11-Model Predictive System

### Neural Networks (4 models)
- **GRU**: Gated Recurrent Unit for complex temporal patterns
- **LSTM**: Long Short-Term Memory for sequential data
- **CNN**: Convolutional Neural Network for pattern recognition  
- **Autoencoder**: Neural compression for anomaly detection

### Statistical Models (2 models)  
- **Holt-Winters**: Exponential smoothing with trend/seasonality
- **ARIMA**: AutoRegressive Integrated Moving Average

### Tree-Based Models (2 models)
- **XGBoost**: Gradient boosting with feature engineering
- **LightGBM**: Efficient gradient boosting for large datasets

### Hybrid/Specialized Models (2 models)
- **StatuScale**: Rule-based with burst detection (Hilman et al.)
- **Prophet**: Facebook's business forecasting library

### Meta-Model (1 model)
- **Ensemble**: Weighted combination with dynamic selection

## Repository Structure

```text
.
├── Autoscaler/                    # 11-Model Predictive Scaler
│   ├── app.py                     # Main application (11-model support)
│   ├── model-config.yaml          # Model selection configuration  
│   ├── advanced_models.py         # ARIMA, CNN, Autoencoder, Prophet, Ensemble
│   ├── lstm_model.py             # LSTM implementation
│   ├── tree_models.py            # XGBoost, LightGBM
│   ├── statuscale_model.py       # StatuScale hybrid model
│   ├── test_11_models.py         # Comprehensive test suite
│   └── Dockerfile                # Multi-model container
├── Monitoring/                    # Prometheus + Grafana setup
├── Product-App/                   # Test application with HPA/Combined modes
├── stress-test/                   # Load testing framework
├── fixed_hpa_baseline_tests.sh    # HPA baseline test suite (30 tests)
├── download_test_results.py       # Results collection utility  
├── working_validation_test.sh     # System validation
└── Research Context Files:
    ├── paper.txt                  # Research paper content
    ├── paper_context.txt          # Research context
    ├── statuscale.txt             # StatuScale algorithm details  
    └── time_series.txt            # Time-series forecasting theory
  ├── load_test.py                 # Load tester script (supports client metrics)
  ├── Dockerfile                   # Container image for auto-deployed load tester
  ├── requirements.txt             # Python deps for image
  ├── entrypoint.sh                # Container startup (fetch latest script, run tester)
  ├── load-tester-deployment.yaml  # Deployment + Service + ServiceMonitor (auto)
  ├── load-test-pod.yaml           # Legacy one-off pod (manual execution)
  ├── load-test-service.yaml       # Legacy Service (metrics)
  └── load-test-servicemonitor.yaml# Legacy ServiceMonitor (metrics)
```

## Research Methodology

### Test Protocol
- **360 Total Tests**: Comprehensive evaluation across all scenarios
  - **30 HPA Baseline Tests**: Traditional reactive autoscaling
  - **330 Predictive Tests**: 11 models × 3 scenarios × 10 iterations
- **Test Duration**: 30 minutes each (total ~180 hours of testing)
- **Load Scenarios**: 
  - LOW: 50-150 RPS (baseline trickle traffic)
  - MEDIUM: 150-350 RPS (moderate Poisson distribution)  
  - HIGH: 350-600 RPS (intense Poisson with burst patterns)

### Metrics Collected
- **Cost Efficiency**: Pod count, pod-minutes, scaling events
- **Quality of Service**: Success rate, response time percentiles
- **Prediction Accuracy**: MSE, MAE, RMSE for each model
- **Model Performance**: Training time, prediction latency

### Infrastructure Requirements
- **AWS K3s Cluster**: 4 nodes (1 master + 3 workers)
- **Node Types**: t3.medium (master), t3.small (workers)  
- **Storage**: Persistent volumes for results collection
- **Monitoring**: Prometheus + Grafana for real-time metrics

## Prerequisites

- A running Kubernetes cluster (Tested with K3s on AWS)
- `kubectl` configured to communicate with your cluster
- `helm` v3 installed
- Docker installed and running (if building images)
- A StorageClass available in your cluster (e.g., `local-path` for K3s)
- Internet access from cluster nodes for Helm charts and image pulls

### Install Helm

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Environment Setup (K3s example)

```bash
# Example for K3s - adjust path if needed
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```

**Important:** Replace `ip-44-221-13-199` in the commands below with your actual node hostname. You can find it using:

```bash
kubectl get nodes -o wide
```

## Deployment Steps

### 1. Build and Push Docker Images 

**Product-App:**
```bash
cd Product-App
docker build -t your-dockerhub-username/product-app:latest .
docker push your-dockerhub-username/product-app:latest
cd ..
```

**Predictive Autoscaler:**
```bash
cd Autoscaler
docker build -t your-dockerhub-username/predictive-scaler:latest .
docker push your-dockerhub-username/predictive-scaler:latest
cd ..
```

**Scaling Controller (for Combined Approach):**
```bash
cd Product-App/controller
docker build -t your-dockerhub-username/scaling-controller:latest .
docker push your-dockerhub-username/scaling-controller:latest
cd ../..
```

> **Note:** Update the image names in all YAML deployment files to match your pushed images:
> - `Autoscaler/predictive-scaler-deployment.yaml`
> - `Product-App/HPA/product-app-hpa.yaml`
> - `Product-App/Combination/product-app-combined-deployment.yaml`
> - `Product-App/Combination/scaling-controller-combined.yaml`

### 2. Deploy Monitoring (Prometheus & Grafana)

**Add Helm repositories:**
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

**Create monitoring namespace:**
```bash
kubectl create namespace monitoring
```

**Create Grafana PVC:**
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-pvc
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
  storageClassName: local-path
EOF
```

**Install kube-prometheus-stack (includes both Prometheus and Grafana):**
```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.nodeSelector."kubernetes\.io/hostname"="YOUR-NODE-HOSTNAME" \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=5Gi \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=local-path \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.accessModes[0]=ReadWriteOnce \
  --set grafana.persistence.enabled=true \
  --set grafana.persistence.existingClaim=grafana-pvc \
  --set grafana.nodeSelector."kubernetes\.io/hostname"="YOUR-NODE-HOSTNAME" \
  --set kubeStateMetrics.enabled=true \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```
> Important: All ServiceMonitor YAMLs in this repo include a `metadata.labels.release` label. Ensure its value matches the Helm release name you use above (default here is `prometheus`). If you install with a different release name (e.g., `monitoring`), update the ServiceMonitor labels or set a matching selector in your Helm values.

**Verify installation:**
```bash
kubectl get pods -n monitoring
kubectl get svc -n monitoring
```

**Get Grafana admin password:**
```bash
kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
```

### 3. Expose Grafana and Prometheus (Optional)

**Expose Grafana:**
```bash
kubectl apply -f Monitoring/grafana/grafana-service.yaml
```

**Expose Prometheus:**
```bash
kubectl apply -f Monitoring/prometheus/prometheus-service.yaml
```

**Alternative - Port forwarding (might not work in AWS):**
```bash
# Grafana
kubectl port-forward --namespace monitoring svc/prometheus-grafana 3000:80

# Prometheus
kubectl port-forward --namespace monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

### 4. Import Grafana Dashboard

1. Access Grafana UI (http://localhost:3000 if using port-forward)
2. Login with username `admin` and the password from step 2
3. Go to **Dashboards** → **Import** → **Upload JSON file**
4. Upload `Monitoring/Monitoring.json`

### 5. Deploy Predictive Scaler

```bash
# Deploy RBAC and ServiceAccount
kubectl apply -f Autoscaler/predictive-scaler-serviceaccount.yaml
kubectl apply -f Autoscaler/predictive-scaler-rbac.yaml

# Deploy Service and Deployment
kubectl apply -f Autoscaler/predictive-scaler-service.yaml
kubectl apply -f Autoscaler/predictive-scaler-deployment.yaml

# Deploy ServiceMonitor (ensure its metadata.labels.release matches your Helm release)
kubectl apply -f Autoscaler/predictive-scaler-servicemonitor.yaml -n monitoring
```

### 6. Deploy Product Application

Choose **ONE** of the following options:

#### Option A: HPA Setup

```bash
# Deploy Product App with HPA
kubectl apply -f Product-App/HPA/product-app-hpa-service.yaml
kubectl apply -f Product-App/HPA/product-app-hpa.yaml
kubectl apply -f Product-App/HPA/product-app-hpa-hpa.yaml

# Deploy ServiceMonitor (ensure it has 'release: prometheus' label)
kubectl apply -f Product-App/HPA/product-app-servicemonitor.yaml -n monitoring

# Optional: HPA Controller
# kubectl apply -f Product-App/Controller/controller-rbac.yaml
# kubectl apply -f Product-App/HPA/scaling-controller-hpa.yaml
```

#### Option B: Combined Approach

```bash
# Deploy RBAC for Scaling Controller
kubectl apply -f Product-App/Controller/controller-rbac.yaml

# Deploy Product App with Combined setup
kubectl apply -f Product-App/Combination/product-app-combined-service.yaml
kubectl apply -f Product-App/Combination/product-app-combined-deployment.yaml

# Deploy ServiceMonitor (ensure it has 'release: prometheus' label)
kubectl apply -f Product-App/Combination/product-app-combined-servicemonitor.yaml -n monitoring

# Deploy Combined Scaling Controller
kubectl apply -f Product-App/Combination/scaling-controller-combined.yaml
```

### 7. Restore Baseline Training Data (For Testing with Pre-trained Baselines)

If you have backed up baseline training data from a previous cluster, you can restore it to skip the 12+ hour data collection process.

**Prerequisites:**
- Baseline backup file (e.g., `baseline_backup_YYYYMMDD_HHMMSS.tar.gz`)
- Predictive scaler pod must be running with persistent volume mounted

**Restore Steps:**

1. **Upload baseline backup to cluster:**
```bash
scp -i "path/to/your-key.pem" baseline_backups/baseline_backup_*.tar.gz ubuntu@your-ec2-host:~/
```

2. **Extract and copy baselines to predictive-scaler pod:**
```bash
# SSH into your cluster
ssh -i "path/to/your-key.pem" ubuntu@your-ec2-host

# Get the predictive-scaler pod name
PRED_POD=$(sudo kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')

# Extract the backup
tar -xzf baseline_backup_*.tar.gz

# Copy each baseline file to the pod
sudo kubectl cp baseline_low.json $PRED_POD:/data/baselines/
sudo kubectl cp baseline_medium.json $PRED_POD:/data/baselines/
sudo kubectl cp baseline_high.json $PRED_POD:/data/baselines/

# Verify files are in place
sudo kubectl exec $PRED_POD -- ls -lh /data/baselines/
```

3. **Copy required Python modules to predictive-scaler (if needed):**
```bash
# These modules are required for the baseline loading API
sudo kubectl cp Autoscaler/baseline_datasets.py $PRED_POD:/app/
sudo kubectl cp Autoscaler/model_variants.py $PRED_POD:/app/
```

4. **Verify baseline can be loaded via API:**
```bash
# Get predictive-scaler service port (usually 5000)
sudo kubectl port-forward svc/predictive-scaler 5000:5000 &

# Test loading a baseline
curl -X POST http://localhost:5000/api/baseline/load \
  -H "Content-Type: application/json" \
  -d '{"scenario":"low"}'

# Expected response: {"success": true, "data_points_loaded": 240, ...}
```

**PowerShell Script for Windows (Automated Restore):**

Create `restore_baseline_local.ps1`:
```powershell
# Configuration
$KeyPath = "C:\path\to\your-key.pem"
$Host = "your-ec2-host.compute-1.amazonaws.com"
$BackupFile = "baseline_backups\baseline_backup_*.tar.gz"

# Upload backup
Write-Host "Uploading baseline backup to cluster..."
scp -i $KeyPath $BackupFile ubuntu@${Host}:~/

# Restore baselines
Write-Host "Restoring baselines to predictive-scaler pod..."
ssh -i $KeyPath ubuntu@$Host @"
  PRED_POD=\`$(sudo kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')\`
  tar -xzf baseline_backup_*.tar.gz
  sudo kubectl cp baseline_low.json \`$PRED_POD:/data/baselines/
  sudo kubectl cp baseline_medium.json \`$PRED_POD:/data/baselines/
  sudo kubectl cp baseline_high.json \`$PRED_POD:/data/baselines/
  sudo kubectl exec \`$PRED_POD -- ls -lh /data/baselines/
"@

Write-Host "Baseline restore complete!"
```

### 8. Load Testing Options

You can run load tests in two ways: Auto-deployed (recommended) or Legacy one-off pod.

#### Option A: Auto-deployed Load Tester (recommended)

1) Build and push the load tester image (replace `your-dockerhub-username`):

```bash
docker build -f stress-test/Dockerfile -t your-dockerhub-username/load-tester:latest .
docker push your-dockerhub-username/load-tester:latest
```

2) Update image in `stress-test/load-tester-deployment.yaml` to your repo and apply:

```bash
kubectl apply -f stress-test/load-tester-deployment.yaml
```

3) Verify rollout and logs:

```bash
kubectl get deploy load-tester
kubectl get pods -l app=load-tester
kubectl logs deploy/load-tester --tail 50 -f
```

4) Optional: expose client metrics locally

```bash
kubectl port-forward svc/load-tester 9105:9105
# Browse http://localhost:9105/metrics
```

5) Optional: always fetch the latest script (no rebuild needed)

```bash
kubectl set env deploy/load-tester LOAD_TEST_SOURCE_URL=https://raw.githubusercontent.com/<owner>/<repo>/main/stress-test/load_test.py
kubectl rollout restart deploy/load-tester
```

> The Deployment uses `imagePullPolicy: Always` so new images are pulled at startup. If you also set `LOAD_TEST_SOURCE_URL`, the container fetches the latest `load_test.py` at boot.

#### Option B: Legacy one-off Pod (manual execution)

```bash
kubectl apply -f stress-test/load-test-pod.yaml
kubectl wait --for=condition=Ready pod/load-test --timeout=60s
kubectl cp stress-test/load_test.py load-test:/tmp/load_test.py
kubectl exec -it load-test -- bash -c "pip install requests numpy pandas prometheus-client && cd /tmp && python load_test.py --duration 600 --target combined --metrics-port 9105"
kubectl logs load-test --tail=50 -f
```

> If you previously used the legacy pod and are switching to the Deployment, remove the old resources to avoid duplicate load:
```bash
kubectl delete pod load-test --ignore-not-found
kubectl delete -f stress-test/load-test-service.yaml --ignore-not-found
kubectl delete -f stress-test/load-test-servicemonitor.yaml --ignore-not-found
```

### 9. Running Automated Test Iterations

After deploying the load-tester and restoring baseline data, you can run automated test iterations for data collection.

**Prerequisites:**
- Load-tester pod deployed and running
- Baseline training data restored to predictive-scaler pod
- Python modules (baseline_datasets.py, model_variants.py) copied to predictive-scaler pod

**Running Test Iterations:**

1. **Create test orchestration script** (save as `run_tests.sh`):
```bash
#!/bin/bash
# Example: Run 10 iterations of high scenario tests

SCENARIO="high"
START_ITER=1
END_ITER=10
OUTPUT_BASE="/results/${SCENARIO}_$(date +%Y%m%d_%H%M%S)"

echo "Running $SCENARIO scenario tests"
echo "Iterations: $START_ITER to $END_ITER"
echo "Output: $OUTPUT_BASE"

for i in $(seq $START_ITER $END_ITER); do
    echo "=== Starting iteration $i ==="
    
    # Load baseline dataset
    curl -X POST http://predictive-scaler:5000/api/baseline/load \
         -H "Content-Type: application/json" \
         -d "{\"scenario\":\"$SCENARIO\"}"
    
    # Train models (~3-5 minutes)
    curl -X POST http://predictive-scaler:5000/api/models/train_all
    
    # Submit test to Control API (30 minute test)
    ITER_DIR="${OUTPUT_BASE}/iteration_$(printf '%02d' $i)"
    curl -X POST http://localhost:8080/start \
         -H "Content-Type: application/json" \
         -d "{\"scenarios\":[\"$SCENARIO\"],\"duration\":1800,\"iterations\":1,\"output_dir\":\"$ITER_DIR\"}"
    
    # Wait for test completion
    while true; do
        STATUS=$(curl -s http://localhost:8080/status | grep -o '"running":[^,}]*')
        if [[ "$STATUS" == *"false"* ]]; then
            echo "Test completed!"
            break
        fi
        echo "Test running..."
        sleep 30
    done
    
    echo "=== Iteration $i completed ==="
    sleep 10
done

echo "ALL TESTS COMPLETED!"
```

2. **Copy script to load-tester pod and run as daemon:**
```bash
# Copy script to pod
kubectl cp run_tests.sh load-tester-POD-NAME:/app/

# Make executable
kubectl exec load-tester-POD-NAME -- chmod +x /app/run_tests.sh

# Create daemon launcher (start_tests.py)
cat > start_tests.py << 'EOF'
import subprocess
import sys
process = subprocess.Popen(
    ['/bin/bash', '/app/run_tests.sh'],
    stdout=open('/app/test_run.log', 'w'),
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print(f"Started tests daemon with PID: {process.pid}")
print(f"Log file: /app/test_run.log")
sys.exit(0)
EOF

# Copy daemon launcher
kubectl cp start_tests.py load-tester-POD-NAME:/app/

# Start tests as background daemon
kubectl exec load-tester-POD-NAME -- python3 /app/start_tests.py
```

3. **Monitor test progress:**
```bash
# Check daemon log
kubectl exec load-tester-POD-NAME -- tail -f /app/test_run.log

# Check Control API status
kubectl exec load-tester-POD-NAME -- curl -s http://localhost:8080/status
```

4. **Download results after completion:**
```bash
# Create tar archive of all results
kubectl exec load-tester-POD-NAME -- bash -c "cd /results && tar -czf /tmp/test_results.tar.gz high_*/"

# Download to local machine
kubectl cp load-tester-POD-NAME:/tmp/test_results.tar.gz ./test_results.tar.gz

# Extract locally
tar -xzf test_results.tar.gz
```

**PowerShell Automation (Windows):**

```powershell
# Configuration
$KeyPath = "C:\path\to\your-key.pem"
$Host = "your-ec2-host.compute-1.amazonaws.com"

# Get load-tester pod name
$PodName = ssh -i $KeyPath ubuntu@$Host "sudo kubectl get pods -l app=load-tester -o jsonpath='{.items[0].metadata.name}'"

# Monitor tests
Write-Host "Monitoring test progress (Ctrl+C to stop)..."
while ($true) {
    ssh -i $KeyPath ubuntu@$Host "sudo kubectl exec $PodName -- tail -20 /app/test_run.log"
    Start-Sleep -Seconds 30
}
```

**Test Duration Estimates:**
- Each iteration: ~35 minutes (5 min model training + 30 min load test + overhead)
- 10 iterations: ~6 hours
- 3 scenarios (low/medium/high) × 10 iterations: ~18 hours total

## 10. Verification and Monitoring

### Check Deployment Status

```bash
# Check all pods
kubectl get pods --all-namespaces

# Check services
kubectl get svc --all-namespaces

# Check HPA status (if using HPA)
kubectl get hpa

# Check ServiceMonitors
kubectl get servicemonitors -n monitoring
```

### Access Monitoring

- **Grafana:** http://your-node-ip (if using LoadBalancer service) or port-forward to localhost:3000
- **Prometheus:** http://your-node-ip:9090 (if using LoadBalancer service) or port-forward to localhost:9090

### Troubleshooting

**Check pod logs:**
```bash
kubectl logs -f deployment/predictive-scaler
kubectl logs -f deployment/product-app-hpa  # or product-app-combined
```

**Check ServiceMonitor discovery:**
```bash
kubectl get servicemonitors -n monitoring -o yaml
```

**Verify Prometheus targets:**
- Access Prometheus UI → Status → Targets
- Ensure your services are listed and UP

**Load Test Troubleshooting:**
```bash
# Check if load test pod is running properly
kubectl get pod load-test
kubectl describe pod load-test

# If using the legacy pod and it is just sleeping, run manually:
kubectl exec -it load-test -- bash -c "
  pip install requests numpy pandas && 
  cd /tmp && 
  python load_test.py --duration 300 --target combined
"

# Monitor CPU usage during load test:
kubectl top pods | grep product-app-combined

# Check autoscaler data collection:
curl http://localhost:5000/status | jq '{current_cpu, data_points, data_points_collected}'
```

## About `entrypoint.sh`

`stress-test/entrypoint.sh` is the container startup script used by the auto-deployed load tester image. It:

- Optionally downloads the latest `load_test.py` from `LOAD_TEST_SOURCE_URL` (if set)
- Composes the Python command using environment variables (targets, peak ranges, duration, concurrency, metrics port)
- Starts the load test and exposes client metrics

Keep `entrypoint.sh` if you are using the load tester container/Deployment. If you only run the script manually with the legacy pod or on your local machine, you can ignore it (but the Dockerfile expects it, so don’t delete it if you plan to build the image).

## Important Notes

- Replace `YOUR-NODE-HOSTNAME` with your actual node hostname
- Ensure all ServiceMonitor YAML files have `release: prometheus` label in metadata
- Only deploy one Product-App configuration (HPA OR Combined) at a time
- Storage class `local-path` is specific to K3s; adjust for other clusters
- All application components deploy to `default` namespace, monitoring to `monitoring` namespace

## Clean Up

To remove all components:

```bash
# Remove Helm releases
helm uninstall prometheus -n monitoring

# Remove namespaces (this removes all resources in them)
kubectl delete namespace monitoring

# Remove default namespace resources
kubectl delete -f Product-App/HPA/ --ignore-not-found=true
kubectl delete -f Product-App/Combination/ --ignore-not-found=true
kubectl delete -f Autoscaler/ --ignore-not-found=true
kubectl delete -f stress-test/ --ignore-not-found=true
```
