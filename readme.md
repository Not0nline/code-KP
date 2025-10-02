# Kubernetes Autoscaling Setup - Reproduction Steps

This README provides the steps to deploy the Kubernetes components for the autoscaling comparison project (Predictive, HPA, Combined), using Helm for Prometheus and Grafana installation.

## Repository Structure

```text
.
├── Autoscaler/         # Predictive Scaler application & K8s manifests
├── Monitoring/         # Monitoring setup (Grafana Dashboard JSON, example Service YAMLs) & K8s manifests
│   ├── Monitoring.json     # Grafana Dashboard Import
│   ├── allow-monitoring.yaml # Example NetworkPolicy
│   ├── grafana/
│   │   └── grafana-service.yaml # For exposing port externally for grafana
│   └── prometheus/
│       └── prometheus-service.yaml # For exposing port externally for prometheus
├── Product-App/        # Product-App application & K8s manifests (including HPA and Combined setups)
│   ├── app.py
│   ├── Dockerfile
│   ├── combination/
│   ├── controller/
│   └── hpa/
└── stress-test/        # Load testing scripts, image, and K8s manifests
  ├── load_test.py                 # Load tester script (supports client metrics)
  ├── Dockerfile                   # Container image for auto-deployed load tester
  ├── requirements.txt             # Python deps for image
  ├── entrypoint.sh                # Container startup (fetch latest script, run tester)
  ├── load-tester-deployment.yaml  # Deployment + Service + ServiceMonitor (auto)
  ├── load-test-pod.yaml           # Legacy one-off pod (manual execution)
  ├── load-test-service.yaml       # Legacy Service (metrics)
  └── load-test-servicemonitor.yaml# Legacy ServiceMonitor (metrics)
```

## Prerequisites

- A running Kubernetes cluster (Tested with K3s)
- `kubectl` configured to communicate with your cluster
- `helm` v3 installed
- Docker installed and running (if you need to build the container images)
- A StorageClass available in your cluster (e.g., `local-path` for K3s)
- Internet access from your cluster nodes for Helm chart downloads and image pulls

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

### 7. Load Testing Options

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

## Verification and Monitoring

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
