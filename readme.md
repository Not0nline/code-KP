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
└── stress-test/        # Load testing scripts & K8s manifests
    ├── load_test.py
    └── load-test-pod.yaml
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
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.serviceMonitorSelector.matchLabels.release=prometheus
```

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

# Deploy ServiceMonitor (ensure it has 'release: prometheus' label)
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

### 7. Deploy and Run Load Tests

**Deploy load test pod:**
```bash
kubectl apply -f stress-test/load-test-pod.yaml
```

**Wait for pod to be ready:**
```bash
kubectl wait --for=condition=Ready pod/load-test --timeout=60s
```

**Copy test script and install dependencies:**
```bash
kubectl cp stress-test/load_test.py load-test:/load_test.py
kubectl exec -it load-test -- pip install requests numpy pandas
```

**Run load test:**
```bash
# Basic test
kubectl exec -it load-test -- python /load_test.py

# Custom parameters (adjust based on your script)
# kubectl exec -it load-test -- python /load_test.py --duration 1800 --target-service product-app-combined-service.default.svc.cluster.local
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
