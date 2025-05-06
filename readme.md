
# Kubernetes Autoscaling Setup - Reproduction Steps

This README provides the steps to deploy the Kubernetes components for the autoscaling comparison project (Predictive, HPA, Combined), using Helm for Prometheus and Grafana installation.

## Repository Structure

.
├── autoscaler/               # Predictive Scaler application & K8s manifests
├── monitoring/                # Monitoring setup (Grafana Dashboard JSON, example Service YAMLs) & K8s manifests
│   ├── Monitoring.json       # Grafana Dashboard Import
│   ├── allow-monitoring.yaml # Example NetworkPolicy
│   ├── grafana/
│   │   └── grafana-service.yaml #(Example, likely managed by Helm)
│   └── prometheus/
│       └── prometheus-service.yaml #(Example, likely managed by Helm)
├── product-app/               # Product-App application & K8s manifests (including HPA and Combined setups)
│   ├── app.py
│   ├── Dockerfile
│   ├── combination/
│   ├── controller/
│   └── hpa/
└── stress-test/              # Load testing scripts & K8s manifests
├── load_test.py
└── load-test-pod.yaml

## Prerequisites

*   A running Kubernetes cluster (Tested with K3s).
*   `kubectl` configured to communicate with your cluster.
*   `helm` v3 installed.
*   Docker installed and running (if you need to build the container images).
*   A StorageClass available in your cluster (e.g., `local-path` was used in the script). Adjust storage settings if using a different StorageClass.
*   Internet access from your cluster nodes for Helm chart downloads and image pulls.
*   **Environment Specific:** You will need to replace `"ip-172-31-27-69"` in the Helm commands with a valid node hostname selector for your environment if you want to pin Prometheus/Grafana to specific nodes. You might also need to adjust StorageClass names and sizes. Set `KUBECONFIG` if necessary (`export KUBECONFIG=/path/to/your/kubeconfig`).

## Deployment Steps

These steps assume you are running commands from the root of the repository. Most application components will be deployed to the `default` namespace, while monitoring components go into the `monitoring` namespace.

**1. Configure Environment (If using K3s example)**

```bash
# Example for K3s - adjust path if needed
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```

2. Build and Push Docker Images

* Product-App:
```bash
cd product-app
docker build -t your-dockerhub-username/product-app:latest .
# docker login your-registry.com # Optional: login if needed
docker push your-dockerhub-username/product-app:latest
cd ..
```

* Predictive Autoscaler:
```bash
cd autoscaler
docker build -t your-dockerhub-username/predictive-scaler:latest .
# docker login your-registry.com # Optional: login if needed
docker push your-dockerhub-username/predictive-scaler:latest
cd ..
```

* Scaling Controller (for Combined Approach):
```bash
cd product-app/controller
# Ensure the Dockerfile here is named 'Dockerfile'
docker build -t your-dockerhub-username/scaling-controller:latest .
# docker login your-registry.com # Optional: login if needed
docker push your-dockerhub-username/scaling-controller:latest
cd ../..
```

Replace your-dockerhub-username with your actual Docker Hub username or your container registry path. Crucially, update the image names in the following YAML files to match the images you just pushed:
*   autoscaler/predictive-scaler-deployment.yaml
*   product-app/hpa/product-app-hpa.yaml (or similar Deployment file in HPA)
*   product-app/combination/product-app-combined-deployment.yaml
*   product-app/combination/scaling-controller-combined.yaml (if controller image is specified there)
*   product-app/hpa/scaling-controller-hpa.yaml (if controller image is specified there)

3. Deploy Monitoring (Prometheus & Grafana via Helm)

* Add Helm Repositories:
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

* Create Monitoring Namespace:
```bash
kubectl create namespace monitoring
```

* Create Grafana Persistent Volume Claim:
```bash
# Adjust storage size and storageClassName if needed
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi # Adjust size if needed
  storageClassName: local-path # IMPORTANT: Use a StorageClass available in your cluster
EOF
```

* Install Prometheus (kube-prometheus-stack):
```bash
# *** Adjust nodeSelector, storage requests/className as needed for your environment ***
# The serviceMonitorSelector config ensures Prometheus finds ServiceMonitors with the 'release=prometheus' label
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.nodeSelector."kubernetes\.io/hostname"="ip-172-31-27-69" \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=5Gi \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=local-path \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.accessModes[0]=ReadWriteOnce \
  --set kubeStateMetrics.enabled=true \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.serviceMonitorSelector.matchLabels.release=prometheus
```

* Install Grafana:
```bash
# *** Adjust nodeSelector as needed for your environment ***
helm upgrade --install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=true \
  --set persistence.existingClaim=grafana \
  --set nodeSelector."kubernetes\.io/hostname"="ip-172-31-27-69"
```

* Apply Network Policy (Optional - if needed for Prometheus scraping):
    
    - Review monitoring/allow-monitoring.yaml and apply if necessary for your cluster's network setup. This policy might allow pods in the monitoring namespace egress access.

```bash
# kubectl apply -f monitoring/allow-monitoring.yaml -n default # Apply to default namespace if needed
```

* Import Grafana Dashboard:

- Get the Grafana admin password:

```bash
kubectl get secret --namespace monitoring grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
# If using kube-prometheus-stack's Grafana, it might be:
# kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
```


- Access the Grafana UI (you might need to port-forward or use an Ingress):

```bash
# Example using port-forward:
kubectl port-forward --namespace monitoring svc/grafana 3000:80
# Or if using kube-prometheus-stack's Grafana:
# kubectl port-forward --namespace monitoring svc/prometheus-grafana 3000:80
```

Access Grafana at http://localhost:3000.
- Log in with username admin and the retrieved password.

- Go to Dashboards -> Import -> Upload JSON file and upload monitoring/Monitoring.json.



4. Deploy the Predictive Scaler

```bash
# Ensure ServiceAccount/RBAC are applied first
kubectl apply -f autoscaler/predictive-scaler-serviceaccount.yaml -n default
kubectl apply -f autoscaler/predictive-scaler-rbac.yaml -n default
kubectl apply -f autoscaler/predictive-scaler-service.yaml -n default
kubectl apply -f autoscaler/predictive-scaler-deployment.yaml -n default

# Apply the ServiceMonitor - Ensure it has 'release: prometheus' label inside the YAML
# Apply it to the monitoring namespace so Prometheus Operator finds it
kubectl apply -f autoscaler/predictive-scaler-servicemonitor.yaml -n monitoring
```

Note: Ensure the predictive-scaler-servicemonitor.yaml file contains the label release: prometheus under metadata.labels for the Helm-installed Prometheus Operator to discover it.

5. Deploy the Product Application and Autoscaling Configurations

Choose one of the following configurations (HPA or Combined) to deploy at a time.

* Option A: Deploy Product-App with HPA

```bash
# Deploy Product App specific resources for HPA
kubectl apply -f product-app/hpa/product-app-hpa-service.yaml -n default
kubectl apply -f product-app/hpa/product-app-hpa.yaml -n default # Deployment
kubectl apply -f product-app/hpa/product-app-hpa-hpa.yaml -n default # HPA object
# Apply the ServiceMonitor - Ensure it has 'release: prometheus' label inside the YAML
# Apply it to the monitoring namespace
kubectl apply -f product-app/hpa/product-app-hpa-servicemonitor.yaml -n monitoring

# Optional: Deploy the HPA-specific controller RBAC/Deployment if needed
# kubectl apply -f product-app/controller/controller-rbac.yaml -n default # Add if RBAC needed for HPA controller
# kubectl apply -f product-app/hpa/scaling-controller-hpa.yaml -n default
```

* Option B: Deploy Product-App with Combined Approach

```bash
# Apply RBAC for the Scaling Controller
kubectl apply -f product-app/controller/controller-rbac.yaml -n default

# Create ConfigMap for the controller code (if using ConfigMap-based deployment)
# kubectl apply -f product-app/controller/scaling-controller-code-configmap.yaml -n default

# Deploy Product App specific resources for Combined
kubectl apply -f product-app/combination/product-app-combined-service.yaml -n default
kubectl apply -f product-app/combination/product-app-combined-deployment.yaml -n default
# Apply the ServiceMonitor - Ensure it has 'release: prometheus' label inside the YAML
# Apply it to the monitoring namespace
kubectl apply -f product-app/combination/product-app-combined-servicemonitor.yaml -n monitoring

# Deploy the Combined scaling controller Deployment
kubectl apply -f product-app/combination/scaling-controller-combined.yaml -n default
```


Note: Ensure the product-app-*-servicemonitor.yaml files contain the label release: prometheus under metadata.labels for the Helm-installed Prometheus Operator to discover them.

6. Deploy and Run Stress Test

* Deploy the load test pod:

```bash
kubectl apply -f stress-test/load-test-pod.yaml -n default
```

Wait for the pod to be in the Running state: kubectl get pod load-test -n default


* Copy the script and install dependencies:

```bash
kubectl cp stress-test/load_test.py load-test:/load_test.py -n default
kubectl exec -it load-test -n default -- pip install requests numpy pandas
```

* Execute the load test script:

```bash
# Example: Run with default settings (adjust script defaults if needed)
kubectl exec -it load-test -n default -- python /load_test.py

# Example: Run with specific parameters (check script for available args)
# kubectl exec -it load-test -n default -- python /load_test.py --duration 1800 --target-service product-app-combined-service.default.svc.cluster.local

```