#!/bin/bash
# Install Helm
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh

# Add Repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus Stack
# Release name: prometheus (to match the DNS name expected by scaler)
# Namespace: monitoring
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
helm install prometheus prometheus-community/kube-prometheus-stack --namespace monitoring --set prometheus.service.type=ClusterIP

# Wait for it
kubectl rollout status statefulset prometheus-prometheus-kube-prometheus-prometheus -n monitoring --timeout=300s
