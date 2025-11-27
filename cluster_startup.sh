#!/bin/bash

echo "🚀 CLUSTER STARTUP AND VALIDATION PROCEDURE"
echo "==========================================="

# Step 1: Check cluster connectivity
echo "1️⃣ CHECKING CLUSTER STATUS"
echo "-------------------------"

echo "Testing cluster connection..."
kubectl get nodes 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Cluster is accessible"
else
    echo "❌ Cluster not accessible - need to start/configure"
    echo ""
    echo "MANUAL STEPS NEEDED:"
    echo "1. Start AWS EC2 instances if stopped"
    echo "2. SSH into master node and check K3s status"
    echo "3. Restart K3s if needed: sudo systemctl start k3s"
    echo "4. Update kubeconfig if IP changed"
    exit 1
fi

# Step 2: Check essential services
echo ""
echo "2️⃣ CHECKING ESSENTIAL SERVICES"
echo "------------------------------"

echo "Checking product-app deployment..."
kubectl get deployment product-app-combined -n default
if [ $? -ne 0 ]; then
    echo "❌ Product app not deployed"
else
    echo "✅ Product app found"
fi

echo ""
echo "Checking HPA status..."
kubectl get hpa -n default
if [ $? -ne 0 ]; then
    echo "⚠️ No HPA found - will deploy for tests"
else
    echo "✅ HPA active"
fi

echo ""
echo "Checking load-tester image..."
kubectl get pods -l app=load-tester 2>/dev/null | head -2
if [ $? -ne 0 ]; then
    echo "⚠️ No active load-tester pods (expected)"
else
    echo "ℹ️ Load-tester pods found"
fi

# Step 3: Deploy/Update essential components
echo ""
echo "3️⃣ DEPLOYING/UPDATING COMPONENTS"
echo "--------------------------------"

# Ensure product-app is running
echo "Deploying product-app if needed..."
kubectl apply -f Product-App/Combination/product-app-combined-deployment.yaml
kubectl apply -f Product-App/Combination/product-app-combined-service.yaml

# Ensure HPA is active
echo "Deploying HPA for baseline tests..."
cat > hpa-for-testing.yaml << 'EOF'
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: product-app-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: product-app-combined
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
EOF

kubectl apply -f hpa-for-testing.yaml

# Wait for pods to be ready
echo ""
echo "Waiting for product-app to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/product-app-combined
if [ $? -eq 0 ]; then
    echo "✅ Product app is ready"
else
    echo "❌ Product app failed to start"
fi

# Step 4: Check load-tester image
echo ""
echo "4️⃣ ENSURING LOAD-TESTER IMAGE"
echo "-----------------------------"

# Check if load-tester image exists
docker images | grep load-tester
if [ $? -ne 0 ]; then
    echo "🔨 Building load-tester image..."
    cd stress-test
    sudo docker build -t load-tester:latest .
    cd ..
else
    echo "✅ Load-tester image exists"
fi

# Step 5: Validation test
echo ""
echo "5️⃣ QUICK CONNECTIVITY TEST"
echo "-------------------------"

# Get service IP
SERVICE_IP=$(kubectl get service product-app-combined-service -o jsonpath='{.spec.clusterIP}')
echo "Product service IP: $SERVICE_IP"

# Test health endpoint
echo "Testing health endpoint..."
kubectl run test-pod --image=curlimages/curl --rm -it --restart=Never -- curl -s http://$SERVICE_IP/health
if [ $? -eq 0 ]; then
    echo "✅ Service is responsive"
else
    echo "❌ Service not responding"
fi

echo ""
echo "🎯 STARTUP COMPLETE"
echo "=================="
echo "Ready for validation tests!"