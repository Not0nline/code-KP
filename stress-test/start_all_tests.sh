#!/bin/bash
# Complete test restart: clean databases and start MEDIUM + HIGH tests
# Results saved to /results/multi_run_results/ (persistent storage)

set -e

echo "========================================"
echo "Complete Test Restart"
echo "========================================"
echo ""

# Step 1: Clean databases
echo "Step 1/2: Cleaning product databases..."
HPA_POD=$(kubectl get pod -l app=product-app-hpa -o jsonpath='{.items[0].metadata.name}')
COMBINED_POD=$(kubectl get pod -l app=product-app-combined -o jsonpath='{.items[0].metadata.name}')

if [ -n "$HPA_POD" ]; then
    echo "  - Cleaning HPA database..."
    kubectl exec $HPA_POD -- curl -s -X POST http://localhost:5000/reset_data
fi

if [ -n "$COMBINED_POD" ]; then
    echo "  - Cleaning Combined database..."
    kubectl exec $COMBINED_POD -- curl -s -X POST http://localhost:5000/reset_data
fi

echo "  ✅ Databases cleaned"
echo ""

# Step 2: Start tests
echo "Step 2/2: Starting MEDIUM + HIGH tests..."
echo "  - MEDIUM: 10 iterations × 30 min"
echo "  - HIGH:   10 iterations × 30 min"
echo "  - Total:  ~10 hours"
echo "  - Auto cleanup after each iteration"
echo ""

# Get load-tester pod name
POD_NAME=$(kubectl get pod -l app=load-tester -o name | sed 's/pod\///')
echo "Using pod: $POD_NAME"
echo ""
kubectl exec $POD_NAME -- curl -X POST http://localhost:8080/start \
  -H 'Content-Type: application/json' \
  -d '{
    "scenarios": ["medium", "high"],
    "iterations": 10,
    "duration": 1800
  }'

echo ""
echo ""
echo "========================================"
echo "✅ Tests Started Successfully!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  kubectl logs -f deployment/load-tester"
echo ""
echo "Check status:"
echo "  kubectl exec $POD_NAME -- curl http://localhost:8080/status"
echo ""
echo "Results location (in cluster):"
echo "  /results/multi_run_results/"
echo ""
echo "Download results later:"
echo "  Run download_all_results.ps1 from your Windows machine"
echo ""
echo "You can now close this terminal!"
echo "========================================"
