#!/bin/bash
# Start MEDIUM + HIGH tests
# Database cleanup happens AUTOMATICALLY after each iteration
# Results saved to /results/multi_run_results/ (persistent storage)

set -e

echo "========================================"
echo "Starting MEDIUM + HIGH Tests"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - MEDIUM: 10 iterations × 30 min"
echo "  - HIGH:   10 iterations × 30 min"
echo "  - Total:  ~10 hours"
echo "  - Auto cleanup: ✅ (after each iteration)"
echo ""

# Get load-tester pod name
POD_NAME=$(kubectl get pod -l app=load-tester -o name | sed 's/pod\///')
echo "Using pod: $POD_NAME"
echo ""

echo "Starting tests..."
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
