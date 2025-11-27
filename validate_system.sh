#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ SYSTEM VALIDATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "1️⃣  Baseline Data Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for SCENARIO in low medium high; do
  POINTS=$(sudo kubectl exec deploy/predictive-scaler -- cat /data/baselines/baseline_$SCENARIO.json 2>/dev/null | grep '"data_points"' | awk -F': ' '{print $2}' | tr -d ',')
  SIZE=$(sudo kubectl exec deploy/predictive-scaler -- du -h /data/baselines/baseline_$SCENARIO.json 2>/dev/null | cut -f1)
  echo "   $SCENARIO: $POINTS points ($SIZE)"
done
echo ""

echo "2️⃣  PVC Storage Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
PVC_STATUS=$(sudo kubectl get pvc load-test-results-pvc -o jsonpath='{.status.phase}')
PVC_SIZE=$(sudo kubectl get pvc load-test-results-pvc -o jsonpath='{.spec.resources.requests.storage}')
echo "   Status: $PVC_STATUS"
echo "   Size: $PVC_SIZE"
echo ""

echo "3️⃣  Test Results in PVC:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# Create browser pod if not exists
sudo kubectl get pod pvc-browser >/dev/null 2>&1 || bash browse_pvc.sh >/dev/null 2>&1

TEST_DIRS=$(sudo kubectl exec pvc-browser -- find /results -type d -name 'test_run_*' 2>/dev/null | wc -l)
CSV_FILES=$(sudo kubectl exec pvc-browser -- find /results -name 'combined_results.csv' 2>/dev/null | wc -l)
TOTAL_SIZE=$(sudo kubectl exec pvc-browser -- du -sh /results 2>/dev/null | cut -f1)

echo "   Test runs: $TEST_DIRS"
echo "   Result files: $CSV_FILES"
echo "   Total size: $TOTAL_SIZE"

if [ $CSV_FILES -gt 0 ]; then
  echo ""
  echo "   Recent tests:"
  sudo kubectl exec pvc-browser -- find /results -name 'combined_results.csv' -exec ls -lh {} \; 2>/dev/null | while read line; do
    echo "     $line"
  done
fi
echo ""

echo "4️⃣  Predictive Scaler Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
POD_STATUS=$(sudo kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].status.phase}')
POD_NAME=$(sudo kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
RESTARTS=$(sudo kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}')
echo "   Pod: $POD_NAME"
echo "   Status: $POD_STATUS"
echo "   Restarts: $RESTARTS"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 READINESS CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

READY=true

# Check baselines
if sudo kubectl exec deploy/predictive-scaler -- cat /data/baselines/baseline_high.json 2>/dev/null | grep -q '"data_points": 2000'; then
  echo "✅ Baselines: 2000 points configured"
else
  echo "❌ Baselines: NOT 2000 points"
  READY=false
fi

# Check PVC
if [ "$PVC_STATUS" = "Bound" ]; then
  echo "✅ PVC: Bound and ready"
else
  echo "❌ PVC: Not bound ($PVC_STATUS)"
  READY=false
fi

# Check test results
if [ $CSV_FILES -gt 0 ]; then
  echo "✅ Test results: $CSV_FILES validation tests completed"
else
  echo "⚠️  Test results: No validation tests yet"
fi

# Check predictive-scaler
if [ "$POD_STATUS" = "Running" ]; then
  echo "✅ Predictive-scaler: Running"
else
  echo "❌ Predictive-scaler: Not running ($POD_STATUS)"
  READY=false
fi

echo ""
if [ "$READY" = true ]; then
  echo "🎉 SYSTEM READY FOR FULL 10x30-MINUTE TEST SUITE!"
  echo ""
  echo "📋 To start the full tests:"
  echo "   bash run_iterative_high_tests.sh"
  echo ""
  echo "⏱  Expected duration: ~5.5 hours"
  echo "💾 Results will be saved to PVC (persistent)"
  echo ""
else
  echo "⚠️  SYSTEM NOT READY - Fix issues above first"
  echo ""
fi
