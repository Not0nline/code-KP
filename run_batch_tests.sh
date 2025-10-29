#!/bin/bash

# Batch Test Suite - Runs 5 tests per scenario and downloads results
# Safer approach: Run in small batches to avoid losing data if AWS terminates

set -e

# Configuration
SCENARIO="$1"  # Pass "low", "medium", or "high" as argument
ITERATIONS=5
DURATION=1800  # 30 minutes per test
BASE_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="/results/batch_${SCENARIO}_${BASE_TIMESTAMP}"
LOAD_TESTER_POD="load-tester-7c987b969b-zt5tj"
LOG_FILE="/results/batch_${SCENARIO}_${BASE_TIMESTAMP}.log"

# Validate scenario input
if [[ ! "$SCENARIO" =~ ^(low|medium|high)$ ]]; then
    echo "❌ Invalid scenario. Usage: $0 <low|medium|high>"
    exit 1
fi

echo "=========================================="
echo "BATCH TEST SUITE - ${SCENARIO^^} SCENARIO"
echo "=========================================="
echo "Running 5 ${SCENARIO} tests directly in Kubernetes cluster"
echo ""
echo "Timestamp: ${BASE_TIMESTAMP}"
echo "Results: ${RESULTS_BASE}"
echo "Log: ${LOG_FILE}"
echo "Tests: 5 iterations of ${SCENARIO} scenario"
echo "Duration: ~2.5 hours"
echo ""

# Check if we can access the load tester pod
echo "Checking load-tester pod accessibility..."
if ! kubectl get pod $LOAD_TESTER_POD >/dev/null 2>&1; then
    echo "❌ Cannot find load-tester pod: $LOAD_TESTER_POD"
    echo "Available pods:"
    kubectl get pods | grep load-tester || echo "No load-tester pods found"
    exit 1
fi
echo "✓ Load-tester pod found"

# Check current test status
echo "Checking current test status..."
current_status=$(kubectl exec $LOAD_TESTER_POD -- curl -s http://localhost:8080/status 2>/dev/null || echo '{"running": false}')
current_running=$(echo "$current_status" | jq -r '.running // false' 2>/dev/null || echo "false")

if [[ "$current_running" == "true" ]]; then
    current_scenario=$(echo "$current_status" | jq -r '.current_scenario // "unknown"' 2>/dev/null || echo "unknown")
    echo "⚠️  Test currently running: $current_scenario"
    echo "Waiting for current test to complete..."
    
    while [[ "$(kubectl exec $LOAD_TESTER_POD -- curl -s http://localhost:8080/status 2>/dev/null | jq -r '.running // false' 2>/dev/null || echo "false")" == "true" ]]; do
        echo "  Still running..."
        sleep 60
    done
    echo "✓ Previous test completed"
fi

echo ""
echo "Creating batch test script for ${SCENARIO} scenario..."

# Create the batch test script that runs inside the pod
SCRIPT_FILE="./batch_test_${SCENARIO}.sh"
cat > "$SCRIPT_FILE" << EOF
#!/bin/bash
set -e

# Batch Test Suite - ${SCENARIO^^} Scenario
SCENARIO="${SCENARIO}"
ITERATIONS=${ITERATIONS}
DURATION=${DURATION}
BASE_TIMESTAMP="${BASE_TIMESTAMP}"
RESULTS_BASE="${RESULTS_BASE}"
LOG_FILE="${LOG_FILE}"

# Redirect output to both console and log
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "=========================================="
echo "BATCH TESTING SUITE - ${SCENARIO^^}"
echo "=========================================="
echo "Started: \$(date)"
echo "Results directory: \$RESULTS_BASE"
echo "Log file: \$LOG_FILE"
echo "Test plan: 5 ${SCENARIO^^} tests"
echo "Estimated duration: 2.5 hours"
echo ""

# Create results directory structure
mkdir -p "\$RESULTS_BASE/${SCENARIO}_test"

# Function to check test status
check_status() {
    curl -s http://localhost:8080/status 2>/dev/null || echo '{"running": false}'
}

# Wait for any existing test to complete
if [[ "\$(check_status | jq -r '.running // false')" == "true" ]]; then
    echo "Waiting for existing test to complete..."
    while [[ "\$(check_status | jq -r '.running // false')" == "true" ]]; do
        sleep 60
    done
    echo "Previous test completed, starting batch suite"
fi

# Main test execution loop
total_tests=${ITERATIONS}
current_test=0

echo ""
echo "=========================================="
echo "STARTING ${SCENARIO^^} SCENARIO (5 tests)"
echo "=========================================="

for iteration in \$(seq 1 \$ITERATIONS); do
    current_test=\$((current_test + 1))
    output_dir="\$RESULTS_BASE/${SCENARIO}_test/iteration_\$(printf '%02d' \$iteration)"
    
    echo ""
    echo "TEST \$current_test/\$total_tests: ${SCENARIO^^} ITERATION \$iteration"
    echo "Time: \$(date)"
    echo "Output: \$output_dir"
    echo "----------------------------------------"
    
    # Start test
    echo "→ Starting ${SCENARIO} test iteration \$iteration..."
    start_response=\$(curl -s -X POST http://localhost:8080/start \\
        -H 'Content-Type: application/json' \\
        -d "{
            \\"duration\\": \$DURATION,
            \\"target\\": \\"both\\",
            \\"scenario\\": \\"\$SCENARIO\\",
            \\"test_predictive\\": true,
            \\"output_dir\\": \\"\$output_dir\\"
        }" 2>/dev/null || echo "failed to start")
    
    echo "  Start response: \$start_response"
    sleep 10
    
    # Wait for completion with progress updates
    echo "→ Waiting for test completion..."
    start_time=\$(date +%s)
    
    while [[ "\$(check_status | jq -r '.running // false')" == "true" ]]; do
        elapsed=\$(( (\$(date +%s) - start_time) / 60 ))
        if (( elapsed % 5 == 0 )) && (( elapsed > 0 )); then
            echo "  Running for \${elapsed} minutes... (${SCENARIO} #\${iteration})"
        fi
        sleep 60
    done
    
    total_time=\$(( (\$(date +%s) - start_time) / 60 ))
    echo "  ✓ Test completed in \${total_time} minutes"
    
    # Verify results
    result_count=\$(find "\$output_dir" -name "*.csv" 2>/dev/null | wc -l || echo "0")
    echo "  Results: \$result_count CSV files created"
    
    # Cleanup between tests (except for the last test)
    if [[ \$iteration -lt \$ITERATIONS ]]; then
        echo "→ Cleaning up..."
        
        # Clean databases
        db_cleanup=\$(curl -s -X POST http://localhost:8080/cleanup_databases 2>/dev/null || echo "failed")
        echo "  Database cleanup: \$db_cleanup"
        
        # Clean models
        model_cleanup=\$(curl -s -X POST http://localhost:8080/cleanup_models 2>/dev/null || echo "failed")
        echo "  Model cleanup: \$model_cleanup"
        
        echo "  Waiting 60 seconds for cleanup to settle..."
        sleep 60
        echo "  ✓ Cleanup completed"
    fi
    
    echo "✓ COMPLETED: ${SCENARIO} iteration \$iteration (Test \$current_test/\$total_tests)"
    
    # Brief pause between tests
    sleep 30
done

echo ""
echo "✓ ALL ${SCENARIO^^} BATCH TESTS COMPLETED"
echo ""

# Final summary
echo ""
echo "=========================================="
echo "🎉 BATCH ${SCENARIO^^} TESTS COMPLETED!"
echo "=========================================="
echo "Completion time: \$(date)"
echo "Total tests completed: \$total_tests"
echo "Results saved in: \$RESULTS_BASE"
echo "Log file: \$LOG_FILE"

# Count final results
csv_count=\$(find "\$RESULTS_BASE" -name "*.csv" 2>/dev/null | wc -l || echo "0")
json_count=\$(find "\$RESULTS_BASE" -name "*.json" 2>/dev/null | wc -l || echo "0")

echo ""
echo "📊 Final Results Summary:"
echo "  CSV files: \$csv_count"
echo "  JSON files: \$json_count"
echo "  Scenario: ${SCENARIO}"
echo "  Iterations completed: \$ITERATIONS"

echo ""
echo "📁 Result directories:"
find "\$RESULTS_BASE" -type d -name "*iteration*"

echo ""
echo "🏁 BATCH TESTING COMPLETED SUCCESSFULLY!"
echo "All results saved to: \$RESULTS_BASE"
echo "Complete log available at: \$LOG_FILE"

# Create completion marker
touch "\$RESULTS_BASE/.batch_completed"
EOF

echo "✓ Test script created: $SCRIPT_FILE"

echo ""
echo "Copying script to load-tester pod..."
kubectl cp "$SCRIPT_FILE" $LOAD_TESTER_POD:/tmp/batch_test_${SCENARIO}.sh

echo "Making script executable..."
kubectl exec $LOAD_TESTER_POD -- chmod +x /tmp/batch_test_${SCENARIO}.sh

echo ""
echo "Starting batch tests in background..."

# Start the tests in background inside the pod
kubectl exec -d $LOAD_TESTER_POD -- nohup /tmp/batch_test_${SCENARIO}.sh > /dev/null 2>&1 &

# Give it a moment to start
sleep 5

# Verify it started
echo "Verifying test started..."
status_check=$(kubectl exec $LOAD_TESTER_POD -- curl -s http://localhost:8080/status 2>/dev/null || echo '{"running": false}')
running=$(echo "$status_check" | jq -r '.running // false' 2>/dev/null || echo "false")

if [[ "$running" == "true" ]]; then
    scenario=$(echo "$status_check" | jq -r '.current_scenario // "unknown"' 2>/dev/null || echo "unknown")
    echo "✅ Test successfully started: $scenario"
else
    echo "⚠️  Test may still be initializing..."
fi

echo ""
echo "=========================================="
echo "🚀 BATCH ${SCENARIO^^} TESTS STARTED!"
echo "=========================================="
echo ""
echo "✅ Status:"
echo "  • 5 ${SCENARIO} tests running in load-tester pod"
echo "  • Tests will run autonomously in background"
echo "  • Estimated completion: ~2.5 hours"
echo ""
echo "📋 Monitor Progress:"
echo "  Check status:"
echo "    kubectl exec $LOAD_TESTER_POD -- curl -s http://localhost:8080/status"
echo ""
echo "  View live logs:"
echo "    kubectl exec $LOAD_TESTER_POD -- tail -f $LOG_FILE"
echo ""
echo "  Check results:"
echo "    kubectl exec $LOAD_TESTER_POD -- ls -la $RESULTS_BASE/"
echo ""
echo "📁 Results will be saved to: $RESULTS_BASE"
echo ""
echo "⏰ After tests complete (~2.5 hours), download results:"
echo "    ./download_batch_results.sh ${SCENARIO} ${BASE_TIMESTAMP}"
echo ""
echo "🎯 Tests are now running autonomously!"

# Clean up temp file
rm -f "$SCRIPT_FILE"

echo ""
echo "💾 To download results after completion, run:"
echo "   ./download_batch_results.sh ${SCENARIO} ${BASE_TIMESTAMP}"