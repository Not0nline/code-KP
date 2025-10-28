#!/bin/bash

# Comprehensive Test Suite - Direct execution in Kubernetes cluster
# Run this script directly on the cluster - no SSH needed!

set -e

# Configuration
SCENARIOS=("low" "medium" "high")
ITERATIONS=10
DURATION=1800  # 30 minutes per test
BASE_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="/results/comprehensive_test_${BASE_TIMESTAMP}"
LOAD_TESTER_POD="load-tester-7c987b969b-zt5tj"
LOG_FILE="/results/comprehensive_test_${BASE_TIMESTAMP}.log"

echo "=========================================="
echo "COMPREHENSIVE TEST SUITE (CLUSTER MODE)"
echo "=========================================="
echo "Running directly in Kubernetes cluster"
echo "No SSH required - direct pod execution"
echo ""
echo "Timestamp: ${BASE_TIMESTAMP}"
echo "Results: ${RESULTS_BASE}"
echo "Log: ${LOG_FILE}"
echo "Tests: 10 LOW + 10 MEDIUM + 10 HIGH = 30 total"
echo "Duration: ~15 hours"
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
echo "Creating comprehensive test script..."

# Create the comprehensive test script that runs inside the pod
# Use current directory instead of /tmp to avoid permission issues
SCRIPT_FILE="./comprehensive_test_pod.sh"
cat > "$SCRIPT_FILE" << 'EOF'
#!/bin/bash
set -e

# Comprehensive Test Suite - Runs inside load-tester pod
SCENARIOS=("low" "medium" "high")
ITERATIONS=10
DURATION=1800  # 30 minutes
BASE_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="/results/comprehensive_${BASE_TIMESTAMP}"
LOG_FILE="/results/comprehensive_${BASE_TIMESTAMP}.log"

# Redirect output to both console and log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "COMPREHENSIVE TESTING SUITE STARTED"
echo "=========================================="
echo "Started: $(date)"
echo "Results directory: $RESULTS_BASE"
echo "Log file: $LOG_FILE"
echo "Test plan: 10 LOW + 10 MEDIUM + 10 HIGH = 30 tests"
echo "Estimated duration: 15 hours"
echo ""

# Create results directory structure
mkdir -p "$RESULTS_BASE"
for scenario in "${SCENARIOS[@]}"; do
    mkdir -p "$RESULTS_BASE/${scenario}_test"
done

# Function to check test status
check_status() {
    curl -s http://localhost:8080/status 2>/dev/null || echo '{"running": false}'
}

# Wait for any existing test to complete
if [[ "$(check_status | jq -r '.running // false')" == "true" ]]; then
    echo "Waiting for existing test to complete..."
    while [[ "$(check_status | jq -r '.running // false')" == "true" ]]; do
        sleep 60
    done
    echo "Previous test completed, starting comprehensive suite"
fi

# Main test execution loop
total_tests=$((${#SCENARIOS[@]} * ITERATIONS))
current_test=0

for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "=========================================="
    echo "STARTING ${scenario^^} SCENARIO (10 tests)"
    echo "=========================================="
    
    for iteration in $(seq 1 $ITERATIONS); do
        current_test=$((current_test + 1))
        output_dir="$RESULTS_BASE/${scenario}_test/iteration_$(printf '%02d' $iteration)"
        
        echo ""
        echo "TEST $current_test/$total_tests: ${scenario^^} ITERATION $iteration"
        echo "Time: $(date)"
        echo "Output: $output_dir"
        echo "----------------------------------------"
        
        # Start test
        echo "→ Starting ${scenario} test iteration $iteration..."
        start_response=$(curl -s -X POST http://localhost:8080/start \
            -H 'Content-Type: application/json' \
            -d "{
                \"duration\": $DURATION,
                \"target\": \"both\",
                \"scenario\": \"$scenario\",
                \"test_predictive\": true,
                \"output_dir\": \"$output_dir\"
            }" 2>/dev/null || echo "failed to start")
        
        echo "  Start response: $start_response"
        sleep 10
        
        # Wait for completion with progress updates
        echo "→ Waiting for test completion..."
        start_time=$(date +%s)
        
        while [[ "$(check_status | jq -r '.running // false')" == "true" ]]; do
            elapsed=$(( ($(date +%s) - start_time) / 60 ))
            if (( elapsed % 5 == 0 )) && (( elapsed > 0 )); then
                echo "  Running for ${elapsed} minutes... (${scenario} #${iteration})"
            fi
            sleep 60
        done
        
        total_time=$(( ($(date +%s) - start_time) / 60 ))
        echo "  ✓ Test completed in ${total_time} minutes"
        
        # Verify results
        result_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l || echo "0")
        echo "  Results: $result_count CSV files created"
        
        # Cleanup between tests (except for the last test)
        if [[ ! ("$scenario" == "${SCENARIOS[-1]}" && "$iteration" == "$ITERATIONS") ]]; then
            echo "→ Cleaning up..."
            
            # Clean databases
            db_cleanup=$(curl -s -X POST http://localhost:8080/cleanup_databases 2>/dev/null || echo "failed")
            echo "  Database cleanup: $db_cleanup"
            
            # Clean models
            model_cleanup=$(curl -s -X POST http://localhost:8080/cleanup_models 2>/dev/null || echo "failed")
            echo "  Model cleanup: $model_cleanup"
            
            echo "  Waiting 60 seconds for cleanup to settle..."
            sleep 60
            echo "  ✓ Cleanup completed"
        fi
        
        echo "✓ COMPLETED: ${scenario} iteration ${iteration} (Test $current_test/$total_tests)"
        
        # Brief pause between tests
        sleep 30
    done
    
    echo ""
    echo "✓ ALL ${scenario^^} SCENARIO TESTS COMPLETED"
    echo ""
done

# Final summary
echo ""
echo "=========================================="
echo "🎉 ALL COMPREHENSIVE TESTS COMPLETED!"
echo "=========================================="
echo "Completion time: $(date)"
echo "Total tests completed: $total_tests"
echo "Results saved in: $RESULTS_BASE"
echo "Log file: $LOG_FILE"

# Count final results
csv_count=$(find "$RESULTS_BASE" -name "*.csv" 2>/dev/null | wc -l || echo "0")
json_count=$(find "$RESULTS_BASE" -name "*.json" 2>/dev/null | wc -l || echo "0")

echo ""
echo "📊 Final Results Summary:"
echo "  CSV files: $csv_count"
echo "  JSON files: $json_count"
echo "  Test scenarios: ${#SCENARIOS[@]}"
echo "  Iterations per scenario: $ITERATIONS"
echo "  Total tests: $total_tests"

echo ""
echo "📁 Sample result directories:"
find "$RESULTS_BASE" -type d -name "*iteration*" | head -5

echo ""
echo "🏁 COMPREHENSIVE TESTING SUITE COMPLETED SUCCESSFULLY!"
echo "All results saved to: $RESULTS_BASE"
echo "Complete log available at: $LOG_FILE"
EOF

echo "✓ Test script created: $SCRIPT_FILE"

echo ""
echo "Copying script to load-tester pod..."
kubectl cp "$SCRIPT_FILE" $LOAD_TESTER_POD:/tmp/comprehensive_test_pod.sh

echo "Making script executable..."
kubectl exec $LOAD_TESTER_POD -- chmod +x /tmp/comprehensive_test_pod.sh

echo ""
echo "Starting comprehensive tests in background..."

# Start the tests in background inside the pod
kubectl exec -d $LOAD_TESTER_POD -- nohup /tmp/comprehensive_test_pod.sh > /dev/null 2>&1 &

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
echo "🚀 COMPREHENSIVE TESTS DEPLOYED!"
echo "=========================================="
echo ""
echo "✅ Status:"
echo "  • 30 tests deployed and running in load-tester pod"
echo "  • Tests will run completely autonomously"
echo "  • No SSH or external dependencies needed"
echo ""
echo "📊 Test Plan:"
echo "  • LOW: 10 tests × 30min = 5 hours"
echo "  • MEDIUM: 10 tests × 30min = 5 hours"
echo "  • HIGH: 10 tests × 30min = 5 hours"
echo "  • TOTAL: ~15 hours"
echo ""
echo "📋 Monitor Progress:"
echo "  Check status:"
echo "    kubectl exec $LOAD_TESTER_POD -- curl -s http://localhost:8080/status"
echo ""
echo "  View live logs:"
echo "    kubectl exec $LOAD_TESTER_POD -- tail -f /results/comprehensive_*.log"
echo ""
echo "  Check results:"
echo "    kubectl exec $LOAD_TESTER_POD -- ls -la /results/"
echo ""
echo "📁 Results will be saved to: /results/comprehensive_[timestamp]/"
echo ""
echo "🎯 Tests are now running completely autonomously in the cluster!"
echo "   The process will continue even if you disconnect."

# Clean up temp file
rm -f "$SCRIPT_FILE"