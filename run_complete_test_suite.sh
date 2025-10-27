#!/bin/bash
# Complete Test Suite Script
# Runs LOW, MEDIUM, HIGH tests with proper workflow:
# Load → Train → Predict → Save → Clean → Repeat

echo "🚀 Starting Complete Test Suite (LOW → MEDIUM → HIGH)"
echo "⏰ Start Time: $(date)"
echo "================================================================================"

# Configuration
BATCH_NAME="complete_test_$(date +%Y%m%d_%H%M%S)"
LOAD_TESTER_POD=$(kubectl get pods -l app=load-tester -o jsonpath='{.items[0].metadata.name}')
RESULTS_BASE="/results/complete_suite_$BATCH_NAME"

echo "📋 Test Configuration:"
echo "   Batch Name: $BATCH_NAME"
echo "   Load Tester Pod: $LOAD_TESTER_POD"
echo "   Results Directory: $RESULTS_BASE"
echo "   Test Duration: 1800s (30 minutes) per scenario"
echo "================================================================================"

# Function to run a single test scenario
run_test_scenario() {
    local scenario=$1
    local scenario_upper=$(echo $scenario | tr '[:lower:]' '[:upper:]')
    
    echo ""
    echo "🎯 STARTING $scenario_upper TEST"
    echo "⏰ Time: $(date)"
    echo "================================================================================"
    
    # Step 1: Clean all services before test
    echo "🧹 Step 1: Cleaning all services..."
    
    # Kill any existing load test processes
    echo "   Stopping any existing load test processes..."
    kubectl exec $LOAD_TESTER_POD -- pkill -f "python load_test.py" || true
    sleep 5
    
    # Reset product databases
    echo "   Resetting product databases..."
    kubectl exec deployment/product-app-hpa -- wget -q -O- --post-data='' http://localhost:5000/reset_data || echo "   ⚠️ HPA reset failed (might not have wget)"
    kubectl exec deployment/product-app-combined -- wget -q -O- --post-data='' http://localhost:5000/reset_data || echo "   ⚠️ Combined reset failed (might not have wget)"
    
    # Reset predictive scaler (if possible)
    echo "   Resetting predictive scaler..."
    kubectl exec -it $LOAD_TESTER_POD -- python -c "
import requests
try:
    response = requests.post('http://predictive-scaler.default.svc.cluster.local:5000/reset_data', timeout=15)
    print(f'   ✅ Predictive reset: {response.status_code}')
except Exception as e:
    print(f'   ⚠️ Predictive reset failed: {e}')
" 2>/dev/null || echo "   ⚠️ Predictive reset skipped"
    
    # Step 2: Wait for system stabilization
    echo "   ⏳ Waiting 30s for system stabilization..."
    sleep 30
    
    # Step 3: Run the test
    echo "🔄 Step 2: Running $scenario_upper test (30 minutes)..."
    local test_output_dir="$RESULTS_BASE/${scenario}_test"
    
    kubectl exec $LOAD_TESTER_POD -- python load_test.py \
        --scenario $scenario \
        --duration 1800 \
        --target both \
        --test-predictive \
        --metrics-port 9106 \
        --output-dir "$test_output_dir" &
    
    local test_pid=$!
    
    # Step 4: Monitor test progress
    echo "📊 Step 3: Monitoring test progress..."
    echo "   Test PID: $test_pid"
    echo "   Monitor with: kubectl logs -f $LOAD_TESTER_POD"
    echo "   HPA scaling: kubectl get hpa -w"
    echo "   Pod scaling: kubectl get pods -w"
    
    # Wait for test to complete
    wait $test_pid
    local test_exit_code=$?
    
    if [ $test_exit_code -eq 0 ]; then
        echo "✅ $scenario_upper test completed successfully"
    else
        echo "❌ $scenario_upper test failed with exit code $test_exit_code"
        return $test_exit_code
    fi
    
    # Step 5: Verify results were saved
    echo "💾 Step 4: Verifying results..."
    kubectl exec -it $LOAD_TESTER_POD -- ls -la "$test_output_dir/" || echo "   ⚠️ Could not list results directory"
    
    echo "✅ $scenario_upper TEST COMPLETED"
    echo "================================================================================"
}

# Function to check system status
check_system_status() {
    echo "🔍 SYSTEM STATUS CHECK"
    echo "================================================================================"
    
    echo "📊 Pod Status:"
    kubectl get pods | grep -E "(product-app|predictive-scaler|load-tester)"
    
    echo ""
    echo "📈 HPA Status:"
    kubectl get hpa
    
    echo ""
    echo "💾 Storage Status:"
    kubectl exec -it $LOAD_TESTER_POD -- df -h /results
    
    echo "================================================================================"
}

# Function to create final summary
create_final_summary() {
    echo ""
    echo "📊 CREATING FINAL SUMMARY"
    echo "================================================================================"
    
    local summary_file="$RESULTS_BASE/complete_suite_summary.json"
    
    kubectl exec -it $LOAD_TESTER_POD -- python -c "
import json
import os
from datetime import datetime

summary = {
    'batch_name': '$BATCH_NAME',
    'start_time': '$start_time',
    'end_time': datetime.now().isoformat(),
    'tests_completed': [],
    'results_location': '$RESULTS_BASE'
}

# Check which tests completed
scenarios = ['low', 'medium', 'high']
for scenario in scenarios:
    test_dir = f'$RESULTS_BASE/{scenario}_test'
    if os.path.exists(test_dir):
        summary['tests_completed'].append(scenario)

# Save summary
os.makedirs('$RESULTS_BASE', exist_ok=True)
with open('$summary_file', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to: $summary_file')
print(f'Tests completed: {summary[\"tests_completed\"]}')
"
    
    echo "✅ Final summary created"
    echo "================================================================================"
}

# Main execution
main() {
    local start_time=$(date -Iseconds)
    
    # Initial system check
    check_system_status
    
    # Update scaling configuration to ensure fairness
    echo "⚖️ UPDATING SCALING CONFIGURATION"
    echo "================================================================================"
    echo "Applying fair scaling thresholds:"
    echo "  - Both HPA & Combined: Scale up at 70% CPU, target 75% CPU"
    echo "  - HPA: 75% CPU target, 80% memory target"
    echo "  - Combined: 70% scale-up threshold, 75% target utilization"
    kubectl apply -f Product-App/scaling-thresholds-configmap.yaml
    kubectl apply -f Product-App/HPA/product-app-hpa-hpa.yaml
    echo "✅ Fair scaling configuration applied"
    echo ""
    
    # Run all three test scenarios
    echo "🎯 RUNNING ALL TEST SCENARIOS"
    echo "================================================================================"
    
    # LOW Test
    if run_test_scenario "low"; then
        echo "✅ LOW test successful"
    else
        echo "❌ LOW test failed - continuing with remaining tests..."
    fi
    
    # Cooldown between tests
    echo "😴 Cooldown: 5 minutes between tests..."
    sleep 300
    
    # MEDIUM Test  
    if run_test_scenario "medium"; then
        echo "✅ MEDIUM test successful"
    else
        echo "❌ MEDIUM test failed - continuing with remaining tests..."
    fi
    
    # Cooldown between tests
    echo "😴 Cooldown: 5 minutes between tests..."
    sleep 300
    
    # HIGH Test
    if run_test_scenario "high"; then
        echo "✅ HIGH test successful"
    else
        echo "❌ HIGH test failed"
    fi
    
    # Final summary
    create_final_summary
    
    # Final system check
    check_system_status
    
    echo ""
    echo "🎉 COMPLETE TEST SUITE FINISHED"
    echo "⏰ End Time: $(date)"
    echo "📊 Results saved in: $RESULTS_BASE"
    echo "================================================================================"
}

# Run main function
main "$@"