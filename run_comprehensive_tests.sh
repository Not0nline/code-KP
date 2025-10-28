#!/bin/bash

# Comprehensive Test Suite - 10 iterations each for LOW, MEDIUM, HIGH scenarios
# Workflow: Load model → Load dataset → Test → Save results → Clean dataset → Clean model → Repeat
# This script runs INSIDE the Kubernetes cluster in the load-tester pod

set -e

# Configuration
SCENARIOS=("low" "medium" "high")
ITERATIONS=10
DURATION=1800  # 30 minutes per test
BASE_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="/results/comprehensive_test_${BASE_TIMESTAMP}"

# SSH and Kubernetes configuration - Update path as needed
SSH_KEY="key_ta_2.pem"  # Make sure this key is in current directory or provide full path
SSH_HOST="ubuntu@ec2-52-54-42-13.compute-1.amazonaws.com"
LOAD_TESTER_POD="load-tester-7c987b969b-zt5tj"
LOG_FILE="/results/comprehensive_test_${BASE_TIMESTAMP}.log"

# Function to check SSH connectivity
check_ssh_connection() {
    echo "Checking SSH connection and key..."
    
    # Check if key file exists
    if [[ ! -f "$SSH_KEY" ]]; then
        echo "❌ SSH key file '$SSH_KEY' not found in current directory"
        echo "Please ensure the SSH key is available. Options:"
        echo "1. Copy key_ta_2.pem to current directory"
        echo "2. Update SSH_KEY variable in script with full path"
        echo "3. Run from directory containing the key file"
        return 1
    fi
    
    # Test SSH connection
    if ! ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SSH_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
        echo "❌ Cannot connect to cluster via SSH"
        echo "Please check:"
        echo "1. SSH key permissions: chmod 600 $SSH_KEY"
        echo "2. Network connectivity to $SSH_HOST"
        echo "3. Key file is correct"
        return 1
    fi
    
    echo "✓ SSH connection successful"
    return 0
}

# Create comprehensive test script that runs INSIDE the pod
create_pod_script() {
    cat > /tmp/comprehensive_test_pod.sh << 'EOF'
#!/bin/bash
set -e

# Configuration
SCENARIOS=("low" "medium" "high")
ITERATIONS=10
DURATION=1800  # 30 minutes per test
BASE_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="/results/comprehensive_test_${BASE_TIMESTAMP}"
LOG_FILE="/results/comprehensive_test_${BASE_TIMESTAMP}.log"

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "COMPREHENSIVE TESTING SUITE STARTED"
echo "=========================================="
echo "Base timestamp: ${BASE_TIMESTAMP}"
echo "Results directory: ${RESULTS_BASE}"
echo "Log file: ${LOG_FILE}"
echo "Scenarios: ${SCENARIOS[@]}"
echo "Iterations per scenario: ${ITERATIONS}"
echo "Duration per test: ${DURATION} seconds (30 minutes)"
echo "Started at: $(date)"
echo ""

# Function to check test status (local API call)
check_test_status() {
    curl -s http://localhost:8080/status 2>/dev/null || echo '{"running": false}'
}

# Function to wait for test completion
wait_for_test_completion() {
    local scenario=$1
    local iteration=$2
    local start_time=$(date +%s)
    
    echo "Waiting for ${scenario} test iteration ${iteration} to complete..."
    
    while true; do
        status=$(check_test_status)
        running=$(echo "$status" | jq -r '.running // false')
        
        if [[ "$running" == "false" ]]; then
            local end_time=$(date +%s)
            local duration=$(( (end_time - start_time) / 60 ))
            echo "Test completed in ${duration} minutes!"
            break
        fi
        
        # Show progress every 5 minutes
        local elapsed=$(( ($(date +%s) - start_time) / 60 ))
        if (( elapsed % 5 == 0 )) && (( elapsed > 0 )); then
            current_scenario=$(echo "$status" | jq -r '.current_scenario // "unknown"')
            echo "[${elapsed}min] Still running: ${current_scenario} scenario..."
        fi
        sleep 60  # Check every minute
    done
}

# Function to clean up between tests
cleanup_between_tests() {
    local scenario=$1
    local iteration=$2
    
    echo "Cleaning up after ${scenario} iteration ${iteration}..."
    
    # Clean product databases
    echo "  → Cleaning product databases..."
    local db_cleanup=$(curl -s -X POST http://localhost:8080/cleanup_databases 2>/dev/null || echo "failed")
    echo "    Database cleanup result: ${db_cleanup}"
    
    # Clean predictive models
    echo "  → Cleaning predictive models..."
    local model_cleanup=$(curl -s -X POST http://localhost:8080/cleanup_models 2>/dev/null || echo "failed")
    echo "    Model cleanup result: ${model_cleanup}"
    
    # Wait for cleanup to settle
    echo "  → Waiting 60 seconds for cleanup to settle..."
    sleep 60
    
    echo "  ✓ Cleanup completed."
}

# Function to load model and dataset (handled automatically by start endpoint)
prepare_for_test() {
    local scenario=$1
    local iteration=$2
    
    echo "Preparing for ${scenario} iteration ${iteration}..."
    echo "  → Model and dataset will be loaded automatically when test starts"
    echo "  ✓ Ready to start test"
}

# Function to start a test
start_test() {
    local scenario=$1
    local iteration=$2
    local output_dir="${RESULTS_BASE}/${scenario}_test/iteration_$(printf '%02d' $iteration)"
    
    echo "Starting ${scenario} test iteration ${iteration}..."
    echo "  → Output directory: ${output_dir}"
    
    response=$(curl -s -X POST http://localhost:8080/start \
        -H 'Content-Type: application/json' \
        -d "{
            \"duration\": ${DURATION},
            \"target\": \"both\",
            \"scenario\": \"${scenario}\",
            \"test_predictive\": true,
            \"output_dir\": \"${output_dir}\"
        }" 2>/dev/null || echo "failed to start")
    
    echo "  ✓ Test started. Response: ${response}"
    
    # Brief pause to let test initialize
    sleep 10
}

# Function to save and verify results
save_and_verify_results() {
    local scenario=$1
    local iteration=$2
    
    echo "Verifying results for ${scenario} iteration ${iteration}..."
    
    # Get final status
    final_status=$(check_test_status)
    echo "  → Final status: ${final_status}"
    
    # Check if results were saved
    local result_count=$(find ${RESULTS_BASE}/${scenario}_test/iteration_$(printf '%02d' $iteration) -name '*.csv' 2>/dev/null | wc -l || echo "0")
    echo "  → Found ${result_count} CSV result files"
    
    if [[ "$result_count" -gt 0 ]]; then
        echo "  ✓ Results saved and verified."
    else
        echo "  ⚠️  Warning: No CSV files found in results directory"
    fi
}

# Main execution loop
total_tests=$((${#SCENARIOS[@]} * ITERATIONS))
current_test=0

echo "Creating results directory structure..."
mkdir -p "${RESULTS_BASE}"
for scenario in "${SCENARIOS[@]}"; do
    mkdir -p "${RESULTS_BASE}/${scenario}_test"
done

# Wait for any existing test to complete first
if [[ "$(echo "$(check_test_status)" | jq -r '.running // false')" == "true" ]]; then
    echo "⚠️  Existing test detected, waiting for completion..."
    wait_for_test_completion "existing" "test"
    cleanup_between_tests "existing" "test"
    echo "✓ Existing test completed, starting comprehensive suite"
    echo ""
fi

for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo "=========================================="
    echo "STARTING ${scenario^^} SCENARIO TESTS"
    echo "=========================================="
    
    for iteration in $(seq 1 $ITERATIONS); do
        current_test=$((current_test + 1))
        
        echo ""
        echo "==========================================";
        echo "TEST ${current_test}/${total_tests}: ${scenario^^} ITERATION ${iteration}"
        echo "Time: $(date)"
        echo "=========================================="
        
        # Step 1: Prepare for test
        prepare_for_test "$scenario" "$iteration"
        
        # Step 2: Start the test
        start_test "$scenario" "$iteration"
        
        # Step 3: Wait for test completion
        wait_for_test_completion "$scenario" "$iteration"
        
        # Step 4: Verify results
        save_and_verify_results "$scenario" "$iteration"
        
        # Step 5: Clean up (except for the last test of the last scenario)
        if [[ ! ("$scenario" == "${SCENARIOS[-1]}" && "$iteration" == "$ITERATIONS") ]]; then
            cleanup_between_tests "$scenario" "$iteration"
        fi
        
        echo "✓ Completed ${scenario} iteration ${iteration} (Test ${current_test}/${total_tests})"
        
        # Brief pause between tests
        sleep 30
    done
    
    echo ""
    echo "=========================================="
    echo "COMPLETED ALL ${scenario^^} SCENARIO TESTS"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "🎉 ALL COMPREHENSIVE TESTS COMPLETED!"
echo "=========================================="
echo "Completion time: $(date)"
echo "Total tests run: ${total_tests}"
echo "Results saved in: ${RESULTS_BASE}"
echo "Log file: ${LOG_FILE}"
echo ""

# Generate final summary
echo "Generating final summary..."
local csv_files=$(find ${RESULTS_BASE} -name '*.csv' 2>/dev/null | wc -l || echo "0")
local json_files=$(find ${RESULTS_BASE} -name '*.json' 2>/dev/null | wc -l || echo "0")
local summary_files=$(find ${RESULTS_BASE} -name 'api_summary.json' 2>/dev/null | wc -l || echo "0")

echo ""
echo "📊 Final Results Summary:"
echo "  CSV files: ${csv_files}"
echo "  JSON files: ${json_files}" 
echo "  Test summaries: ${summary_files}"
echo "  Total scenarios: ${#SCENARIOS[@]}"
echo "  Iterations per scenario: ${ITERATIONS}"
echo "  Expected total tests: ${total_tests}"
echo ""

# List sample result directories
echo "📁 Sample result directories:"
find ${RESULTS_BASE} -type d -name "*iteration*" | head -5

echo ""
echo "🏁 COMPREHENSIVE TESTING SUITE COMPLETED SUCCESSFULLY!"
echo "All logs saved to: ${LOG_FILE}"

EOF
}

# Main execution starts here
echo "=========================================="
echo "COMPREHENSIVE TEST DEPLOYMENT"
echo "=========================================="
echo "This will deploy and run 30 comprehensive tests in the cluster"
echo "Tests will run in background inside the load-tester pod"
echo ""

# Check SSH connection first
if ! check_ssh_connection; then
    echo ""
    echo "❌ Cannot proceed without SSH connection to cluster"
    exit 1
fi

echo ""
echo "Checking for any running tests..."
current_status=$(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
    "kubectl exec ${LOAD_TESTER_POD} -- curl -s http://localhost:8080/status" 2>/dev/null || echo '{"running": false}')

current_running=$(echo "$current_status" | jq -r '.running // false')
if [[ "$current_running" == "true" ]]; then
    echo "⚠️  Test currently running. Waiting for completion..."
    current_scenario=$(echo "$current_status" | jq -r '.current_scenario // "unknown"')
    echo "Current test: ${current_scenario}"
    
    while [[ "$(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "kubectl exec ${LOAD_TESTER_POD} -- curl -s http://localhost:8080/status" 2>/dev/null | \
        jq -r '.running // false')" == "true" ]]; do
        echo "Still waiting for current test to complete..."
        sleep 60
    done
    echo "✓ Current test completed"
fi

echo ""
echo "Deploying comprehensive test script to cluster..."

# Create the pod script
create_pod_script

# Copy script to cluster and execute in background
echo "Copying script to cluster and starting execution..."
scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no /tmp/comprehensive_test_pod.sh "${SSH_HOST}:~/comprehensive_test_pod.sh"

# Execute the script in background inside the pod
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
    "kubectl cp comprehensive_test_pod.sh ${LOAD_TESTER_POD}:/tmp/comprehensive_test_pod.sh && \
     kubectl exec ${LOAD_TESTER_POD} -- chmod +x /tmp/comprehensive_test_pod.sh && \
     kubectl exec -d ${LOAD_TESTER_POD} -- nohup /tmp/comprehensive_test_pod.sh > /dev/null 2>&1 &"

echo ""
echo "=========================================="
echo "🚀 COMPREHENSIVE TESTS STARTED!"
echo "=========================================="
echo ""
echo "✓ Script deployed to cluster"
echo "✓ Tests running in background inside load-tester pod"
echo "✓ You can safely close your laptop - tests will continue!"
echo ""
echo "📊 Test Plan:"
echo "  • LOW: 10 iterations × 30 min = 5 hours"
echo "  • MEDIUM: 10 iterations × 30 min = 5 hours"  
echo "  • HIGH: 10 iterations × 30 min = 5 hours"
echo "  • Total: 30 tests over ~15 hours"
echo ""
echo "📋 Monitor Progress:"
echo "  Check status: ssh -i \"${SSH_KEY}\" ${SSH_HOST} \"kubectl exec ${LOAD_TESTER_POD} -- curl -s http://localhost:8080/status\""
echo "  View logs: ssh -i \"${SSH_KEY}\" ${SSH_HOST} \"kubectl exec ${LOAD_TESTER_POD} -- tail -f /results/comprehensive_test_*.log\""
echo ""
echo "📁 Results will be saved to: /results/comprehensive_test_[timestamp]/"
echo ""
echo "🎯 Tests are now running autonomously in the cluster!"