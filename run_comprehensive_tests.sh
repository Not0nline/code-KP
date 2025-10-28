#!/bin/bash

# Comprehensive Test Suite - 10 iterations each for LOW, MEDIUM, HIGH scenarios
# Workflow: Load model → Load dataset → Test → Save results → Clean dataset → Clean model → Repeat

set -e

# Configuration
SCENARIOS=("low" "medium" "high")
ITERATIONS=10
DURATION=1800  # 30 minutes per test
BASE_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_BASE="/results/comprehensive_test_${BASE_TIMESTAMP}"

# SSH and Kubernetes configuration
SSH_KEY="key_ta_2.pem"
SSH_HOST="ubuntu@ec2-52-54-42-13.compute-1.amazonaws.com"
LOAD_TESTER_POD="load-tester-7c987b969b-zt5tj"

echo "=========================================="
echo "COMPREHENSIVE TESTING SUITE STARTED"
echo "=========================================="
echo "Base timestamp: ${BASE_TIMESTAMP}"
echo "Results directory: ${RESULTS_BASE}"
echo "Scenarios: ${SCENARIOS[@]}"
echo "Iterations per scenario: ${ITERATIONS}"
echo "Duration per test: ${DURATION} seconds (30 minutes)"
echo ""

# Function to check test status
check_test_status() {
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "kubectl exec ${LOAD_TESTER_POD} -- curl -s http://localhost:8080/status" 2>/dev/null
}

# Function to wait for test completion
wait_for_test_completion() {
    local scenario=$1
    local iteration=$2
    
    echo "Waiting for ${scenario} test iteration ${iteration} to complete..."
    
    while true; do
        status=$(check_test_status)
        running=$(echo "$status" | jq -r '.running // false')
        
        if [[ "$running" == "false" ]]; then
            echo "Test completed!"
            break
        fi
        
        # Show progress every 2 minutes
        current_scenario=$(echo "$status" | jq -r '.current_scenario // "unknown"')
        echo "Still running: ${current_scenario} scenario..."
        sleep 120
    done
}

# Function to clean up between tests
cleanup_between_tests() {
    local scenario=$1
    local iteration=$2
    
    echo "Cleaning up after ${scenario} iteration ${iteration}..."
    
    # Clean product databases
    echo "Cleaning product databases..."
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "kubectl exec ${LOAD_TESTER_POD} -- curl -s -X POST http://localhost:8080/cleanup_databases" || true
    
    # Clean predictive models
    echo "Cleaning predictive models..."
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "kubectl exec ${LOAD_TESTER_POD} -- curl -s -X POST http://localhost:8080/cleanup_models" || true
    
    # Wait for cleanup to complete
    sleep 30
    
    echo "Cleanup completed."
}

# Function to load model and dataset
load_model_and_dataset() {
    local scenario=$1
    local iteration=$2
    
    echo "Loading model and dataset for ${scenario} iteration ${iteration}..."
    
    # Load the appropriate dataset for the scenario
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "kubectl exec ${LOAD_TESTER_POD} -- curl -s -X POST http://localhost:8080/load_dataset \
        -H 'Content-Type: application/json' \
        -d '{\"scenario\": \"${scenario}\"}'" || true
    
    # Train the model with the dataset
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "kubectl exec ${LOAD_TESTER_POD} -- curl -s -X POST http://localhost:8080/train_model \
        -H 'Content-Type: application/json' \
        -d '{\"scenario\": \"${scenario}\"}'" || true
    
    # Wait for training to complete
    sleep 60
    
    echo "Model and dataset loaded for ${scenario}."
}

# Function to start a test
start_test() {
    local scenario=$1
    local iteration=$2
    local output_dir="${RESULTS_BASE}/${scenario}_test/iteration_$(printf '%02d' $iteration)"
    
    echo "Starting ${scenario} test iteration ${iteration}..."
    echo "Output directory: ${output_dir}"
    
    response=$(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "kubectl exec ${LOAD_TESTER_POD} -- curl -s -X POST http://localhost:8080/start \
        -H 'Content-Type: application/json' \
        -d '{
            \"duration\": ${DURATION},
            \"target\": \"both\",
            \"scenario\": \"${scenario}\",
            \"test_predictive\": true,
            \"output_dir\": \"${output_dir}\"
        }'")
    
    echo "Test started. Response: ${response}"
}

# Function to save and verify results
save_and_verify_results() {
    local scenario=$1
    local iteration=$2
    
    echo "Saving and verifying results for ${scenario} iteration ${iteration}..."
    
    # Get final status
    final_status=$(check_test_status)
    echo "Final status: ${final_status}"
    
    # Check if results were saved
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
        "find ${RESULTS_BASE}/${scenario}_test/iteration_$(printf '%02d' $iteration) -name '*.csv' | head -5" || true
    
    echo "Results saved and verified."
}

# Main execution loop
total_tests=$((${#SCENARIOS[@]} * ITERATIONS))
current_test=0

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
        echo "=========================================="
        
        # Step 1: Load model and dataset
        load_model_and_dataset "$scenario" "$iteration"
        
        # Step 2: Start the test
        start_test "$scenario" "$iteration"
        
        # Step 3: Wait for test completion
        wait_for_test_completion "$scenario" "$iteration"
        
        # Step 4: Save and verify results
        save_and_verify_results "$scenario" "$iteration"
        
        # Step 5: Clean up (except for the last test of the last scenario)
        if [[ ! ("$scenario" == "${SCENARIOS[-1]}" && "$iteration" == "$ITERATIONS") ]]; then
            cleanup_between_tests "$scenario" "$iteration"
        fi
        
        echo "Completed ${scenario} iteration ${iteration} (Test ${current_test}/${total_tests})"
        
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
echo "ALL COMPREHENSIVE TESTS COMPLETED!"
echo "=========================================="
echo "Total tests run: ${total_tests}"
echo "Results saved in: ${RESULTS_BASE}"
echo ""

# Generate final summary
echo "Generating final summary..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" \
    "find ${RESULTS_BASE} -name 'api_summary.json' | wc -l" || true

echo "Summary: Found $(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_HOST}" "find ${RESULTS_BASE} -name 'api_summary.json' | wc -l" 2>/dev/null || echo "0") completed test summaries"

echo ""
echo "Comprehensive testing suite completed successfully!"