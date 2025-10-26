#!/bin/bash
# Run a single scenario test
# Usage: ./run_single_scenario.sh <low|medium|high> [iterations] [duration]

SCENARIO=$1
ITERATIONS=${2:-10}
DURATION=${3:-1800}

if [ -z "$SCENARIO" ]; then
    echo "Usage: ./run_single_scenario.sh <low|medium|high> [iterations] [duration]"
    echo ""
    echo "Examples:"
    echo "  ./run_single_scenario.sh low           # 10 iterations, 1800s each"
    echo "  ./run_single_scenario.sh medium 5 900  # 5 iterations, 900s each"
    exit 1
fi

# Validate scenario
if [[ ! "$SCENARIO" =~ ^(low|medium|high)$ ]]; then
    echo "❌ Invalid scenario. Must be: low, medium, or high"
    exit 1
fi

LOAD_TESTER_POD="deployment/load-tester"
PREDICTIVE_SCALER="http://predictive-scaler:5000"
CONTROL_API="http://localhost:8080"
RESULTS_DIR="/data/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCENARIO_DIR="${RESULTS_DIR}/${SCENARIO}_${TIMESTAMP}"

echo "========================================"
echo "Single Scenario Test Runner"
echo "========================================"
echo "Scenario:   $SCENARIO"
echo "Iterations: $ITERATIONS"
echo "Duration:   ${DURATION}s ($(($DURATION / 60)) min per test)"
echo "Total time: ~$(($ITERATIONS * $DURATION / 60)) minutes"
echo "Results:    $SCENARIO_DIR"
echo "========================================"
echo ""

# Create directory
echo "1. Creating results directory..."
kubectl exec -it $LOAD_TESTER_POD -- mkdir -p "$SCENARIO_DIR"
echo "   ✅ Directory created"
echo ""

# Load baseline
echo "2. Loading ${SCENARIO} baseline..."
kubectl exec -it $LOAD_TESTER_POD -- curl -X POST \
    "${PREDICTIVE_SCALER}/api/baseline/load" \
    -H 'Content-Type: application/json' \
    -d "{\"scenario\":\"${SCENARIO}\"}"

echo ""
sleep 5

# Verify baseline
echo "3. Verifying baseline..."
BASELINE_POINTS=$(kubectl exec -it $LOAD_TESTER_POD -- curl -s \
    "${PREDICTIVE_SCALER}/status" | jq -r '.training_dataset_points // 0')

if [ "$BASELINE_POINTS" -lt 100 ]; then
    echo "   ❌ ERROR: Baseline not loaded (points: $BASELINE_POINTS)"
    exit 1
fi
echo "   ✅ Baseline loaded: $BASELINE_POINTS points"
echo ""

# Run iterations
for i in $(seq 1 $ITERATIONS); do
    echo "========================================"
    echo "Iteration $i / $ITERATIONS"
    echo "========================================"
    
    RESULT_FILE="${SCENARIO_DIR}/${SCENARIO}_${i}.json"
    START_TIME=$(date)
    
    echo "Start time: $START_TIME"
    echo "Result will be saved to: $RESULT_FILE"
    echo ""
    
    # Start test
    echo "Starting test..."
    TEST_RESPONSE=$(kubectl exec -it $LOAD_TESTER_POD -- curl -s -X POST \
        "${CONTROL_API}/start" \
        -H 'Content-Type: application/json' \
        -d "{
            \"iterations\": 1,
            \"duration\": ${DURATION},
            \"scenario\": \"${SCENARIO}\",
            \"result_file\": \"${RESULT_FILE}\"
        }")
    
    echo "API Response: $TEST_RESPONSE"
    echo ""
    
    # Wait for completion
    WAIT_TIME=$(($DURATION + 60))
    echo "Waiting ${WAIT_TIME}s for test to complete..."
    
    # Show progress bar
    for j in $(seq 1 $(($WAIT_TIME / 10))); do
        sleep 10
        echo -n "."
    done
    echo ""
    
    END_TIME=$(date)
    echo "End time: $END_TIME"
    echo ""
    
    # Verify result
    if kubectl exec -it $LOAD_TESTER_POD -- test -f "$RESULT_FILE"; then
        echo "✅ Result saved successfully!"
        echo ""
        echo "Quick summary:"
        kubectl exec -it $LOAD_TESTER_POD -- cat "$RESULT_FILE" | jq -C '{
            scenario,
            iteration,
            duration,
            total_requests,
            successful_requests,
            failed_requests,
            success_rate,
            avg_response_time
        }' 2>/dev/null || echo "   (Could not parse JSON)"
    else
        echo "⚠️  WARNING: Result file not created!"
        echo "Checking Control API status..."
        kubectl exec -it $LOAD_TESTER_POD -- curl -s "${CONTROL_API}/status" | jq -C
    fi
    
    echo ""
    echo "Pausing 10s before next iteration..."
    sleep 10
    echo ""
done

# Summary
echo "========================================"
echo "SCENARIO COMPLETE!"
echo "========================================"
echo "Scenario: $SCENARIO"
echo "Results directory: $SCENARIO_DIR"
echo ""

# Count saved files
SAVED_COUNT=$(kubectl exec -it $LOAD_TESTER_POD -- sh -c \
    "ls ${SCENARIO_DIR}/*.json 2>/dev/null | wc -l" | tr -d '\r' | xargs)

echo "Files saved: $SAVED_COUNT / $ITERATIONS"
echo ""

if [ "$SAVED_COUNT" -eq "$ITERATIONS" ]; then
    echo "✅ All iterations completed successfully!"
else
    echo "⚠️  Some iterations may have failed."
    echo "Check logs and verify files."
fi

echo ""
echo "Download results:"
SCENARIO_NAME=$(basename "$SCENARIO_DIR")
echo "  kubectl cp load-tester:${SCENARIO_DIR} ./${SCENARIO_NAME}"
echo ""
echo "Verify files:"
echo "  kubectl exec -it $LOAD_TESTER_POD -- ls -lh $SCENARIO_DIR"
echo ""
