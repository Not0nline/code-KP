#!/bin/bash
# Run a single scenario test in BACKGROUND
# Usage: ./run_single_scenario_bg.sh <low|medium|high> [iterations] [duration]

SCENARIO=$1
ITERATIONS=${2:-10}
DURATION=${3:-1800}

if [ -z "$SCENARIO" ]; then
    echo "Usage: ./run_single_scenario_bg.sh <low|medium|high> [iterations] [duration]"
    exit 1
fi

if [[ ! "$SCENARIO" =~ ^(low|medium|high)$ ]]; then
    echo "❌ Invalid scenario. Must be: low, medium, or high"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCENARIO}_test_${TIMESTAMP}.log"

echo "========================================"
echo "Starting ${SCENARIO} test in BACKGROUND"
echo "========================================"
echo "Scenario:   $SCENARIO"
echo "Iterations: $ITERATIONS"
echo "Duration:   ${DURATION}s per test"
echo "Log file:   $LOG_FILE"
echo "========================================"
echo ""

# Run in background with nohup
nohup ./run_single_scenario.sh "$SCENARIO" "$ITERATIONS" "$DURATION" > "$LOG_FILE" 2>&1 &

PID=$!
echo "✅ Test started in background!"
echo "Process ID: $PID"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if still running:"
echo "  ps -p $PID"
echo ""
echo "Verify results:"
echo "  ./verify_tests.sh"
echo ""
echo "Stop test (if needed):"
echo "  kill $PID"
echo ""

# Save PID for reference
echo $PID > "${SCENARIO}_test.pid"
echo "PID saved to: ${SCENARIO}_test.pid"
