#!/bin/bash
# Verify test completion and saved results
# Usage: ./verify_tests.sh [session_dir]

LOAD_TESTER_POD="deployment/load-tester"

echo "========================================"
echo "Test Results Verification"
echo "========================================"
echo ""

# Find latest session directory if not specified
if [ -z "$1" ]; then
    echo "Finding latest test session..."
    LATEST_SESSION=$(kubectl exec -it $LOAD_TESTER_POD -- sh -c \
        "ls -td /data/test_results/session_* 2>/dev/null | head -n1" 2>&1 | tr -d '\r')
    
    if [ -z "$LATEST_SESSION" ] || [[ "$LATEST_SESSION" == *"No such file"* ]]; then
        echo "❌ No test sessions found in /data/test_results/"
        echo ""
        echo "Checking /data directory:"
        kubectl exec -it $LOAD_TESTER_POD -- ls -lah /data/
        exit 1
    fi
    
    SESSION_DIR="$LATEST_SESSION"
else
    SESSION_DIR="$1"
fi

echo "📁 Session directory: $SESSION_DIR"
echo ""

# Count results per scenario
echo "Test Completion Status:"
echo "----------------------------------------"

for SCENARIO in low medium high; do
    COUNT=$(kubectl exec -it $LOAD_TESTER_POD -- sh -c \
        "ls ${SESSION_DIR}/${SCENARIO}/${SCENARIO}_*.json 2>/dev/null | wc -l" 2>&1 | tr -d '\r' | xargs)
    
    if [ "$COUNT" -eq 10 ]; then
        STATUS="✅"
        COLOR="\033[0;32m"  # Green
    elif [ "$COUNT" -gt 0 ]; then
        STATUS="⚠️"
        COLOR="\033[0;33m"  # Yellow
    else
        STATUS="❌"
        COLOR="\033[0;31m"  # Red
    fi
    
    echo -e "${COLOR}${STATUS} $(echo $SCENARIO | tr '[:lower:]' '[:upper:]'): $COUNT / 10 tests\033[0m"
    
    if [ "$COUNT" -gt 0 ]; then
        echo "   Files:"
        kubectl exec -it $LOAD_TESTER_POD -- ls -lh ${SESSION_DIR}/${SCENARIO}/ 2>/dev/null | grep ".json" | sed 's/^/     /'
    fi
    echo ""
done

# Total count
TOTAL=$(kubectl exec -it $LOAD_TESTER_POD -- sh -c \
    "find ${SESSION_DIR} -name '*.json' -type f 2>/dev/null | wc -l" 2>&1 | tr -d '\r' | xargs)

echo "========================================"
echo "TOTAL: $TOTAL / 30 tests completed"
echo "========================================"
echo ""

# Show directory structure
echo "Directory Structure:"
echo "----------------------------------------"
kubectl exec -it $LOAD_TESTER_POD -- ls -lR "$SESSION_DIR"
echo ""

# Download instructions
echo "========================================"
echo "Download Results:"
echo "========================================"
SESSION_NAME=$(basename "$SESSION_DIR")
echo "kubectl cp load-tester:${SESSION_DIR} ./${SESSION_NAME}"
echo ""

# Next steps
if [ "$TOTAL" -eq 30 ]; then
    echo "✅ All tests complete! Ready for analysis."
elif [ "$TOTAL" -eq 0 ]; then
    echo "❌ No tests completed. Check logs and restart."
else
    echo "⚠️  Tests partially complete. Resume or restart?"
fi
