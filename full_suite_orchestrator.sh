#!/bin/bash

# FULL SUITE ORCHESTRATOR
# Usage: ./full_suite_orchestrator.sh <CLUSTER_MODE> [SUDO_PREFIX]
# CLUSTER_MODE: CLUSTER2 (HPA + GRU/HW), CLUSTER3 (11 Models)

MODE=$1
SUDO_CMD=$2
BASE_DIR="/home/ubuntu/saved_tests"
SCENARIOS=("low" "medium" "high")
MAX_COUNT=10

if [ -z "$MODE" ]; then
    echo "Usage: $0 <CLUSTER2|CLUSTER3> [sudo]"
    exit 1
fi

# Function to run a test
run_test() {
    local TYPE=$1
    local SCENARIO=$2
    local ITERATION=$3
    local SAVE_PATH=$4
    local USE_PREDICTIVE=$5
    
    TEST_NAME="${TYPE}-${SCENARIO}-${ITERATION}"
    
    if [ -d "$SAVE_PATH/$TEST_NAME" ]; then
        echo "Γ£à $TEST_NAME already exists. Skipping."
        return
    fi

    echo "========================================================================"
    echo "≡ƒÜÇ STARTING TEST: $TEST_NAME"
    echo "≡ƒôé Target Path: $SAVE_PATH/$TEST_NAME"
    echo "========================================================================"

    # --- PREDICTIVE SCALER SETUP ---
    if [ "$USE_PREDICTIVE" == "true" ]; then
        echo "≡ƒöä [Phase 1] Restarting Scaler..."
        $SUDO_CMD kubectl rollout restart deployment predictive-scaler
        $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
        sleep 10 # Give it a moment to settle
        
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
        echo "   New Scaler Pod: $SCALER_POD"

        echo "≡ƒºá [Phase 2] Loading Baseline Dataset ($SCENARIO)..."
        $SUDO_CMD kubectl exec $SCALER_POD -- python -c "import requests; print(requests.post('http://localhost:5000/api/baseline/load', json={'scenario': '$SCENARIO'}).text)"
        
        echo "≡ƒºá [Phase 2b] Triggering Training..."
        $SUDO_CMD kubectl exec $SCALER_POD -- python -c "import requests; print(requests.post('http://localhost:5000/api/models/train_all').text)"

        echo "ΓÅ│ [Phase 3] Waiting 300s (5 mins) for training to complete..."
        sleep 300
    else
        # Ensure scaler is NOT interfering for HPA tests (Scale it to 0 or ensure HPA is active)
        # For safety, we can scale the predictive deployment to 0, but HPA tests usually target a different service/deployment ('product-app-hpa')
        # checking if we need to disable predictive scaler:
        # The load test script targets specific URLs. 'hpa' target hits 'product-app-hpa-service'.
        # Predictive scaler scales 'product-app-combined'.
        # So they shouldn't conflict, but let's be clean.
        echo "≡ƒÆñ [Phase 1] Skipping Scaler Setup (HPA Test)"
    fi

    # --- EXECUTE LOAD TEST ---
    echo "≡ƒöÑ [Phase 4] Launching Load Test Pod..."
    
    # Clean up old pod
    $SUDO_CMD kubectl delete pod $TEST_NAME --ignore-not-found=true --force --grace-period=0
    sleep 5

    # Determine TARGET env var
    if [ "$USE_PREDICTIVE" == "true" ]; then
        TARGET_ENV="combined"
    else
        TARGET_ENV="hpa"
    fi

    $SUDO_CMD kubectl run $TEST_NAME \
        --image=4dri4l/stress-test:latest \
        --image-pull-policy=Always \
        --restart=Never \
        --labels="app=load-tester" \
        --env="TARGET=$TARGET_ENV" \
        --env="SCENARIO=$SCENARIO" \
        --env="DURATION=1800" \
        --env="TEST_NAME=$TEST_NAME" \
        --env="TEST_PREDICTIVE=$USE_PREDICTIVE" \
        --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 3600"

    # Wait for completion
    echo "ΓÅ│ [Phase 5] Test running (30 mins)..."
    while true; do
        if $SUDO_CMD kubectl logs $TEST_NAME 2>/dev/null | grep -q "TEST_FINISHED_MARKER"; then
            echo "Γ£à Test Completed!"
            PHASE="Succeeded"
            break
        fi
        
        POD_STATUS=$($SUDO_CMD kubectl get pod $TEST_NAME -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$POD_STATUS" == "Failed" ] || [ "$POD_STATUS" == "Unknown" ]; then
            echo "Γ¥î Test Failed ($POD_STATUS)!"
            PHASE="Failed"
            break
        fi
        sleep 60
    done

    # --- SAVE RESULTS ---
    if [ "$PHASE" == "Succeeded" ]; then
        echo "≡ƒÆ╛ [Phase 6] Saving results..."
        mkdir -p "$SAVE_PATH/$TEST_NAME"
        $SUDO_CMD kubectl cp "${TEST_NAME}:/app/load_test_results" "$SAVE_PATH/$TEST_NAME/"
        
        if [ "$USE_PREDICTIVE" == "true" ]; then
            SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
            if [ -n "$SCALER_POD" ]; then
                echo "≡ƒöÑ Copying metrics from $SCALER_POD..."
                $SUDO_CMD kubectl cp "default/$SCALER_POD:/data/metrics_export.txt" "$SAVE_PATH/$TEST_NAME/predictive_scaler_metrics.txt"
            fi
        fi
        
        # Cleanup
        echo "≡ƒùæ [Phase 7] Deleting test pod..."
        $SUDO_CMD kubectl delete pod $TEST_NAME --grace-period=0 --force
    fi
    
    echo "zzz Sleeping 10s..."
    sleep 10
}

# === MAIN LOOP ===

if [ "$MODE" == "CLUSTER2" ]; then
    # Cluster 2: HPA (10x3) AND Holt_GRU (10x3)
    
    # 1. HPA Tests
    for SCENARIO in "${SCENARIOS[@]}"; do
        SAVE_DIR="$BASE_DIR/hpa/$SCENARIO"
        mkdir -p "$SAVE_DIR"
        for i in $(seq -f "%02g" 1 $MAX_COUNT);
 do
            run_test "hpa" "$SCENARIO" "$i" "$SAVE_DIR" "false"
        done
    done

    # 2. Holt_GRU Tests (Ensemble)
    for SCENARIO in "${SCENARIOS[@]}"; do
        SAVE_DIR="$BASE_DIR/holt_gru/$SCENARIO"
        mkdir -p "$SAVE_DIR"
        for i in $(seq -f "%02g" 1 $MAX_COUNT);
 do
            run_test "ensemble" "$SCENARIO" "$i" "$SAVE_DIR" "true"
        done
    done

elif [ "$MODE" == "CLUSTER3" ]; then
    # Cluster 3: 11 Models (10x3)
    
    for SCENARIO in "${SCENARIOS[@]}"; do
        SAVE_DIR="$BASE_DIR/11models/$SCENARIO"
        mkdir -p "$SAVE_DIR"
        for i in $(seq -f "%02g" 1 $MAX_COUNT);
 do
            run_test "11models" "$SCENARIO" "$i" "$SAVE_DIR" "true"
        done
    done
fi

echo "≡ƒÄë ALL TESTS COMPLETED FOR $MODE ≡ƒÄë"
