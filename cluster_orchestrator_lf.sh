#!/bin/bash

# CLUSTER ORCHESTRATOR - AUTONOMOUS TESTING AGENT
# Usage: ./cluster_orchestrator.sh <TEST_TYPE> [SUDO_PREFIX]
# TEST_TYPE: ensemble, hpa, individual, inverse

TEST_TYPE=$1
SUDO_CMD=$2
RESULTS_DIR="/home/ubuntu/test_results_final/${TEST_TYPE}"
SCENARIOS=("low" "medium" "high")
MAX_COUNT=10

# Validate Input
if [ -z "$TEST_TYPE" ]; then
    echo "Usage: $0 <ensemble|hpa|individual|inverse> [sudo]"
    exit 1
fi

mkdir -p "$RESULTS_DIR"
echo "=== STARTING AUTONOMOUS TEST CYCLE FOR: $TEST_TYPE ==="
echo "Results will be saved to: $RESULTS_DIR"

# Define scenarios based on test type
if [ "$TEST_TYPE" == "inverse" ]; then
    # Custom schedule for Inverse: HPA (High) -> Ensemble (Low)
    # We will construct a custom loop or logic here
    echo "≡ƒöä Running INVERSE Mode: HPA (High) + Ensemble (Low)"
    
    # Define the specific tests to run
    # Format: "TYPE:SCENARIO"
    TEST_QUEUE=("hpa:high" "ensemble:low")
else
    # Standard mode: Run all scenarios for the given type
    TEST_QUEUE=()
    for SCENARIO in "${SCENARIOS[@]}"; do
        TEST_QUEUE+=("$TEST_TYPE:$SCENARIO")
    done
fi

for ITEM in "${TEST_QUEUE[@]}"; do
    # Parse Type and Scenario
    CURRENT_TYPE=${ITEM%%:*}
    SCENARIO=${ITEM##*:}
    
    for i in $(seq -f "%02g" 1 $MAX_COUNT); do
        
        # Construct Test Name & Config
        if [ "$CURRENT_TYPE" == "hpa" ]; then
            TEST_NAME="hpa-test-${SCENARIO}-${i}"
            TARGET="hpa"
            PREDICTIVE="false"
        elif [ "$CURRENT_TYPE" == "individual" ]; then
            TEST_NAME="individual-${SCENARIO}-${i}"
            TARGET="combined"
            PREDICTIVE="true"
        elif [ "$CURRENT_TYPE" == "ensemble" ]; then
            TEST_NAME="ensemble-${SCENARIO}-${i}"
            TARGET="combined"
            PREDICTIVE="true"
        else
             echo "Unknown type: $CURRENT_TYPE"
             continue
        fi

        # Check if already done
        if [ -d "${RESULTS_DIR}/${TEST_NAME}" ]; then
            echo "Γ£à ${TEST_NAME} already exists. Skipping."
            continue
        fi

        echo "------------------------------------------------"
        echo "≡ƒÜÇ STARTING TEST: ${TEST_NAME}"
        echo "------------------------------------------------"

        # --- STEP 1-3 & 7-8: CLEAN SLATE & TRAIN (Only for Predictive) ---
        if [ "$PREDICTIVE" == "true" ]; then
            echo "≡ƒº╣ [Phase 1] Cleaning previous model state..."
            
            # 1. Find Scaler Pod
            SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
            
            if [ ! -z "$SCALER_POD" ]; then
                # 2. Delete persistent data to force fresh baseline load
                echo "   Purging /data/ inside $SCALER_POD..."
                $SUDO_CMD kubectl exec $SCALER_POD -- rm -rf /data/traffic_data.csv /data/gru_model.h5 /data/scaler_X.pkl /data/scaler_y.pkl
            fi

            # 3. Restart Scaler to trigger fresh training
            echo "≡ƒöä [Phase 2] Restarting Scaler to load baseline & train..."
            $SUDO_CMD kubectl rollout restart deployment predictive-scaler
            $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
            
            # 4. Wait for training to complete (approx 60s buffer)
            echo "ΓÅ│ [Phase 3] Waiting 60s for model training..."
            sleep 60
        fi

        # --- STEP 4: EXECUTE LOAD TEST ---
        echo "≡ƒöÑ [Phase 4] Launching Load Test Pod..."
        
        # FORCE DELETE EXISTING POD IF IT EXISTS
        echo "≡ƒº╣ Cleaning up any existing pod with name $TEST_NAME..."
        $SUDO_CMD kubectl delete pod $TEST_NAME --ignore-not-found=true --force --grace-period=0
        sleep 5

        $SUDO_CMD kubectl run $TEST_NAME \
            --image=4dri41/stress-test:latest \
            --restart=Never \
            --env="TARGET=$TARGET" \
            --env="SCENARIO=$SCENARIO" \
            --env="DURATION=1800" \
            --env="TEST_NAME=$TEST_NAME" \
            --env="TEST_PREDICTIVE=$PREDICTIVE" \
            --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 3600"

        # Wait for completion (Log Check)
        echo "ΓÅ│ [Phase 5] Test running (30 mins)..."
        while true; do
            # Check for completion marker
            if $SUDO_CMD kubectl logs $TEST_NAME 2>/dev/null | grep -q "TEST_FINISHED_MARKER"; then
                echo "Γ£à Test Completed (Log marker found)!"
                PHASE="Succeeded"
                break
            fi

            # Check for Pod Failure
            POD_STATUS=$($SUDO_CMD kubectl get pod $TEST_NAME -o jsonpath='{.status.phase}' 2>/dev/null)
            if [ "$POD_STATUS" == "Failed" ] || [ "$POD_STATUS" == "Unknown" ]; then
                echo "Γ¥î Test Failed (Pod Status: $POD_STATUS)!"
                PHASE="Failed"
                break
            fi
            
            sleep 60 # Check every minute
        done

        # --- STEP 5: SAVE RESULTS ---
        if [ "$PHASE" == "Succeeded" ]; then
            echo "≡ƒÆ╛ [Phase 6] Saving results to persistent storage..."
            mkdir -p "${RESULTS_DIR}/${TEST_NAME}"
            $SUDO_CMD kubectl cp "${TEST_NAME}:/app/load_test_results" "${RESULTS_DIR}/${TEST_NAME}/"
            
            # Verify
            if [ -f "${RESULTS_DIR}/${TEST_NAME}/combined_results.csv" ] || [ -f "${RESULTS_DIR}/${TEST_NAME}/hpa_results.csv" ] || [ "$(ls -A ${RESULTS_DIR}/${TEST_NAME})" ]; then
                echo "Γ£à Results verified on disk."
                # --- STEP 6: CLEANUP ---
                echo "≡ƒùæ∩╕Å [Phase 7] Deleting test pod..."
                $SUDO_CMD kubectl delete pod $TEST_NAME --grace-period=0 --force
            else
                echo "ΓÜá∩╕Å WARNING: Results missing or empty!"
            fi
        else
            echo "ΓÜá∩╕Å Skipping save due to failure."
        fi
        
        echo "zzz Sleeping 10s before next cycle..."
        sleep 10
    done
done

echo "≡ƒÄë≡ƒÄë≡ƒÄë ALL TESTS COMPLETED FOR $TEST_TYPE ≡ƒÄë≡ƒÄë≡ƒÄë"
