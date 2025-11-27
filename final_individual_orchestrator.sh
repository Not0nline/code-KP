#!/bin/bash

# INDIVIDUAL MODEL MARATHON (FINAL)
# Usage: ./run_individual_tests.sh [SUDO_PREFIX]

SUDO_CMD=$1
RESULTS_DIR="/home/ubuntu/test_results_final/individual"
SCENARIOS=("low" "medium" "high")
MAX_COUNT=10

echo "=== STARTING INDIVIDUAL MODEL MARATHON (FINAL) ==="
echo "Results will be saved to: $RESULTS_DIR"

for SCENARIO in "${SCENARIOS[@]}"; do
    for i in $(seq -f "%02g" 1 $MAX_COUNT); do
        TEST_NAME="individual-${SCENARIO}-${i}"
        TARGET="combined"

        # Check if already done
        if [ -d "${RESULTS_DIR}/${TEST_NAME}" ]; then
            echo "Γ£à ${TEST_NAME} already exists. Skipping."
            continue
        fi

        echo "------------------------------------------------"
        echo "≡ƒÜÇ STARTING TEST: ${TEST_NAME}"
        echo "------------------------------------------------"

        # --- STEP 1: RESTART SCALER ---
        echo "≡ƒöä [Phase 1] Restarting Scaler to ensure clean state..."
        $SUDO_CMD kubectl rollout restart deployment predictive-scaler
        $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
        sleep 5
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
        echo "   New Scaler Pod: $SCALER_POD"

        # --- STEP 2: ATOMIC LOAD AND TRAIN ---
        echo "≡ƒôÑ [Phase 2] Atomically loading baseline and triggering training for scenario ($SCENARIO)..."
        $SUDO_CMD kubectl exec $SCALER_POD -- python -c "import requests; print(requests.post('http://localhost:5000/api/load_and_train', json={'scenario': '$SCENARIO'}).text)"

        # --- STEP 3: WAIT FOR TRAINING ---
        echo "ΓÅ│ [Phase 3] Waiting 300s (5 mins) for training to complete..."
        sleep 300

        # --- STEP 4: EXECUTE LOAD TEST ---
        echo "≡ƒöÑ [Phase 4] Launching Load Test Pod..."
        $SUDO_CMD kubectl delete pod $TEST_NAME --ignore-not-found=true --force --grace-period=0
        sleep 5

        $SUDO_CMD kubectl run $TEST_NAME \
            --image=4dri41/stress-test:latest \
            --restart=Never \
            --env="TARGET=$TARGET" \
            --env="SCENARIO=$SCENARIO" \
            --env="DURATION=1800" \
            --env="TEST_NAME=$TEST_NAME" \
            --env="TEST_PREDICTIVE=true" \
            --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 3600"

        # --- STEP 5: Wait for completion ---
        echo "ΓÅ│ [Phase 5] Test running (30 mins)..."
        PHASE="Unknown"
        while true; do
            if $SUDO_CMD kubectl logs $TEST_NAME 2>/dev/null | grep -q "TEST_FINISHED_MARKER"; then
                echo "Γ£à Test Completed (Log marker found)!"
                PHASE="Succeeded"
                break
            fi
            POD_STATUS=$($SUDO_CMD kubectl get pod $TEST_NAME -o jsonpath='{.status.phase}' 2>/dev/null)
            if [ "$POD_STATUS" == "Failed" ] || [ "$POD_STATUS" == "Unknown" ]; then
                echo "Γ¥î Test Failed (Pod Status: $POD_STATUS)!"
                PHASE="Failed"
                break
            fi
            sleep 60
        done

        # --- STEP 6: SAVE RESULTS ---
        if [ "$PHASE" == "Succeeded" ]; then
            echo "≡ƒÆ╛ [Phase 6] Saving results..."
            mkdir -p "${RESULTS_DIR}/${TEST_NAME}"
            $SUDO_CMD kubectl cp "${TEST_NAME}:/app/load_test_results" "${RESULTS_DIR}/${TEST_NAME}/"
            echo "≡ƒùæ∩╕Å [Phase 7] Deleting test pod..."
            $SUDO_CMD kubectl delete pod $TEST_NAME --grace-period=0 --force
        else
            echo "ΓÜá∩╕Å Skipping save due to failure."
        fi

        echo "zzz Sleeping 10s before next cycle..."
        sleep 10
    done
done

echo "≡ƒÄë≡ƒÄë≡ƒÄë ALL INDIVIDUAL TESTS COMPLETED ≡ƒÄë≡ƒÄë≡ƒÄë"
