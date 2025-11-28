#!/bin/bash

# INDIVIDUAL MODEL MARATHON (MINHEAP STYLE)
# Usage: ./run_individual_tests.sh [SUDO_PREFIX]
# Runs 10 iterations of Low, Medium, High scenarios.
# Enables ALL 11 models simultaneously (MinHeap) to collect data for all of them in parallel.

SUDO_CMD=$1
RESULTS_DIR="/home/ubuntu/test_results_final/individual"
SCENARIOS=("low" "medium" "high")
MAX_COUNT=10

echo "=== STARTING INDIVIDUAL MODEL MARATHON (MINHEAP STYLE) ==="
echo "Results will be saved to: $RESULTS_DIR"
echo "Configuration: All 11 models enabled, running concurrently."

mkdir -p "$RESULTS_DIR"

# 1. Create Model Config (Enable ALL models)
echo "------------------------------------------------"
echo "≡ƒöä CONFIGURING SYSTEM: ENABLING ALL 11 MODELS"
echo "------------------------------------------------"
cat <<EOF > model-config.yaml
models:
  gru: true
  holt_winters: true
  lstm: true
  lightgbm: true
  xgboost: true
  statuscale: true
  arima: true
  cnn: true
  autoencoder: true
  prophet: true
  ensemble: true
EOF

# 2. Apply ConfigMap
echo "Applying ConfigMap..."
$SUDO_CMD kubectl create configmap model-config --from-file=model-config.yaml --dry-run=client -o yaml | $SUDO_CMD kubectl apply -f -

# 3. Run Tests
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

        # --- STEP 1: CLEAN SLATE ---
        echo "≡ƒº╣ [Phase 1] Cleaning previous model state..."
        
        # Find Scaler Pod
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
        
        if [ ! -z "$SCALER_POD" ]; then
            # Delete persistent data
            echo "   Purging /data/ inside $SCALER_POD..."
            $SUDO_CMD kubectl exec $SCALER_POD -- rm -rf /data/traffic_data.csv /data/gru_model.h5 /data/scaler_X.pkl /data/scaler_y.pkl
        fi

        # --- STEP 2: RESTART SCALER ---
        echo "≡ƒöä [Phase 2] Restarting Scaler to load config..."
        $SUDO_CMD kubectl rollout restart deployment predictive-scaler
        $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
        
        # Get new pod name
        sleep 5
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
        echo "   New Scaler Pod: $SCALER_POD"

        # --- STEP 3: LOAD BASELINE ---
        echo "≡ƒôÑ [Phase 3] Loading Baseline Dataset ($SCENARIO)..."
        # Use python inside the pod to call the API
        $SUDO_CMD kubectl exec $SCALER_POD -- python -c "import requests; print(requests.post('http://localhost:5000/api/baseline/load', json={'scenario': '$SCENARIO'}).text)"

        # --- STEP 4: TRAIN MODELS ---
        echo "≡ƒºá [Phase 4] Triggering Training for ALL 11 Models..."
        $SUDO_CMD kubectl cp trigger_training.sh $SCALER_POD:/tmp/trigger_training.sh && $SUDO_CMD kubectl exec $SCALER_POD -- /bin/bash /tmp/trigger_training.sh
        
        # --- STEP 5: WAIT FOR TRAINING ---
        echo "ΓÅ│ [Phase 5] Waiting 300s (5 mins) for training to complete..."
        sleep 900

        # --- STEP 6: EXECUTE LOAD TEST ---
        echo "≡ƒöÑ [Phase 6] Launching Load Test Pod..."
        
        # FORCE DELETE EXISTING POD IF IT EXISTS
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
            --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 300"

        # Wait for completion (Log Check)
        echo "ΓÅ│ [Phase 7] Test running (30 mins)..."
        PHASE="Unknown"
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

        # --- STEP 7: SAVE RESULTS ---
        if [ "$PHASE" == "Succeeded" ]; then
            echo "≡ƒÆ╛ [Phase 8] Saving results to persistent storage..."
            mkdir -p "${RESULTS_DIR}/${TEST_NAME}"
            $SUDO_CMD kubectl cp "${TEST_NAME}:/app/load_test_results" "${RESULTS_DIR}/${TEST_NAME}/"
            
            # ADD THIS PART: Copy metrics from the predictive-scaler pod
            SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
            if [ -n "$SCALER_POD" ]; then
                echo "≡ƒöÑ Copying metrics from $SCALER_POD..."
                $SUDO_CMD kubectl cp "default/$SCALER_POD:/data/metrics_export.txt" "${RESULTS_DIR}/${TEST_NAME}/predictive_scaler_metrics.txt"
            else
                echo "ΓÜá∩╕Å Could not find predictive-scaler pod, skipping metrics copy."
            fi
            
            # Verify
            if [ "$(ls -A ${RESULTS_DIR}/${TEST_NAME})" ]; then
                echo "Γ£à Results verified on disk."
                # --- STEP 8: CLEANUP ---
                echo "≡ƒùæ∩╕Å [Phase 9] Deleting test pod..."
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

echo "≡ƒÄë≡ƒÄë≡ƒÄë ALL INDIVIDUAL TESTS COMPLETED ≡ƒÄë≡ƒÄë≡ƒÄë"
