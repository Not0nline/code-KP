#!/bin/bash

# INDIVIDUAL MODEL ORCHESTRATOR
# Usage: ./run_individual_tests.sh [SUDO_PREFIX]
# Runs 11 models sequentially, 3 scenarios each, 10 iterations each.

SUDO_CMD=$1
RESULTS_DIR="/home/ubuntu/test_results_final/individual"
MODELS=("gru" "lstm" "holt_winters" "lightgbm" "xgboost" "statuscale" "arima" "cnn" "autoencoder" "prophet" "ensemble")
SCENARIOS=("low" "medium" "high")
MAX_COUNT=10

echo "=== STARTING INDIVIDUAL MODEL MARATHON ==="
echo "Results will be saved to: $RESULTS_DIR"

mkdir -p "$RESULTS_DIR"

for MODEL in "${MODELS[@]}"; do
    echo "------------------------------------------------"
    echo "≡ƒöä CONFIGURING SYSTEM FOR MODEL: $MODEL"
    echo "------------------------------------------------"
    
    # 1. Create Model Config (Enable ONLY the current model)
    echo "Creating model-config.yaml..."
    cat <<EOF > model-config.yaml
models:
  gru: $( [ "$MODEL" == "gru" ] && echo "true" || echo "false" )
  holt_winters: $( [ "$MODEL" == "holt_winters" ] && echo "true" || echo "false" )
  lstm: $( [ "$MODEL" == "lstm" ] && echo "true" || echo "false" )
  lightgbm: $( [ "$MODEL" == "lightgbm" ] && echo "true" || echo "false" )
  xgboost: $( [ "$MODEL" == "xgboost" ] && echo "true" || echo "false" )
  statuscale: $( [ "$MODEL" == "statuscale" ] && echo "true" || echo "false" )
  arima: $( [ "$MODEL" == "arima" ] && echo "true" || echo "false" )
  cnn: $( [ "$MODEL" == "cnn" ] && echo "true" || echo "false" )
  autoencoder: $( [ "$MODEL" == "autoencoder" ] && echo "true" || echo "false" )
  prophet: $( [ "$MODEL" == "prophet" ] && echo "true" || echo "false" )
  ensemble: $( [ "$MODEL" == "ensemble" ] && echo "true" || echo "false" )
EOF

    # 2. Apply ConfigMap
    echo "Applying ConfigMap..."
    $SUDO_CMD kubectl create configmap model-config --from-file=model-config.yaml --dry-run=client -o yaml | $SUDO_CMD kubectl apply -f -
    
    # 3. Run Tests for this Model
    for SCENARIO in "${SCENARIOS[@]}"; do
        for i in $(seq -f "%02g" 1 $MAX_COUNT); do
            TEST_NAME="indiv-${MODEL}-${SCENARIO}-${i}"
            TARGET="combined"
            
            # Check if already done
            if [ -d "${RESULTS_DIR}/${MODEL}/${TEST_NAME}" ]; then
                echo "Γ£à ${TEST_NAME} already exists. Skipping."
                continue
            fi

            echo "------------------------------------------------"
            echo "≡ƒÜÇ STARTING TEST: ${TEST_NAME}"
            echo "------------------------------------------------"

            # --- STEP 1-3: CLEAN SLATE & TRAIN ---
            echo "≡ƒº╣ [Phase 1] Cleaning previous model state..."
            
            # Find Scaler Pod
            SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
            
            if [ ! -z "$SCALER_POD" ]; then
                # Delete persistent data to force fresh baseline load
                echo "   Purging /data/ inside $SCALER_POD..."
                $SUDO_CMD kubectl exec $SCALER_POD -- rm -rf /data/traffic_data.csv /data/gru_model.h5 /data/scaler_X.pkl /data/scaler_y.pkl
            fi

            # Restart Scaler to trigger fresh training AND load new ConfigMap
            echo "≡ƒöä [Phase 2] Restarting Scaler to load config & train..."
            $SUDO_CMD kubectl rollout restart deployment predictive-scaler
            $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
            
            # Wait for training to complete (approx 60s buffer)
            echo "ΓÅ│ [Phase 3] Waiting 60s for model training..."
            sleep 60

            # --- STEP 4: EXECUTE LOAD TEST ---
            echo "≡ƒöÑ [Phase 4] Launching Load Test Pod..."
            
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
                --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 3600"

            # Wait for completion (Log Check)
            echo "ΓÅ│ [Phase 5] Test running (30 mins)..."
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

            # --- STEP 5: SAVE RESULTS ---
            if [ "$PHASE" == "Succeeded" ]; then
                echo "≡ƒÆ╛ [Phase 6] Saving results to persistent storage..."
                mkdir -p "${RESULTS_DIR}/${MODEL}/${TEST_NAME}"
                $SUDO_CMD kubectl cp "${TEST_NAME}:/app/load_test_results" "${RESULTS_DIR}/${MODEL}/${TEST_NAME}/"
                
                # Verify
                if [ "$(ls -A ${RESULTS_DIR}/${MODEL}/${TEST_NAME})" ]; then
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
done

echo "≡ƒÄë≡ƒÄë≡ƒÄë ALL INDIVIDUAL TESTS COMPLETED ≡ƒÄë≡ƒÄë≡ƒÄë"
