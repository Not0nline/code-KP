#!/bin/bash

# INDIVIDUAL MODEL MARATHON (2-MODEL ENSEMBLE STYLE)
# Usage: ./run_individual_tests.sh [SUDO_PREFIX]

SUDO_CMD=$1
RESULTS_DIR="/home/ubuntu/test_results_final/individual"
SCENARIOS=("low" "medium" "high")
MAX_COUNT=10

echo "=== STARTING 2-MODEL ENSEMBLE MARATHON ==="
echo "Results will be saved to: $RESULTS_DIR"

mkdir -p "$RESULTS_DIR"

# 1. Create Model Config (Only GRU + Holt-Winters)
echo "------------------------------------------------"
echo "≡ƒöä CONFIGURING SYSTEM: GRU + HOLT-WINTERS ONLY"
echo "------------------------------------------------"
cat <<EOC > model-config.yaml
models:
  gru: true
  holt_winters: true
  lstm: false
  lightgbm: false
  xgboost: false
  statuscale: false
  arima: false
  cnn: false
  autoencoder: false
  prophet: false
  ensemble: false
EOC

# 2. Apply ConfigMap
echo "Applying ConfigMap..."
$SUDO_CMD kubectl create configmap model-config --from-file=model-config.yaml --dry-run=client -o yaml | $SUDO_CMD kubectl apply -f - 

# 3. Run Tests
for SCENARIO in "${SCENARIOS[@]}"; do
    for i in $(seq -f "%02g" 1 $MAX_COUNT); do
        TEST_NAME="individual-${SCENARIO}-${i}"
        TARGET="combined"

        if [ -d "${RESULTS_DIR}/${TEST_NAME}" ]; then
            echo "Γ£à ${TEST_NAME} already exists. Skipping."
            continue
        fi

        echo "------------------------------------------------"
echo "≡ƒÜÇ STARTING TEST: ${TEST_NAME}"
echo "------------------------------------------------"

        # --- STEP 1: CLEAN SLATE ---
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
        if [ ! -z "$SCALER_POD" ]; then
            $SUDO_CMD kubectl exec $SCALER_POD -- rm -rf /data/traffic_data.csv /data/gru_model.h5 /data/scaler_X.pkl /data/scaler_y.pkl
        fi

        # --- STEP 2: RESTART SCALER ---
        $SUDO_CMD kubectl rollout restart deployment predictive-scaler
        $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
        sleep 5
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')

        echo "+++ Injecting baseline data into new pod +++"
        $SUDO_CMD kubectl exec $SCALER_POD -- mkdir -p /data/baselines
        $SUDO_CMD kubectl cp /home/ubuntu/code-KP/baseline_backups/fair_2000/baseline_low.json $SCALER_POD:/data/baselines/baseline_low.json
        $SUDO_CMD kubectl cp /home/ubuntu/code-KP/baseline_backups/fair_2000/baseline_medium.json $SCALER_POD:/data/baselines/baseline_medium.json
        $SUDO_CMD kubectl cp /home/ubuntu/code-KP/baseline_backups/fair_2000/baseline_high.json $SCALER_POD:/data/baselines/baseline_high.json
        echo "+++ Verifying baseline data in pod... +++"
        $SUDO_CMD kubectl exec $SCALER_POD -- ls -l /data/baselines/

        # --- STEP 3: LOAD BASELINE ---
        $SUDO_CMD kubectl exec $SCALER_POD -- python -c "import requests; print(requests.post('http://localhost:5000/api/baseline/load', json={'scenario': '$SCENARIO'}).text)"

        # --- STEP 4: TRAIN MODELS ---
        $SUDO_CMD kubectl cp trigger_training.sh $SCALER_POD:/tmp/trigger_training.sh && $SUDO_CMD kubectl exec $SCALER_POD -- /bin/bash /tmp/trigger_training.sh

        # --- STEP 5: WAIT FOR TRAINING ---
        echo "ΓÅ│ Waiting 300s for training..."
        sleep 300

        # --- STEP 6: EXECUTE LOAD TEST ---
        $SUDO_CMD kubectl delete pod $TEST_NAME --ignore-not-found=true --force --grace-period=0
        sleep 5
        $SUDO_CMD kubectl run $TEST_NAME \
            --image=4dri41/stress-test:latest \
            --restart=Never \
            --labels="app=load-tester" \
            --env="TARGET=$TARGET" \
            --env="SCENARIO=$SCENARIO" \
            --env="DURATION=1800" \
            --env="TEST_NAME=$TEST_NAME" \
            --env="TEST_PREDICTIVE=true" \
            --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 300"

        # Wait for completion
        PHASE="Unknown"
        while true; do
            if $SUDO_CMD kubectl logs $TEST_NAME 2>/dev/null | grep -q "TEST_FINISHED_MARKER"; then
                PHASE="Succeeded"
                break
            fi
            POD_STATUS=$($SUDO_CMD kubectl get pod $TEST_NAME -o jsonpath='{.status.phase}' 2>/dev/null)
            if [ "$POD_STATUS" == "Failed" ] || [ "$POD_STATUS" == "Unknown" ]; then
                PHASE="Failed"
                break
            fi
            sleep 60
        done

        # --- STEP 7: SAVE RESULTS ---
        if [ "$PHASE" == "Succeeded" ]; then
            mkdir -p "${RESULTS_DIR}/${TEST_NAME}"
            $SUDO_CMD kubectl cp "${TEST_NAME}:/app/load_test_results" "${RESULTS_DIR}/${TEST_NAME}/"
            
            SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
            if [ -n "$SCALER_POD" ]; then
                $SUDO_CMD kubectl cp "default/$SCALER_POD:/data/metrics_export.txt" "${RESULTS_DIR}/${TEST_NAME}/predictive_scaler_metrics.txt"
            fi
            
            # Cleanup
            $SUDO_CMD kubectl delete pod $TEST_NAME --grace-period=0 --force
        fi
        sleep 10
    done
done
