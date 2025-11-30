#!/bin/bash

SUDO_CMD=$1
RESULTS_DIR="/home/ubuntu/test_results_final/ensemble"
SCENARIOS=("low" "medium" "high")
MAX_COUNT=10

echo "=== STARTING ENSEMBLE (GRU + Holt-Winters) MARATHON ==="
echo "Results will be saved to: $RESULTS_DIR"

mkdir -p "$RESULTS_DIR"

echo "------------------------------------------------"
echo "CONFIGURING SYSTEM: ENABLING ENSEMBLE MODELS"
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
  ensemble: true
EOC

# Apply ConfigMap
echo "Applying ConfigMap..."
$SUDO_CMD kubectl create configmap model-config --from-file=model-config.yaml --dry-run=client -o yaml | $SUDO_CMD kubectl apply -f -
# Run Tests
for SCENARIO in "${SCENARIOS[@]}"; do
    for i in $(seq -f "%02g" 1 $MAX_COUNT); do
        TEST_NAME="ensemble-${SCENARIO}-${i}"
        TARGET="combined"
        if [ -d "${RESULTS_DIR}/${TEST_NAME}" ]; then
            echo "${TEST_NAME} already exists. Skipping."
            continue
        fi
        echo "------------------------------------------------"
        echo "STARTING TEST: ${TEST_NAME}"
        echo "------------------------------------------------"
        $SUDO_CMD kubectl rollout restart deployment predictive-scaler
        $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
        sleep 5
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
        echo "+++ Injecting baseline data into new pod +++"
        $SUDO_CMD kubectl exec $SCALER_POD -- mkdir -p /data/baselines
        $SUDO_CMD kubectl cp /home/ubuntu/code-KP/baseline_backups/fair_2000/baseline_low.json $SCALER_POD:/data/baselines/baseline_low.json        
        $SUDO_CMD kubectl cp /home/ubuntu/code-KP/baseline_backups/fair_2000/baseline_medium.json $SCALER_POD:/data/baselines/baseline_medium.json  
        $SUDO_CMD kubectl cp /home/ubuntu/code-KP/baseline_backups/fair_2000/baseline_high.json $SCALER_POD:/data/baselines/baseline_high.json      
        $SUDO_CMD kubectl exec $SCALER_POD -- python -c "import requests; print(requests.post('http://localhost:5000/api/baseline/load',json={'scenario': '$SCENARIO'}).text)"
        $SUDO_CMD kubectl exec $SCALER_POD -- python -c "import requests; print(requests.post('http://localhost:5000/api/models/train_all').text)"  
        echo "Waiting 300s for training..."
        sleep 300
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
        if [ "$PHASE" == "Succeeded" ]; then
            mkdir -p "${RESULTS_DIR}/${TEST_NAME}"
            $SUDO_CMD kubectl cp "${TEST_NAME}:/app/load_test_results" "${RESULTS_DIR}/${TEST_NAME}/"
            SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
            if [ -n "$SCALER_POD" ]; then
                   $SUDO_CMD kubectl cp "default/$SCALER_POD:/data/metrics_export.txt" "${RESULTS_DIR}/${TEST_NAME}/predictive_scaler_metrics.txt"     
               fi
                  $SUDO_CMD kubectl delete pod $TEST_NAME --grace-period=0 --force
           fi
           sleep 10
       done
   do