#!/bin/bash
# INVERSE LOAD ORCHESTRATOR
# Usage: ./inverse_load_orchestrator.sh [sudo]

SUDO_CMD=$1
RESULTS_DIR_HPA="/home/ubuntu/test_results_final/hpa"
RESULTS_DIR_ENSEMBLE="/home/ubuntu/test_results_final/ensemble"
MAX_COUNT=10

# Define Batches: HPA_SCENARIO ENSEMBLE_SCENARIO
BATCHES=(
    "low high"
    "medium medium"
    "high low"
)

mkdir -p "$RESULTS_DIR_HPA"
mkdir -p "$RESULTS_DIR_ENSEMBLE"

for BATCH in "${BATCHES[@]}"; do
    read -r HPA_SCENARIO ENSEMBLE_SCENARIO <<< "$BATCH"
    
    echo "=== STARTING BATCH: HPA=$HPA_SCENARIO | ENSEMBLE=$ENSEMBLE_SCENARIO ==="

    for i in $(seq -f "%02g" 1 $MAX_COUNT); do
        
        HPA_TEST_NAME="hpa-test-${HPA_SCENARIO}-${i}"
        ENSEMBLE_TEST_NAME="ensemble-${ENSEMBLE_SCENARIO}-${i}"
        
        # Check if both already done (skip if both exist)
        if [ -d "${RESULTS_DIR_HPA}/${HPA_TEST_NAME}" ] && [ -d "${RESULTS_DIR_ENSEMBLE}/${ENSEMBLE_TEST_NAME}" ]; then
             echo "Skipping ${HPA_TEST_NAME} and ${ENSEMBLE_TEST_NAME} (Already exist)"
             continue
        fi

        echo "------------------------------------------------"
        echo "STARTING PAIR: ${HPA_TEST_NAME} & ${ENSEMBLE_TEST_NAME}"
        echo "------------------------------------------------"

        # --- PREPARE ENSEMBLE (Predictive) ---
        echo "[Phase 1] Preparing Ensemble (Cleaning model state)..."
        SCALER_POD=$($SUDO_CMD kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}')
        if [ ! -z "$SCALER_POD" ]; then
            $SUDO_CMD kubectl exec $SCALER_POD -- rm -rf /data/traffic_data.csv /data/gru_model.h5 /data/scaler_X.pkl /data/scaler_y.pkl
        fi
        $SUDO_CMD kubectl rollout restart deployment predictive-scaler
        $SUDO_CMD kubectl rollout status deployment predictive-scaler --timeout=180s
        echo "Waiting 60s for model training..."
        sleep 60

        # --- LAUNCH TESTS ---
        echo "[Phase 2] Launching Test Pods..."
        
        # Cleanup existing pods
        $SUDO_CMD kubectl delete pod $HPA_TEST_NAME --ignore-not-found=true --force --grace-period=0
        $SUDO_CMD kubectl delete pod $ENSEMBLE_TEST_NAME --ignore-not-found=true --force --grace-period=0
        sleep 5

        # Launch HPA Test
        $SUDO_CMD kubectl run $HPA_TEST_NAME \
            --image=4dri41/stress-test:latest \
            --restart=Never \
            --env="TARGET=hpa" \
            --env="SCENARIO=$HPA_SCENARIO" \
            --env="DURATION=1800" \
            --env="TEST_NAME=$HPA_TEST_NAME" \
            --env="TEST_PREDICTIVE=false" \
            --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 3600" &

        # Launch Ensemble Test
        $SUDO_CMD kubectl run $ENSEMBLE_TEST_NAME \
            --image=4dri41/stress-test:latest \
            --restart=Never \
            --env="TARGET=combined" \
            --env="SCENARIO=$ENSEMBLE_SCENARIO" \
            --env="DURATION=1800" \
            --env="TEST_NAME=$ENSEMBLE_TEST_NAME" \
            --env="TEST_PREDICTIVE=true" \
            --command -- /bin/bash -c "python /app/load_test.py; echo 'TEST_FINISHED_MARKER'; sleep 3600" &

        wait # Wait for kubectl run commands to finish
        
        # Wait for completion
        echo "[Phase 3] Tests running (30 mins)..."
        
        HPA_DONE=false
        ENSEMBLE_DONE=false
        
        while true; do
            if [ "$HPA_DONE" = false ]; then
                if $SUDO_CMD kubectl logs $HPA_TEST_NAME 2>/dev/null | grep -q "TEST_FINISHED_MARKER"; then
                    echo "HPA Test Completed!"
                    HPA_DONE=true
                fi
            fi
            
            if [ "$ENSEMBLE_DONE" = false ]; then
                if $SUDO_CMD kubectl logs $ENSEMBLE_TEST_NAME 2>/dev/null | grep -q "TEST_FINISHED_MARKER"; then
                    echo "Ensemble Test Completed!"
                    ENSEMBLE_DONE=true
                fi
            fi
            
            if [ "$HPA_DONE" = true ] && [ "$ENSEMBLE_DONE" = true ]; then
                break
            fi
            
            sleep 60
        done

        # --- SAVE RESULTS ---
        echo "[Phase 4] Saving results..."
        
        mkdir -p "${RESULTS_DIR_HPA}/${HPA_TEST_NAME}"
        $SUDO_CMD kubectl cp "${HPA_TEST_NAME}:/app/load_test_results" "${RESULTS_DIR_HPA}/${HPA_TEST_NAME}/"
        
        mkdir -p "${RESULTS_DIR_ENSEMBLE}/${ENSEMBLE_TEST_NAME}"
        $SUDO_CMD kubectl cp "${ENSEMBLE_TEST_NAME}:/app/load_test_results" "${RESULTS_DIR_ENSEMBLE}/${ENSEMBLE_TEST_NAME}/"

        # --- CLEANUP ---
        $SUDO_CMD kubectl delete pod $HPA_TEST_NAME --grace-period=0 --force
        $SUDO_CMD kubectl delete pod $ENSEMBLE_TEST_NAME --grace-period=0 --force
        
        echo "Sleeping 10s..."
        sleep 10
    done
done
