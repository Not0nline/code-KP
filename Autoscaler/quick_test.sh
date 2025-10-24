#!/bin/bash
# Quick Reference Script for Multi-Model Testing
# Usage: ./quick_test.sh [command]

AUTOSCALER_URL="http://predictive-scaler.default.svc.cluster.local:5000"
LOAD_TESTER_API="http://load-tester.default.svc.cluster.local:8080"

case "$1" in
    "check")
        echo "=== Checking Service Status ==="
        echo ""
        echo "Autoscaler:"
        curl -s ${AUTOSCALER_URL}/health | jq '.'
        echo ""
        echo "Load Tester:"
        curl -s ${LOAD_TESTER_API}/health | jq '.'
        ;;
        
    "list-baselines")
        echo "=== Available Baseline Datasets ==="
        curl -s ${AUTOSCALER_URL}/api/baseline/list | jq '.baselines'
        ;;
        
    "clear-data")
        echo "=== Clearing All Data ==="
        echo "⚠️  This will delete all traffic data, models, and predictions!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            curl -X POST ${AUTOSCALER_URL}/api/baseline/clear | jq '.'
        else
            echo "Cancelled."
        fi
        ;;
        
    "load-baseline")
        if [ -z "$2" ]; then
            echo "Usage: $0 load-baseline <scenario>"
            echo "Scenarios: low, medium, high"
            exit 1
        fi
        echo "=== Loading Baseline: $2 ==="
        curl -X POST ${AUTOSCALER_URL}/api/baseline/load \
            -H "Content-Type: application/json" \
            -d "{\"scenario\": \"$2\"}" | jq '.'
        ;;
        
    "save-baseline")
        if [ -z "$2" ]; then
            echo "Usage: $0 save-baseline <scenario>"
            echo "Scenarios: low, medium, high"
            exit 1
        fi
        echo "=== Saving Current Data as Baseline: $2 ==="
        curl -X POST ${AUTOSCALER_URL}/api/baseline/save \
            -H "Content-Type: application/json" \
            -d "{\"scenario\": \"$2\"}" | jq '.'
        ;;
        
    "list-models")
        echo "=== Available Model Types ==="
        curl -s ${AUTOSCALER_URL}/api/models/list | jq '.models'
        ;;
        
    "train-model")
        if [ -z "$2" ]; then
            echo "Usage: $0 train-model <model_name>"
            echo "Models: xgboost, catboost, lightgbm, gru, holt_winters"
            exit 1
        fi
        echo "=== Training Model: $2 ==="
        curl -X POST ${AUTOSCALER_URL}/api/models/train/$2 | jq '.'
        ;;
        
    "train-all")
        echo "=== Training All Models ==="
        curl -X POST ${AUTOSCALER_URL}/api/models/train_all | jq '.'
        ;;
        
    "status")
        echo "=== Autoscaler Status ==="
        curl -s ${AUTOSCALER_URL}/status | jq '{
            data_points: .data_points,
            model_trained: .model_trained,
            current_replicas: .current_replicas,
            gru_mse: .gru_mse,
            holt_winters_mse: .holt_winters_mse
        }'
        ;;
        
    "quick-test")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: $0 quick-test <scenario> <model>"
            echo "Example: $0 quick-test medium xgboost"
            exit 1
        fi
        
        SCENARIO=$2
        MODEL=$3
        
        echo "=== Quick Test: $MODEL on $SCENARIO ===" 
        echo ""
        echo "Step 1: Loading baseline..."
        curl -X POST ${AUTOSCALER_URL}/api/baseline/load \
            -H "Content-Type: application/json" \
            -d "{\"scenario\": \"$SCENARIO\"}" | jq '.message'
        
        echo ""
        echo "Step 2: Training model..."
        curl -X POST ${AUTOSCALER_URL}/api/models/train/$MODEL | jq '.message'
        
        echo ""
        echo "Step 3: Ready for load test!"
        echo "Run this command in another terminal:"
        echo ""
        echo "kubectl exec -it deployment/load-tester -- python3 load_test.py --scenario $SCENARIO --duration 1800 --target combined"
        ;;
        
    "collect-baseline")
        if [ -z "$2" ]; then
            DURATION=14400  # Default 4 hours = 240 points
        else
            DURATION=$2
        fi
        
        echo "=== Collecting All Baselines (Duration: ${DURATION}s = $((DURATION/3600)) hours each) ==="
        echo "Expected data points: $((DURATION/60)) per baseline"
        echo ""
        
        for SCENARIO in low medium high; do
            echo "=========================================="
            echo "Collecting: $SCENARIO"
            echo "=========================================="
            
            echo "  1. Clearing old data..."
            curl -X POST ${AUTOSCALER_URL}/api/baseline/clear
            
            echo ""
            echo "  2. Starting load test in background..."
            nohup kubectl exec -it deployment/load-tester -- python3 load_test.py \
                --scenario $SCENARIO \
                --duration $DURATION \
                --target combined \
                --test-predictive false \
                > ~/baseline_${SCENARIO}.log 2>&1 &
            
            LOAD_PID=$!
            echo $LOAD_PID > ~/baseline_${SCENARIO}.pid
            echo "  Load test started (PID: $LOAD_PID)"
            
            echo ""
            echo "  3. Waiting for test to complete (${DURATION}s = $((DURATION/3600)) hours)..."
            echo "     Monitor with: tail -f ~/baseline_${SCENARIO}.log"
            echo "     Check progress: curl -s ${AUTOSCALER_URL}/status | jq '.data_points'"
            sleep $(($DURATION + 120))
            
            echo ""
            echo "  4. Saving baseline..."
            curl -X POST ${AUTOSCALER_URL}/api/baseline/save \
                -H "Content-Type: application/json" \
                -d "{\"scenario\": \"$SCENARIO\"}" | jq '.message'
            
            if [ "$SCENARIO" != "high" ]; then
                echo ""
                echo "  5. Waiting 30 minutes before next collection..."
                sleep 1800
            fi
        done
        
        echo ""
        echo "=== Baseline Collection Complete ==="
        echo "Verify with: $0 list-baselines"
        ;;
        
    "help"|*)
        echo "Multi-Model Testing Quick Reference"
        echo "===================================="
        echo ""
        echo "Service Management:"
        echo "  $0 check                       - Check service health"
        echo "  $0 status                      - Show autoscaler status"
        echo ""
        echo "Baseline Management:"
        echo "  $0 list-baselines              - List available baselines"
        echo "  $0 clear-data                  - Clear all data (use before collection)"
        echo "  $0 load-baseline <scenario>    - Load a baseline dataset"
        echo "  $0 save-baseline <scenario>    - Save current data as baseline"
        echo "  $0 collect-baseline [duration] - Collect all baselines (default: 14400s = 4h)"
        echo ""
        echo "Model Management:"
        echo "  $0 list-models                 - List available models"
        echo "  $0 train-model <model>         - Train specific model"
        echo "  $0 train-all                   - Train all models"
        echo ""
        echo "Testing:"
        echo "  $0 quick-test <scenario> <model> - Run a quick test"
        echo ""
        echo "Examples:"
        echo "  $0 load-baseline medium"
        echo "  $0 train-model xgboost"
        echo "  $0 quick-test medium xgboost"
        echo "  $0 collect-baseline 1800"
        ;;
esac
