#!/bin/bash

# Fixed Baseline Predictive Scaler Tests - GRU + Holt-Winters Only
# Tests: 10 Low + 10 Medium + 10 High = 30 total baseline tests

set -e

NAMESPACE=${NAMESPACE:-"default"}
BASELINE_RESULTS_DIR="/home/ubuntu/baseline_results"

# Create results directory
mkdir -p "$BASELINE_RESULTS_DIR"

echo "=== Starting FIXED Baseline Predictive Scaler Tests ==="
echo "Models: GRU + Holt-Winters only"
echo "Tests: 10 Low + 10 Medium + 10 High"
echo "Results Directory: $BASELINE_RESULTS_DIR"
echo "Start Time: $(date)"

# Function to run a single baseline test with FIXED math
run_baseline_test() {
    local test_name="$1"
    local rps_min="$2"
    local rps_max="$3"
    local duration_minutes="$4"
    local category="$5"
    
    echo "=== Starting Baseline Test: $test_name ==="
    echo "Category: $category"
    echo "Load: ${rps_min}-${rps_max} RPS"
    echo "Duration: $duration_minutes minutes"
    
    # Clean up any existing jobs/pods
    sudo kubectl delete job "$test_name" --ignore-not-found=true
    sudo kubectl delete pod -l job-name="$test_name" --ignore-not-found=true
    
    # Wait for cleanup
    sleep 10
    
    # Create job with predictive scaler testing - FIXED MATH
    cat <<EOF | sudo kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: $test_name
  namespace: $NAMESPACE
spec:
  template:
    metadata:
      labels:
        app: baseline-load-test
        test: $test_name
        category: $category
    spec:
      nodeSelector:
        kubernetes.io/hostname: "ip-172-31-22-167"
      restartPolicy: Never
      containers:
      - name: load-tester
        image: curlimages/curl:latest
        command: ["/bin/sh"]
        args:
        - -c
        - |
          echo "=== FIXED Baseline Test: $test_name Started ==="
          echo "Target: product-app-combined service"
          echo "Load Pattern: ${rps_min}-${rps_max} RPS"
          echo "Duration: $duration_minutes minutes"
          echo "Models: GRU + Holt-Winters (Baseline)"
          echo "Start Time: \$(date)"
          
          # Install required tools
          apk add --no-cache bc
          
          # Test duration in seconds  
          DURATION=\$((${duration_minutes} * 60))
          REQUESTS_SENT=0
          START_TIME=\$(date +%s)
          
          # Create results directory in pod
          mkdir -p /tmp/results
          
          echo "timestamp,response_time_ms,status_code,rps_target,test_name" > /tmp/results/requests_detailed.csv
          
          # FIXED: Simple variable RPS load generation without complex math
          while [ \$(($(date +%s) - START_TIME)) -lt \$DURATION ]; do
            CURRENT_TIME=\$(($(date +%s) - START_TIME))
            
            # Simple RPS calculation using integer math only
            RPS_RANGE=\$((${rps_max} - ${rps_min}))
            TIME_FACTOR=\$((CURRENT_TIME % 60))  # 0-59 second cycle
            
            # Create sine-like pattern using integer math
            if [ \$TIME_FACTOR -lt 15 ]; then
              RPS_OFFSET=\$((RPS_RANGE * TIME_FACTOR / 15))
            elif [ \$TIME_FACTOR -lt 30 ]; then
              RPS_OFFSET=\$RPS_RANGE
            elif [ \$TIME_FACTOR -lt 45 ]; then
              RPS_OFFSET=\$((RPS_RANGE * (45 - TIME_FACTOR) / 15))
            else
              RPS_OFFSET=0
            fi
            
            CURRENT_RPS=\$((${rps_min} + RPS_OFFSET))
            
            # Ensure minimum 1 RPS and maximum safety
            if [ "\$CURRENT_RPS" -lt 1 ]; then
              CURRENT_RPS=1
            elif [ "\$CURRENT_RPS" -gt ${rps_max} ]; then
              CURRENT_RPS=${rps_max}
            fi
            
            echo "Time: \${CURRENT_TIME}s/\${DURATION}s, Target RPS: \$CURRENT_RPS"
            
            # Send requests for current second (simplified)
            for i in \$(seq 1 \$CURRENT_RPS); do
              REQUEST_START=\$(date +%s%3N)
              HTTP_CODE=\$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
                http://product-app-combined.default.svc.cluster.local:80/api/products || echo "000")
              REQUEST_END=\$(date +%s%3N)
              RESPONSE_TIME=\$((REQUEST_END - REQUEST_START))
              
              echo "\$(date -Iseconds),\$RESPONSE_TIME,\$HTTP_CODE,\$CURRENT_RPS,$test_name" >> /tmp/results/requests_detailed.csv
              REQUESTS_SENT=\$((REQUESTS_SENT + 1))
              
              # Small delay between requests (0.1 second max)
              sleep 0.1
            done
            
            # Sleep remainder of second (but max 1 second total per loop)
            sleep 1
          done
          
          echo "=== Baseline Test: $test_name Completed ==="
          echo "Total Requests Sent: \$REQUESTS_SENT"
          echo "End Time: \$(date)"
          echo "Results saved to /tmp/results/"
          
          # Keep container alive for result collection
          sleep 300
EOF
    
    # Wait for job to start
    echo "Waiting for job to start..."
    sudo kubectl wait --for=condition=ready pod -l job-name="$test_name" --timeout=60s
    
    # Monitor job progress
    echo "Job started. Running for $duration_minutes minutes..."
    
    # Wait for completion with timeout
    TIMEOUT_SECONDS=$((duration_minutes * 60 + 600))  # Test duration + 10 minutes buffer
    sudo kubectl wait --for=condition=complete job "$test_name" --timeout=${TIMEOUT_SECONDS}s || {
        echo "Job $test_name timed out or failed"
        sudo kubectl describe job "$test_name"
        sudo kubectl logs job/"$test_name" || true
        return 1
    }
    
    echo "Job completed successfully. Collecting results..."
    
    # Create results directory for this test
    TEST_RESULT_DIR="$BASELINE_RESULTS_DIR/$test_name"
    mkdir -p "$TEST_RESULT_DIR"
    
    # Copy results from pod
    POD_NAME=$(sudo kubectl get pods -l job-name="$test_name" -o jsonpath='{.items[0].metadata.name}')
    sudo kubectl cp "$POD_NAME:/tmp/results/requests_detailed.csv" "$TEST_RESULT_DIR/requests_detailed.csv" || echo "Failed to copy detailed requests"
    
    # Collect pod metrics during test
    sudo kubectl top pod -l app=product-app-combined --no-headers > "$TEST_RESULT_DIR/pod_metrics.txt" || echo "Failed to collect pod metrics"
    
    # Get scaling events
    sudo kubectl get events --field-selector involvedObject.name=product-app-combined --sort-by='.metadata.creationTimestamp' > "$TEST_RESULT_DIR/scaling_events.txt" || echo "Failed to collect scaling events"
    
    # Get pod count over time (from logs)
    sudo kubectl logs deployment/predictive-scaler --tail=100 > "$TEST_RESULT_DIR/scaling_decisions.log" || echo "No scaling decisions found"
    
    # Create test summary
    cat > "$TEST_RESULT_DIR/test_summary.txt" <<SUMMARY
Test Name: $test_name
Category: $category  
Type: Baseline (GRU + Holt-Winters)
Load Pattern: ${rps_min}-${rps_max} RPS
Duration: $duration_minutes minutes
Execution Time: $(date)
Pod Name: $POD_NAME
SUMMARY
    
    echo "Results collected for $test_name"
    
    # Cleanup job
    sudo kubectl delete job "$test_name" --ignore-not-found=true
    
    echo "=== Baseline Test: $test_name Complete ==="
    echo ""
}

# Start with just a few tests first to verify fix
echo "=== STARTING FIXED LOW LOAD BASELINE TESTS (First 3) ==="
for i in 01 02 03; do
    run_baseline_test "baseline-low-$i" 45 105 15 "low"  # Shorter 15-minute tests for verification
    sleep 30  # 30 seconds between tests
done

echo "=== FIRST 3 BASELINE TESTS COMPLETED ==="
echo "If successful, continue with remaining tests..."