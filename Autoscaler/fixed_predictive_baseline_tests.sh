#!/bin/bash

# FIXED Predictive Scaler Baseline Tests with Proper Storage
# Ensures all metrics are saved persistently outside cluster
# Tests: GRU + Holt-Winters baseline with proper data collection

echo "🤖 FIXED PREDICTIVE SCALER BASELINE TESTS"
echo "=========================================="
echo "Features: Persistent storage, real-time metric collection"
echo "Models: GRU + Holt-Winters only (baseline configuration)"
echo "Started: $(date)"
echo ""

# Create persistent results directory outside cluster
RESULTS_BASE="/home/ubuntu/predictive_scaler_results"
BACKUP_DIR="/home/ubuntu/backup_results"
mkdir -p "$RESULTS_BASE"
mkdir -p "$BACKUP_DIR"

echo "📁 Results will be saved to: $RESULTS_BASE"
echo "💾 Backup location: $BACKUP_DIR"

# Verify predictive scaler configuration
echo "🔍 Verifying baseline configuration..."
sudo kubectl get configmap baseline-model-config -o yaml | grep -A 15 "models:" || echo "⚠️ Baseline config not found"
echo ""

# Simplified test configurations (fewer tests for verification)
declare -A LOW_TESTS=(
    ["baseline-low-01"]="45,95"
    ["baseline-low-02"]="50,100"
    ["baseline-low-03"]="40,90"
)

declare -A MEDIUM_TESTS=(
    ["baseline-medium-01"]="150,250"
    ["baseline-medium-02"]="160,260"
)

declare -A HIGH_TESTS=(
    ["baseline-high-01"]="350,450"
)

# Function to collect real-time metrics outside the pod
collect_external_metrics() {
    local test_name=$1
    local results_dir=$2
    local duration_seconds=$3
    
    echo "📊 Starting external metric collection for $test_name..."
    
    # Create metrics collection script
    cat > /tmp/collect_metrics.sh << 'EOF'
#!/bin/bash
TEST_NAME=$1
RESULTS_DIR=$2
DURATION=$3

START_TIME=$(date +%s)
echo "timestamp,pod_count,cpu_total,memory_total,predictive_scaler_status" > "$RESULTS_DIR/external_metrics.csv"

while [ $(($(date +%s) - START_TIME)) -lt $DURATION ]; do
    TIMESTAMP=$(date -Iseconds)
    
    # Get pod count
    POD_COUNT=$(sudo kubectl get pods -l app=product-app-combined --no-headers 2>/dev/null | grep Running | wc -l || echo "0")
    
    # Get total CPU and memory usage
    CPU_TOTAL=$(sudo kubectl top pods -l app=product-app-combined --no-headers 2>/dev/null | awk '{sum += $2} END {print sum}' | sed 's/m//' || echo "0")
    MEMORY_TOTAL=$(sudo kubectl top pods -l app=product-app-combined --no-headers 2>/dev/null | awk '{sum += $3} END {print sum}' | sed 's/Mi//' || echo "0")
    
    # Check predictive scaler status
    SCALER_STATUS=$(sudo kubectl get pods -l app=predictive-scaler --no-headers 2>/dev/null | grep Running | wc -l || echo "0")
    
    echo "$TIMESTAMP,$POD_COUNT,$CPU_TOTAL,$MEMORY_TOTAL,$SCALER_STATUS" >> "$RESULTS_DIR/external_metrics.csv"
    
    sleep 30  # Collect every 30 seconds
done

echo "External metrics collection completed for $TEST_NAME"
EOF
    
    chmod +x /tmp/collect_metrics.sh
    nohup bash /tmp/collect_metrics.sh "$test_name" "$results_dir" "$duration_seconds" > "$results_dir/metrics_collection.log" 2>&1 &
    METRICS_PID=$!
    echo $METRICS_PID > "$results_dir/metrics_pid.txt"
    
    echo "📈 External metrics collection started (PID: $METRICS_PID)"
}

# Function to run a single test with fixed storage
run_single_test() {
    local test_name=$1
    local min_rps=$2
    local max_rps=$3
    local category=$4
    local test_number=$5
    local total_tests=$6
    
    echo ""
    echo "🚀 STARTING FIXED PREDICTIVE SCALER TEST $test_number/$total_tests: $test_name"
    echo "Category: $category Load ($min_rps-$max_rps RPS)"
    echo "Models: GRU + Holt-Winters (Baseline)"
    echo "Started: $(date)"
    echo "----------------------------------------"
    
    # Create dedicated results directory
    local results_dir="$RESULTS_BASE/${test_name}"
    mkdir -p "$results_dir"
    
    # Create test metadata
    cat > "$results_dir/test_info.json" << EOF
{
  "test_name": "$test_name",
  "category": "$category",
  "min_rps": $min_rps,
  "max_rps": $max_rps,
  "duration_minutes": 15,
  "models": ["gru", "holt_winters"],
  "started_at": "$(date -Iseconds)",
  "test_number": $test_number,
  "total_tests": $total_tests
}
EOF
    
    # Start external metrics collection
    collect_external_metrics "$test_name" "$results_dir" 900  # 15 minutes = 900 seconds
    
    # Clean any existing resources
    sudo kubectl delete job "$test_name" --ignore-not-found=true >/dev/null 2>&1
    sleep 5
    
    # Create SIMPLIFIED test job without volume issues
    cat > "${test_name}-job.yaml" << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $test_name
spec:
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        kubernetes.io/hostname: ip-172-31-22-167
      containers:
      - name: load-generator
        image: curlimages/curl:latest
        command: ["/bin/sh"]
        args:
        - -c
        - |
          echo "=== $test_name Load Test Started ===" 
          echo "Target: product-app-combined"
          echo "Load: $min_rps-$max_rps RPS"
          echo "Duration: 15 minutes"
          
          # Simple variables
          TOTAL_REQUESTS=0
          SUCCESS_REQUESTS=0
          START_TIME=\$(date +%s)
          DURATION=900  # 15 minutes
          
          echo "Starting load generation..."
          
          # Simple load generation loop
          while [ \$(($(date +%s) - START_TIME)) -lt \$DURATION ]; do
            ELAPSED=\$(($(date +%s) - START_TIME))
            
            # Simple RPS calculation (integer math only)
            CYCLE=\$((ELAPSED / 60))  # Minute number
            if [ \$((CYCLE % 2)) -eq 0 ]; then
              TARGET_RPS=$max_rps
            else
              TARGET_RPS=$min_rps
            fi
            
            # Make requests
            for i in \$(seq 1 \$TARGET_RPS); do
              RESPONSE_CODE=\$(curl -s -w "%{http_code}" -o /dev/null --max-time 2 \
                "http://product-app-combined.default.svc.cluster.local:80/api/products" 2>/dev/null || echo "000")
              
              TOTAL_REQUESTS=\$((TOTAL_REQUESTS + 1))
              if [ "\$RESPONSE_CODE" = "200" ]; then
                SUCCESS_REQUESTS=\$((SUCCESS_REQUESTS + 1))
              fi
              
              # Quick delay between requests
              sleep 0.1
            done
            
            # Log progress every 5 minutes
            if [ \$((ELAPSED % 300)) -eq 0 ]; then
              SUCCESS_RATE=0
              if [ \$TOTAL_REQUESTS -gt 0 ]; then
                SUCCESS_RATE=\$((SUCCESS_REQUESTS * 100 / TOTAL_REQUESTS))
              fi
              echo "[\$(date +'%H:%M:%S')] \$ELAPSED/\$DURATION sec, \$TOTAL_REQUESTS requests, \$SUCCESS_RATE% success"
            fi
            
            # Sleep remainder of second
            sleep 1
          done
          
          echo "=== Load Test $test_name Completed ==="
          echo "Total Requests: \$TOTAL_REQUESTS"
          echo "Success Requests: \$SUCCESS_REQUESTS"
          echo "Duration: 15 minutes"
          
          # Keep pod alive briefly for any final data collection
          sleep 60
EOF
    
    # Apply the job
    sudo kubectl apply -f "${test_name}-job.yaml" >/dev/null
    
    # Wait for job completion
    echo "⏳ Test started, waiting for completion (15 minutes)..."
    
    # Monitor test progress
    timeout 1200 sudo kubectl wait --for=condition=complete job "$test_name" >/dev/null 2>&1
    JOB_RESULT=$?
    
    if [ $JOB_RESULT -eq 0 ]; then
        echo "✅ Test $test_name completed successfully"
    else
        echo "⚠️ Test $test_name finished (may have timed out)"
    fi
    
    # Stop metrics collection
    if [ -f "$results_dir/metrics_pid.txt" ]; then
        METRICS_PID=$(cat "$results_dir/metrics_pid.txt")
        kill $METRICS_PID 2>/dev/null
        echo "📊 External metrics collection stopped"
    fi
    
    # Collect final system state
    echo "📥 Collecting final system state..."
    
    # Get predictive scaler logs
    sudo kubectl logs deployment/predictive-scaler --tail=100 > "$results_dir/predictive_scaler_logs.txt" 2>/dev/null
    
    # Get pod information
    sudo kubectl get pods -l app=product-app-combined -o wide > "$results_dir/final_pod_state.txt" 2>/dev/null
    
    # Get events during test
    sudo kubectl get events --sort-by='.metadata.creationTimestamp' --field-selector involvedObject.name=product-app-combined > "$results_dir/scaling_events.txt" 2>/dev/null
    
    # Get job logs if available
    POD_NAME=$(sudo kubectl get pods -l job-name="$test_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ ! -z "$POD_NAME" ]; then
        sudo kubectl logs "$POD_NAME" > "$results_dir/load_test_logs.txt" 2>/dev/null
    fi
    
    # Create summary
    cat > "$results_dir/test_summary.txt" << SUMMARY
Test Summary: $test_name
============================
Category: $category Load
RPS Range: $min_rps - $max_rps
Duration: 15 minutes
Models: GRU + Holt-Winters (Baseline)
Started: $(date -Iseconds)
Status: Completed
Results Directory: $results_dir

Files Created:
- test_info.json: Test configuration
- external_metrics.csv: Pod count, CPU, memory metrics
- predictive_scaler_logs.txt: Scaler decision logs
- final_pod_state.txt: Final pod configuration
- scaling_events.txt: Kubernetes scaling events
- load_test_logs.txt: Load generator logs
- test_summary.txt: This summary
SUMMARY
    
    # Create backup
    cp -r "$results_dir" "$BACKUP_DIR/"
    
    echo "💾 Results saved to: $results_dir"
    echo "💾 Backup created in: $BACKUP_DIR"
    
    # Cleanup job
    sudo kubectl delete job "$test_name" --ignore-not-found=true >/dev/null 2>&1
    rm -f "${test_name}-job.yaml"
    
    echo "🧹 Cleanup completed"
    echo "⏱️ Waiting 30 seconds before next test..."
    sleep 30
}

# Run simplified test suite
test_counter=1
total_tests=6

echo "🏁 Starting FIXED LOW load tests..."
for test_name in "${!LOW_TESTS[@]}"; do
    IFS=',' read min_rps max_rps <<< "${LOW_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "Low" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo "🏁 Starting FIXED MEDIUM load tests..."
for test_name in "${!MEDIUM_TESTS[@]}"; do
    IFS=',' read min_rps max_rps <<< "${MEDIUM_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "Medium" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo "🏁 Starting FIXED HIGH load tests..."
for test_name in "${!HIGH_TESTS[@]}"; do
    IFS=',' read min_rps max_rps <<< "${HIGH_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "High" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo ""
echo "🎉 ALL FIXED PREDICTIVE SCALER TESTS COMPLETED!"
echo "=============================================="
echo "Total Tests: 6 (3 Low + 2 Medium + 1 High)"
echo "Models Tested: GRU + Holt-Winters (Baseline)"
echo "Results Directory: $RESULTS_BASE"
echo "Backup Directory: $BACKUP_DIR"
echo "Completed: $(date)"
echo ""
echo "✅ All metrics saved persistently outside cluster"
echo "✅ Data will survive cluster restarts"
echo "Next: Download results for analysis"