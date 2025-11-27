#!/bin/bash

# Clean Sequential PREDICTIVE SCALER Baseline Tests
# Adapted from working HPA test framework
# Tests: GRU + Holt-Winters baseline vs product-app-combined with predictive scaler
# Runs 30 tests total: 10 Low + 10 Medium + 10 High Load tests

echo "🤖 PREDICTIVE SCALER BASELINE TESTS (GRU + Holt-Winters)"
echo "==========================================================="
echo "Total Tests: 30 (10 Low + 10 Medium + 10 High)"
echo "Duration: ~15 hours (30 minutes per test)"
echo "Models: GRU + Holt-Winters only (baseline configuration)"
echo "Started: $(date)"
echo ""

# Verify predictive scaler configuration
echo "🔍 Verifying baseline configuration..."
kubectl get configmap baseline-model-config -o yaml | grep -A 15 "models:" || echo "⚠️ Baseline config not found"
echo ""

# Test configurations (same RPS as HPA for comparison)
declare -A LOW_TESTS=(
    ["baseline-low-01"]="45,95"
    ["baseline-low-02"]="50,100"
    ["baseline-low-03"]="40,90"
    ["baseline-low-04"]="55,105"
    ["baseline-low-05"]="35,85"
    ["baseline-low-06"]="60,110"
    ["baseline-low-07"]="30,80"
    ["baseline-low-08"]="65,115"
    ["baseline-low-09"]="25,75"
    ["baseline-low-10"]="70,120"
)

declare -A MEDIUM_TESTS=(
    ["baseline-medium-01"]="150,250"
    ["baseline-medium-02"]="160,260"
    ["baseline-medium-03"]="140,240"
    ["baseline-medium-04"]="170,270"
    ["baseline-medium-05"]="130,230"
    ["baseline-medium-06"]="180,280"
    ["baseline-medium-07"]="120,220"
    ["baseline-medium-08"]="190,290"
    ["baseline-medium-09"]="110,210"
    ["baseline-medium-10"]="200,300"
)

declare -A HIGH_TESTS=(
    ["baseline-high-01"]="350,450"
    ["baseline-high-02"]="370,500"
    ["baseline-high-03"]="400,550"
    ["baseline-high-04"]="320,420"
    ["baseline-high-05"]="450,600"
    ["baseline-high-06"]="380,480"
    ["baseline-high-07"]="420,520"
    ["baseline-high-08"]="360,460"
    ["baseline-high-09"]="500,650"
    ["baseline-high-10"]="330,430"
)

# Function to run a single predictive scaler test
run_single_test() {
    local test_name=$1
    local min_rps=$2
    local max_rps=$3
    local category=$4
    local test_number=$5
    local total_tests=$6
    
    echo ""
    echo "🚀 STARTING PREDICTIVE SCALER TEST $test_number/$total_tests: $test_name"
    echo "Category: $category Load ($min_rps-$max_rps RPS)"
    echo "Models: GRU + Holt-Winters (Baseline)"
    echo "Started: $(date)"
    echo "----------------------------------------"
    
    # Clean any existing test resources
    sudo kubectl delete job $test_name --ignore-not-found=true --force --grace-period=0 >/dev/null 2>&1
    sudo kubectl delete pod -l job-name=$test_name --ignore-not-found=true --force --grace-period=0 >/dev/null 2>&1
    
    # Create results directory
    local results_dir="/home/ubuntu/predictive_baseline_results/${test_name//-/_}"
    sudo rm -rf $results_dir
    mkdir -p $results_dir
    
    # Create predictive scaler test job (adapted from HPA framework)
    cat > ${test_name}-job.yaml << EOF
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
      - name: predictive-load-generator
        image: curlimages/curl:latest
        command: ["/bin/sh"]
        args:
        - -c
        - |
          echo "=== PREDICTIVE SCALER BASELINE TEST: $test_name ===" > /results/test_metadata.txt
          echo "Category: $category Load" >> /results/test_metadata.txt
          echo "Test Number: $test_number of $total_tests" >> /results/test_metadata.txt
          echo "RPS Range: $min_rps - $max_rps" >> /results/test_metadata.txt
          echo "Duration: 1800 seconds (30 minutes)" >> /results/test_metadata.txt
          echo "Started: \$(date)" >> /results/test_metadata.txt
          echo "Purpose: Predictive Scaler baseline (GRU + Holt-Winters)" >> /results/test_metadata.txt
          echo "Target: product-app-combined with predictive scaler" >> /results/test_metadata.txt
          echo "" >> /results/test_metadata.txt
          
          # Install bc for math calculations
          apk add --no-cache bc >/dev/null 2>&1
          
          # Initialize CSV files (same structure as HPA tests)
          echo "timestamp,elapsed_seconds,pod_count" > /results/pod_counts.csv
          echo "timestamp,response_time_ms,status_code,success,request_id" > /results/requests_detailed.csv
          echo "service,total_requests,success_requests,error_requests,avg_response_ms,p95_response_ms,success_rate" > /results/summary_overall.csv
          echo "timestamp,requests_per_second" > /results/request_rate.csv
          echo "minute,total_requests,avg_response_ms,success_rate,pod_count" > /results/summary_by_minute.csv
          
          mkdir -p /results/logs
          echo "Predictive Scaler Test $test_name initialization complete" > /results/logs/execution.log
          echo "Target: $min_rps-$max_rps RPS ($category load)" >> /results/logs/execution.log
          echo "Models: GRU + Holt-Winters (Baseline)" >> /results/logs/execution.log
          echo "" >> /results/logs/execution.log
          
          # Initialize counters
          TOTAL_REQUESTS=0
          SUCCESS_REQUESTS=0
          ERROR_REQUESTS=0
          RESPONSE_TIME_SUM=0
          START_TIME=\$(date +%s)
          LAST_MINUTE=-1
          LAST_POD_COUNT=1
          
          echo "[\$(date +'%H:%M:%S')] Starting $category load generation (Predictive Scaler)..." | tee -a /results/logs/execution.log
          
          # Main 30-minute load generation loop (same logic as HPA tests)
          while [ \$(($(date +%s) - START_TIME)) -lt 1800 ]; do
            CURRENT_TIME=\$(date +%s)
            ELAPSED=\$((CURRENT_TIME - START_TIME))
            MINUTE=\$((ELAPSED / 60))
            
            # Calculate RPS with realistic variation (using integer math)
            CYCLE_POS=\$((ELAPSED % 120))  # 2-minute cycles
            if [ \$CYCLE_POS -lt 30 ]; then
              # Ramp up
              TARGET_RPS=\$(($min_rps + (($max_rps - $min_rps) * \$CYCLE_POS / 30)))
            elif [ \$CYCLE_POS -lt 60 ]; then
              # Peak
              TARGET_RPS=$max_rps
            elif [ \$CYCLE_POS -lt 90 ]; then
              # Ramp down
              TARGET_RPS=\$(($max_rps - (($max_rps - $min_rps) * (\$CYCLE_POS - 60) / 30)))
            else
              # Low period
              TARGET_RPS=$min_rps
            fi
            
            # Ensure within bounds
            if [ \$TARGET_RPS -lt $min_rps ]; then TARGET_RPS=$min_rps; fi
            if [ \$TARGET_RPS -gt $max_rps ]; then TARGET_RPS=$max_rps; fi
            
            # Make HTTP request with timing
            REQUEST_START=\$(date +%s%3N)
            TIMESTAMP=\$(date -Iseconds)
            
            # Vary endpoints for realistic load (same as HPA tests)
            ENDPOINT_NUM=\$((TOTAL_REQUESTS % 5))
            case \$ENDPOINT_NUM in
              0) ENDPOINT="/api/products/1" ;;
              1) ENDPOINT="/api/products/2" ;;
              2) ENDPOINT="/api/products" ;;
              3) ENDPOINT="/health" ;;
              4) ENDPOINT="/api/products/\$((1 + TOTAL_REQUESTS % 10))" ;;
            esac
            
            # Target product-app-combined service (predictive scaler target)
            RESPONSE_CODE=\$(timeout 3 curl -s -w "%{http_code}" -o /dev/null "http://product-app-combined.default.svc.cluster.local:80\$ENDPOINT" 2>/dev/null || echo "000")
            REQUEST_END=\$(date +%s%3N)
            RESPONSE_TIME=\$((REQUEST_END - REQUEST_START))
            
            # Process response (same as HPA)
            if [ "\$RESPONSE_CODE" = "200" ]; then
              SUCCESS_FLAG=1
              SUCCESS_REQUESTS=\$((SUCCESS_REQUESTS + 1))
              RESPONSE_TIME_SUM=\$((RESPONSE_TIME_SUM + RESPONSE_TIME))
            else
              SUCCESS_FLAG=0
              ERROR_REQUESTS=\$((ERROR_REQUESTS + 1))
            fi
            
            TOTAL_REQUESTS=\$((TOTAL_REQUESTS + 1))
            
            # Log detailed request
            echo "\$TIMESTAMP,\$RESPONSE_TIME,\$RESPONSE_CODE,\$SUCCESS_FLAG,${test_name}_req_\$TOTAL_REQUESTS" >> /results/requests_detailed.csv
            
            # Monitor pod count every 15 seconds
            if [ \$((ELAPSED % 15)) -eq 0 ]; then
              POD_COUNT=\$(timeout 10 kubectl get pods -l app=product-app-combined --no-headers 2>/dev/null | grep -c Running || echo "\$LAST_POD_COUNT")
              if [ "\$POD_COUNT" != "\$LAST_POD_COUNT" ]; then
                echo "[\$(date +'%H:%M:%S')] Pod count changed: \$LAST_POD_COUNT -> \$POD_COUNT (Predictive Scaler)" | tee -a /results/logs/execution.log
                LAST_POD_COUNT=\$POD_COUNT
              fi
              echo "\$(date -Iseconds),\$ELAPSED,\$POD_COUNT" >> /results/pod_counts.csv
            fi
            
            # Log rate every 60 seconds
            if [ \$((ELAPSED % 60)) -eq 0 ] && [ \$MINUTE -ne \$LAST_MINUTE ]; then
              AVG_RESPONSE=0
              if [ \$SUCCESS_REQUESTS -gt 0 ]; then
                AVG_RESPONSE=\$((RESPONSE_TIME_SUM / SUCCESS_REQUESTS))
              fi
              
              SUCCESS_RATE=0
              if [ \$TOTAL_REQUESTS -gt 0 ]; then
                SUCCESS_RATE=\$((SUCCESS_REQUESTS * 100 / TOTAL_REQUESTS))
              fi
              
              echo "\$(date -Iseconds),\$TARGET_RPS" >> /results/request_rate.csv
              echo "\$MINUTE,\$TOTAL_REQUESTS,\$AVG_RESPONSE,\$SUCCESS_RATE,\$POD_COUNT" >> /results/summary_by_minute.csv
              
              echo "[\$(date +'%H:%M:%S')] Min \$MINUTE: \$TOTAL_REQUESTS req, \$SUCCESS_RATE% success, \$AVG_RESPONSE ms avg, \$POD_COUNT pods" | tee -a /results/logs/execution.log
              LAST_MINUTE=\$MINUTE
            fi
            
            # Adaptive delay to achieve target RPS
            DELAY_MS=\$(echo "1000 / \$TARGET_RPS" | bc -l | cut -d'.' -f1)
            if [ "\$DELAY_MS" = "" ] || [ \$DELAY_MS -lt 10 ]; then DELAY_MS=10; fi
            if [ \$DELAY_MS -gt 1000 ]; then DELAY_MS=1000; fi
            
            sleep 0.\$DELAY_MS 2>/dev/null || sleep 1
          done
          
          # Final summary
          AVG_RESPONSE_FINAL=0
          if [ \$SUCCESS_REQUESTS -gt 0 ]; then
            AVG_RESPONSE_FINAL=\$((RESPONSE_TIME_SUM / SUCCESS_REQUESTS))
          fi
          
          SUCCESS_RATE_FINAL=0  
          if [ \$TOTAL_REQUESTS -gt 0 ]; then
            SUCCESS_RATE_FINAL=\$((SUCCESS_REQUESTS * 100 / TOTAL_REQUESTS))
          fi
          
          echo "product-app-combined,\$TOTAL_REQUESTS,\$SUCCESS_REQUESTS,\$ERROR_REQUESTS,\$AVG_RESPONSE_FINAL,\$AVG_RESPONSE_FINAL,\$SUCCESS_RATE_FINAL" >> /results/summary_overall.csv
          
          echo "" >> /results/logs/execution.log
          echo "=== PREDICTIVE SCALER TEST $test_name COMPLETED ===" >> /results/logs/execution.log
          echo "Duration: 30 minutes" >> /results/logs/execution.log
          echo "Total Requests: \$TOTAL_REQUESTS" >> /results/logs/execution.log
          echo "Success Rate: \$SUCCESS_RATE_FINAL%" >> /results/logs/execution.log
          echo "Average Response Time: \$AVG_RESPONSE_FINAL ms" >> /results/logs/execution.log
          echo "Final Pod Count: \$POD_COUNT" >> /results/logs/execution.log
          echo "Completed: \$(date)" >> /results/logs/execution.log
          
          echo "Test completed. Results saved to /results/"
          
          # Keep container running for result collection
          sleep 300
      volumes:
      - name: results-storage
        hostPath:
          path: $results_dir
      - name: kube-config
        hostPath:
          path: /etc/rancher/k3s/k3s.yaml
EOF

    # Apply the job
    sudo kubectl apply -f ${test_name}-job.yaml >/dev/null
    
    # Wait for pod to be ready
    echo "⏳ Waiting for test pod to start..."
    timeout 120 sudo kubectl wait --for=condition=ready pod -l job-name=$test_name >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Test $test_name started successfully"
        echo "📊 Running for 30 minutes..."
        
        # Wait for job completion with 35-minute timeout
        timeout 2100 sudo kubectl wait --for=condition=complete job $test_name >/dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ Test $test_name completed successfully"
            
            # Copy results from pod
            POD_NAME=$(sudo kubectl get pods -l job-name=$test_name -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
            if [ ! -z "$POD_NAME" ]; then
                echo "📥 Collecting results from $POD_NAME..."
                
                # Copy all result files
                sudo kubectl cp $POD_NAME:/results/ $results_dir/ 2>/dev/null || echo "⚠️ Partial result collection"
                
                # Get predictive scaler logs during test
                sudo kubectl logs deployment/predictive-scaler --tail=200 > $results_dir/predictive_scaler_logs.txt 2>/dev/null
                
                # Get scaling events
                sudo kubectl get events --field-selector involvedObject.name=product-app-combined --sort-by='.metadata.creationTimestamp' > $results_dir/scaling_events.txt 2>/dev/null
                
                echo "📋 Results saved to: $results_dir"
            fi
        else
            echo "❌ Test $test_name timed out or failed"
        fi
    else
        echo "❌ Test $test_name failed to start"
    fi
    
    # Cleanup job
    sudo kubectl delete job $test_name --ignore-not-found=true >/dev/null 2>&1
    rm -f ${test_name}-job.yaml
    
    echo "🧹 Cleanup completed for $test_name"
    echo "⏱️ Waiting 60 seconds before next test..."
    sleep 60
}

# Create results directory
mkdir -p /home/ubuntu/predictive_baseline_results

# Run all tests sequentially
test_counter=1
total_tests=30

echo "🏁 Starting LOW load tests..."
for test_name in "${!LOW_TESTS[@]}"; do
    IFS=',' read min_rps max_rps <<< "${LOW_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "Low" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo "🏁 Starting MEDIUM load tests..."
for test_name in "${!MEDIUM_TESTS[@]}"; do
    IFS=',' read min_rps max_rps <<< "${MEDIUM_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "Medium" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo "🏁 Starting HIGH load tests..."
for test_name in "${!HIGH_TESTS[@]}"; do
    IFS=',' read min_rps max_rps <<< "${HIGH_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "High" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo ""
echo "🎉 ALL PREDICTIVE SCALER BASELINE TESTS COMPLETED!"
echo "==========================================="
echo "Total Tests: 30 (10 Low + 10 Medium + 10 High)"
echo "Models Tested: GRU + Holt-Winters (Baseline)"
echo "Results Directory: /home/ubuntu/predictive_baseline_results"
echo "Completed: $(date)"
echo ""
echo "Next Step: Compare with HPA results for research paper analysis"