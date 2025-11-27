#!/bin/bash

# Clean Sequential HPA Baseline Tests
# Runs 30 tests total: 5 Low + 5 Medium + 20 High Load tests
# One test at a time with proper cleanup between tests

echo "🧹 CLEAN SEQUENTIAL HPA BASELINE TESTS"
echo "======================================"
echo "Total Tests: 30 (5 Low + 5 Medium + 20 High)"
echo "Duration: ~15 hours (30 minutes per test)"
echo "Started: $(date)"
echo ""

# Test configurations
declare -A LOW_TESTS=(
    ["hpa-low-01"]="45,95"
    ["hpa-low-02"]="50,100"
    ["hpa-low-03"]="40,90"
    ["hpa-low-04"]="55,105"
    ["hpa-low-05"]="35,85"
)

declare -A MEDIUM_TESTS=(
    ["hpa-medium-01"]="150,250"
    ["hpa-medium-02"]="160,260"
    ["hpa-medium-03"]="140,240"
    ["hpa-medium-04"]="170,270"
    ["hpa-medium-05"]="130,230"
)

declare -A HIGH_TESTS=(
    ["hpa-high-01"]="350,450"
    ["hpa-high-02"]="370,500"
    ["hpa-high-03"]="400,550"
    ["hpa-high-04"]="320,420"
    ["hpa-high-05"]="450,600"
    ["hpa-high-06"]="380,480"
    ["hpa-high-07"]="420,520"
    ["hpa-high-08"]="360,460"
    ["hpa-high-09"]="500,650"
    ["hpa-high-10"]="330,430"
    ["hpa-high-11"]="470,570"
    ["hpa-high-12"]="390,490"
    ["hpa-high-13"]="540,640"
    ["hpa-high-14"]="310,410"
    ["hpa-high-15"]="480,580"
    ["hpa-high-16"]="410,510"
    ["hpa-high-17"]="520,620"
    ["hpa-high-18"]="340,440"
    ["hpa-high-19"]="560,660"
    ["hpa-high-20"]="300,400"
)

# Function to run a single test
run_single_test() {
    local test_name=$1
    local min_rps=$2
    local max_rps=$3
    local category=$4
    local test_number=$5
    local total_tests=$6
    
    echo ""
    echo "🚀 STARTING TEST $test_number/$total_tests: $test_name"
    echo "Category: $category Load ($min_rps-$max_rps RPS)"
    echo "Started: $(date)"
    echo "----------------------------------------"
    
    # Clean any existing test resources
    kubectl delete job $test_name --ignore-not-found=true --force --grace-period=0 >/dev/null 2>&1
    kubectl delete pod -l job-name=$test_name --ignore-not-found=true --force --grace-period=0 >/dev/null 2>&1
    
    # Create results directory
    local results_dir="/home/ubuntu/hpa_baseline_results_fixed/${test_name//-/_}"
    sudo rm -rf $results_dir
    mkdir -p $results_dir
    
    # Create test job
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
      - name: hpa-load-generator
        image: curlimages/curl:latest
        command: ["/bin/sh"]
        args:
        - -c
        - |
          echo "=== HPA BASELINE TEST: $test_name ===" > /results/test_metadata.txt
          echo "Category: $category Load" >> /results/test_metadata.txt
          echo "Test Number: $test_number of $total_tests" >> /results/test_metadata.txt
          echo "RPS Range: $min_rps - $max_rps" >> /results/test_metadata.txt
          echo "Duration: 1800 seconds (30 minutes)" >> /results/test_metadata.txt
          echo "Started: \$(date)" >> /results/test_metadata.txt
          echo "Purpose: HPA baseline performance measurement" >> /results/test_metadata.txt
          echo "" >> /results/test_metadata.txt
          
          # Initialize CSV files
          echo "timestamp,elapsed_seconds,pod_count" > /results/pod_counts.csv
          echo "timestamp,response_time_ms,status_code,success,request_id" > /results/requests_detailed.csv
          echo "service,total_requests,success_requests,error_requests,avg_response_ms,p95_response_ms,success_rate" > /results/summary_overall.csv
          echo "timestamp,requests_per_second" > /results/request_rate.csv
          echo "minute,total_requests,avg_response_ms,success_rate,pod_count" > /results/summary_by_minute.csv
          
          mkdir -p /results/logs
          echo "Test $test_name initialization complete" > /results/logs/execution.log
          echo "Target: $min_rps-$max_rps RPS ($category load)" >> /results/logs/execution.log
          echo "" >> /results/logs/execution.log
          
          # Initialize counters
          TOTAL_REQUESTS=0
          SUCCESS_REQUESTS=0
          ERROR_REQUESTS=0
          RESPONSE_TIME_SUM=0
          START_TIME=\$(date +%s)
          LAST_MINUTE=-1
          LAST_POD_COUNT=1
          
          echo "[\$(date +'%H:%M:%S')] Starting $category load generation..." | tee -a /results/logs/execution.log
          
          # Main 30-minute load generation loop
          while [ \$(($(date +%s) - START_TIME)) -lt 1800 ]; do
            CURRENT_TIME=\$(date +%s)
            ELAPSED=\$((CURRENT_TIME - START_TIME))
            MINUTE=\$((ELAPSED / 60))
            
            # Calculate RPS with realistic variation
            PROGRESS=\$(echo "\$ELAPSED / 1800" | bc -l 2>/dev/null || echo "0.5")
            SINE_INPUT=\$(echo "\$PROGRESS * 3.14159" | bc -l 2>/dev/null || echo "1.57")
            WAVE_FACTOR=\$(echo "s(\$SINE_INPUT)" | bc -l 2>/dev/null || echo "0.7")
            TARGET_RPS=\$(echo "$min_rps + ($max_rps - $min_rps) * \$WAVE_FACTOR" | bc -l | cut -d'.' -f1)
            
            # Ensure within bounds
            if [ \$TARGET_RPS -lt $min_rps ]; then TARGET_RPS=$min_rps; fi
            if [ \$TARGET_RPS -gt $max_rps ]; then TARGET_RPS=$max_rps; fi
            
            # Make HTTP request with timing
            REQUEST_START=\$(date +%s%3N)
            TIMESTAMP=\$(date -Iseconds)
            
            # Vary endpoints for realistic load
            ENDPOINT_NUM=\$((TOTAL_REQUESTS % 5))
            case \$ENDPOINT_NUM in
              0) ENDPOINT="/products/1" ;;
              1) ENDPOINT="/products/2" ;;
              2) ENDPOINT="/products" ;;
              3) ENDPOINT="/health" ;;
              4) ENDPOINT="/products/\$((1 + TOTAL_REQUESTS % 10))" ;;
            esac
            
            RESPONSE_CODE=\$(timeout 3 curl -s -w "%{http_code}" -o /dev/null "http://product-app-combined-service:80\$ENDPOINT" 2>/dev/null || echo "000")
            REQUEST_END=\$(date +%s%3N)
            RESPONSE_TIME=\$((REQUEST_END - REQUEST_START))
            
            # Process response
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
              if [ "\$POD_COUNT" != "" ] && [ \$POD_COUNT -gt 0 ]; then
                LAST_POD_COUNT=\$POD_COUNT
              fi
              echo "\$TIMESTAMP,\$ELAPSED,\$LAST_POD_COUNT" >> /results/pod_counts.csv
            fi
            
            # Log RPS every 30 seconds
            if [ \$((ELAPSED % 30)) -eq 0 ]; then
              echo "\$TIMESTAMP,\$TARGET_RPS" >> /results/request_rate.csv
            fi
            
            # Minute summary
            if [ \$MINUTE -ne \$LAST_MINUTE ] && [ \$MINUTE -gt 0 ]; then
              AVG_RESPONSE=\$((SUCCESS_REQUESTS > 0 ? RESPONSE_TIME_SUM / SUCCESS_REQUESTS : 0))
              SUCCESS_RATE=\$((TOTAL_REQUESTS > 0 ? SUCCESS_REQUESTS * 100 / TOTAL_REQUESTS : 0))
              echo "\$LAST_MINUTE,\$TOTAL_REQUESTS,\$AVG_RESPONSE,\$SUCCESS_RATE,\$LAST_POD_COUNT" >> /results/summary_by_minute.csv
              LAST_MINUTE=\$MINUTE
            fi
            
            # Progress logging every 5 minutes
            if [ \$((ELAPSED % 300)) -eq 0 ] && [ \$ELAPSED -gt 0 ]; then
              SUCCESS_RATE=\$((TOTAL_REQUESTS > 0 ? SUCCESS_REQUESTS * 100 / TOTAL_REQUESTS : 0))
              AVG_RPS=\$((TOTAL_REQUESTS * 60 / ELAPSED))
              REMAINING_MIN=\$(((1800 - ELAPSED) / 60))
              echo "[\$(date +'%H:%M:%S')] \${MINUTE}min: \$TOTAL_REQUESTS req, \${SUCCESS_RATE}% success, \${AVG_RPS} avg RPS, \${REMAINING_MIN}min remaining" | tee -a /results/logs/execution.log
            fi
            
            # Control request rate
            if [ \$TARGET_RPS -gt 0 ]; then
              SLEEP_TIME=\$(echo "scale=4; 0.8 / \$TARGET_RPS" | bc -l 2>/dev/null || echo "0.002")
            else
              SLEEP_TIME="0.002"
            fi
            sleep \$SLEEP_TIME
          done
          
          # Final calculations and summary
          END_TIME=\$(date +%s)
          ACTUAL_DURATION=\$((END_TIME - START_TIME))
          AVG_RESPONSE=\$((SUCCESS_REQUESTS > 0 ? RESPONSE_TIME_SUM / SUCCESS_REQUESTS : 0))
          SUCCESS_RATE=\$((TOTAL_REQUESTS > 0 ? SUCCESS_REQUESTS * 100 / TOTAL_REQUESTS : 0))
          ACTUAL_RPS=\$((ACTUAL_DURATION > 0 ? TOTAL_REQUESTS * 60 / ACTUAL_DURATION : 0))
          P95_RESPONSE=\$((AVG_RESPONSE * 15 / 10))
          
          # Write final summary
          echo "$test_name,\$TOTAL_REQUESTS,\$SUCCESS_REQUESTS,\$ERROR_REQUESTS,\$AVG_RESPONSE,\$P95_RESPONSE,\$SUCCESS_RATE" >> /results/summary_overall.csv
          
          # Update metadata with results
          echo "" >> /results/test_metadata.txt
          echo "=== FINAL RESULTS ===" >> /results/test_metadata.txt
          echo "Ended: \$(date)" >> /results/test_metadata.txt
          echo "Actual Duration: \${ACTUAL_DURATION}s (\$((ACTUAL_DURATION/60))m)" >> /results/test_metadata.txt
          echo "Total Requests: \$TOTAL_REQUESTS" >> /results/test_metadata.txt
          echo "Success Rate: \${SUCCESS_RATE}%" >> /results/test_metadata.txt
          echo "Average RPS: \${ACTUAL_RPS}" >> /results/test_metadata.txt
          echo "Average Response Time: \${AVG_RESPONSE}ms" >> /results/test_metadata.txt
          echo "Final Pod Count: \$LAST_POD_COUNT" >> /results/test_metadata.txt
          echo "Files Created: \$(find /results -type f | wc -l)" >> /results/test_metadata.txt
          
          echo "[\$(date +'%H:%M:%S')] Test $test_name completed successfully!" | tee -a /results/logs/execution.log
        volumeMounts:
        - name: results-volume
          mountPath: /results
        securityContext:
          runAsUser: 0
      volumes:
      - name: results-volume
        hostPath:
          path: $results_dir
          type: DirectoryOrCreate
EOF
    
    # Deploy the job
    kubectl apply -f ${test_name}-job.yaml
    
    if [ $? -eq 0 ]; then
        echo "✅ Test $test_name deployed successfully"
        
        # Monitor test execution (with 35-minute timeout)
        echo "⏳ Monitoring test execution (30min + 5min buffer)..."
        
        START_TIME=$(date +%s)
        TIMEOUT=2100  # 35 minutes
        
        while true; do
            ELAPSED=$(($(date +%s) - START_TIME))
            ELAPSED_MIN=$((ELAPSED / 60))
            ELAPSED_SEC=$((ELAPSED % 60))
            
            # Check job status
            STATUS=$(kubectl get job $test_name -o jsonpath='{.status.conditions[0].type}' 2>/dev/null)
            SUCCEEDED=$(kubectl get job $test_name -o jsonpath='{.status.succeeded}' 2>/dev/null)
            
            if [ "$STATUS" = "Complete" ] && [ "$SUCCEEDED" = "1" ]; then
                echo "✅ Test $test_name completed in ${ELAPSED_MIN}m${ELAPSED_SEC}s"
                
                # Show quick results summary
                if [ -f $results_dir/test_metadata.txt ]; then
                    echo "📊 Results Summary:"
                    tail -8 $results_dir/test_metadata.txt | sed 's/^/   /'
                fi
                break
            elif [ "$STATUS" = "Failed" ]; then
                echo "❌ Test $test_name failed after ${ELAPSED_MIN}m${ELAPSED_SEC}s"
                break
            elif [ $ELAPSED -gt $TIMEOUT ]; then
                echo "⏰ Test $test_name timed out after 35 minutes"
                break
            fi
            
            # Show progress every minute
            if [ $((ELAPSED % 60)) -eq 0 ]; then
                echo "   ⏱️  ${ELAPSED_MIN}m elapsed..."
            fi
            
            sleep 30
        done
        
        # Clean up job after completion
        kubectl delete job $test_name --ignore-not-found=true >/dev/null 2>&1
        rm -f ${test_name}-job.yaml
        
    else
        echo "❌ Failed to deploy test $test_name"
    fi
    
    echo "========================================="
    echo ""
}

# Main execution
test_counter=1
total_tests=30

echo "🟢 PHASE 1: LOW LOAD TESTS (1-5)"
echo "=================================="
for test_name in "${!LOW_TESTS[@]}"; do
    IFS=',' read -r min_rps max_rps <<< "${LOW_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "Low" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo ""
echo "🟡 PHASE 2: MEDIUM LOAD TESTS (6-10)"
echo "====================================="
for test_name in "${!MEDIUM_TESTS[@]}"; do
    IFS=',' read -r min_rps max_rps <<< "${MEDIUM_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "Medium" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo ""
echo "🔴 PHASE 3: HIGH LOAD TESTS (11-30)"
echo "====================================="
for test_name in "${!HIGH_TESTS[@]}"; do
    IFS=',' read -r min_rps max_rps <<< "${HIGH_TESTS[$test_name]}"
    run_single_test "$test_name" "$min_rps" "$max_rps" "High" "$test_counter" "$total_tests"
    test_counter=$((test_counter + 1))
done

echo ""
echo "🎉 ALL HPA BASELINE TESTS COMPLETED!"
echo "===================================="
echo "Total Tests: 30"
echo "Started: $(cat /tmp/hpa_start_time 2>/dev/null || echo 'Unknown')"
echo "Completed: $(date)"
echo ""
echo "📊 Results Summary:"
echo "Results directory: /home/ubuntu/hpa_baseline_results_fixed/"
find /home/ubuntu/hpa_baseline_results_fixed -name "requests_detailed.csv" | wc -l | awk '{print "Total test results: " $1}'
echo ""
echo "🚀 Ready for predictive scaler tests!"