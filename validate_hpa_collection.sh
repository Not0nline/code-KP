#!/bin/bash

# QUICK HPA TEST VALIDATION - 3 × 5-minute tests
# Debug results collection mechanism before full tests

echo "🧪 HPA RESULTS COLLECTION VALIDATION"
echo "===================================="
echo "Target: 3 quick tests × 5 minutes each"
echo "Load: 100-200 RPS (simple validation load)"
echo "Purpose: Debug volume mounting and results collection"
echo ""

# Test parameters
LOAD_TYPE="validation"
ITERATIONS=3
MIN_RPS=100
MAX_RPS=200
DURATION=300  # 5 minutes for quick validation

# Create validation directory
mkdir -p ~/hpa_validation_tests

echo "📋 Validation Configuration:"
echo "   Test Type: $LOAD_TYPE"
echo "   Iterations: $ITERATIONS" 
echo "   RPS Range: $MIN_RPS - $MAX_RPS"
echo "   Duration: $DURATION seconds (5 min)"
echo "   Results Dir: ~/hpa_validation_tests"
echo ""

for i in $(seq 1 $ITERATIONS); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " VALIDATION TEST $i/$ITERATIONS (5-minute HPA test)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏰ Started: $(date +'%H:%M:%S')"
    
    # Create test directory with explicit permissions
    TEST_DIR="hpa_validation_tests/test_$i"
    mkdir -p $TEST_DIR
    chmod 777 $TEST_DIR  # Ensure write permissions
    
    echo "📁 Created test directory: $TEST_DIR (permissions: 777)"
    
    echo ""
    echo "━━━ Launch Validation Load Test (${DURATION}s) ━━━"
    
    # Create load test job with explicit volume configuration
    cat > hpa-validation-$i.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: load-test-hpa-validation-$i
spec:
  ttlSecondsAfterFinished: 60
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: load-tester
        image: load-tester:latest
        env:
        - name: TARGET_URL
          value: "http://product-app-combined-service:80"
        - name: MIN_RPS
          value: "$MIN_RPS"
        - name: MAX_RPS  
          value: "$MAX_RPS"
        - name: DURATION
          value: "$DURATION"
        - name: DISTRIBUTION
          value: "poisson"
        - name: TEST_NAME
          value: "hpa_validation_$i"
        - name: OUTPUT_DIR
          value: "/test_results"
        - name: DEBUG_MODE
          value: "true"
        volumeMounts:
        - name: results-volume
          mountPath: /test_results
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        securityContext:
          runAsUser: 0
          runAsGroup: 0
      volumes:
      - name: results-volume
        hostPath:
          path: /home/ubuntu/$TEST_DIR
          type: DirectoryOrCreate
      nodeSelector:
        kubernetes.io/hostname: "$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')"
EOF

    # Apply the job
    kubectl apply -f hpa-validation-$i.yaml
    
    if [ $? -eq 0 ]; then
        echo "✅ Job created: load-test-hpa-validation-$i"
    else
        echo "❌ Failed to create job"
        continue
    fi
    
    # Wait a moment for job to start
    sleep 10
    
    echo "📊 Job status check:"
    kubectl get job load-test-hpa-validation-$i -o wide
    
    echo ""
    echo "🔍 Pod details:"
    kubectl get pods -l job-name=load-test-hpa-validation-$i -o wide
    
    echo ""
    echo "⏳ Monitoring test progress (checking every 30s)..."
    
    # Monitor job completion
    START_TIME=$(date +%s)
    TIMEOUT=$((DURATION + 180))  # Add 3 min buffer
    
    while true; do
        ELAPSED=$(($(date +%s) - START_TIME))
        ELAPSED_MIN=$((ELAPSED / 60))
        TIMEOUT_MIN=$((TIMEOUT / 60))
        
        STATUS=$(kubectl get job load-test-hpa-validation-$i -o jsonpath='{.status.conditions[0].type}' 2>/dev/null)
        SUCCEEDED=$(kubectl get job load-test-hpa-validation-$i -o jsonpath='{.status.succeeded}' 2>/dev/null)
        
        if [ "$STATUS" = "Complete" ] || [ "$SUCCEEDED" = "1" ]; then
            echo "✅ Test completed successfully after ${ELAPSED_MIN}m"
            break
        elif [ "$STATUS" = "Failed" ]; then
            echo "❌ Test failed"
            echo "📋 Pod logs:"
            kubectl logs -l job-name=load-test-hpa-validation-$i --tail=10
            break
        elif [ $ELAPSED -gt $TIMEOUT ]; then
            echo "⏰ Test timed out after ${TIMEOUT_MIN}m"
            break
        else
            echo "[$(date +'%H:%M:%S')] Test running... (${ELAPSED_MIN}m / $((DURATION/60))m target)"
            
            # Quick volume check
            if [ -d "$TEST_DIR" ]; then
                FILE_COUNT=$(find $TEST_DIR -type f 2>/dev/null | wc -l)
                echo "   📁 Files in results dir: $FILE_COUNT"
            fi
            
            sleep 30
        fi
    done
    
    echo ""
    echo "🔍 DETAILED RESULTS VERIFICATION:"
    echo "   Directory: $TEST_DIR"
    
    if [ -d "$TEST_DIR" ]; then
        echo "   ✅ Directory exists"
        
        # List all files
        FILES=$(find $TEST_DIR -type f 2>/dev/null)
        FILE_COUNT=$(echo "$FILES" | grep -v '^$' | wc -l)
        
        echo "   📊 Total files found: $FILE_COUNT"
        
        if [ $FILE_COUNT -gt 0 ]; then
            echo "   📁 Files list:"
            echo "$FILES" | while read file; do
                if [ -f "$file" ]; then
                    SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
                    echo "      - $(basename $file) (${SIZE} bytes)"
                fi
            done
            
            # Check for key files
            if [ -f "$TEST_DIR/summary_overall.csv" ]; then
                echo "   ✅ Key file found: summary_overall.csv"
                echo "      Content preview:"
                head -3 "$TEST_DIR/summary_overall.csv" | sed 's/^/      /'
            else
                echo "   ⚠️  Missing: summary_overall.csv"
            fi
            
            if [ -f "$TEST_DIR/combined_pod_counts.csv" ]; then
                echo "   ✅ Key file found: combined_pod_counts.csv"
                LINE_COUNT=$(wc -l < "$TEST_DIR/combined_pod_counts.csv")
                echo "      Lines: $LINE_COUNT"
            else
                echo "   ⚠️  Missing: combined_pod_counts.csv"
            fi
        else
            echo "   ❌ NO FILES FOUND - Results collection failed!"
            
            # Debug information
            echo ""
            echo "🔧 DEBUG INFORMATION:"
            echo "   Pod logs (last 20 lines):"
            kubectl logs -l job-name=load-test-hpa-validation-$i --tail=20 | sed 's/^/      /'
            
            echo ""
            echo "   Volume mount details:"
            kubectl describe pod -l job-name=load-test-hpa-validation-$i | grep -A5 -B5 "Volume\|Mount" | sed 's/^/      /'
        fi
    else
        echo "   ❌ Directory does not exist!"
    fi
    
    echo ""
    echo "🧹 Cleaning up job..."
    kubectl delete job load-test-hpa-validation-$i 2>/dev/null
    
    END_TIME=$(date +'%H:%M:%S')
    echo ""
    echo "✅ Validation test $i complete (Started: $(date -d "@$START_TIME" +'%H:%M:%S'), Ended: $END_TIME)"
    echo ""
    
    # Pause between tests
    if [ $i -lt $ITERATIONS ]; then
        echo "⏳ 30-second pause before next validation test..."
        sleep 30
        echo ""
    fi
done

echo "🏁 VALIDATION COMPLETE"
echo "====================="
echo ""
echo "📊 SUMMARY:"
for i in $(seq 1 $ITERATIONS); do
    TEST_DIR="hpa_validation_tests/test_$i"
    if [ -d "$TEST_DIR" ]; then
        FILE_COUNT=$(find $TEST_DIR -type f 2>/dev/null | wc -l)
        if [ $FILE_COUNT -gt 0 ]; then
            echo "   Test $i: ✅ SUCCESS ($FILE_COUNT files collected)"
        else
            echo "   Test $i: ❌ FAILED (no files collected)"
        fi
    else
        echo "   Test $i: ❌ FAILED (directory missing)"
    fi
done

echo ""
echo "🎯 NEXT STEPS:"
echo "   - If all tests successful: Proceed with full 30-minute tests"
echo "   - If tests failed: Review debug output above and fix issues"
echo "   - Check ~/hpa_validation_tests/ for collected data"