#!/bin/bash
# Quick 3-minute validation test with very low RPS (won't affect HPA test)
# This test just verifies all 5+ models are being tracked in model_metrics.csv

echo "🧪 Running quick 3-minute validation test (very low RPS)"
echo "   This won't affect the HPA test running in parallel"
echo ""

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: load-test-models-validation
spec:
  ttlSecondsAfterFinished: 600
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      volumes:
      - name: results
        persistentVolumeClaim:
          claimName: load-test-results-pvc
      containers:
      - name: load-tester
        image: 4dri4l/load-tester:latest
        imagePullPolicy: Always
        env:
        - name: TARGET
          value: "both"
        - name: DURATION
          value: "180"
        - name: SCENARIO
          value: "low"
        - name: TEST_PREDICTIVE
          value: "true"
        - name: PREDICTIVE_URL
          value: "http://predictive-scaler.default.svc.cluster.local:5000"
        - name: OUTPUT_DIR
          value: "models_validation"
        volumeMounts:
        - name: results
          mountPath: /results
EOF

echo "✅ Job created: load-test-models-validation"
echo "⏳ Waiting for test to complete (max 5 minutes)..."

kubectl wait --for=condition=complete --timeout=300s job/load-test-models-validation

echo ""
echo "✅ Test completed! Checking results..."
echo ""

# Check for model_metrics.csv
kubectl run file-checker --image=busybox --rm -i --restart=Never \
  --overrides='{
    "spec": {
      "volumes": [{"name": "results", "persistentVolumeClaim": {"claimName": "load-test-results-pvc"}}],
      "containers": [{
        "name": "checker",
        "image": "busybox",
        "command": ["sh", "-c", "echo \"📊 Model Metrics File:\"; find /results/models_validation -name model_metrics.csv -exec ls -lh {} \\; && echo \"\" && echo \"📝 Unique models found:\"; find /results/models_validation -name model_metrics.csv -exec tail -n +2 {} \\; | cut -d, -f3 | sort -u | nl"],
        "volumeMounts": [{" name": "results", "mountPath": "/results"}]
      }]
    }
  }'

echo ""
echo "🧹 Cleaning up job..."
kubectl delete job load-test-models-validation

echo ""
echo "✅ Validation complete!"
