#!/bin/bash
# Fix predictive-scaler code by mounting correct app.py via ConfigMap

APP_FILE="Autoscaler/app.py"
REMOTE_APP_PATH="/home/ubuntu/app.py"

# Function to apply fix on a cluster
apply_fix() {
    HOST=$1
    KEY=$2
    NAME=$3
    
    echo "=== Applying Fix to $NAME ($HOST) ==="
    
    # 1. Upload app.py
    echo "Uploading app.py..."
    scp -i "$KEY" "$APP_FILE" "ubuntu@$HOST:$REMOTE_APP_PATH"
    
    # 2. Create ConfigMap
    echo "Creating ConfigMap..."
    ssh -i "$KEY" "ubuntu@$HOST" "sudo kubectl create configmap predictive-scaler-code-fix --from-file=/app/app.py=$REMOTE_APP_PATH --dry-run=client -o yaml | sudo kubectl apply -f -"
    
    # 3. Patch Deployment
    echo "Patching Deployment..."
    ssh -i "$KEY" "ubuntu@$HOST" "sudo kubectl patch deployment predictive-scaler --patch '
spec:
  template:
    spec:
      containers:
      - name: predictive-scaler
        volumeMounts:
        - name: code-volume
          mountPath: /app/app.py
          subPath: app.py
      volumes:
      - name: code-volume
        configMap:
          name: predictive-scaler-code-fix
'"
    
    echo "Waiting for rollout..."
    ssh -i "$KEY" "ubuntu@$HOST" "sudo kubectl rollout status deployment predictive-scaler"
    echo "Done with $NAME."
}

# Cluster 2 (HPA/INVERSE)
# apply_fix "ec2-34-239-4-82.compute-1.amazonaws.com" "keys/key_ta4.pem" "Cluster 2"

# Cluster 3 (Individual)
apply_fix "ec2-13-216-71-213.compute-1.amazonaws.com" "keys/key_ta5.pem" "Cluster 3"
