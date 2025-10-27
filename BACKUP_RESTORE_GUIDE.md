# Dataset Backup & Restore Guide

## Quick Commands

### Backup to GitHub
```bash
# SSH to EC2 server
ssh -i key_ta_2.pem ec2-user@ec2-52-54-42-13.compute-1.amazonaws.com

# Create backup
kubectl exec -it $(kubectl get pods -l app=predictive-scaler -o jsonpath='{.items[0].metadata.name}') -- \
  tar -czf /tmp/autoscaler_backup.tar.gz -C /data .

# Copy to local
kubectl cp predictive-scaler-pod:/tmp/autoscaler_backup.tar.gz ./autoscaler_backup.tar.gz

# Upload to GitHub (from your local machine)
git add autoscaler_backup.tar.gz
git commit -m "Backup autoscaler data"
git push
```

### Restore from GitHub
```bash
# Download backup
git pull
kubectl cp ./autoscaler_backup.tar.gz predictive-scaler-pod:/tmp/

# Restore data
kubectl exec -it predictive-scaler-pod -- \
  tar -xzf /tmp/autoscaler_backup.tar.gz -C /data

# Restart to load data
kubectl rollout restart deployment/predictive-scaler
```

## What Gets Backed Up

- **traffic_data.csv**: Current 240 data points
- **traffic_data_training.csv**: Full baseline dataset  
- **gru_model.h5**: Trained GRU model
- **gru_scaler.pkl**: Model preprocessing scaler
- **hw_model_*.pkl**: Holt-Winters models
- **performance_metrics.json**: MSE history

Total size: ~500 KB - 2 MB
├── scaler_X.pkl                  # Input scaler (~5 KB)
├── scaler_y.pkl                  # Output scaler (~2 KB)
├── predictions_history.json      # MSE calculation data
├── collection_status.json        # Collection metadata
└── baseline_datasets/            # All saved baselines
    ├── low_baseline.csv
    ├── medium_baseline.csv
    └── high_baseline.csv
```

**Size is perfect for GitHub!** (Under 2 MB compressed)

---

## Quick Start

### Backup (Before Switching AWS Accounts)

```bash
# On your Ubuntu server
cd ~/code-KP
chmod +x backup_autoscaler.sh
./backup_autoscaler.sh
```

This creates: `~/autoscaler_backup/autoscaler_data_<timestamp>.tar.gz`

### Upload to GitHub

```bash
mkdir -p ~/code-KP/backup_data
cp ~/autoscaler_backup/autoscaler_data_*.tar.gz ~/code-KP/backup_data/
cd ~/code-KP
git add backup_data/
git commit -m "Backup: Trained models and baselines"
git push
```

### Restore (On New AWS Account)

```bash
# Clone your repo on new server
git clone https://github.com/Not0nline/code-KP.git
cd code-KP

# Deploy autoscaler first
kubectl apply -f Autoscaler/predictive-scaler-pvc.yaml
kubectl apply -f Autoscaler/predictive-scaler-rbac.yaml
kubectl apply -f Autoscaler/predictive-scaler-deployment.yaml
kubectl apply -f Autoscaler/predictive-scaler-service.yaml

# Wait for pod
kubectl wait --for=condition=ready pod -l app=predictive-scaler --timeout=120s

# Restore data
chmod +x restore_autoscaler.sh
./restore_autoscaler.sh backup_data/autoscaler_data_*.tar.gz
```

### Verify Restoration

```bash
# Check baselines are available
kubectl exec -it deployment/predictive-scaler -- \
  curl http://localhost:5000/api/baseline/list

# Should return: {"baselines": ["low", "medium", "high"], "count": 3}

# Train models (will use restored data)
kubectl exec -it deployment/predictive-scaler -- \
  curl -X POST http://localhost:5000/api/models/train_all
```

---

## Manual Backup (Without Scripts)

### Backup

```bash
# Find pod name
POD=$(kubectl get pod -l app=predictive-scaler -o name | cut -d'/' -f2)

# Copy data
mkdir -p ~/autoscaler_backup
kubectl cp default/$POD:/data ~/autoscaler_backup/data

# Compress
cd ~/autoscaler_backup
tar -czf autoscaler_data_$(date +%Y%m%d).tar.gz data/
```

### Restore

```bash
# Extract
tar -xzf autoscaler_data_*.tar.gz

# Find new pod
POD=$(kubectl get pod -l app=predictive-scaler -o name | cut -d'/' -f2)

# Copy back
kubectl cp data/ default/$POD:/data

# Restart pod
kubectl rollout restart deployment/predictive-scaler
```

---

## Testing After Restore

Once data is restored, you can immediately run tests:

```bash
# Access load tester pod
kubectl exec -it deployment/load-tester -- /bin/bash
cd /app

# Start Control API
nohup python load_test.py > control_api.log 2>&1 &

# Run comprehensive tests
curl -X POST http://localhost:8080/start \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": ["low", "medium", "high"],
    "iterations": 1,
    "duration": 1800,
    "cooldown": 300
  }'
```

**No need to recollect baselines!** They're already restored.

---

## Backup Schedule Recommendations

### Critical Backups (Always Do):
1. **After collecting all baselines** (low, medium, high)
2. **Before switching AWS accounts**
3. **Before destroying/recreating cluster**

### Optional Backups:
- After successful model training
- Before major configuration changes
- After running comprehensive tests (if you want to preserve MSE history)

---

## Troubleshooting

### Backup Failed - Pod Not Found
```bash
# Check if pod is running
kubectl get pods -l app=predictive-scaler

# If not deployed yet:
kubectl apply -f Autoscaler/predictive-scaler-deployment.yaml
```

### Restore Failed - No /data Directory
```bash
# Check PVC is mounted
kubectl describe pod -l app=predictive-scaler | grep -A 5 Mounts

# If PVC missing:
kubectl apply -f Autoscaler/predictive-scaler-pvc.yaml
kubectl rollout restart deployment/predictive-scaler
```

### Baselines Not Showing After Restore
```bash
# Check files were copied
kubectl exec -it deployment/predictive-scaler -- ls -lh /data/baseline_datasets/

# If empty, retry copy
kubectl cp data/ default/$POD:/data

# Restart pod
kubectl rollout restart deployment/predictive-scaler
```

### Archive Too Large for GitHub
GitHub has a 100 MB file limit. Your backup should be ~2 MB, but if it grows:

```bash
# Check archive size
ls -lh ~/autoscaler_backup/autoscaler_data_*.tar.gz

# If over 100 MB, use alternative storage:
# - AWS S3
# - Google Drive
# - External hard drive
```

---

## What You Save

### Without Backup:
- ❌ 12+ hours to recollect baselines
- ❌ Lose all training progress
- ❌ Can't compare across clusters
- ❌ Start from zero every time

### With Backup:
- ✅ Restore in <1 minute
- ✅ Keep all training data
- ✅ Compare results across AWS accounts
- ✅ Resume testing immediately

---

## File Locations Summary

| What | Where (Old Cluster) | Where (New Cluster) |
|------|-------------------|---------------------|
| Training data | Pod: `/data` | GitHub: `backup_data/*.tar.gz` |
| Backup script | `~/code-KP/backup_autoscaler.sh` | Same |
| Restore script | `~/code-KP/restore_autoscaler.sh` | Same |
| Local backup | `~/autoscaler_backup/` | Same |

---

## Quick Commands Reference

```bash
# Backup
./backup_autoscaler.sh

# Upload to GitHub
git add backup_data/ && git commit -m "Backup" && git push

# Restore (on new cluster)
./restore_autoscaler.sh backup_data/autoscaler_data_*.tar.gz

# Verify
kubectl exec -it deployment/predictive-scaler -- curl http://localhost:5000/api/baseline/list
```

---

**Remember:** Always backup before switching AWS accounts!
