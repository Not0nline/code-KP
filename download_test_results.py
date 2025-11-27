#!/usr/bin/env python3
"""
Download test results from Prometheus for completed tests.
Downloads metrics for LOW 1-10, MEDIUM 1-10, HIGH 1-5
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
import time

# SSH connection details
SSH_KEY = r"C:\Users\Adrial\Documents\ta\key_ta3.pem"
SSH_HOST = "ubuntu@ec2-18-214-68-201.compute-1.amazonaws.com"

# Test configuration
TESTS = {
    "low": list(range(1, 11)),      # Tests 1-10
    "medium": list(range(1, 11)),   # Tests 1-10
    "high": list(range(1, 6))       # Tests 1-5
}

# Output directory
OUTPUT_DIR = r"saved_tests\Six_Models"

# Test duration (30 minutes = 1800 seconds)
TEST_DURATION = 1800

# Metrics to collect
METRICS = {
    "pod_count": 'kube_deployment_status_replicas{deployment="product-app-combined"}',
    "cpu_usage": 'sum(rate(container_cpu_usage_seconds_total{pod=~"product-app-combined.*"}[1m]))',
    "memory_usage": 'sum(container_memory_working_set_bytes{pod=~"product-app-combined.*"})',
    "request_rate": 'sum(rate(http_requests_total{job="product-app-combined-service"}[1m]))',
    "response_time_p50": 'histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{job="product-app-combined-service"}[1m])) by (le))',
    "response_time_p95": 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="product-app-combined-service"}[1m])) by (le))',
    "response_time_p99": 'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job="product-app-combined-service"}[1m])) by (le))',
    "success_rate": '(sum(rate(http_requests_total{job="product-app-combined-service",status="200"}[1m])) / sum(rate(http_requests_total{job="product-app-combined-service"}[1m]))) * 100',
    "predicted_load": 'predictive_scaler_predicted_load',
    "required_pods": 'predictive_scaler_required_pods',
    "best_model": 'predictive_scaler_best_model',
    "model_mse_gru": 'predictive_scaler_model_mse{model="gru"}',
    "model_mse_holt_winters": 'predictive_scaler_model_mse{model="holt_winters"}',
    "model_mse_lstm": 'predictive_scaler_model_mse{model="lstm"}',
    "model_mse_lightgbm": 'predictive_scaler_model_mse{model="lightgbm"}',
    "model_mse_xgboost": 'predictive_scaler_model_mse{model="xgboost"}',
    "model_mse_statuscale": 'predictive_scaler_model_mse{model="statuscale"}',
}


def get_test_timestamps():
    """
    Parse the test log to get start times for each test.
    Returns dict: {(level, number): start_timestamp}
    """
    print("📅 Extracting test timestamps from log...")
    
    # Download the log file
    cmd = f'ssh -i "{SSH_KEY}" {SSH_HOST} "cat test_6models_fresh_start.log"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to get log file: {result.stderr}")
        return None
    
    log_content = result.stdout
    
    # Parse timestamps from log
    # We need to estimate based on test sequence and 30min duration + 30s pause
    # Each test takes 1800s + 30s = 1830s = 30.5 minutes
    
    timestamps = {}
    
    # Get the first test start time from the log
    # Look for "STARTING ALL 30 TESTS" or first test creation
    lines = log_content.split('\n')
    
    # Find first job creation to establish baseline
    first_test_time = None
    for line in lines:
        if "job.batch/load-test-low-1 created" in line:
            # We'll need to get this from kubectl events or estimate
            break
    
    # Since we can't get exact timestamps from the log, we'll need to get them from Kubernetes events
    print("⚠️  Log doesn't have timestamps. Will use Kubernetes events to find test times...")
    return None


def get_test_times_from_k8s():
    """
    Get test start times from Kubernetes events.
    This is more reliable than parsing logs.
    """
    print("📅 Getting test times from Kubernetes events...")
    
    # Get all job creation events
    cmd = f'ssh -i "{SSH_KEY}" {SSH_HOST} "sudo kubectl get events --sort-by=.metadata.creationTimestamp -o json"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to get events: {result.stderr}")
        return None
    
    try:
        events = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("❌ Failed to parse events JSON")
        return None
    
    test_times = {}
    
    for event in events.get('items', []):
        if event.get('reason') == 'SuccessfulCreate' and 'involvedObject' in event:
            obj = event['involvedObject']
            if obj.get('kind') == 'Job' and 'load-test-' in obj.get('name', ''):
                job_name = obj['name']
                timestamp = event['metadata']['creationTimestamp']
                
                # Parse job name: load-test-{level}-{number}
                parts = job_name.replace('load-test-', '').split('-')
                if len(parts) == 2:
                    level, number = parts[0], int(parts[1])
                    test_times[(level, number)] = timestamp
                    print(f"  Found: {level.upper()} Test {number} at {timestamp}")
    
    return test_times


def query_prometheus(query, start_time, end_time, step='15s'):
    """
    Query Prometheus for a metric over a time range.
    Returns the JSON response.
    """
    # Convert ISO timestamps to Unix timestamps
    if isinstance(start_time, str):
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        start_ts = int(start_dt.timestamp())
    else:
        start_ts = int(start_time)
    
    if isinstance(end_time, str):
        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        end_ts = int(end_dt.timestamp())
    else:
        end_ts = int(end_time)
    
    # Build Prometheus query URL
    prom_url = f"http://localhost:9090/api/v1/query_range?query={query}&start={start_ts}&end={end_ts}&step={step}"
    
    # Execute via kubectl exec
    cmd = f'ssh -i "{SSH_KEY}" {SSH_HOST} "sudo kubectl exec -n monitoring prometheus-prometheus-kube-prometheus-prometheus-0 -c prometheus -- wget -qO- \'{prom_url}\'"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"    ⚠️  Query failed: {result.stderr}")
        return None
    
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"    ⚠️  Invalid JSON response")
        return None


def download_test_metrics(level, number, start_time, end_time):
    """
    Download all metrics for a single test.
    """
    print(f"\n📊 Downloading {level.upper()} Test {number}...")
    
    test_dir = os.path.join(OUTPUT_DIR, level, f"test_{number}")
    os.makedirs(test_dir, exist_ok=True)
    
    results = {
        "test_info": {
            "level": level,
            "number": number,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": TEST_DURATION
        },
        "metrics": {}
    }
    
    # Download each metric
    for metric_name, metric_query in METRICS.items():
        print(f"  - {metric_name}...", end=" ", flush=True)
        
        data = query_prometheus(metric_query, start_time, end_time)
        
        if data and data.get('status') == 'success':
            results["metrics"][metric_name] = data['data']
            print("✅")
        else:
            print("❌")
            results["metrics"][metric_name] = None
    
    # Save results
    output_file = os.path.join(test_dir, "metrics.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  💾 Saved to {output_file}")
    
    return results


def main():
    print("=" * 80)
    print("DOWNLOADING TEST RESULTS FROM PROMETHEUS")
    print("=" * 80)
    
    # Create output directories
    for level in TESTS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, level), exist_ok=True)
    
    # Get test times from Kubernetes
    test_times = get_test_times_from_k8s()
    
    if not test_times:
        print("\n❌ Could not determine test times from Kubernetes events.")
        print("⚠️  Using estimated times based on log file creation...")
        
        # Use the log file birth time as first test start
        # 2025-11-06 08:46:35 UTC (from stat test_6models_fresh_start.log)
        first_test_start = datetime(2025, 11, 6, 8, 46, 35)
        
        # Generate estimated times for all tests
        test_times = {}
        current_time = first_test_start
        
        for level in ["low", "medium", "high"]:
            for number in TESTS[level]:
                test_times[(level, number)] = current_time.isoformat() + 'Z'
                current_time += timedelta(seconds=1830)  # 30 min test + 30 sec pause
    
    # Download results for each completed test
    total_tests = sum(len(numbers) for numbers in TESTS.values())
    completed = 0
    
    for level in ["low", "medium", "high"]:
        for number in TESTS[level]:
            if (level, number) not in test_times:
                print(f"\n⚠️  Skipping {level.upper()} Test {number} - no timestamp found")
                continue
            
            start_time = test_times[(level, number)]
            
            # Calculate end time (start + 30 minutes)
            if isinstance(start_time, str):
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            else:
                start_dt = start_time
            
            end_dt = start_dt + timedelta(seconds=TEST_DURATION)
            end_time = end_dt.isoformat().replace('+00:00', 'Z')
            
            # Download metrics
            download_test_metrics(level, number, start_time, end_time)
            
            completed += 1
            print(f"\n📈 Progress: {completed}/{total_tests} tests downloaded")
            
            # Small delay to avoid overwhelming Prometheus
            time.sleep(1)
    
    print("\n" + "=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
