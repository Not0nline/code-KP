import os
import pandas as pd
import json

BASE_DIR = "saved_test"
TEST_TYPES = {
    "Ensemble": {
        "path": os.path.join(BASE_DIR, "11models"),
        "prefix": "ensemble",
        "result_file": "combined_results.csv"
    },
    "HPA": {
        "path": os.path.join(BASE_DIR, "HPA"),
        "prefix": "hpa-test",
        "result_file": "hpa_results.csv"
    },
    "Individual": {
        "path": os.path.join(BASE_DIR, "GRU_HW"),
        "prefix": "individual",
        "result_file": "combined_results.csv"
    }
}

SCENARIOS = ["low", "medium", "high"]
TEST_COUNT = 10
MIN_DURATION_SECONDS = 1700  # Allow some buffer for 30 mins (1800s)

def verify_test(test_path, result_filename):
    # Search for result file recursively
    found_files = []
    for root, dirs, files in os.walk(test_path):
        if result_filename in files:
            found_files.append(os.path.join(root, result_filename))
    
    if not found_files:
        return False, "Missing result file"
    
    # Check the most recent one or all? Let's check if ANY is valid.
    valid_runs = []
    for result_path in found_files:
        try:
            df = pd.read_csv(result_path)
            
            duration = 0
            if 'elapsed_seconds' in df.columns:
                duration = df['elapsed_seconds'].max()
            elif 'timestamp' in df.columns:
                # Try to parse timestamp
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
                except:
                    pass
            
            if duration >= MIN_DURATION_SECONDS:
                valid_runs.append(duration)
        except:
            continue
            
    if valid_runs:
        return True, f"Valid ({max(valid_runs):.2f}s)"
    
    return False, "No valid run found (duration too short or invalid CSV)"

report = {}

for type_name, config in TEST_TYPES.items():
    report[type_name] = {
        "valid": [],
        "invalid": [],
        "missing": []
    }
    
    base_path = config["path"]
    if not os.path.exists(base_path):
        print(f"Warning: Path not found {base_path}")
        # Mark all as missing
        for scenario in SCENARIOS:
            for i in range(1, TEST_COUNT + 1):
                test_name = f"{config['prefix']}-{scenario}-{i:02d}"
                report[type_name]["missing"].append(test_name)
        continue

    for scenario in SCENARIOS:
        for i in range(1, TEST_COUNT + 1):
            test_name = f"{config['prefix']}-{scenario}-{i:02d}"
            test_folder = os.path.join(base_path, test_name)
            
            if not os.path.exists(test_folder):
                report[type_name]["missing"].append(test_name)
            else:
                is_valid, message = verify_test(test_folder, config["result_file"])
                if is_valid:
                    report[type_name]["valid"].append(test_name)
                else:
                    report[type_name]["invalid"].append({"name": test_name, "reason": message})

print(json.dumps(report, indent=2))
