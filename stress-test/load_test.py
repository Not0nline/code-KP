#!/usr/bin/env python3
import requests
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import logging
import os
import sys
import csv
import shutil
import json
from requests.exceptions import ConnectTimeout, ReadTimeout
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('load_test.log')
    ]
)
logger = logging.getLogger("load-tester")

# --- URLs and Test Targets ---
DEFAULT_HPA_URL = "http://product-app-hpa-service.default.svc.cluster.local"
DEFAULT_COMBINED_URL = "http://product-app-combined-service.default.svc.cluster.local"
DEFAULT_PREDICTIVE_URL = "http://predictive-scaler.default.svc.cluster.local:5000"

TEST_TARGETS = {
    "hpa": False,
    "combined": True,
    "predictive": True
}

# --- Traffic Profile Configuration ---
class TrafficProfile(Enum):
    PREDICTABLE_CYCLES = "predictable_cycles"
    VOLATILE_SPIKES = "volatile_spikes"
    GRADUAL_GROWTH = "gradual_growth"

# --- General Load Configuration ---
PROFILE_SWITCH_INTERVAL = 600      # Switch profile every 10 minutes (600s)
NORMAL_LOAD_RANGE = (5, 15)        # Baseline load for most patterns
REQUEST_TIMEOUT = 60

# --- Profile-Specific Settings ---
# PREDICTABLE_CYCLES (Daily pattern with peaks)
PREDICTABLE_PEAK_LOAD_RANGE = (80, 120)
PREDICTABLE_PEAK_INTERVAL = 300    # Peak every 5 minutes
PREDICTABLE_PEAK_DURATION = 60     # Peak lasts for 1 minute

# VOLATILE_SPIKES (Chaotic, unpredictable traffic)
VOLATILE_SPIKE_CHANCE = 0.10       # 10% chance of a spike each second
VOLATILE_SPIKE_LOAD_RANGE = (70, 110)

# GRADUAL_GROWTH (Simulates a ramp-up event)
GRADUAL_GROWTH_TARGET = 100        # Target RPS at the end of the growth period

# --- Endpoint Distribution ---
ENDPOINT_WEIGHTS = {
    "product_list": 0.45,
    "product_create": 0.35,
    "load": 0.1,
    "health": 0.1
}

class LoadTester:
    """
    Advanced load tester with dynamic traffic profiles to test predictive scaling models.
    """
    def __init__(self, hpa_url, combined_url, predictive_url=None,
                 duration=3600, output_dir="./load_test_results",
                 test_hpa=True, test_combined=True, test_predictive=True):
        """Initialize the load tester."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"test_run_{timestamp}")
        self.hpa_url = hpa_url
        self.combined_url = combined_url
        self.predictive_url = predictive_url
        self.duration = duration
        self.stop_event = threading.Event()

        self.test_hpa = test_hpa
        self.test_combined = test_combined
        self.test_predictive = test_predictive and predictive_url is not None

        self.hpa_results, self.combined_results, self.predictive_results = [], [], []
        self.hpa_products, self.combined_products = [], []

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "patterns"), exist_ok=True)
        self.start_time = None

        self.hpa_lock = threading.Lock()
        self.combined_lock = threading.Lock()
        self.current_request_rate = 0

        active_services = [name for name, active in {"HPA": self.test_hpa, "Combined": self.test_combined, "Predictive": self.test_predictive}.items() if active]
        logger.info(f"Testing services: {', '.join(active_services)}")
        logger.info(f"Results will be saved to: {self.output_dir}")

    def generate_traffic_pattern(self):
        """
        Generate a dynamic traffic pattern by switching between different profiles.
        """
        time_points = list(range(self.duration))
        request_rates = []
        load_types = []
        
        profiles = [TrafficProfile.PREDICTABLE_CYCLES, TrafficProfile.VOLATILE_SPIKES, TrafficProfile.GRADUAL_GROWTH]
        
        for second in time_points:
            # Determine current profile
            profile_index = (second // PROFILE_SWITCH_INTERVAL) % len(profiles)
            current_profile = profiles[profile_index]
            
            rate = 0
            
            if current_profile == TrafficProfile.PREDICTABLE_CYCLES:
                # Daily sinusoidal pattern + sharp peaks
                is_peak_time = (second % PREDICTABLE_PEAK_INTERVAL) < PREDICTABLE_PEAK_DURATION
                if is_peak_time:
                    base_load = np.random.uniform(*PREDICTABLE_PEAK_LOAD_RANGE)
                    load_type = "PREDICTABLE_PEAK"
                else:
                    base_load = np.random.uniform(*NORMAL_LOAD_RANGE)
                    load_type = "PREDICTABLE_NORMAL"
                
                # Add a slow daily cycle
                daily_cycle = 10 * np.sin(2 * np.pi * second / (60 * 60)) # Hourly cycle for shorter tests
                rate = max(1, round(base_load + daily_cycle))

            elif current_profile == TrafficProfile.VOLATILE_SPIKES:
                # Low baseline with random, sharp spikes
                if np.random.random() < VOLATILE_SPIKE_CHANCE:
                    rate = round(np.random.uniform(*VOLATILE_SPIKE_LOAD_RANGE))
                    load_type = "VOLATILE_SPIKE"
                else:
                    rate = round(np.random.uniform(*NORMAL_LOAD_RANGE))
                    load_type = "VOLATILE_NORMAL"

            elif current_profile == TrafficProfile.GRADUAL_GROWTH:
                # Linearly increase load over the profile interval
                time_in_profile = second % PROFILE_SWITCH_INTERVAL
                growth_factor = time_in_profile / PROFILE_SWITCH_INTERVAL
                target_rate = NORMAL_LOAD_RANGE[0] + (GRADUAL_GROWTH_TARGET - NORMAL_LOAD_RANGE[0]) * growth_factor
                rate = max(1, round(np.random.uniform(target_rate - 5, target_rate + 5)))
                load_type = "GRADUAL_GROWTH"

            request_rates.append(rate)
            load_types.append(load_type)

        rate_map = {second: rate for second, rate in zip(time_points, request_rates)}
        
        # Save pattern details for analysis
        self.save_pattern_details(time_points, request_rates, load_types)
        logger.info("Generated dynamic traffic pattern with multiple profiles.")
        return rate_map

    def save_pattern_details(self, time_points, request_rates, load_types):
        """Save the generated traffic pattern to CSV and a summary file."""
        # Save detailed CSV
        df = pd.DataFrame({'second': time_points, 'requests': request_rates, 'type': load_types})
        df.to_csv(os.path.join(self.output_dir, "patterns", "full_traffic_pattern.csv"), index=False)

        # Save summary
        with open(os.path.join(self.output_dir, "patterns", "pattern_summary.txt"), 'w') as f:
            f.write("DYNAMIC TRAFFIC PATTERN SUMMARY\n")
            f.write("===============================\n\n")
            f.write(f"Test Duration: {self.duration} seconds\n")
            f.write(f"Profile Switch Interval: {PROFILE_SWITCH_INTERVAL} seconds\n\n")
            
            profiles_used = sorted(list(set(load_types)), key=lambda x: x.split('_')[0])
            for profile_type in profiles_used:
                profile_rates = [r for r, t in zip(request_rates, load_types) if t == profile_type]
                if profile_rates:
                    f.write(f"--- Profile: {profile_type} ---\n")
                    f.write(f"  - Duration: {len(profile_rates)} seconds\n")
                    f.write(f"  - Average Load: {np.mean(profile_rates):.2f} req/sec\n")
                    f.write(f"  - Min/Max Load: {np.min(profile_rates)}/{np.max(profile_rates)} req/sec\n\n")

    def generate_product_data(self):
        """Generate random product data for create requests."""
        product_types = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse"]
        brands = ["Apple", "Samsung", "Dell", "HP", "Logitech", "Sony"]
        return {
            "name": f"{np.random.choice(brands)}_{np.random.choice(product_types)}_{np.random.randint(100, 999)}",
            "description": "High performance computing device.",
            "price": round(np.random.uniform(100, 2000), 2)
        }

    def make_predictive_request(self, url, results_list):
        """Send a GET request to the predictive scaler's /predict endpoint."""
        endpoint = "/predict"
        full_url = f"{url}{endpoint}"
        try:
            start_time = time.time()
            response = requests.get(full_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            status_code = response.status_code
            response_time_ms = (time.time() - start_time) * 1000
        except (ConnectTimeout, ReadTimeout) as e:
            logger.error(f"Predictive request timed out to {full_url}: {e}")
            status_code, response_time_ms, error_msg = -1, -1, "Timeout"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making predictive request to {full_url}: {e}")
            status_code = e.response.status_code if e.response else -1
            response_time_ms = (time.time() - start_time) * 1000 if 'start_time' in locals() else -1
            error_msg = str(e)
        
        results_list.append({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()) if self.start_time else 0,
            "url": full_url, "method": "GET", "endpoint": endpoint,
            "status_code": status_code, "response_time_ms": response_time_ms,
            "error": locals().get('error_msg'), "service": "Predictive", "endpoint_type": "predict"
        })

    def make_requests(self, url, second, request_count, results_list, products_list, lock, name):
        """Make a fixed number of requests to a target URL."""
        successful, failed = 0, 0
        self.current_request_rate = request_count
        if request_count == 0: return
        
        delay = 0.8 / (request_count + 1)
        for _ in range(request_count):
            time.sleep(delay)
            endpoint_type = np.random.choice(list(ENDPOINT_WEIGHTS.keys()), p=list(ENDPOINT_WEIGHTS.values()))
            method, data, endpoint = "GET", None, "/health"
            
            if endpoint_type == "product_list":
                endpoint = "/product/list"
            elif endpoint_type == "product_create":
                method, endpoint, data = "POST", "/product/create", self.generate_product_data()
            elif endpoint_type == "load":
                endpoint = f"/load?duration=1&intensity={np.random.randint(10, 30)}"

            full_url = f"{url}{endpoint}"
            start_time = time.time()
            try:
                if method == "GET":
                    response = requests.get(full_url, timeout=REQUEST_TIMEOUT)
                else:
                    response = requests.post(full_url, json=data, headers={'Content-Type': 'application/json'}, timeout=REQUEST_TIMEOUT)
                
                status_code = response.status_code
                if 200 <= status_code < 400:
                    successful += 1
                    if endpoint_type == "product_create" and status_code == 201:
                        with lock: products_list.append(response.json().get('product', {}).get('id'))
                else:
                    failed += 1
                
            except (ConnectTimeout, ReadTimeout) as e:
                failed += 1; status_code = -1; logger.error(f"Timeout to {full_url}: {e}")
            except Exception as e:
                failed += 1; status_code = -1; logger.error(f"Error to {full_url}: {e}")

            results_list.append({
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()),
                "url": url, "method": method, "endpoint": endpoint, "status_code": status_code,
                "response_time_ms": (time.time() - start_time) * 1000,
                "service": name, "endpoint_type": endpoint_type
            })

    def run_test(self):
        """Run the load test, switching between traffic profiles."""
        logger.info(f"Starting dynamic load test for {self.duration} seconds")
        self.request_pattern = self.generate_traffic_pattern()
        self.start_time = datetime.now()
        
        current_profile_name = ""
        for second in range(self.duration):
            if self.stop_event.is_set(): break
            
            # Log profile changes
            profile_index = (second // PROFILE_SWITCH_INTERVAL) % len(list(TrafficProfile))
            profile_name = list(TrafficProfile)[profile_index].value
            if profile_name != current_profile_name:
                logger.info(f"Second {second}: Switching to traffic profile -> {profile_name.upper()}")
                current_profile_name = profile_name

            request_count = self.request_pattern.get(second, 1)
            self.current_request_rate = request_count
            
            threads = []
            if self.test_hpa:
                threads.append(threading.Thread(target=self.make_requests, args=(self.hpa_url, second, request_count, self.hpa_results, self.hpa_products, self.hpa_lock, "HPA")))
            if self.test_combined:
                threads.append(threading.Thread(target=self.make_requests, args=(self.combined_url, second, request_count, self.combined_results, self.combined_products, self.combined_lock, "Combined")))
            if self.test_predictive:
                threads.append(threading.Thread(target=self.make_predictive_request, args=(self.predictive_url, self.predictive_results)))

            for t in threads: t.start()
            for t in threads: t.join(timeout=0.95)
            
            if second % 60 == 0 and second > 0:
                logger.info(f"Progress: {second}/{self.duration}s. Current Rate: {request_count}/sec. Profile: {current_profile_name.upper()}")

        end_time = datetime.now()
        logger.info(f"Test finished at {end_time.isoformat()}. Total duration: {(end_time - self.start_time).total_seconds():.1f}s")
        self.save_results()
        try:
            shutil.copy('load_test.log', os.path.join(self.output_dir, 'load_test.log'))
        except Exception as e:
            logger.error(f"Could not copy log file: {e}")

    def save_results(self):
        """Save all results to CSV files and generate summaries."""
        try:
            if self.test_hpa and self.hpa_results:
                pd.DataFrame(self.hpa_results).to_csv(os.path.join(self.output_dir, "hpa_results.csv"), index=False)
            if self.test_combined and self.combined_results:
                pd.DataFrame(self.combined_results).to_csv(os.path.join(self.output_dir, "combined_results.csv"), index=False)
            if self.test_predictive and self.predictive_results:
                pd.DataFrame(self.predictive_results).to_csv(os.path.join(self.output_dir, "predictive_results.csv"), index=False)
            
            self.create_summary_statistics()
            logger.info(f"All results saved to {self.output_dir}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def create_summary_statistics(self):
        """Generate and save summary statistics from the test results."""
        all_results = []
        if self.test_hpa: all_results.extend(self.hpa_results)
        if self.test_combined: all_results.extend(self.combined_results)
        if not all_results: return

        df = pd.DataFrame(all_results)
        df['minute'] = df['elapsed_seconds'] // 60
        
        summary = df.groupby(['service', 'minute', 'endpoint_type']).agg(
            total_requests=('status_code', 'count'),
            success_rate=('status_code', lambda x: np.mean([1 if 200 <= s < 400 else 0 for s in x])),
            avg_response_time_ms=('response_time_ms', 'mean'),
            p95_response_time_ms=('response_time_ms', lambda x: np.percentile(x, 95) if len(x) > 0 else 0)
        ).reset_index()
        
        summary.to_csv(os.path.join(self.output_dir, "summary_by_minute.csv"), index=False)
        
        overall_summary = df.groupby('service').agg(
            total_requests=('status_code', 'count'),
            success_rate=('status_code', lambda x: np.mean([1 if 200 <= s < 400 else 0 for s in x])),
            avg_response_time_ms=('response_time_ms', 'mean'),
            p95_response_time_ms=('response_time_ms', lambda x: np.percentile(x, 95) if len(x) > 0 else 0)
        ).reset_index()
        
        logger.info("--- Overall Test Summary ---")
        logger.info(overall_summary.to_string(index=False))
        overall_summary.to_csv(os.path.join(self.output_dir, "summary_overall.csv"), index=False)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run dynamic load test for predictive autoscaling.')
    parser.add_argument('--duration', type=int, default=1800, help='Test duration in seconds (default: 1800)')
    parser.add_argument('--output-dir', type=str, default='./load_test_results', help='Output directory for results')
    parser.add_argument('--hpa-url', type=str, default=DEFAULT_HPA_URL)
    parser.add_argument('--combined-url', type=str, default=DEFAULT_COMBINED_URL)
    parser.add_argument('--predictive-url', type=str, default=DEFAULT_PREDICTIVE_URL)
    return parser.parse_args()

def main():
    """Main function to run the load tester."""
    args = parse_arguments()
    
    tester = LoadTester(
        hpa_url=args.hpa_url,
        combined_url=args.combined_url,
        predictive_url=args.predictive_url,
        duration=args.duration,
        output_dir=args.output_dir,
        test_hpa=TEST_TARGETS["hpa"],
        test_combined=TEST_TARGETS["combined"],
        test_predictive=TEST_TARGETS["predictive"]
    )

    try:
        tester.run_test()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user. Saving partial results...")
        tester.stop_event.set()
        tester.save_results()
    except Exception as e:
        logger.error(f"Test failed with an unexpected error: {e}", exc_info=True)
        tester.stop_event.set()
        tester.save_results()

if __name__ == '__main__':
    main()
