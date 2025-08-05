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

# Default endpoints
DEFAULT_HPA_URL = "http://product-app-hpa-service.default.svc.cluster.local"
DEFAULT_COMBINED_URL = "http://product-app-combined-service.default.svc.cluster.local"
DEFAULT_PREDICTIVE_URL = "http://predictive-scaler.default.svc.cluster.local:5000"

# Traffic pattern configuration – lower, more manageable load
NORMAL_LOAD_RANGE = (15, 25)       # Normal load requests per second
PEAK_LOAD_RANGE = (30, 50)         # Peak load requests per second
DISTURBANCE_INTERVAL = 25          # Seconds between peak disturbances (about 25 sec)
DISTURBANCE_DURATION = 10          # Peak duration in seconds
VOLATILITY_FACTOR = 1              # Volatility factor for minor spikes/dips

# Endpoint weights (probability distribution)
ENDPOINT_WEIGHTS = {
    "product_list": 0.45,       # 45% of requests to /product/list
    "product_create": 0.35,     # 35% of requests to /product/create
    "load": 0.1,              # 10% to generate load via /load
    "health": 0.1             # 10% to /health endpoint
}

# Request timeout (in seconds; increased to accommodate load endpoint)
REQUEST_TIMEOUT = 60

# Updated config for autoscaler (if used in integration tests)
# In your autoscaler config the CPU threshold is set to 3%
CPU_THRESHOLD = 3.0  # 3%

class LoadTester:
    """
    Load testing implementation that tests all application endpoints
    at a manageable load level using a moderately volatile traffic pattern.
    """
    def __init__(self, hpa_url, combined_url, predictive_url=None,
                 duration=3600, output_dir="./load_test_results"):
        """Initialize the load tester."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"test_run_{timestamp}")
        self.hpa_url = hpa_url
        self.combined_url = combined_url
        self.predictive_url = predictive_url
        self.duration = duration
        self.stop_event = threading.Event()

        # Results storage
        self.hpa_results = []
        self.combined_results = []
        self.predictive_results = []

        # Created products tracking
        self.hpa_products = []
        self.combined_products = []

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "patterns"), exist_ok=True)
        self.start_time = None

        # Thread safety for product lists
        self.hpa_lock = threading.Lock()
        self.combined_lock = threading.Lock()

        self.current_request_rate = 0

        logger.info(f"Results will be saved to: {self.output_dir}")

    def generate_traffic_pattern(self):
        """
        Generate a traffic pattern with ramp-up, normal, and periodic disturbance (peak) periods.
        The resulting pattern is saved as CSV files for reference.
        """
        time_points = list(range(self.duration))
        request_rates = []
        load_types = []

        normal_min, normal_max = NORMAL_LOAD_RANGE
        peak_min, peak_max = PEAK_LOAD_RANGE

        ramp_up_duration = min(300, self.duration // 6)
        for second in time_points:
            if second < ramp_up_duration:
                # Gradually ramp up
                ramp_factor = second / ramp_up_duration
                current_normal_min = max(1, normal_min * ramp_factor)
                current_normal_max = max(1, normal_max * ramp_factor)
                current_peak_min = max(1, peak_min * ramp_factor)
                current_peak_max = max(2, peak_max * ramp_factor)
            else:
                current_normal_min = normal_min
                current_normal_max = normal_max
                current_peak_min = peak_min
                current_peak_max = peak_max

            is_disturbance = (second % DISTURBANCE_INTERVAL < DISTURBANCE_DURATION)
            current_time = datetime.now() + timedelta(seconds=second)
            is_business_hours = 9 <= current_time.hour < 17
            
            if is_disturbance:
                # Use peak values during disturbances
                if is_business_hours:
                    base_value = np.random.uniform(current_peak_min * 1.2, current_peak_max * 1.2)
                    load_type = "BUSINESS_PEAK"
                else:
                    base_value = np.random.uniform(current_peak_min, current_peak_max)
                    load_type = "NORMAL_PEAK"
            else:
                # Normal load
                if is_business_hours:
                    base_value = np.random.uniform(current_normal_min * 1.5, current_normal_max * 1.5)
                    load_type = "BUSINESS_NORMAL"
                else:
                    base_value = np.random.uniform(current_normal_min, current_normal_max)
                    load_type = "NORMAL"

            # Add small random volatility
            if np.random.random() < VOLATILITY_FACTOR:
                volatility = np.random.uniform(1, 3) if np.random.random() < 0.7 else np.random.uniform(-1, -0.5)
            else:
                volatility = 0

            # Sinusoidal variation over a 5-minute cycle
            variation = 2 * np.sin(2 * np.pi * second / (5 * 60))

            final_load = max(1, round(base_value + volatility + variation))
            request_rates.append(final_load)
            load_types.append(load_type)

        rate_map = {second: rate for second, rate in zip(time_points, request_rates)}
        # Save detailed pattern
        with open(os.path.join(self.output_dir, "patterns", "detailed_pattern.csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['second', 'requests', 'type'])
            for s, r, t in zip(time_points, request_rates, load_types):
                writer.writerow([s, r, t])

        # Save a summary
        pattern_summary = {
            "NORMAL": {
                "count": load_types.count("NORMAL"),
                "avg_requests": round(np.mean([r for r, t in zip(request_rates, load_types) if t == "NORMAL"]), 2) if load_types.count("NORMAL") else 0
            },
            "BUSINESS_NORMAL": {
                "count": load_types.count("BUSINESS_NORMAL"),
                "avg_requests": round(np.mean([r for r, t in zip(request_rates, load_types) if t == "BUSINESS_NORMAL"]), 2) if load_types.count("BUSINESS_NORMAL") else 0
            },
            "NORMAL_PEAK": {
                "count": load_types.count("NORMAL_PEAK"),
                "avg_requests": round(np.mean([r for r, t in zip(request_rates, load_types) if t == "NORMAL_PEAK"]), 2) if load_types.count("NORMAL_PEAK") else 0
            },
            "BUSINESS_PEAK": {
                "count": load_types.count("BUSINESS_PEAK"),
                "avg_requests": round(np.mean([r for r, t in zip(request_rates, load_types) if t == "BUSINESS_PEAK"]), 2) if load_types.count("BUSINESS_PEAK") else 0
            }
        }
        with open(os.path.join(self.output_dir, "patterns", "pattern_summary.txt"), 'w') as f:
            f.write("TRAFFIC PATTERN SUMMARY\n")
            f.write("======================\n\n")
            f.write(f"Test Duration: {self.duration} seconds\n")
            f.write(f"Normal Load Range: {NORMAL_LOAD_RANGE} req/sec\n")
            f.write(f"Peak Load Range: {PEAK_LOAD_RANGE} req/sec\n")
            f.write(f"Disturbance Interval: Every {DISTURBANCE_INTERVAL} seconds\n")
            f.write(f"Disturbance Duration: {DISTURBANCE_DURATION} seconds\n")
            f.write(f"Volatility Factor: {VOLATILITY_FACTOR}\n\n")
            f.write("PATTERN BREAKDOWN\n")
            f.write("----------------\n")
            for pt, stats in pattern_summary.items():
                f.write(f"{pt}: {stats['count']} seconds, Avg: {stats['avg_requests']} req/sec\n")
        self.save_pattern_chart(time_points, request_rates, load_types)
        logger.info("Generated moderate traffic pattern with ramp-up")
        return rate_map

    def save_pattern_chart(self, time_points, request_rates, load_types):
        """Save the pattern as a CSV for visualization."""
        try:
            df = pd.DataFrame({
                'second': time_points,
                'requests': request_rates,
                'type': load_types
            })
            df.to_csv(os.path.join(self.output_dir, "patterns", "pattern.csv"), index=False)
        except Exception as e:
            logger.error(f"Failed to save pattern chart: {e}")

    def generate_product_data(self):
        """Generate random product data for create requests."""
        product_types = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse",
                         "Headphones", "Speaker", "Camera", "Printer"]
        brands = ["Apple", "Samsung", "Dell", "HP", "Logitech", "Sony", "LG", "Asus",
                  "Lenovo", "Microsoft"]
        descriptions = [
            "High performance computing device.",
            "Compact and powerful mobile phone.",
            "Versatile device for work and entertainment.",
            "Sharp display for clear visuals.",
            "Ergonomic and responsive input device.",
            "Precision pointing device.",
            "Immersive sound experience.",
            "Rich audio output.",
            "Capture life's moments with clarity.",
            "Reliable document output."
        ]
        product_type = np.random.choice(product_types)
        brand = np.random.choice(brands)
        model = f"Model-{np.random.randint(100, 999)}"
        price = round(np.random.uniform(10, 2000), 2)
        description = np.random.choice(descriptions)
        return {
            "name": f"{brand}_{product_type}_{model}",
            "description": description,
            "price": price
        }

    def make_predictive_request(self, url, results_list):
        """Send a GET request to the predictive scaler’s /predict endpoint."""
        endpoint = "/predict"
        full_url = f"{url}{endpoint}"
        try:
            start_time = time.time()
            response = requests.get(full_url, timeout=REQUEST_TIMEOUT)
            end_time = time.time()
            response.raise_for_status()
            status_code = response.status_code
            response_time_ms = (end_time - start_time) * 1000
            try:
                response_body = response.json()
                logger.debug(f"Predictive response: {response_body}")
            except Exception:
                logger.debug(f"Predictive response (non-JSON): {response.text}")
        except (ConnectTimeout, ReadTimeout) as e:
            logger.error(f"Predictive request timed out to {full_url}: {e}")
            status_code = -1
            response_time_ms = -1
            error_msg = "Timeout"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making predictive request to {full_url}: {e}")
            status_code = e.response.status_code if e.response is not None else -1
            response_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
        results_list.append({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()) if self.start_time else 0,
            "url": full_url,
            "method": "GET",
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "error": locals().get('error_msg', None),
            "service": "Predictive",
            "endpoint_type": "predict"
        })

    def make_requests(self, url, second, request_count, results_list, products_list, lock, name):
        """
        Make a fixed number of requests to a target URL.
        Requests are evenly distributed within the second using a small delay.
        """
        successful = 0
        failed = 0
        self.current_request_rate = request_count
        delay = 0.8 / (request_count + 1)
        for i in range(request_count):
            try:
                if i > 0:
                    time.sleep(delay)
                endpoint_type = np.random.choice(
                    list(ENDPOINT_WEIGHTS.keys()),
                    p=list(ENDPOINT_WEIGHTS.values())
                )
                method = "GET"
                data = None
                endpoint = "/health"
                if endpoint_type == "product_list":
                    endpoint = "/product/list"
                elif endpoint_type == "product_create":
                    method = "POST"
                    endpoint = "/product/create"
                    data = self.generate_product_data()
                elif endpoint_type == "load":
                    duration_param = 1
                    intensity = np.random.randint(10, 30)
                    endpoint = f"/load?duration={duration_param}&intensity={intensity}"
                elif endpoint_type == "health":
                    endpoint = "/health"
                start_time = time.time()
                full_url = f"{url}{endpoint}"
                if method == "GET":
                    response = requests.get(full_url, timeout=REQUEST_TIMEOUT)
                else:
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(full_url, data=json.dumps(data), headers=headers, timeout=REQUEST_TIMEOUT)
                end_time = time.time()
                status_code = response.status_code
                response_time_ms = (end_time - start_time) * 1000
                if 200 <= status_code < 400:
                    successful += 1
                    if endpoint_type == "product_create" and status_code == 201:
                        try:
                            resp_data = response.json()
                            if 'product' in resp_data and 'id' in resp_data['product']:
                                product_id = resp_data['product']['id']
                                with lock:
                                    products_list.append(product_id)
                                logger.debug(f"Created product {product_id} for {name}")
                        except Exception as e:
                            logger.warning(f"Could not extract product ID: {e}")
                else:
                    failed += 1
                    logger.warning(f"Request to {full_url} failed with status {status_code}")
                results_list.append({
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()),
                    "url": url,
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "response_time_ms": response_time_ms,
                    "request_count": request_count,
                    "service": name,
                    "endpoint_type": endpoint_type
                })
            except (ConnectTimeout, ReadTimeout) as e:
                failed += 1
                logger.error(f"Timeout error making request to {url}{endpoint}: {e}")
                results_list.append({
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()),
                    "url": url,
                    "method": method if 'method' in locals() else "unknown",
                    "endpoint": endpoint if 'endpoint' in locals() else "unknown",
                    "status_code": -1,
                    "response_time_ms": -1,
                    "error": str(e),
                    "request_count": request_count,
                    "service": name,
                    "endpoint_type": endpoint_type if 'endpoint_type' in locals() else "unknown"
                })
            except Exception as e:
                failed += 1
                logger.error(f"Error making request to {url}{endpoint}: {e}")
                results_list.append({
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()),
                    "url": url,
                    "method": method if 'method' in locals() else "unknown",
                    "endpoint": endpoint if 'endpoint' in locals() else "unknown",
                    "status_code": -1,
                    "response_time_ms": -1,
                    "error": str(e),
                    "request_count": request_count,
                    "service": name,
                    "endpoint_type": endpoint_type if 'endpoint_type' in locals() else "unknown"
                })
        if request_count > 0:
            total = successful + failed
            success_rate = (successful / total) * 100 if total > 0 else 0
            logger.debug(f"Second {second}: Made {total} requests to {name} (S:{successful}/F:{failed}, {success_rate:.1f}% success)")
        return successful, failed

    def run_test(self):
        """
        Run the load test for the specified duration.
        For each second, the number of requests is determined by the traffic pattern.
        Results are stored to disk periodically.
        """
        logger.info(f"Starting functional load test for {self.duration} seconds")
        logger.info(f"HPA URL: {self.hpa_url}")
        logger.info(f"Combined URL: {self.combined_url}")
        if self.predictive_url:
            logger.info(f"Predictive Scaler URL: {self.predictive_url}")
        self.request_pattern = self.generate_traffic_pattern()
        self.start_time = datetime.now()
        logger.info(f"Test started at {self.start_time.isoformat()}")
        logger.info(f"Testing endpoints with weights: {ENDPOINT_WEIGHTS}")
        current_pattern_type = "NORMAL"
        rate_history = []
        for second in range(self.duration):
            if self.stop_event.is_set():
                logger.info("Stopping test early due to stop event")
                break
            request_count = self.request_pattern.get(second, 1)
            rate_history.append(request_count)
            if len(rate_history) > 120:
                rate_history.pop(0)
            self.current_request_rate = request_count
            is_disturbance = (second % DISTURBANCE_INTERVAL < DISTURBANCE_DURATION)
            current_time = datetime.now() + timedelta(seconds=second)
            is_business_hours = (9 <= current_time.hour < 17)
            if is_disturbance and is_business_hours:
                pattern_type = "BUSINESS_PEAK"
            elif is_disturbance:
                pattern_type = "NORMAL_PEAK"
            elif is_business_hours:
                pattern_type = "BUSINESS_NORMAL"
            else:
                pattern_type = "NORMAL"
            if pattern_type != current_pattern_type:
                logger.info(f"Second {second}: Pattern changed from {current_pattern_type} to {pattern_type}")
                current_pattern_type = pattern_type
            hpa_thread = threading.Thread(
                target=lambda: self.make_requests(
                    self.hpa_url, second, request_count,
                    self.hpa_results, self.hpa_products, self.hpa_lock, "HPA"
                )
            )
            combined_thread = threading.Thread(
                target=lambda: self.make_requests(
                    self.combined_url, second, request_count,
                    self.combined_results, self.combined_products, self.combined_lock, "Combined"
                )
            )
            threads = [hpa_thread, combined_thread]
            if self.predictive_url:
                pred_thread = threading.Thread(
                    target=lambda: self.make_predictive_request(self.predictive_url, self.predictive_results)
                )
                threads.append(pred_thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=0.95)
            if second % 30 == 0 and second > 0:
                elapsed = second
                remaining = self.duration - second
                progress = elapsed / self.duration * 100
                hpa_count = len(self.hpa_results)
                combined_count = len(self.combined_results)
                hpa_successes = sum(1 for r in self.hpa_results if 200 <= r.get("status_code", 0) < 400)
                combined_successes = sum(1 for r in self.combined_results if 200 <= r.get("status_code", 0) < 400)
                hpa_success_rate = hpa_successes / hpa_count * 100 if hpa_count > 0 else 0
                combined_success_rate = combined_successes / combined_count * 100 if combined_count > 0 else 0
                avg_rate = sum(rate_history[-30:]) / min(30, len(rate_history[-30:]))
                logger.info(f"Progress: {elapsed}/{self.duration} seconds ({progress:.1f}%). Remaining: {remaining} seconds")
                logger.info(f"Current request rate: {request_count}/sec, Avg last 30s: {avg_rate:.1f}/sec")
                logger.info(f"Requests made: HPA={hpa_count}, Combined={combined_count}")
                logger.info(f"Success rates: HPA={hpa_success_rate:.1f}%, Combined={combined_success_rate:.1f}%")
                logger.info(f"Products created: HPA={len(self.hpa_products)}, Combined={len(self.combined_products)}")
                if second % 300 == 0 and second > 0:
                    self.save_partial_results(f"partial_{second}")
        end_time = datetime.now()
        logger.info(f"Test ended at {end_time.isoformat()}")
        logger.info(f"Total duration: {(end_time - self.start_time).total_seconds():.1f} seconds")
        total_hpa = len(self.hpa_results)
        total_combined = len(self.combined_results)
        logger.info(f"Total requests: HPA={total_hpa}, Combined={total_combined}")
        logger.info(f"Products created: HPA={len(self.hpa_products)}, Combined={len(self.combined_products)}")
        self.save_results()
        try:
            shutil.copy('load_test.log', os.path.join(self.output_dir, 'load_test.log'))
            logger.info(f"Load test log copied to {self.output_dir}/load_test.log")
        except Exception as e:
            logger.error(f"Could not copy log file: {e}")
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "hpa_requests": total_hpa,
            "combined_requests": total_combined,
            "hpa_success_rate": self.calculate_success_rate(self.hpa_results),
            "combined_success_rate": self.calculate_success_rate(self.combined_results),
            "hpa_products_created": len(self.hpa_products),
            "combined_products_created": len(self.combined_products),
            "output_directory": self.output_dir
        }

    def calculate_success_rate(self, results):
        """Calculate success rate excluding health endpoint requests."""
        if not results:
            return 0
        functional_requests = [r for r in results if r.get("endpoint_type", "") != "health"]
        if not functional_requests:
            functional_requests = results
        successes = sum(1 for r in functional_requests if 200 <= r.get("status_code", 0) < 400)
        return successes / len(functional_requests)

    def save_partial_results(self, prefix):
        """Save partial results to disk."""
        try:
            partial_dir = os.path.join(self.output_dir, "partial")
            os.makedirs(partial_dir, exist_ok=True)
            if self.hpa_results:
                pd.DataFrame(self.hpa_results).to_csv(os.path.join(partial_dir, f"{prefix}_hpa_results.csv"), index=False)
            if self.combined_results:
                pd.DataFrame(self.combined_results).to_csv(os.path.join(partial_dir, f"{prefix}_combined_results.csv"), index=False)
            if self.predictive_results:
                pd.DataFrame(self.predictive_results).to_csv(os.path.join(partial_dir, f"{prefix}_predictive_results.csv"), index=False)
            logger.info(f"Saved partial results with prefix '{prefix}'")
        except Exception as e:
            logger.error(f"Error saving partial results: {e}")

    def save_results(self):
        """Save all results to CSV and JSON files."""
        try:
            if self.hpa_results:
                hpa_df = pd.DataFrame(self.hpa_results)
                hpa_df.to_csv(os.path.join(self.output_dir, "hpa_results.csv"), index=False)
                logger.info(f"Saved HPA results to {self.output_dir}/hpa_results.csv")
            if self.combined_results:
                combined_df = pd.DataFrame(self.combined_results)
                combined_df.to_csv(os.path.join(self.output_dir, "combined_results.csv"), index=False)
                logger.info(f"Saved Combined results to {self.output_dir}/combined_results.csv")
            if self.predictive_results:
                predictive_df = pd.DataFrame(self.predictive_results)
                predictive_df.to_csv(os.path.join(self.output_dir, "predictive_results.csv"), index=False)
                logger.info(f"Saved Predictive results to {self.output_dir}/predictive_results.csv")
            with open(os.path.join(self.output_dir, "hpa_products.json"), 'w') as f:
                json.dump(self.hpa_products, f)
            with open(os.path.join(self.output_dir, "combined_products.json"), 'w') as f:
                json.dump(self.combined_products, f)
            self.create_summary_statistics()
            self.save_request_rate_data()
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def save_request_rate_data(self):
        """Save the per-second request count for future analysis."""
        try:
            results_df = pd.DataFrame(self.hpa_results + self.combined_results)
            if results_df.empty:
                logger.warning("No results to create request rate data")
                return
            results_df['second'] = results_df['elapsed_seconds']
            rate_data = results_df.groupby('second').size().reset_index(name='request_count')
            rate_data.to_csv(os.path.join(self.output_dir, "request_rate.csv"), index=False)
            logger.info(f"Saved request rate data to {self.output_dir}/request_rate.csv")
            rate_data['timestamp'] = rate_data['second'].apply(lambda x: (self.start_time + timedelta(seconds=int(x))).isoformat())
            rate_data.to_csv(os.path.join(self.output_dir, "request_rate_timestamped.csv"), index=False)
        except Exception as e:
            logger.error(f"Error saving request rate data: {e}")

    def create_summary_statistics(self):
        """Generate summary statistics from the test results and save them."""
        try:
            all_results = []
            if self.hpa_results:
                for result in self.hpa_results:
                    r_copy = result.copy()
                    r_copy["service"] = "HPA"
                    all_results.append(r_copy)
            if self.combined_results:
                for result in self.combined_results:
                    r_copy = result.copy()
                    r_copy["service"] = "Combined"
                    all_results.append(r_copy)
            if not all_results:
                logger.warning("No results to create summary statistics")
                return
            results_df = pd.DataFrame(all_results)
            results_df['minute'] = results_df['elapsed_seconds'] // 60
            endpoint_summary = results_df.groupby(['minute', 'service', 'endpoint_type']).agg({
                'status_code': [
                    ('total', 'count'),
                    ('success', lambda x: sum(1 for s in x if 200 <= s < 400)),
                    ('error', lambda x: sum(1 for s in x if s < 200 or s >= 400))
                ],
                'response_time_ms': [
                    ('avg', 'mean'),
                    ('p95', lambda x: np.percentile(x, 95) if len(x) > 0 else 0)
                ]
            }).reset_index()
            endpoint_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in endpoint_summary.columns.values]
            endpoint_summary['success_rate'] = endpoint_summary['status_code_success'] / endpoint_summary['status_code_total']
            endpoint_summary.to_csv(os.path.join(self.output_dir, "endpoint_summary.csv"), index=False)
            time_summary = results_df.groupby(['minute', 'service']).agg({
                'status_code': [
                    ('total', 'count'),
                    ('success', lambda x: sum(1 for s in x if 200 <= s < 400)),
                    ('error', lambda x: sum(1 for s in x if s < 200 or s >= 400))
                ],
                'response_time_ms': [
                    ('avg', 'mean'),
                    ('median', 'median'),
                    ('p95', lambda x: np.percentile(x, 95) if len(x) > 0 else 0),
                    ('min', lambda x: np.min(x) if len(x) > 0 else 0),
                    ('max', lambda x: np.max(x) if len(x) > 0 else 0)
                ]
            }).reset_index()
            time_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in time_summary.columns.values]
            time_summary['success_rate'] = time_summary['status_code_success'] / time_summary['status_code_total']
            time_summary.to_csv(os.path.join(self.output_dir, "summary_by_minute.csv"), index=False)
            results_df['is_health'] = results_df['endpoint_type'] == 'health'
            overall = results_df.groupby(['service', 'is_health']).agg({
                'status_code': [
                    ('total', 'count'),
                    ('success', lambda x: sum(1 for s in x if 200 <= s < 400)),
                    ('error', lambda x: sum(1 for s in x if s < 200 or s >= 400))
                ],
                'response_time_ms': [
                    ('avg', 'mean'),
                    ('median', 'median'),
                    ('p95', lambda x: np.percentile(x, 95) if len(x) > 0 else 0),
                    ('min', lambda x: np.min(x) if len(x) > 0 else 0),
                    ('max', lambda x: np.max(x) if len(x) > 0 else 0)
                ]
            }).reset_index()
            overall.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in overall.columns.values]
            overall['success_rate'] = overall['status_code_success'] / overall['status_code_total']
            overall.to_csv(os.path.join(self.output_dir, "summary_overall.csv"), index=False)
            functional_results = results_df[~results_df['is_health']]
            functional_summary = functional_results.groupby(['service']).agg({
                'status_code': [
                    ('total', 'count'),
                    ('success', lambda x: sum(1 for s in x if 200 <= s < 400)),
                    ('error', lambda x: sum(1 for s in x if s < 200 or s >= 400))
                ],
                'response_time_ms': [
                    ('avg', 'mean'),
                    ('p95', lambda x: np.percentile(x, 95) if len(x) > 0 else 0)
                ]
            }).reset_index()
            functional_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in functional_summary.columns.values]
            functional_summary['success_rate'] = functional_summary['status_code_success'] / functional_summary['status_code_total']
            functional_summary['endpoint_type'] = 'functional'
            all_summary = results_df.groupby(['service']).agg({
                'status_code': [
                    ('total', 'count'),
                    ('success', lambda x: sum(1 for s in x if 200 <= s < 400)),
                    ('error', lambda x: sum(1 for s in x if s < 200 or s >= 400))
                ],
                'response_time_ms': [
                    ('avg', 'mean'),
                    ('p95', lambda x: np.percentile(x, 95) if len(x) > 0 else 0)
                ]
            }).reset_index()
            all_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in all_summary.columns.values]
            all_summary['success_rate'] = all_summary['status_code_success'] / all_summary['status_code_total']
            all_summary['endpoint_type'] = 'all'
            combined_summary = pd.concat([functional_summary, all_summary])
            combined_summary.to_csv(os.path.join(self.output_dir, "simple_summary.csv"), index=False)
            logger.info("Summary statistics created and saved:")
            for _, row in functional_summary.iterrows():
                svc = row['service']
                total = row['status_code_total']
                success = row['status_code_success']
                success_rate = row['success_rate'] * 100
                avg_resp = row['response_time_ms_avg']
                p95_resp = row['response_time_ms_p95']
                logger.info(f"{svc}: {total} requests, {success_rate:.1f}% success, {avg_resp:.1f}ms avg, {p95_resp:.1f}ms p95")
            endpoint_counts = results_df.groupby(['service', 'endpoint_type']).size().reset_index(name='count')
            for svc in endpoint_counts['service'].unique():
                logger.info(f"\n{svc} Service:")
                svc_data = endpoint_counts[endpoint_counts['service'] == svc]
                total = svc_data['count'].sum()
                for _, row in svc_data.iterrows():
                    endpoint = row['endpoint_type']
                    count = row['count']
                    pct = (count / total) * 100
                    logger.info(f"  {endpoint}: {count} requests ({pct:.1f}%)")
        except Exception as e:
            logger.error(f"Error creating summary statistics: {e}")
            logger.exception(e)

def parse_arguments():
    """Parse command line arguments for the load test."""
    parser = argparse.ArgumentParser(description='Run functional load test for product app')
    parser.add_argument('--hpa-url', type=str, default=DEFAULT_HPA_URL, help='URL for HPA service')
    parser.add_argument('--combined-url', type=str, default=DEFAULT_COMBINED_URL, help='URL for Combined service')
    parser.add_argument('--predictive-url', type=str, default=DEFAULT_PREDICTIVE_URL, help='URL for Predictive Scaler service')
    parser.add_argument('--duration', type=int, default=1800, help='Test duration in seconds (default: 1800)')
    parser.add_argument('--output-dir', type=str, default='./load_test_results', help='Output directory for results')
    parser.add_argument('--normal-min', type=int, default=NORMAL_LOAD_RANGE[0], help=f'Minimum normal requests (default: {NORMAL_LOAD_RANGE[0]})')
    parser.add_argument('--normal-max', type=int, default=NORMAL_LOAD_RANGE[1], help=f'Maximum normal requests (default: {NORMAL_LOAD_RANGE[1]})')
    parser.add_argument('--peak-min', type=int, default=PEAK_LOAD_RANGE[0], help=f'Minimum peak requests (default: {PEAK_LOAD_RANGE[0]})')
    parser.add_argument('--peak-max', type=int, default=PEAK_LOAD_RANGE[1], help=f'Maximum peak requests (default: {PEAK_LOAD_RANGE[1]})')
    parser.add_argument('--disturbance-interval', type=int, default=DISTURBANCE_INTERVAL, help=f'Seconds between disturbances (default: {DISTURBANCE_INTERVAL})')
    parser.add_argument('--disturbance-duration', type=int, default=DISTURBANCE_DURATION, help=f'Duration of each disturbance (default: {DISTURBANCE_DURATION} seconds)')
    parser.add_argument('--volatility', type=float, default=VOLATILITY_FACTOR, help=f'Volatility factor (default: {VOLATILITY_FACTOR})')
    parser.add_argument('--timeout', type=int, default=REQUEST_TIMEOUT, help=f'Request timeout in seconds (default: {REQUEST_TIMEOUT})')
    return parser.parse_args()

def main():
    """Main function – parse arguments and start the load test."""
    args = parse_arguments()
    global NORMAL_LOAD_RANGE, PEAK_LOAD_RANGE, DISTURBANCE_INTERVAL, DISTURBANCE_DURATION, VOLATILITY_FACTOR, REQUEST_TIMEOUT
    NORMAL_LOAD_RANGE = (args.normal_min, args.normal_max)
    PEAK_LOAD_RANGE = (args.peak_min, args.peak_max)
    DISTURBANCE_INTERVAL = args.disturbance_interval
    DISTURBANCE_DURATION = args.disturbance_duration
    VOLATILITY_FACTOR = args.volatility
    REQUEST_TIMEOUT = args.timeout

    logger.info("Starting Moderate Load Test")
    logger.info("=========================")
    logger.info(f"HPA URL: {args.hpa_url}")
    logger.info(f"Combined URL: {args.combined_url}")
    logger.info(f"Predictive URL: {args.predictive_url}")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Normal load range: {NORMAL_LOAD_RANGE} req/sec")
    logger.info(f"Peak load range: {PEAK_LOAD_RANGE} req/sec")
    logger.info(f"Disturbance interval: {DISTURBANCE_INTERVAL} seconds")
    logger.info(f"Disturbance duration: {DISTURBANCE_DURATION} seconds")
    logger.info(f"Volatility factor: {VOLATILITY_FACTOR}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT} seconds")
    logger.info(f"Endpoint weights: {ENDPOINT_WEIGHTS}")
    logger.info("=========================")

    tester = LoadTester(
        hpa_url=args.hpa_url,
        combined_url=args.combined_url,
        predictive_url=args.predictive_url,
        duration=args.duration,
        output_dir=args.output_dir
    )

    try:
        summary = tester.run_test()
        logger.info("Load test completed successfully")
        logger.info("==============================")
        logger.info(f"Start time: {summary['start_time']}")
        logger.info(f"End time: {summary['end_time']}")
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info(f"Total HPA requests: {summary['hpa_requests']}")
        logger.info(f"Total Combined requests: {summary['combined_requests']}")
        logger.info(f"HPA success rate: {summary['hpa_success_rate'] * 100:.2f}%")
        logger.info(f"Combined success rate: {summary['combined_success_rate'] * 100:.2f}%")
        logger.info(f"HPA products created: {summary['hpa_products_created']}")
        logger.info(f"Combined products created: {summary['combined_products_created']}")
        logger.info(f"Results saved to: {summary['output_directory']}")
        logger.info("==============================")
        logger.info("\nTo retrieve the results, run:")
        logger.info(f"kubectl cp load-test:<pod_name>:{summary['output_directory']} ./")
    except KeyboardInterrupt:
        logger.info("Test interrupted by user. Saving partial results...")
        tester.stop_event.set()
        tester.save_results()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.exception(e)
        tester.stop_event.set()
        tester.save_results()
        sys.exit(2)

if __name__ == '__main__':
    main()