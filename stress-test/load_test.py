#!/usr/bin/env python3
import requests
from requests.adapters import HTTPAdapter
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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Optional
try:
    from prometheus_client import start_http_server as prom_start_http_server, Counter as PromCounter, Gauge as PromGauge
    PROM_AVAILABLE = True
except Exception:
    PROM_AVAILABLE = False
    prom_start_http_server = None
    PromCounter = None
    PromGauge = None

def _coerce_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


CONFIG_SCHEMA = {
    "TARGET": ("both", str),
    "DURATION": (1800, int),
    "HPA_URL": ("http://product-app-hpa-service.default.svc.cluster.local", str),
    "COMBINED_URL": ("http://product-app-combined-service.default.svc.cluster.local", str),
    "PREDICTIVE_URL": ("http://predictive-scaler.default.svc.cluster.local:5000", str),
    "TEST_PREDICTIVE": (False, _coerce_bool),
    "MAX_CONCURRENCY": (384, int),
    "METRICS_PORT": (9105, int),
    "NORMAL_MIN": (1, int),
    "NORMAL_MAX": (3, int),
    "PEAK_MIN": (5, int),
    "PEAK_MAX": (10, int),
    "SEASON_DURATION": (300, int),
    "PEAK_DURATION": (60, int),
    "PEAK_OFFSET": (60, int),
    "VOLATILITY": (0.1, float),
    "TIMEOUT": (60.0, float),
    "LOAD_INTENSITY_NORMAL_MIN": (3, int),
    "LOAD_INTENSITY_NORMAL_MAX": (8, int),
    "LOAD_INTENSITY_PEAK_MIN": (15, int),
    "LOAD_INTENSITY_PEAK_MAX": (30, int),
    "LOAD_DURATION_NORMAL": (1.0, float),
    "LOAD_DURATION_PEAK": (0.5, float),
    "CLIENT_TIMEOUT_NORMAL": (15.0, float),
    "CLIENT_TIMEOUT_PEAK": (10.0, float),
    "TICK_GRACE": (2.0, float),
    "MAX_RPS": (400, int),
    "CHAOS_ERROR_RATE": (0.0, float),
    "BURST_ON_PEAK_SECONDS": (2, int),
    "BURST_ON_PEAK_MULTIPLIER": (1.5, float),
    "CPU_THRESHOLD": (3.0, float),
    "RPS_PER_POD": (50, int),
}


def _cast_value(raw_value, caster, default):
    try:
        return caster(raw_value)
    except (TypeError, ValueError):
        return default


def _load_config():
    config = {}
    for key, (default, caster) in CONFIG_SCHEMA.items():
        raw = os.getenv(key)
        if raw is None or raw == "":
            config[key] = default
        else:
            config[key] = _cast_value(raw, caster, default)
    return config


CONFIG = _load_config()

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

# Default endpoints sourced from configuration
DEFAULT_HPA_URL = CONFIG["HPA_URL"]
DEFAULT_COMBINED_URL = CONFIG["COMBINED_URL"]
DEFAULT_PREDICTIVE_URL = CONFIG["PREDICTIVE_URL"]
TEST_TARGETS = {
    "hpa": False,        # Test HPA service
    "combined": True,    # Test Combined service  
    "predictive": True   # Test Predictive service
}

# Traffic pattern configuration - seasons with a single, sharp peak
NORMAL_LOAD_RANGE = (CONFIG["NORMAL_MIN"], CONFIG["NORMAL_MAX"])
PEAK_LOAD_RANGE = (CONFIG["PEAK_MIN"], CONFIG["PEAK_MAX"])
SEASON_DURATION = CONFIG["SEASON_DURATION"]
PEAK_DURATION = CONFIG["PEAK_DURATION"]
PEAK_START_OFFSET = CONFIG["PEAK_OFFSET"]
VOLATILITY_FACTOR = CONFIG["VOLATILITY"]

# Extra burst right at peak start to emulate a DDoS-like surge
BURST_ON_PEAK_SECONDS = CONFIG["BURST_ON_PEAK_SECONDS"]
BURST_ON_PEAK_MULTIPLIER = CONFIG["BURST_ON_PEAK_MULTIPLIER"]

# Endpoint weights (probability distribution)
ENDPOINT_WEIGHTS = {
    "product_list": 0.45,       # 45% of requests to /product/list
    "product_create": 0.35,     # 35% of requests to /product/create
    "load": 0.1,                # 10% to generate load via /load
    "health": 0.1               # 10% to /health endpoint
}

# Heavier distribution during spike windows (more CPU-heavy traffic)
PEAK_ENDPOINT_WEIGHTS = {
    "product_list": 0.25,
    "product_create": 0.20,
    "load": 0.50,
    "health": 0.05,
}

# Intensity ranges for /load endpoint - DRASTICALLY reduced to prevent timeouts
# The /load endpoint does intensity*200 iterations, each with 100 random ops
# intensity=80 = 1.6 MILLION operations = 3-5 seconds response time!
LOAD_INTENSITY_NORMAL = (
    CONFIG["LOAD_INTENSITY_NORMAL_MIN"],
    CONFIG["LOAD_INTENSITY_NORMAL_MAX"],
)
LOAD_INTENSITY_PEAK = (
    CONFIG["LOAD_INTENSITY_PEAK_MIN"],
    CONFIG["LOAD_INTENSITY_PEAK_MAX"],
)

# Duration ranges (seconds) for /load endpoint
# Longer duration with lower intensity = more CPU work spread over time
LOAD_DURATION_NORMAL = CONFIG["LOAD_DURATION_NORMAL"]
LOAD_DURATION_PEAK = CONFIG["LOAD_DURATION_PEAK"]

# Request timeout (in seconds; increased to accommodate load endpoint)
REQUEST_TIMEOUT = CONFIG["TIMEOUT"]

# Updated config for autoscaler (if used in integration tests)
# In your autoscaler config the CPU threshold is set to 3%
CPU_THRESHOLD = CONFIG["CPU_THRESHOLD"]

class LoadTester:
    """
    Load testing implementation that tests selected application endpoints
    at a manageable load level using a moderately volatile traffic pattern.
    """
    def __init__(self, hpa_url, combined_url, predictive_url=None,
                 duration=None, output_dir="./load_test_results",
                 test_hpa=True, test_combined=True, test_predictive=True,
                 max_concurrency: int = CONFIG["MAX_CONCURRENCY"],
                 metrics_enabled: bool = False,
                 chaos_error_rate: float = CONFIG["CHAOS_ERROR_RATE"],
                 client_timeout_normal: float = CONFIG["CLIENT_TIMEOUT_NORMAL"],
                 client_timeout_peak: float = CONFIG["CLIENT_TIMEOUT_PEAK"],
                 tick_grace: float = CONFIG["TICK_GRACE"],
                 max_rps: int = CONFIG["MAX_RPS"]):
        """Initialize the load tester."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"test_run_{timestamp}")
        self.hpa_url = hpa_url
        self.combined_url = combined_url
        self.predictive_url = predictive_url
        self.duration = duration if duration is not None else CONFIG["DURATION"]
        self.stop_event = threading.Event()
        self.max_concurrency = max(1, int(max_concurrency))
        self.metrics_enabled = bool(metrics_enabled and PROM_AVAILABLE)
        self.chaos_error_rate = max(0.0, min(1.0, float(chaos_error_rate)))

        # Client behavior tuning
        self.client_timeout_normal = float(client_timeout_normal)
        self.client_timeout_peak = float(client_timeout_peak)
        self.tick_grace = float(tick_grace)
        self.max_rps = int(max(1, max_rps))

        # Test target flags
        self.test_hpa = test_hpa
        self.test_combined = test_combined
        self.test_predictive = test_predictive and predictive_url is not None

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

        # Thread safety for product lists AND results lists
        self.hpa_lock = threading.Lock()
        self.combined_lock = threading.Lock()
        # CRITICAL: Need locks for results lists too - multiple threads append simultaneously!
        self.hpa_results_lock = threading.Lock()
        self.combined_results_lock = threading.Lock()

        self.current_request_rate = 0

        # HTTP sessions with pooled connections per service for higher throughput
        self.sessions = {}
        adapter = HTTPAdapter(pool_connections=self.max_concurrency, pool_maxsize=self.max_concurrency)
        if self.test_hpa:
            s = requests.Session()
            s.mount('http://', adapter)
            s.mount('https://', adapter)
            self.sessions['HPA'] = s
        if self.test_combined:
            s = requests.Session()
            s.mount('http://', adapter)
            s.mount('https://', adapter)
            self.sessions['Combined'] = s
        if self.test_predictive:
            s = requests.Session()
            s.mount('http://', adapter)
            s.mount('https://', adapter)
            self.sessions['Predictive'] = s

        # Metrics setup
        self.prev_indices = {}
        if self.metrics_enabled:
            self.metrics_requests_total = PromCounter(
                'load_tester_requests_total', 'Requests made by load tester', ['service', 'outcome']
            )
            self.metrics_rps = PromGauge(
                'load_tester_rps', 'Per-second achieved RPS by service', ['service']
            )
            self.metrics_scheduled_rps = PromGauge(
                'load_tester_scheduled_rps', 'Per-second scheduled RPS by service', ['service']
            )
            self.metrics_error_rate = PromGauge(
                'load_tester_error_rate', 'Per-second error rate by service', ['service']
            )
            if self.test_hpa:
                self.prev_indices['HPA'] = 0
            if self.test_combined:
                self.prev_indices['Combined'] = 0

        # Log which services will be tested
        active_services = []
        if self.test_hpa:
            active_services.append("HPA")
        if self.test_combined:
            active_services.append("Combined")
        if self.test_predictive:
            active_services.append("Predictive")
        
        logger.info(f"Testing services: {', '.join(active_services)}")
        logger.info(f"Results will be saved to: {self.output_dir}")

    def _update_client_metrics_for_service(self, service_name: str, results_list: list, lock: threading.Lock):
        if not self.metrics_enabled:
            return
        # CRITICAL: Lock while reading results_list to prevent race conditions
        with lock:
            start_idx = self.prev_indices.get(service_name, 0)
            new_entries = results_list[start_idx:]
            total = len(new_entries)
            successes = sum(1 for r in new_entries if 200 <= r.get('status_code', 0) < 400)
            errors = total - successes
            # Advance index while still holding lock
            self.prev_indices[service_name] = start_idx + total
        
        # Set gauges for this second (outside lock - metrics are thread-safe)
        self.metrics_rps.labels(service=service_name).set(total)
        self.metrics_error_rate.labels(service=service_name).set((errors / total) if total > 0 else 0.0)
        # Increment counters cumulatively
        if successes:
            self.metrics_requests_total.labels(service=service_name, outcome='success').inc(successes)
        if errors:
            self.metrics_requests_total.labels(service=service_name, outcome='error').inc(errors)

    def update_client_metrics(self):
        if not self.metrics_enabled:
            return
        if self.test_hpa:
            self._update_client_metrics_for_service('HPA', self.hpa_results, self.hpa_results_lock)
        if self.test_combined:
            self._update_client_metrics_for_service('Combined', self.combined_results, self.combined_results_lock)

    def generate_traffic_pattern(self):
        """
        Generate a seasonal traffic pattern with a sudden spike during the peak window.
        No smoothing: at PEAK_START_OFFSET an instant jump to PEAK_LOAD_RANGE occurs
        for PEAK_DURATION seconds, then it drops back to NORMAL_LOAD_RANGE.
        """
        time_points = list(range(self.duration))
        request_rates = []
        load_types = []

        normal_min, normal_max = NORMAL_LOAD_RANGE
        peak_min, peak_max = PEAK_LOAD_RANGE

        # Calculate number of complete seasons
        num_complete_seasons = self.duration // SEASON_DURATION
        peak_start = PEAK_START_OFFSET
        peak_end = PEAK_START_OFFSET + PEAK_DURATION
        logger.info(f"Generating {num_complete_seasons} seasons of {SEASON_DURATION} seconds each")
        logger.info(f"Sudden spike pattern: Peak window is [{peak_start}, {peak_end}) each season")
        logger.info(f"Volatility factor: {VOLATILITY_FACTOR}")

        for second in time_points:
            # Position within current season
            season_second = second % SEASON_DURATION
            in_peak = (peak_start <= season_second < peak_end)

            # Choose base rate range instantly switching at boundaries
            if in_peak:
                base = np.random.randint(peak_min, peak_max + 1)
                load_types.append("SEASONAL_PEAK")
            else:
                base = np.random.randint(normal_min, normal_max + 1)
                load_types.append("SEASONAL_NORMAL")

            # Apply symmetric jitter if configured
            if VOLATILITY_FACTOR > 0:
                jitter_span = int(base * VOLATILITY_FACTOR)
                if jitter_span > 0:
                    jitter = np.random.randint(-jitter_span, jitter_span + 1)
                    base = max(0, base + jitter)

            request_rates.append(int(base))

        pattern_summary = {
            "SEASONAL_NORMAL": {
                "count": load_types.count("SEASONAL_NORMAL"),
                "avg_requests": round(np.mean([r for r, t in zip(request_rates, load_types) if t == "SEASONAL_NORMAL"]), 2) if load_types.count("SEASONAL_NORMAL") else 0
            },
            "SEASONAL_PEAK": {
                "count": load_types.count("SEASONAL_PEAK"),
                "avg_requests": round(np.mean([r for r, t in zip(request_rates, load_types) if t == "SEASONAL_PEAK"]), 2) if load_types.count("SEASONAL_PEAK") else 0
            }
        }

        peak_seconds_per_season = PEAK_DURATION
        normal_seconds_per_season = SEASON_DURATION - PEAK_DURATION

        with open(os.path.join(self.output_dir, "patterns", "pattern_summary.txt"), 'w') as f:
            f.write("SUDDEN-SPIKE SEASONAL TRAFFIC PATTERN\n")
            f.write("=====================================\n\n")
            f.write("Pattern for triggering fast scale-up then allowing scale-down\n\n")
            f.write(f"Test Duration: {self.duration} seconds\n")
            f.write(f"Season Duration: {SEASON_DURATION} seconds\n")
            f.write(f"Complete Seasons: {num_complete_seasons}\n")
            f.write(f"Peak Window: {PEAK_START_OFFSET}-{PEAK_START_OFFSET + PEAK_DURATION}s into each season\n")
            f.write(f"Peak Duration: {PEAK_DURATION} seconds\n")
            f.write(f"Normal Load: {NORMAL_LOAD_RANGE} req/sec\n")
            f.write(f"Peak Load: {PEAK_LOAD_RANGE} req/sec\n")
            f.write(f"Volatility: {VOLATILITY_FACTOR}\n\n")
            f.write("PATTERN BREAKDOWN\n")
            f.write("----------------\n")
            for pt, stats in pattern_summary.items():
                f.write(f"{pt}: {stats['count']} seconds, Avg: {stats['avg_requests']} req/sec\n")
            f.write(f"\nPer Season Breakdown:\n")
            f.write(f"- Normal: {normal_seconds_per_season} seconds per season\n")
            f.write(f"- Peak: {peak_seconds_per_season} seconds per season\n")

        # Optional chart
        self.save_pattern_chart(time_points, request_rates, load_types)
        logger.info("Generated sudden-spike seasonal traffic pattern")
        logger.info(f"✅ {num_complete_seasons} seasons, each {SEASON_DURATION}s with peak at {PEAK_START_OFFSET}s")
        rate_map = {second: rate for second, rate in zip(time_points, request_rates)}
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
        """Send a GET request to the predictive scaler's /predict endpoint."""
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
            logger.error(f"GRU & Holt-Winters prediction models timed out to {full_url}: {e}")
            status_code = -1
            response_time_ms = -1
            error_msg = "GRU & Holt-Winters models timeout"
        except requests.exceptions.RequestException as e:
            logger.error(f"GRU & Holt-Winters prediction models failed to {full_url}: {e}")
            status_code = e.response.status_code if e.response is not None else -1
            response_time_ms = (time.time() - start_time) * 1000
            error_msg = f"GRU & Holt-Winters models failed: {str(e)}"
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

    def make_requests(self, url, second, request_count, results_list, products_list, lock, name, task_list=None, is_peak: bool = False, start_barrier: Optional[threading.Barrier] = None):
        """
        Make 'request_count' requests to a target URL, using concurrency so we can
        actually hit high RPS during peaks. Results are appended to results_list.
        """
        self.current_request_rate = request_count

        if request_count <= 0:
            return 0, 0

        # Choose weights based on peak vs normal window
        selected_weights = PEAK_ENDPOINT_WEIGHTS if is_peak else ENDPOINT_WEIGHTS

        # Slightly longer per-request timeouts (configurable) to reduce premature client errors
        per_request_timeout = min(
            REQUEST_TIMEOUT,
            self.client_timeout_peak if is_peak else self.client_timeout_normal
        )

        def do_single_request(task=None):
            # Optional: inject a client-side failure before sending any request
            try:
                if self.chaos_error_rate > 0.0 and np.random.rand() < self.chaos_error_rate:
                    raise RuntimeError("chaos: injected failure")
            except Exception as e:
                # Record synthetic failure and return
                endpoint_type = 'chaos'
                method = 'GET'
                endpoint = '/chaos'
                # CRITICAL: Lock when appending to results_list
                with lock:
                    results_list.append({
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()),
                        "url": url,
                        "method": method,
                        "endpoint": endpoint,
                        "status_code": -1,
                        "response_time_ms": -1,
                        "error": str(e),
                        "request_count": request_count,
                        "service": name,
                        "endpoint_type": endpoint_type
                    })
                return 0
            if task is not None:
                endpoint_type = task.get('endpoint_type', 'health')
                method = task.get('method', 'GET')
                data = task.get('data')
                endpoint = task.get('endpoint', '/health')
            else:
                endpoint_type = np.random.choice(
                    list(selected_weights.keys()),
                    p=list(selected_weights.values())
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
                    duration_param = LOAD_DURATION_PEAK if is_peak else LOAD_DURATION_NORMAL
                    low, high = LOAD_INTENSITY_PEAK if is_peak else LOAD_INTENSITY_NORMAL
                    intensity = np.random.randint(low, high)
                    endpoint = f"/load?duration={duration_param}&intensity={intensity}"
                elif endpoint_type == "health":
                    endpoint = "/health"

            start_time = time.time()
            full_url = f"{url}{endpoint}"
            try:
                session = self.sessions.get(name)
                if method == "GET":
                    response = (session.get if session else requests.get)(full_url, timeout=per_request_timeout)
                else:
                    headers = {'Content-Type': 'application/json'}
                    response = (session.post if session else requests.post)(full_url, data=json.dumps(data), headers=headers, timeout=per_request_timeout)
                end_time = time.time()
                status_code = response.status_code
                response_time_ms = (end_time - start_time) * 1000
                if 200 <= status_code < 400 and endpoint_type == "product_create" and status_code == 201:
                    try:
                        resp_data = response.json()
                        if 'product' in resp_data and 'id' in resp_data['product']:
                            product_id = resp_data['product']['id']
                            with lock:
                                products_list.append(product_id)
                            logger.debug(f"Created product {product_id} for {name}")
                    except Exception as e:
                        logger.debug(f"Could not extract product ID: {e}")
                # CRITICAL: Lock when appending to results_list
                with lock:
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
                return 1 if 200 <= status_code < 400 else 0
            except (ConnectTimeout, ReadTimeout) as e:
                logger.error(f"{name} service timed out to {full_url}: {e}")
                # CRITICAL: Lock when appending to results_list
                with lock:
                    results_list.append({
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()),
                        "url": url,
                        "method": method,
                        "endpoint": endpoint,
                        "status_code": -1,
                        "response_time_ms": -1,
                        "error": f"{name} service timeout: {str(e)}",
                        "request_count": request_count,
                        "service": name,
                        "endpoint_type": endpoint_type
                    })
                return 0
            except Exception as e:
                logger.error(f"[{name} service] failed to {full_url}: {e}")
                # CRITICAL: Lock when appending to results_list
                with lock:
                    results_list.append({
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_seconds": int((datetime.now() - self.start_time).total_seconds()),
                        "url": url,
                        "method": method,
                        "endpoint": endpoint,
                        "status_code": -1,
                        "response_time_ms": -1,
                        "error": f"{name} service failed: {str(e)}",
                        "request_count": request_count,
                        "service": name,
                        "endpoint_type": endpoint_type
                    })
                return 0

        # Synchronize batch start across services to improve fairness
        if start_barrier is not None:
            try:
                start_barrier.wait(timeout=2.0)
            except Exception:
                pass

        # Use per-second executor with proper rate limiting - wait for all tasks to complete
        # This ensures we don't accumulate tasks across seconds and both services get equal treatment
        max_workers = min(self.max_concurrency, request_count)
        batch_deadline = self.start_time + timedelta(seconds=second + 1 + self.tick_grace)
        
        successes = 0
        failures = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if task_list is not None and len(task_list) == request_count:
                futures = [executor.submit(do_single_request, task) for task in task_list]
            else:
                futures = [executor.submit(do_single_request) for _ in range(request_count)]
            
            # Wait for completion with grace period, but don't forcefully cancel
            # Just let slow requests finish naturally via their own timeouts
            timeout_remaining = (batch_deadline - datetime.now()).total_seconds()
            if timeout_remaining > 0:
                wait(futures, timeout=timeout_remaining)
            
            # Count completed futures - don't force-cancel incomplete ones
            for f in futures:
                if f.done():
                    try:
                        result = f.result(timeout=0)
                        if result == 1:
                            successes += 1
                        else:
                            failures += 1
                    except Exception:
                        failures += 1
        
        if request_count > 0:
            logger.debug(f"Second {second}: {name} scheduled {request_count}, completed {successes + failures} (S:{successes}/F:{failures})")
        
        return successes, failures

    def run_test(self):
        """
        Run the load test for the specified duration.
        Only tests the selected target services.
        """
        logger.info(f"Starting functional load test for {self.duration} seconds")
        
        active_services = []
        if self.test_hpa:
            logger.info(f"HPA URL: {self.hpa_url}")
            active_services.append("HPA")
        if self.test_combined:
            logger.info(f"Combined URL: {self.combined_url}")
            active_services.append("Combined")
        if self.test_predictive:
            logger.info(f"Predictive Scaler URL: {self.predictive_url}")
            active_services.append("Predictive")
            
        logger.info(f"Active services: {', '.join(active_services)}")
        
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
            
            # Determine pattern type based on seasonal position
            season_position = second % SEASON_DURATION
            is_peak = (PEAK_START_OFFSET <= season_position < PEAK_START_OFFSET + PEAK_DURATION)
            # Apply burst multiplier for the first few seconds of the peak window
            if is_peak:
                seconds_into_peak = season_position - PEAK_START_OFFSET
                if 0 <= seconds_into_peak < BURST_ON_PEAK_SECONDS:
                    request_count = int(request_count * BURST_ON_PEAK_MULTIPLIER)

            # Clamp final intended RPS to configured maximum
            if request_count > self.max_rps:
                request_count = self.max_rps

            # Publish scheduled RPS for visibility (final intended per-second sends after burst/clamp)
            if self.metrics_enabled and hasattr(self, 'metrics_scheduled_rps'):
                if self.test_hpa:
                    self.metrics_scheduled_rps.labels(service='HPA').set(request_count)
                if self.test_combined:
                    self.metrics_scheduled_rps.labels(service='Combined').set(request_count)
            
            # Log the scheduled count every 10 seconds to verify fairness
            if second % 10 == 0:
                logger.info(f"Second {second}: Scheduling {request_count} requests to EACH service (HPA + Combined)")
            
            pattern_type = "SEASONAL_PEAK" if is_peak else "SEASONAL_NORMAL"
            if pattern_type != current_pattern_type:
                logger.info(f"Second {second}: Pattern changed from {current_pattern_type} to {pattern_type}, scheduling {request_count} req/s")
                current_pattern_type = pattern_type

            # Pre-build identical batch for both services if both are enabled
            task_batch = None
            if self.test_hpa and self.test_combined:
                task_batch = []
                for _ in range(request_count):
                    weight_dict = PEAK_ENDPOINT_WEIGHTS if is_peak else ENDPOINT_WEIGHTS
                    endpoint_type = np.random.choice(list(weight_dict.keys()), p=list(weight_dict.values()))
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
                        duration_param = LOAD_DURATION_PEAK if is_peak else LOAD_DURATION_NORMAL
                        low, high = LOAD_INTENSITY_PEAK if is_peak else LOAD_INTENSITY_NORMAL
                        intensity = np.random.randint(low, high)
                        endpoint = f"/load?duration={duration_param}&intensity={intensity}"
                    elif endpoint_type == "health":
                        endpoint = "/health"
                    task_batch.append({'endpoint_type': endpoint_type, 'method': method, 'endpoint': endpoint, 'data': data})

            # Create threads only for selected services
            threads = []
            
            # If testing both services, synchronize their batch start with a barrier
            start_barrier = threading.Barrier(2) if (self.test_hpa and self.test_combined) else None

            if self.test_hpa:
                hpa_thread = threading.Thread(
                    target=lambda s=second, rc=request_count, tb=task_batch, ip=is_peak, sb=start_barrier: self.make_requests(
                        self.hpa_url, s, rc,
                        self.hpa_results, self.hpa_products, self.hpa_lock, "HPA",
                        task_list=tb if tb is not None else None,
                        is_peak=ip,
                        start_barrier=sb
                    )
                )
                threads.append(hpa_thread)

            if self.test_combined:
                combined_thread = threading.Thread(
                    target=lambda s=second, rc=request_count, tb=task_batch, ip=is_peak, sb=start_barrier: self.make_requests(
                        self.combined_url, s, rc,
                        self.combined_results, self.combined_products, self.combined_lock, "Combined",
                        task_list=tb if tb is not None else None,
                        is_peak=ip,
                        start_barrier=sb
                    )
                )
                threads.append(combined_thread)

            if self.test_predictive:
                pred_thread = threading.Thread(
                    target=lambda: self.make_predictive_request(self.predictive_url, self.predictive_results)
                )
                threads.append(pred_thread)

            # Start all threads
            for thread in threads:
                thread.start()
            # Wait for threads to complete
            for thread in threads:
                thread.join()

            # Update client Prometheus metrics per second
            self.update_client_metrics()

            # Pace the loop to approximately one-second ticks
            elapsed_in_second = (datetime.now() - (self.start_time + timedelta(seconds=second))).total_seconds()
            if elapsed_in_second < 1.0:
                time.sleep(1.0 - elapsed_in_second)
            
            # Progress logs
            if second % 30 == 0 and second > 0:
                elapsed = second
                remaining = self.duration - second
                progress = elapsed / self.duration * 100
                counts = {}
                success_rates = {}
                if self.test_hpa:
                    # CRITICAL: Lock when reading results_list to prevent race conditions
                    with self.hpa_results_lock:
                        hpa_count = len(self.hpa_results)
                        hpa_successes = sum(1 for r in self.hpa_results if 200 <= r.get("status_code", 0) < 400)
                    counts["HPA"] = hpa_count
                    success_rates["HPA"] = (hpa_successes / hpa_count * 100) if hpa_count > 0 else 0
                if self.test_combined:
                    # CRITICAL: Lock when reading results_list to prevent race conditions
                    with self.combined_results_lock:
                        combined_count = len(self.combined_results)
                        combined_successes = sum(1 for r in self.combined_results if 200 <= r.get("status_code", 0) < 400)
                    counts["Combined"] = combined_count
                    success_rates["Combined"] = (combined_successes / combined_count * 100) if combined_count > 0 else 0
                avg_rate = sum(rate_history[-30:]) / min(30, len(rate_history[-30:]))
                logger.info(f"Progress: {elapsed}/{self.duration} seconds ({progress:.1f}%). Remaining: {remaining} seconds")
                logger.info(f"Current request rate: {request_count}/sec, Avg last 30s: {avg_rate:.1f}/sec")
                count_msg = ", ".join([f"{svc}={count}" for svc, count in counts.items()])
                success_msg = ", ".join([f"{svc}={rate:.1f}%" for svc, rate in success_rates.items()])
                logger.info(f"Requests made: {count_msg}")
                logger.info(f"Success rates: {success_msg}")
                product_msg = []
                if self.test_hpa:
                    product_msg.append(f"HPA={len(self.hpa_products)}")
                if self.test_combined:
                    product_msg.append(f"Combined={len(self.combined_products)}")
                if product_msg:
                    logger.info(f"Products created: {', '.join(product_msg)}")
                if second % 300 == 0 and second > 0:
                    self.save_partial_results(f"partial_{second}")

        end_time = datetime.now()
        logger.info(f"Test ended at {end_time.isoformat()}")
        logger.info(f"Total duration: {(end_time - self.start_time).total_seconds():.1f} seconds")
        final_counts = {}
        if self.test_hpa:
            with self.hpa_results_lock:
                final_counts["HPA"] = len(self.hpa_results)
        if self.test_combined:
            with self.combined_results_lock:
                final_counts["Combined"] = len(self.combined_results)
        logger.info(
            f"Total requests: {', '.join([f'{svc}={count}' for svc, count in final_counts.items()])}"
        )
        product_msg = []
        if self.test_hpa:
            product_msg.append(f"HPA={len(self.hpa_products)}")
        if self.test_combined:
            product_msg.append(f"Combined={len(self.combined_products)}")
        if product_msg:
            logger.info(f"Products created: {', '.join(product_msg)}")
        
        self.save_results()
        try:
            shutil.copy('load_test.log', os.path.join(self.output_dir, 'load_test.log'))
            logger.info(f"Load test log copied to {self.output_dir}/load_test.log")
        except Exception as e:
            logger.error(f"Could not copy log file: {e}")
        # CRITICAL: Lock when reading results for final summary
        hpa_req_count = 0
        combined_req_count = 0
        hpa_success = 0.0
        combined_success = 0.0
        if self.test_hpa:
            with self.hpa_results_lock:
                hpa_req_count = len(self.hpa_results)
                hpa_success = self.calculate_success_rate(self.hpa_results)
        if self.test_combined:
            with self.combined_results_lock:
                combined_req_count = len(self.combined_results)
                combined_success = self.calculate_success_rate(self.combined_results)
        
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - self.start_time).total_seconds(),
            "hpa_requests": hpa_req_count,
            "combined_requests": combined_req_count,
            "hpa_success_rate": hpa_success,
            "combined_success_rate": combined_success,
            "hpa_products_created": len(self.hpa_products) if self.test_hpa else 0,
            "combined_products_created": len(self.combined_products) if self.test_combined else 0,
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
            if self.test_hpa and self.hpa_results:
                pd.DataFrame(self.hpa_results).to_csv(os.path.join(partial_dir, f"{prefix}_hpa_results.csv"), index=False)
            if self.test_combined and self.combined_results:
                pd.DataFrame(self.combined_results).to_csv(os.path.join(partial_dir, f"{prefix}_combined_results.csv"), index=False)
            if self.test_predictive and self.predictive_results:
                pd.DataFrame(self.predictive_results).to_csv(os.path.join(partial_dir, f"{prefix}_predictive_results.csv"), index=False)
            logger.info(f"Saved partial results with prefix '{prefix}'")
        except Exception as e:
            logger.error(f"Error saving partial results: {e}")

    def save_results(self):
        """Save all results to CSV and JSON files."""
        try:
            if self.test_hpa and self.hpa_results:
                hpa_df = pd.DataFrame(self.hpa_results)
                hpa_df.to_csv(os.path.join(self.output_dir, "hpa_results.csv"), index=False)
                logger.info(f"Saved HPA results to {self.output_dir}/hpa_results.csv")
            if self.test_combined and self.combined_results:
                combined_df = pd.DataFrame(self.combined_results)
                combined_df.to_csv(os.path.join(self.output_dir, "combined_results.csv"), index=False)
                logger.info(f"Saved Combined results to {self.output_dir}/combined_results.csv")
            if self.test_predictive and self.predictive_results:
                predictive_df = pd.DataFrame(self.predictive_results)
                predictive_df.to_csv(os.path.join(self.output_dir, "predictive_results.csv"), index=False)
                logger.info(f"Saved Predictive results to {self.output_dir}/predictive_results.csv")
            
            if self.test_hpa:
                with open(os.path.join(self.output_dir, "hpa_products.json"), 'w') as f:
                    json.dump(self.hpa_products, f)
            if self.test_combined:
                with open(os.path.join(self.output_dir, "combined_products.json"), 'w') as f:
                    json.dump(self.combined_products, f)
            
            self.create_summary_statistics()
            self.save_request_rate_data()
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def save_request_rate_data(self):
        """Save the per-second request count for future analysis."""
        try:
            all_results = []
            if self.test_hpa:
                all_results.extend(self.hpa_results)
            if self.test_combined:
                all_results.extend(self.combined_results)
                
            results_df = pd.DataFrame(all_results)
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
            if self.test_hpa and self.hpa_results:
                for result in self.hpa_results:
                    r_copy = result.copy()
                    r_copy["service"] = "HPA"
                    all_results.append(r_copy)
            if self.test_combined and self.combined_results:
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
    parser = argparse.ArgumentParser(
        description='Run functional load test for product app',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Service selection arguments
    parser.add_argument('--target', type=str, choices=['hpa', 'combined', 'both'], default=CONFIG["TARGET"],
                      help="Which service(s) to test: hpa, combined, or both")
    parser.add_argument('--hpa-url', type=str, default=DEFAULT_HPA_URL, help='URL for HPA service')
    parser.add_argument('--combined-url', type=str, default=DEFAULT_COMBINED_URL, help='URL for Combined service')
    parser.add_argument('--predictive-url', type=str, default=DEFAULT_PREDICTIVE_URL, help='URL for Predictive Scaler service')
    parser.add_argument('--test-predictive', action='store_true', default=False, help='Include predictive scaler in testing')
    
    # Test configuration arguments
    parser.add_argument('--duration', type=int, default=CONFIG["DURATION"],
                      help='Test duration in seconds')
    parser.add_argument('--output-dir', type=str, default='./load_test_results', help='Output directory for results')
    parser.add_argument('--max-concurrency', type=int, default=CONFIG["MAX_CONCURRENCY"],
                      help='Max parallel requests per service per second')
    parser.add_argument('--chaos-error-rate', type=float, default=CONFIG["CHAOS_ERROR_RATE"],
                      help='Probability [0-1] to inject a client-side failure to increase error rate')
    parser.add_argument('--metrics-port', type=int, default=CONFIG["METRICS_PORT"],
                      help='If > 0, expose Prometheus metrics on this port')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible traffic pattern')
    parser.add_argument('--auto-peak', action='store_true', default=False,
                        help='Auto-calc peak req/s to target roughly eighty percent utilization of 10 pods')
    parser.add_argument('--rps-per-pod', type=int, default=CONFIG["RPS_PER_POD"],
                      help='Estimated sustainable RPS per pod at 70 percent CPU (used for --auto-peak)')
    
    # Seasonal pattern arguments
    parser.add_argument('--normal-min', type=int, default=NORMAL_LOAD_RANGE[0], help='Minimum normal requests per second')
    parser.add_argument('--normal-max', type=int, default=NORMAL_LOAD_RANGE[1], help='Maximum normal requests per second')
    parser.add_argument('--peak-min', type=int, default=PEAK_LOAD_RANGE[0], help='Minimum peak requests per second')
    parser.add_argument('--peak-max', type=int, default=PEAK_LOAD_RANGE[1], help='Maximum peak requests per second')
    parser.add_argument('--season-duration', type=int, default=SEASON_DURATION, help='Duration of each season in seconds')
    parser.add_argument('--peak-duration', type=int, default=PEAK_DURATION, help='Duration of peak window in seconds')
    parser.add_argument('--peak-offset', type=int, default=PEAK_START_OFFSET, help='Offset (in seconds) before each peak begins')
    parser.add_argument('--volatility', type=float, default=VOLATILITY_FACTOR, help='Volatility factor (0 for perfectly repeatable traffic)')
    parser.add_argument('--timeout', type=float, default=REQUEST_TIMEOUT, help='Request timeout in seconds')
    # Client behavior tuning
    parser.add_argument('--client-timeout-normal', type=float, default=CONFIG["CLIENT_TIMEOUT_NORMAL"],
                      help='Per-request timeout during normal windows (seconds)')
    parser.add_argument('--client-timeout-peak', type=float, default=CONFIG["CLIENT_TIMEOUT_PEAK"],
                      help='Per-request timeout during peak windows (seconds)')
    parser.add_argument('--tick-grace', type=float, default=CONFIG["TICK_GRACE"],
                      help='Extra seconds added to the 1s per-second deadline so slow requests can finish')
    parser.add_argument('--max-rps', type=int, default=CONFIG["MAX_RPS"],
                      help='Hard cap for scheduled requests per second (after burst)')
    parser.set_defaults(test_predictive=CONFIG["TEST_PREDICTIVE"])

    return parser.parse_args()

def main():
    """Main function – parse arguments and start the load test."""
    args = parse_arguments()
    
    # Update global settings based on command-line args
    global NORMAL_LOAD_RANGE, PEAK_LOAD_RANGE, SEASON_DURATION, PEAK_DURATION, PEAK_START_OFFSET, VOLATILITY_FACTOR, REQUEST_TIMEOUT
    NORMAL_LOAD_RANGE = (args.normal_min, args.normal_max)
    # Optional: auto calculate peak range to target ~80% utilization of 10 pods
    if args.auto_peak:
        # target pods/maxPods ~= 10, target utilization ~= 0.8
        target_rps = int(10 * args.rps_per_pod * 0.8)
        # center +/- 10%
        low = max(1, int(target_rps * 0.9))
        high = max(low + 1, int(target_rps * 1.1))
        PEAK_LOAD_RANGE = (low, high)
    else:
        PEAK_LOAD_RANGE = (args.peak_min, args.peak_max)
    SEASON_DURATION = args.season_duration
    PEAK_DURATION = args.peak_duration
    PEAK_START_OFFSET = args.peak_offset
    VOLATILITY_FACTOR = args.volatility
    REQUEST_TIMEOUT = args.timeout

    # Determine which services to test based on target argument
    test_hpa = args.target in ['hpa', 'both']
    test_combined = args.target in ['combined', 'both']
    test_predictive = args.test_predictive

    logger.info("Starting Selective Load Test")
    logger.info("===========================")
    logger.info(f"Target services: {args.target}")
    if test_hpa:
        logger.info(f"HPA URL: {args.hpa_url}")
    if test_combined:
        logger.info(f"Combined URL: {args.combined_url}")
    if test_predictive:
        logger.info(f"Predictive URL: {args.predictive_url}")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Seasonal traffic pattern (single peak per season)")
    logger.info(f"Normal load range: {NORMAL_LOAD_RANGE} req/sec")
    logger.info(f"Peak load range: {PEAK_LOAD_RANGE} req/sec" + (" (auto)" if args.auto_peak else ""))
    if args.auto_peak:
        logger.info(f"Auto-peak targeting ~80% of 10 pods with ~{args.rps_per_pod} RPS/pod assumption => target≈{int(10*args.rps_per_pod*0.8)} RPS")
    logger.info(f"Season duration: {SEASON_DURATION} seconds ({SEASON_DURATION/60:.1f} minutes)")
    logger.info(f"Peak duration: {PEAK_DURATION} seconds")
    logger.info(f"Peak offset: {PEAK_START_OFFSET} seconds into each season")
    logger.info(f"Volatility factor: {VOLATILITY_FACTOR}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT} seconds")
    logger.info(f"Client timeouts: normal={args.client_timeout_normal}s, peak={args.client_timeout_peak}s; tick grace={args.tick_grace}s")
    logger.info(f"Max RPS cap: {args.max_rps}")
    logger.info(f"Max concurrency: {args.max_concurrency}")
    logger.info(f"Endpoint weights: {ENDPOINT_WEIGHTS}")
    if args.seed is not None:
        try:
            np.random.seed(args.seed)
            logger.info(f"Random seed set to: {args.seed}")
        except Exception as e:
            logger.warning(f"Failed to set random seed: {e}")
    logger.info("===========================")

    # Start optional Prometheus metrics server
    metrics_enabled = args.metrics_port > 0 and PROM_AVAILABLE
    if args.metrics_port > 0:
        if PROM_AVAILABLE:
            prom_start_http_server(args.metrics_port)
            logger.info(f"Client metrics exporter started on port {args.metrics_port}")
        else:
            logger.warning("Prometheus client not available; install 'prometheus_client' to enable metrics")

    tester = LoadTester(
        hpa_url=args.hpa_url,
        combined_url=args.combined_url,
        predictive_url=args.predictive_url,
        duration=args.duration,
        output_dir=args.output_dir,
        test_hpa=test_hpa,
        test_combined=test_combined,
        test_predictive=test_predictive,
        max_concurrency=args.max_concurrency,
        metrics_enabled=metrics_enabled,
        chaos_error_rate=args.chaos_error_rate,
        client_timeout_normal=args.client_timeout_normal,
        client_timeout_peak=args.client_timeout_peak,
        tick_grace=args.tick_grace,
        max_rps=args.max_rps
    )

    try:
        summary = tester.run_test()
        logger.info("Load test completed successfully")
        logger.info("==============================")
        logger.info(f"Start time: {summary['start_time']}")
        logger.info(f"End time: {summary['end_time']}")
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        if test_hpa:
            logger.info(f"Total HPA requests: {summary['hpa_requests']}")
            logger.info(f"HPA success rate: {summary['hpa_success_rate'] * 100:.2f}%")
            logger.info(f"HPA products created: {summary['hpa_products_created']}")
        if test_combined:
            logger.info(f"Total Combined requests: {summary['combined_requests']}")
            logger.info(f"Combined success rate: {summary['combined_success_rate'] * 100:.2f}%")
            logger.info(f"Combined products created: {summary['combined_products_created']}")
        logger.info(f"Results saved to: {summary['output_directory']}")
        logger.info("==============================")
        logger.info("\nTo retrieve the results, run:")
        logger.info(f"kubectl cp load-test:{summary['output_directory']} ./")
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
