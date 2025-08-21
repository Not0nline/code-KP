#!/usr/bin/env python3
import requests
import time
import threading
import random
import json
import argparse
import signal
import sys
import csv
import os
from datetime import datetime, timedelta
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
import statistics
from collections import defaultdict, deque
import sqlite3

SERVICE_MAPPING = {
    'hpa': 'http://product-app-hpa-service:80',
    'combined': 'http://product-app-combined-service:80',
    'both': ['http://product-app-hpa-service:80', 'http://product-app-combined-service:80']
}

AUTOSCALER_URL = 'http://predictive-scaler:5000'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'stress_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrafficPattern:
    name: str
    duration_minutes: int
    base_rps: float
    peak_rps: float
    pattern_type: str  # 'linear', 'exponential', 'sine', 'step', 'spike'
    description: str

@dataclass
class RequestResult:
    id: int
    timestamp: str
    pattern_name: str
    cycle: int
    success: bool
    status_code: int
    response_time: float
    endpoint: str
    error: Optional[str] = None
    thread_id: Optional[str] = None
    target_rps: Optional[float] = None
    actual_rps: Optional[float] = None

@dataclass
class MetricsSnapshot:
    timestamp: str
    pattern_name: str
    cycle: int
    target_rps: float
    actual_rps: float
    active_threads: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    current_success_rate: float
    avg_response_time: float
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    replicas: Optional[int] = None
    autoscaler_status: Optional[str] = None

class StressTestRunner:
    def __init__(self, target_url: str, autoscaler_url: str = None):
        self.target_url = target_url.rstrip('/')
        self.autoscaler_url = autoscaler_url.rstrip('/') if autoscaler_url else None
        self.running = False
        self.start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize comprehensive stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'start_time': None,
            'end_time': None,
            'current_rps': 0,
            'pattern_history': [],
            'error_counts': defaultdict(int),
            'status_code_counts': defaultdict(int),
            'endpoint_stats': defaultdict(lambda: {'requests': 0, 'successes': 0, 'failures': 0, 'total_time': 0}),
            'hourly_stats': defaultdict(lambda: {'requests': 0, 'successes': 0, 'failures': 0}),
            'rps_history': deque(maxlen=3600),  # Keep last hour of RPS data
            'response_time_history': deque(maxlen=1000),  # Keep last 1000 response times for rolling stats
        }
        
        # Current state tracking
        self.current_pattern = None
        self.current_cycle = 0
        self.active_threads = 0
        self.request_counter = 0
        self.last_rps_calculation = time.time()
        self.requests_in_last_second = 0
        
        # File paths for outputs
        self.output_dir = f"stress_test_output_{self.start_timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.csv_requests_file = os.path.join(self.output_dir, "requests_detailed.csv")
        self.csv_metrics_file = os.path.join(self.output_dir, "metrics_timeline.csv")
        self.csv_summary_file = os.path.join(self.output_dir, "pattern_summary.csv")
        self.csv_errors_file = os.path.join(self.output_dir, "error_analysis.csv")
        self.json_report_file = os.path.join(self.output_dir, "comprehensive_report.json")
        self.db_file = os.path.join(self.output_dir, "stress_test.db")
        
        # Initialize CSV files and database
        self.init_csv_files()
        self.init_database()
        
        # Thread-safe locks
        self.stats_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        
        # Realistic traffic patterns for better autoscaler training
        self.traffic_patterns = [
            TrafficPattern(
                name="gradual_warmup",
                duration_minutes=8,
                base_rps=0.1,
                peak_rps=2.0,
                pattern_type="linear",
                description="Gradual traffic increase to allow model training"
            ),
            TrafficPattern(
                name="steady_low",
                duration_minutes=5,
                base_rps=1.0,
                peak_rps=1.5,
                pattern_type="sine",
                description="Steady low traffic with minor variations"
            ),
            TrafficPattern(
                name="moderate_load",
                duration_minutes=6,
                base_rps=2.0,
                peak_rps=4.0,
                pattern_type="linear",
                description="Moderate load increase"
            ),
            TrafficPattern(
                name="high_load_sustained",
                duration_minutes=4,
                base_rps=8.0,
                peak_rps=8.5,
                pattern_type="sine",
                description="High sustained load for scale-up testing"
            ),
            TrafficPattern(
                name="gradual_cooldown",
                duration_minutes=6,
                base_rps=8.0,
                peak_rps=1.0,
                pattern_type="exponential",
                description="Gradual cooldown for scale-down testing"
            ),
            TrafficPattern(
                name="idle_period",
                duration_minutes=4,
                base_rps=0.05,
                peak_rps=0.1,
                pattern_type="step",
                description="Very low traffic to test idle scaling"
            ),
            TrafficPattern(
                name="traffic_spike",
                duration_minutes=3,
                base_rps=0.5,
                peak_rps=12.0,
                pattern_type="spike",
                description="Sharp traffic spike"
            ),
            TrafficPattern(
                name="recovery",
                duration_minutes=5,
                base_rps=3.0,
                peak_rps=2.0,
                pattern_type="linear",
                description="Traffic recovery to normal levels"
            ),
            TrafficPattern(
                name="business_hours_sim",
                duration_minutes=8,
                base_rps=1.0,
                peak_rps=6.0,
                pattern_type="sine",
                description="Simulate business hours traffic pattern"
            ),
            TrafficPattern(
                name="final_cooldown",
                duration_minutes=6,
                base_rps=4.0,
                peak_rps=0.2,
                pattern_type="exponential",
                description="Final cooldown to minimal traffic"
            )
        ]

    def init_csv_files(self):
        """Initialize CSV files with headers."""
        try:
            # Detailed requests CSV
            with open(self.csv_requests_file, 'w', newline='') as csvfile:
                fieldnames = ['request_id', 'timestamp', 'pattern_name', 'cycle', 'success', 'status_code', 
                             'response_time', 'endpoint', 'error', 'thread_id', 'target_rps', 'actual_rps']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            # Metrics timeline CSV
            with open(self.csv_metrics_file, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'pattern_name', 'cycle', 'target_rps', 'actual_rps', 'active_threads',
                             'total_requests', 'successful_requests', 'failed_requests', 'current_success_rate',
                             'avg_response_time', 'cpu_usage', 'memory_usage', 'replicas', 'autoscaler_status']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            # Pattern summary CSV
            with open(self.csv_summary_file, 'w', newline='') as csvfile:
                fieldnames = ['pattern_name', 'cycle', 'duration_minutes', 'total_requests', 'successful_requests',
                             'failed_requests', 'success_rate', 'avg_response_time', 'min_response_time',
                             'max_response_time', 'p50_response_time', 'p90_response_time', 'p95_response_time',
                             'p99_response_time', 'avg_rps', 'max_rps', 'start_time', 'end_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            # Error analysis CSV  
            with open(self.csv_errors_file, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'pattern_name', 'cycle', 'error_type', 'status_code', 'error_message',
                             'endpoint', 'response_time', 'occurrence_count']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
            logger.info(f"Initialized CSV files in directory: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV files: {e}")

    def init_database(self):
        """Initialize SQLite database for advanced analytics."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Requests table
            cursor.execute('''
                CREATE TABLE requests (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    pattern_name TEXT,
                    cycle INTEGER,
                    success BOOLEAN,
                    status_code INTEGER,
                    response_time REAL,
                    endpoint TEXT,
                    error TEXT,
                    thread_id TEXT,
                    target_rps REAL,
                    actual_rps REAL
                )
            ''')
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE metrics (
                    timestamp TEXT PRIMARY KEY,
                    pattern_name TEXT,
                    cycle INTEGER,
                    target_rps REAL,
                    actual_rps REAL,
                    active_threads INTEGER,
                    total_requests INTEGER,
                    successful_requests INTEGER,
                    failed_requests INTEGER,
                    current_success_rate REAL,
                    avg_response_time REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    replicas INTEGER,
                    autoscaler_status TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Initialized database: {self.db_file}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def calculate_rps(self, pattern: TrafficPattern, elapsed_ratio: float) -> float:
        """Calculate requests per second based on pattern and elapsed time."""
        if pattern.pattern_type == "linear":
            return pattern.base_rps + (pattern.peak_rps - pattern.base_rps) * elapsed_ratio
        
        elif pattern.pattern_type == "exponential":
            if pattern.peak_rps > pattern.base_rps:
                # Exponential growth
                return pattern.base_rps * (pattern.peak_rps / pattern.base_rps) ** elapsed_ratio
            else:
                # Exponential decay
                return pattern.base_rps * (pattern.peak_rps / pattern.base_rps) ** elapsed_ratio
        
        elif pattern.pattern_type == "sine":
            # Sine wave between base and peak
            mid_point = (pattern.base_rps + pattern.peak_rps) / 2
            amplitude = (pattern.peak_rps - pattern.base_rps) / 2
            return mid_point + amplitude * math.sin(elapsed_ratio * 2 * math.pi)
        
        elif pattern.pattern_type == "step":
            # Step function - stay at base for first half, peak for second half
            return pattern.base_rps if elapsed_ratio < 0.5 else pattern.peak_rps
        
        elif pattern.pattern_type == "spike":
            # Sharp spike in the middle
            if 0.3 <= elapsed_ratio <= 0.7:
                return pattern.peak_rps
            else:
                return pattern.base_rps
        
        return pattern.base_rps

    def make_request(self, request_id: int, pattern_name: str, target_rps: float, cycle: int) -> RequestResult:
        """Make a single HTTP request and return comprehensive timing info."""
        thread_id = threading.current_thread().ident
        start_time = time.time()
        
        with self.stats_lock:
            self.active_threads += 1
        
        try:
            # Vary the endpoints to simulate real traffic
            endpoints = [
                '/health',
                '/api/products',  
                '/api/users',
                '/api/orders',
                '/',
                '/api/categories',
                '/api/search',
                '/api/cart',
                '/api/auth/login',
                '/api/recommendations'
            ]
            endpoint = random.choice(endpoints)
            url = f"{self.target_url}{endpoint}"
            
            # Add some query parameters occasionally for realism
            if random.random() < 0.3:
                params = {
                    'page': random.randint(1, 10),
                    'limit': random.choice([10, 20, 50]),
                    'sort': random.choice(['name', 'date', 'price'])
                }
                response = requests.get(url, params=params, timeout=30)
            else:
                response = requests.get(url, timeout=30)
            
            end_time = time.time()
            response_time = end_time - start_time
            success = response.status_code < 400
            
            result = RequestResult(
                id=request_id,
                timestamp=datetime.now().isoformat(),
                pattern_name=pattern_name,
                cycle=cycle,
                success=success,
                status_code=response.status_code,
                response_time=response_time,
                endpoint=endpoint,
                thread_id=str(thread_id),
                target_rps=target_rps,
                actual_rps=self.calculate_actual_rps()
            )
            
            # Update comprehensive stats
            with self.stats_lock:
                self.stats['total_requests'] += 1
                if success:
                    self.stats['successful_requests'] += 1
                    self.stats['endpoint_stats'][endpoint]['successes'] += 1
                else:
                    self.stats['failed_requests'] += 1
                    self.stats['endpoint_stats'][endpoint]['failures'] += 1
                
                self.stats['endpoint_stats'][endpoint]['requests'] += 1
                self.stats['endpoint_stats'][endpoint]['total_time'] += response_time
                self.stats['response_times'].append(response_time)
                self.stats['response_time_history'].append(response_time)
                self.stats['status_code_counts'][response.status_code] += 1
                
                # Update hourly stats
                hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
                self.stats['hourly_stats'][hour_key]['requests'] += 1
                if success:
                    self.stats['hourly_stats'][hour_key]['successes'] += 1
                else:
                    self.stats['hourly_stats'][hour_key]['failures'] += 1
            
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            error_msg = str(e)
            
            result = RequestResult(
                id=request_id,
                timestamp=datetime.now().isoformat(),
                pattern_name=pattern_name,
                cycle=cycle,
                success=False,
                status_code=0,
                response_time=response_time,
                endpoint=endpoint if 'endpoint' in locals() else 'unknown',
                error=error_msg,
                thread_id=str(thread_id),
                target_rps=target_rps,
                actual_rps=self.calculate_actual_rps()
            )
            
            # Update error stats
            with self.stats_lock:
                self.stats['total_requests'] += 1
                self.stats['failed_requests'] += 1
                self.stats['error_counts'][error_msg] += 1
                self.stats['response_times'].append(response_time)
                self.stats['response_time_history'].append(response_time)
                
                if 'endpoint' in locals():
                    self.stats['endpoint_stats'][endpoint]['requests'] += 1
                    self.stats['endpoint_stats'][endpoint]['failures'] += 1
                    self.stats['endpoint_stats'][endpoint]['total_time'] += response_time
            
            return result
        
        finally:
            with self.stats_lock:
                self.active_threads -= 1

    def calculate_actual_rps(self) -> float:
        """Calculate actual RPS based on recent request history."""
        current_time = time.time()
        if current_time - self.last_rps_calculation >= 1.0:
            with self.stats_lock:
                self.stats['rps_history'].append(self.requests_in_last_second)
                self.requests_in_last_second = 0
                self.last_rps_calculation = current_time
        
        # Return average RPS over last 10 seconds
        if len(self.stats['rps_history']) > 0:
            recent_samples = list(self.stats['rps_history'])[-10:]
            return sum(recent_samples) / len(recent_samples)
        return 0.0

    def log_request_to_csv(self, result: RequestResult):
        """Log individual request to CSV file."""
        try:
            with self.csv_lock:
                with open(self.csv_requests_file, 'a', newline='') as csvfile:
                    fieldnames = ['request_id', 'timestamp', 'pattern_name', 'cycle', 'success', 'status_code',
                                 'response_time', 'endpoint', 'error', 'thread_id', 'target_rps', 'actual_rps']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    row = {
                        'request_id': result.id,
                        'timestamp': result.timestamp,
                        'pattern_name': result.pattern_name,
                        'cycle': result.cycle,
                        'success': result.success,
                        'status_code': result.status_code,
                        'response_time': result.response_time,
                        'endpoint': result.endpoint,
                        'error': result.error or '',
                        'thread_id': result.thread_id,
                        'target_rps': result.target_rps,
                        'actual_rps': result.actual_rps
                    }
                    writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to log request to CSV: {e}")

    def log_request_to_db(self, result: RequestResult):
        """Log request to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO requests (id, timestamp, pattern_name, cycle, success, status_code,
                                    response_time, endpoint, error, thread_id, target_rps, actual_rps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (result.id, result.timestamp, result.pattern_name, result.cycle, result.success,
                  result.status_code, result.response_time, result.endpoint, result.error,
                  result.thread_id, result.target_rps, result.actual_rps))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log request to database: {e}")

    def log_metrics_snapshot(self, snapshot: MetricsSnapshot):
        """Log metrics snapshot to CSV and database."""
        try:
            # CSV logging
            with self.csv_lock:
                with open(self.csv_metrics_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=asdict(snapshot).keys())
                    writer.writerow(asdict(snapshot))
            
            # Database logging
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(asdict(snapshot).values()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log metrics snapshot: {e}")

    def get_autoscaler_status(self) -> Optional[Dict]:
        """Get current autoscaler status."""
        if not self.autoscaler_url:
            return None
        
        try:
            response = requests.get(f"{self.autoscaler_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get autoscaler status: {e}")
        return None

    def get_autoscaler_prediction(self) -> Optional[Dict]:
        """Get current autoscaler prediction."""
        if not self.autoscaler_url:
            return None
        
        try:
            response = requests.get(f"{self.autoscaler_url}/predict_combined", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get autoscaler prediction: {e}")
        return None

    def calculate_success_rate(self, window_size: int = None) -> float:
        """Calculate success rate, optionally over a recent window."""
        with self.stats_lock:
            if window_size and len(self.stats['response_times']) > window_size:
                recent_total = window_size
                recent_successes = self.stats['successful_requests'] - (self.stats['total_requests'] - window_size)
                return (recent_successes / recent_total) * 100 if recent_total > 0 else 0
            else:
                return (self.stats['successful_requests'] / self.stats['total_requests']) * 100 if self.stats['total_requests'] > 0 else 0

    def run_traffic_pattern(self, pattern: TrafficPattern):
        """Execute a specific traffic pattern with comprehensive tracking."""
        logger.info(f"Starting pattern: {pattern.name} - {pattern.description}")
        logger.info(f"Duration: {pattern.duration_minutes} minutes, RPS: {pattern.base_rps} -> {pattern.peak_rps}")
        
        self.current_pattern = pattern
        start_time = time.time()
        end_time = start_time + (pattern.duration_minutes * 60)
        
        pattern_stats = {
            'pattern_name': pattern.name,
            'cycle': self.current_cycle,
            'start_time': datetime.now().isoformat(),
            'duration_minutes': pattern.duration_minutes,
            'requests_sent': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'rps_samples': [],
            'success_rate': 0,
            'avg_response_time': 0,
            'min_response_time': float('inf'),
            'max_response_time': 0,
            'error_details': defaultdict(int)
        }
        
        request_id = self.request_counter
        last_metrics_log = time.time()
        
        while time.time() < end_time and self.running:
            current_time = time.time()
            elapsed_ratio = (current_time - start_time) / (pattern.duration_minutes * 60)
            target_rps = self.calculate_rps(pattern, elapsed_ratio)
            
            self.stats['current_rps'] = target_rps
            pattern_stats['rps_samples'].append(target_rps)
            
            # Calculate how many requests to send this second
            requests_this_second = max(1, int(target_rps))
            delay_between_requests = 1.0 / requests_this_second if requests_this_second > 0 else 1.0
            
            # Send requests for this second
            second_start = time.time()
            with ThreadPoolExecutor(max_workers=min(50, requests_this_second * 3)) as executor:
                futures = []
                
                for _ in range(requests_this_second):
                    if not self.running:
                        break
                    future = executor.submit(self.make_request, request_id, pattern.name, target_rps, self.current_cycle)
                    futures.append(future)
                    request_id += 1
                    
                    # Small delay between requests within the same second
                    if delay_between_requests < 1.0 and delay_between_requests > 0.01:
                        time.sleep(delay_between_requests)
                
                # Collect results
                for future in as_completed(futures):
                    if not self.running:
                        break
                    
                    result = future.result()
                    pattern_stats['requests_sent'] += 1
                    pattern_stats['response_times'].append(result.response_time)
                    
                    if result.success:
                        pattern_stats['successful_requests'] += 1
                    else:
                        pattern_stats['failed_requests'] += 1
                        if result.error:
                            pattern_stats['error_details'][result.error] += 1
                        pattern_stats['error_details'][f"HTTP_{result.status_code}"] += 1
                    
                    # Update min/max response times
                    pattern_stats['min_response_time'] = min(pattern_stats['min_response_time'], result.response_time)
                    pattern_stats['max_response_time'] = max(pattern_stats['max_response_time'], result.response_time)
                    
                    # Log to CSV and database
                    self.log_request_to_csv(result)
                    self.log_request_to_db(result)
                    
                    # Increment requests counter for RPS calculation
                    with self.stats_lock:
                        self.requests_in_last_second += 1
            
            self.request_counter = request_id
            
            # Log metrics snapshot every 10 seconds
            if current_time - last_metrics_log >= 10:
                autoscaler_status = self.get_autoscaler_status()
                
                with self.stats_lock:
                    current_success_rate = self.calculate_success_rate(100)  # Last 100 requests
                    avg_rt = statistics.mean(list(self.stats['response_time_history'])) if self.stats['response_time_history'] else 0
                
                snapshot = MetricsSnapshot(
                    timestamp=datetime.now().isoformat(),
                    pattern_name=pattern.name,
                    cycle=self.current_cycle,
                    target_rps=target_rps,
                    actual_rps=self.calculate_actual_rps(),
                    active_threads=self.active_threads,
                    total_requests=self.stats['total_requests'],
                    successful_requests=self.stats['successful_requests'],
                    failed_requests=self.stats['failed_requests'],
                    current_success_rate=current_success_rate,
                    avg_response_time=avg_rt,
                    cpu_usage=autoscaler_status.get('current_cpu') if autoscaler_status else None,
                    memory_usage=autoscaler_status.get('memory_usage') if autoscaler_status else None,
                    replicas=autoscaler_status.get('current_replicas') if autoscaler_status else None,
                    autoscaler_status=autoscaler_status.get('status', 'unknown') if autoscaler_status else None
                )
                
                self.log_metrics_snapshot(snapshot)
                last_metrics_log = current_time
            
            # Log progress every 30 seconds
            if int(current_time - start_time) % 30 == 0 and int(current_time - start_time) > 0:
                remaining_minutes = (end_time - current_time) / 60
                current_pattern_success_rate = (pattern_stats['successful_requests'] / pattern_stats['requests_sent']) * 100 if pattern_stats['requests_sent'] > 0 else 0
                logger.info(f"Pattern {pattern.name}: {remaining_minutes:.1f} min remaining, "
                           f"RPS: {target_rps:.1f}, Requests: {pattern_stats['requests_sent']}, "
                           f"Success Rate: {current_pattern_success_rate:.1f}%")
                
                # Get and log autoscaler status
                autoscaler_status = self.get_autoscaler_status()
                if autoscaler_status:
                    logger.info(f"Autoscaler: CPU: {autoscaler_status.get('current_cpu', 'N/A')}%, "
                               f"Replicas: {autoscaler_status.get('current_replicas', 'N/A')}, "
                               f"Model: {autoscaler_status.get('mse_stats', {}).get('best_model', 'N/A')}")
            
            # Sleep to maintain timing
            elapsed_this_second = time.time() - second_start
            if elapsed_this_second < 1.0:
                time.sleep(1.0 - elapsed_this_second)
        
        # Pattern completion statistics
        pattern_stats['end_time'] = datetime.now().isoformat()
        if pattern_stats['response_times']:
            pattern_stats['avg_response_time'] = statistics.mean(pattern_stats['response_times'])
            sorted_times = sorted(pattern_stats['response_times'])
            pattern_stats['p50_response_time'] = sorted_times[int(len(sorted_times) * 0.5)]
            pattern_stats['p90_response_time'] = sorted_times[int(len(sorted_times) * 0.9)]
            pattern_stats['p95_response_time'] = sorted_times[int(len(sorted_times) * 0.95)]
            pattern_stats['p99_response_time'] = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            pattern_stats['p50_response_time'] = 0
            pattern_stats['p90_response_time'] = 0
            pattern_stats['p95_response_time'] = 0
            pattern_stats['p99_response_time'] = 0
        
        if pattern_stats['min_response_time'] == float('inf'):
            pattern_stats['min_response_time'] = 0
            
        pattern_stats['success_rate'] = (pattern_stats['successful_requests'] / pattern_stats['requests_sent']) * 100 if pattern_stats['requests_sent'] > 0 else 0
        pattern_stats['avg_rps'] = statistics.mean(pattern_stats['rps_samples']) if pattern_stats['rps_samples'] else 0
        pattern_stats['max_rps'] = max(pattern_stats['rps_samples']) if pattern_stats['rps_samples'] else 0
        
        # Log pattern summary to CSV
        self.log_pattern_summary_to_csv(pattern_stats)
        
        # Log errors if any
        if pattern_stats['error_details']:
            self.log_errors_to_csv(pattern, pattern_stats['error_details'])
        
        self.stats['pattern_history'].append(pattern_stats)
        
        logger.info(f"Completed pattern {pattern.name}: "
                   f"{pattern_stats['requests_sent']} requests, "
                   f"{pattern_stats['success_rate']:.1f}% success rate, "
                   f"{pattern_stats['avg_response_time']:.3f}s avg response time")

    def log_pattern_summary_to_csv(self, pattern_stats: Dict):
        """Log pattern summary to CSV."""
        try:
            with self.csv_lock:
                with open(self.csv_summary_file, 'a', newline='') as csvfile:
                    fieldnames = ['pattern_name', 'cycle', 'duration_minutes', 'total_requests', 'successful_requests',
                                 'failed_requests', 'success_rate', 'avg_response_time', 'min_response_time',
                                 'max_response_time', 'p50_response_time', 'p90_response_time', 'p95_response_time',
                                 'p99_response_time', 'avg_rps', 'max_rps', 'start_time', 'end_time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    row = {
                        'pattern_name': pattern_stats['pattern_name'],
                        'cycle': pattern_stats['cycle'],
                        'duration_minutes': pattern_stats['duration_minutes'],
                        'total_requests': pattern_stats['requests_sent'],
                        'successful_requests': pattern_stats['successful_requests'],
                        'failed_requests': pattern_stats['failed_requests'],
                        'success_rate': pattern_stats['success_rate'],
                        'avg_response_time': pattern_stats['avg_response_time'],
                        'min_response_time': pattern_stats['min_response_time'],
                        'max_response_time': pattern_stats['max_response_time'],
                        'p50_response_time': pattern_stats['p50_response_time'],
                        'p90_response_time': pattern_stats['p90_response_time'],
                        'p95_response_time': pattern_stats['p95_response_time'],
                        'p99_response_time': pattern_stats['p99_response_time'],
                        'avg_rps': pattern_stats['avg_rps'],
                        'max_rps': pattern_stats['max_rps'],
                        'start_time': pattern_stats['start_time'],
                        'end_time': pattern_stats['end_time']
                    }
                    writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to log pattern summary to CSV: {e}")

    def log_errors_to_csv(self, pattern: TrafficPattern, error_details: Dict):
        """Log error details to CSV."""
        try:
            with self.csv_lock:
                with open(self.csv_errors_file, 'a', newline='') as csvfile:
                    fieldnames = ['timestamp', 'pattern_name', 'cycle', 'error_type', 'status_code', 'error_message',
                                 'endpoint', 'response_time', 'occurrence_count']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    for error, count in error_details.items():
                        if error.startswith('HTTP_'):
                            status_code = error.split('_')[1]
                            error_type = 'HTTP_ERROR'
                            error_message = f"HTTP {status_code} Error"
                        else:
                            status_code = '0'
                            error_type = 'CONNECTION_ERROR'
                            error_message = error
                        
                        row = {
                            'timestamp': datetime.now().isoformat(),
                            'pattern_name': pattern.name,
                            'cycle': self.current_cycle,
                            'error_type': error_type,
                            'status_code': status_code,
                            'error_message': error_message,
                            'endpoint': 'various',
                            'response_time': 'N/A',
                            'occurrence_count': count
                        }
                        writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to log errors to CSV: {e}")

    def run_test(self, cycles: int = 1):
        """Run the complete stress test with comprehensive tracking."""
        logger.info(f"Starting enhanced stress test with {cycles} cycle(s)")
        logger.info(f"Target URL: {self.target_url}")
        logger.info(f"Output directory: {self.output_dir}")
        if self.autoscaler_url:
            logger.info(f"Autoscaler URL: {self.autoscaler_url}")
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        try:
            for cycle in range(cycles):
                if not self.running:
                    break
                
                self.current_cycle = cycle + 1
                
                logger.info(f"\n{'='*50}")
                logger.info(f"Starting cycle {cycle + 1}/{cycles}")
                logger.info(f"{'='*50}")
                
                # Get initial autoscaler status
                initial_status = self.get_autoscaler_status()
                if initial_status:
                    logger.info(f"Initial autoscaler state: "
                               f"CPU: {initial_status.get('current_cpu', 'N/A')}%, "
                               f"Replicas: {initial_status.get('current_replicas', 'N/A')}")
                
                # Run each traffic pattern
                for i, pattern in enumerate(self.traffic_patterns):
                    if not self.running:
                        break
                    
                    logger.info(f"\nPattern {i+1}/{len(self.traffic_patterns)} in cycle {cycle+1}")
                    self.run_traffic_pattern(pattern)
                    
                    # Brief pause between patterns to allow metrics collection
                    if self.running and i < len(self.traffic_patterns) - 1:
                        logger.info("Pausing 30 seconds between patterns...")
                        time.sleep(30)
                
                if cycle < cycles - 1:
                    logger.info(f"Completed cycle {cycle + 1}. Pausing 60 seconds before next cycle...")
                    time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        finally:
            self.running = False
            self.stats['end_time'] = datetime.now()
            self.generate_final_reports()

    def generate_final_reports(self):
        """Generate comprehensive final reports in multiple formats."""
        logger.info("Generating final reports...")
        
        # Generate JSON report
        self.generate_json_report()
        
        # Generate summary statistics
        self.print_final_report()
        
        # Generate additional CSV analyses
        self.generate_additional_csv_reports()
        
        logger.info(f"All reports generated in directory: {self.output_dir}")

    def generate_json_report(self):
        """Generate comprehensive JSON report."""
        if not self.stats['start_time']:
            logger.error("No test data to report")
            return
        
        duration = self.stats['end_time'] - self.stats['start_time']
        
        # Calculate comprehensive statistics
        with self.stats_lock:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100 if self.stats['total_requests'] > 0 else 0
            avg_response_time = statistics.mean(self.stats['response_times']) if self.stats['response_times'] else 0
            
            percentiles = {}
            if self.stats['response_times']:
                sorted_times = sorted(self.stats['response_times'])
                percentiles = {
                    'p50': sorted_times[int(len(sorted_times) * 0.5)],
                    'p75': sorted_times[int(len(sorted_times) * 0.75)],
                    'p90': sorted_times[int(len(sorted_times) * 0.9)],
                    'p95': sorted_times[int(len(sorted_times) * 0.95)],
                    'p99': sorted_times[int(len(sorted_times) * 0.99)],
                    'min': min(sorted_times),
                    'max': max(sorted_times)
                }
            
            # Endpoint statistics
            endpoint_stats = {}
            for endpoint, stats in self.stats['endpoint_stats'].items():
                if stats['requests'] > 0:
                    endpoint_stats[endpoint] = {
                        'total_requests': stats['requests'],
                        'successful_requests': stats['successes'],
                        'failed_requests': stats['failures'],
                        'success_rate': (stats['successes'] / stats['requests']) * 100,
                        'avg_response_time': stats['total_time'] / stats['requests']
                    }
        
        # Get final autoscaler status
        final_autoscaler_status = self.get_autoscaler_status()
        final_prediction = self.get_autoscaler_prediction()
        
        report_data = {
            'test_info': {
                'start_time': self.stats['start_time'].isoformat(),
                'end_time': self.stats['end_time'].isoformat(),
                'duration_seconds': duration.total_seconds(),
                'target_url': self.target_url,
                'autoscaler_url': self.autoscaler_url,
                'output_directory': self.output_dir
            },
            'overall_stats': {
                'total_requests': self.stats['total_requests'],
                'successful_requests': self.stats['successful_requests'],
                'failed_requests': self.stats['failed_requests'],
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'total_patterns': len(self.traffic_patterns),
                'total_cycles': self.current_cycle
            },
            'response_time_stats': {
                'average': avg_response_time,
                'percentiles': percentiles
            },
            'endpoint_stats': endpoint_stats,
            'status_code_distribution': dict(self.stats['status_code_counts']),
            'error_distribution': dict(self.stats['error_counts']),
            'hourly_stats': dict(self.stats['hourly_stats']),
            'pattern_history': self.stats['pattern_history'],
            'final_autoscaler_status': final_autoscaler_status,
            'final_prediction': final_prediction,
            'files_generated': {
                'detailed_requests': self.csv_requests_file,
                'metrics_timeline': self.csv_metrics_file,
                'pattern_summary': self.csv_summary_file,
                'error_analysis': self.csv_errors_file,
                'database': self.db_file,
                'comprehensive_report': self.json_report_file
            }
        }
        
        try:
            with open(self.json_report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            logger.info(f"JSON report saved to: {self.json_report_file}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")

    def generate_additional_csv_reports(self):
        """Generate additional CSV analysis reports."""
        try:
            # Endpoint performance report
            endpoint_perf_file = os.path.join(self.output_dir, "endpoint_performance.csv")
            with open(endpoint_perf_file, 'w', newline='') as csvfile:
                fieldnames = ['endpoint', 'total_requests', 'successful_requests', 'failed_requests',
                             'success_rate', 'avg_response_time', 'total_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for endpoint, stats in self.stats['endpoint_stats'].items():
                    if stats['requests'] > 0:
                        writer.writerow({
                            'endpoint': endpoint,
                            'total_requests': stats['requests'],
                            'successful_requests': stats['successes'],
                            'failed_requests': stats['failures'],
                            'success_rate': (stats['successes'] / stats['requests']) * 100,
                            'avg_response_time': stats['total_time'] / stats['requests'],
                            'total_time': stats['total_time']
                        })
            
            # Hourly statistics report
            hourly_stats_file = os.path.join(self.output_dir, "hourly_statistics.csv")
            with open(hourly_stats_file, 'w', newline='') as csvfile:
                fieldnames = ['hour', 'total_requests', 'successful_requests', 'failed_requests', 'success_rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for hour, stats in sorted(self.stats['hourly_stats'].items()):
                    writer.writerow({
                        'hour': hour,
                        'total_requests': stats['requests'],
                        'successful_requests': stats['successes'],
                        'failed_requests': stats['failures'],
                        'success_rate': (stats['successes'] / stats['requests']) * 100 if stats['requests'] > 0 else 0
                    })
            
            logger.info("Additional CSV reports generated")
        except Exception as e:
            logger.error(f"Failed to generate additional CSV reports: {e}")

    def print_final_report(self):
        """Print comprehensive test results to console."""
        if not self.stats['start_time']:
            logger.error("No test data to report")
            return
        
        duration = self.stats['end_time'] - self.stats['start_time']
        success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100 if self.stats['total_requests'] > 0 else 0
        avg_response_time = statistics.mean(self.stats['response_times']) if self.stats['response_times'] else 0
        
        print(f"\n{'='*70}")
        print("COMPREHENSIVE STRESS TEST FINAL REPORT")
        print(f"{'='*70}")
        print(f"Test Duration: {duration}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Target URL: {self.target_url}")
        if self.autoscaler_url:
            print(f"Autoscaler URL: {self.autoscaler_url}")
        
        print(f"\n{'='*50}")
        print("OVERALL STATISTICS")
        print(f"{'='*50}")
        print(f"Total Requests: {self.stats['total_requests']:,}")
        print(f"Successful Requests: {self.stats['successful_requests']:,}")
        print(f"Failed Requests: {self.stats['failed_requests']:,}")
        print(f"Overall Success Rate: {success_rate:.2f}%")
        print(f"Average Response Time: {avg_response_time:.3f}s")
        print(f"Total Cycles Completed: {self.current_cycle}")
        print(f"Total Patterns Executed: {len(self.stats['pattern_history'])}")
        
        if self.stats['response_times']:
            sorted_times = sorted(self.stats['response_times'])
            print(f"\n{'='*50}")
            print("RESPONSE TIME PERCENTILES")
            print(f"{'='*50}")
            print(f"Min Response Time: {min(sorted_times):.3f}s")
            print(f"P50 (Median): {sorted_times[int(len(sorted_times) * 0.5)]:.3f}s")
            print(f"P75: {sorted_times[int(len(sorted_times) * 0.75)]:.3f}s")
            print(f"P90: {sorted_times[int(len(sorted_times) * 0.9)]:.3f}s")
            print(f"P95: {sorted_times[int(len(sorted_times) * 0.95)]:.3f}s")
            print(f"P99: {sorted_times[int(len(sorted_times) * 0.99)]:.3f}s")
            print(f"Max Response Time: {max(sorted_times):.3f}s")
        
        print(f"\n{'='*50}")
        print("PATTERN PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"{'Pattern':<20} {'Cycle':<6} {'Requests':<10} {'Success %':<10} {'Avg RT':<10} {'Max RT':<10}")
        print("-" * 70)
        for pattern in self.stats['pattern_history']:
            print(f"{pattern['pattern_name']:<20} "
                  f"{pattern['cycle']:<6} "
                  f"{pattern['requests_sent']:<10} "
                  f"{pattern['success_rate']:<10.1f} "
                  f"{pattern['avg_response_time']:<10.3f} "
                  f"{pattern['max_response_time']:<10.3f}")
        
        if self.stats['endpoint_stats']:
            print(f"\n{'='*50}")
            print("ENDPOINT PERFORMANCE")
            print(f"{'='*50}")
            print(f"{'Endpoint':<25} {'Requests':<10} {'Success %':<10} {'Avg RT':<10}")
            print("-" * 60)
            for endpoint, stats in sorted(self.stats['endpoint_stats'].items()):
                if stats['requests'] > 0:
                    success_rate_ep = (stats['successes'] / stats['requests']) * 100
                    avg_rt_ep = stats['total_time'] / stats['requests']
                    print(f"{endpoint:<25} {stats['requests']:<10} {success_rate_ep:<10.1f} {avg_rt_ep:<10.3f}")
        
        if self.stats['status_code_counts']:
            print(f"\n{'='*50}")
            print("HTTP STATUS CODE DISTRIBUTION")
            print(f"{'='*50}")
            for status_code, count in sorted(self.stats['status_code_counts'].items()):
                percentage = (count / self.stats['total_requests']) * 100
                print(f"HTTP {status_code}: {count:,} requests ({percentage:.1f}%)")
        
        if self.stats['error_counts']:
            print(f"\n{'='*50}")
            print("ERROR ANALYSIS")
            print(f"{'='*50}")
            for error, count in sorted(self.stats['error_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / self.stats['failed_requests']) * 100 if self.stats['failed_requests'] > 0 else 0
                print(f"{error[:50]}: {count} occurrences ({percentage:.1f}% of errors)")
        
        # Get final autoscaler status
        if self.autoscaler_url:
            final_status = self.get_autoscaler_status()
            prediction = self.get_autoscaler_prediction()
            
            print(f"\n{'='*50}")
            print("FINAL AUTOSCALER STATUS")
            print(f"{'='*50}")
            if final_status:
                mse_stats = final_status.get('mse_stats', {})
                print(f"Current CPU Usage: {final_status.get('current_cpu', 'N/A')}%")
                print(f"Current Replicas: {final_status.get('current_replicas', 'N/A')}")
                print(f"Current Traffic: {final_status.get('current_traffic', 'N/A')}")
                print(f"Best Performing Model: {mse_stats.get('best_model', 'N/A')}")
                print(f"GRU Model MSE: {mse_stats.get('gru_mse', 'N/A')}")
                print(f"Holt-Winters MSE: {mse_stats.get('holt_winters_mse', 'N/A')}")
                
                matching = mse_stats.get('prediction_matching', {})
                if matching:
                    gru_accuracy = (matching.get('gru_matched', 0) / matching.get('gru_total', 1)) * 100
                    hw_accuracy = (matching.get('hw_matched', 0) / matching.get('hw_total', 1)) * 100
                    print(f"GRU Prediction Accuracy: {gru_accuracy:.1f}%")
                    print(f"Holt-Winters Accuracy: {hw_accuracy:.1f}%")
            else:
                print("Autoscaler status not available")
            
            if prediction:
                print(f"Final Scaling Recommendation: {prediction.get('recommended_replicas', 'N/A')} replicas")
                print(f"Prediction Method Used: {prediction.get('method_used', 'N/A')}")
                print(f"Confidence Level: {prediction.get('confidence', 'N/A')}")
        
        print(f"\n{'='*50}")
        print("FILES GENERATED")
        print(f"{'='*50}")
        print(f"Detailed Requests: {self.csv_requests_file}")
        print(f"Metrics Timeline: {self.csv_metrics_file}")
        print(f"Pattern Summary: {self.csv_summary_file}")
        print(f"Error Analysis: {self.csv_errors_file}")
        print(f"Endpoint Performance: {os.path.join(self.output_dir, 'endpoint_performance.csv')}")
        print(f"Hourly Statistics: {os.path.join(self.output_dir, 'hourly_statistics.csv')}")
        print(f"JSON Report: {self.json_report_file}")
        print(f"SQLite Database: {self.db_file}")
        
        print(f"\n{'='*70}")
        print(f"Test completed successfully! Check {self.output_dir} for detailed reports.")
        print(f"{'='*70}")

def signal_handler(signum, frame):
    """Handle interrupt signal gracefully."""
    logger.info("Received interrupt signal, stopping test...")
    if 'runner' in globals():
        runner.running = False
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Stress Test for Predictive Autoscaler')
    parser.add_argument('--target', required=True, help='Target application URL (e.g., http://localhost:8080)')
    parser.add_argument('--autoscaler', help='Autoscaler URL (e.g., http://localhost:5000)')
    parser.add_argument('--cycles', type=int, default=1, help='Number of test cycles to run')
    parser.add_argument('--dry-run', action='store_true', help='Show traffic patterns without running test')
    
    args = parser.parse_args()
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    global runner
    runner = StressTestRunner(args.target, args.autoscaler)
    
    if args.dry_run:
        print("Traffic patterns that would be executed:")
        for i, pattern in enumerate(runner.traffic_patterns):
            print(f"{i+1}. {pattern.name} ({pattern.duration_minutes} min): {pattern.description}")
            print(f"   RPS: {pattern.base_rps} -> {pattern.peak_rps} ({pattern.pattern_type})")
        total_duration = sum(p.duration_minutes for p in runner.traffic_patterns)
        print(f"\nTotal duration per cycle: {total_duration} minutes")
        print(f"Total duration for {args.cycles} cycles: {total_duration * args.cycles} minutes")
        return
    
    # Verify target is accessible
    try:
        response = requests.get(f"{args.target}/health", timeout=5)
        logger.info(f"Target health check: {response.status_code}")
    except Exception as e:
        logger.error(f"Cannot reach target URL {args.target}: {e}")
        sys.exit(1)
    
    # Verify autoscaler is accessible if provided
    if args.autoscaler:
        try:
            response = requests.get(f"{args.autoscaler}/health", timeout=5)
            logger.info(f"Autoscaler health check: {response.status_code}")
        except Exception as e:
            logger.warning(f"Cannot reach autoscaler URL {args.autoscaler}: {e}")
    
    # Run the test
    runner.run_test(args.cycles)

if __name__ == "__main__":
    main()