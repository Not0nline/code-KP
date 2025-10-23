#!/usr/bin/env python3
"""
Test Orchestrator for HPA vs Combined Comparison
Runs 10 iterations of 30-minute load tests with 5-minute warmup
Collects comprehensive statistics and generates summary reports
"""

import subprocess
import time
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('orchestrator.log')
    ]
)
logger = logging.getLogger("orchestrator")

class TestOrchestrator:
    """Orchestrates multiple test runs and collects comprehensive statistics."""
    
    def __init__(self, num_iterations=10, test_duration=1800, warmup_duration=300, 
                 output_base_dir="./multi_run_results", skip_warmup=False):
        """
        Initialize the test orchestrator.
        
        Args:
            num_iterations: Number of test iterations to run (default: 10)
            test_duration: Duration of each test in seconds (default: 1800 = 30 min)
            warmup_duration: Warmup period before data collection in seconds (default: 300 = 5 min)
            output_base_dir: Base directory for all test results
            skip_warmup: If True, skip warmup and use full duration for data collection
        """
        self.num_iterations = num_iterations
        self.test_duration = test_duration
        self.warmup_duration = warmup_duration
        self.skip_warmup = skip_warmup
        self.data_collection_duration = test_duration if skip_warmup else (test_duration - warmup_duration)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_base_dir, f"batch_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results tracking
        self.iteration_results = []
        self.iteration_summaries = []
        
        logger.info(f"Test Orchestrator initialized:")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Test duration: {test_duration}s ({test_duration/60:.1f} min)")
        logger.info(f"  Warmup duration: {warmup_duration}s ({warmup_duration/60:.1f} min)")
        logger.info(f"  Data collection: {self.data_collection_duration}s ({self.data_collection_duration/60:.1f} min)")
        logger.info(f"  Output directory: {self.output_dir}")
        
    def run_single_test(self, iteration):
        """Run a single load test iteration."""
        logger.info(f"\n{'='*80}")
        logger.info(f"ITERATION {iteration}/{self.num_iterations} - Starting")
        logger.info(f"{'='*80}")
        
        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration:02d}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable, "load_test.py",
            "--target", "both",
            "--duration", str(self.test_duration),
            "--output-dir", iteration_dir,
            "--metrics-port", "0"  # Disable Prometheus metrics to avoid port conflicts
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        start_time = datetime.now()
        
        try:
            # Run the test
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=os.path.dirname(__file__)
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"Test completed in {duration:.1f} seconds")
            logger.info(f"Output: {iteration_dir}")
            
            # Extract results from the test run directory
            results = self._extract_iteration_results(iteration_dir, iteration, start_time, end_time)
            self.iteration_results.append(results)
            
            return results
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Test failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def fetch_autoscaler_metrics(self, predictive_url="http://predictive-scaler.default.svc.cluster.local:5000"):
        """
        Fetch model performance metrics from the predictive autoscaler.
        
        Returns dict with:
        - mse_per_model: {model_name: mse_data}
        - model_comparison: comparison data
        - selected_model: which model was chosen
        """
        import requests
        metrics = {}
        
        try:
            # Get MSE status for each model
            mse_response = requests.get(f"{predictive_url}/debug/mse_status", timeout=10)
            if mse_response.status_code == 200:
                mse_data = mse_response.json()
                metrics['mse_per_model'] = {}
                
                for model_name in ['gru', 'holt_winters']:
                    model_key = f"{model_name}_mse"
                    if model_key in mse_data:
                        metrics['mse_per_model'][model_name] = {
                            'current_mse': mse_data[model_key].get('current', 0),
                            'min_mse': mse_data[model_key].get('min', 0),
                            'max_mse': mse_data[model_key].get('max', 0),
                            'avg_mse': mse_data[model_key].get('avg', 0),
                            'samples': mse_data[model_key].get('samples', 0),
                        }
            
            # Get model comparison data (includes selection and prediction times)
            comparison_response = requests.get(f"{predictive_url}/debug/model_comparison", timeout=10)
            if comparison_response.status_code == 200:
                comp_data = comparison_response.json()
                metrics['model_comparison'] = comp_data
                metrics['selected_model'] = comp_data.get('selected_model', 'unknown')
                
                # Extract prediction times from model_comparison
                if 'models' in comp_data:
                    metrics['prediction_times'] = {}
                    for model_name, model_data in comp_data['models'].items():
                        if 'prediction_time_ms' in model_data:
                            metrics['prediction_times'][model_name] = model_data['prediction_time_ms']
            
            # Get general status
            status_response = requests.get(f"{predictive_url}/status", timeout=10)
            if status_response.status_code == 200:
                status_data = status_response.json()
                metrics['autoscaler_status'] = {
                    'is_trained': status_data.get('is_trained', False),
                    'data_points': status_data.get('data_points', 0),
                    'collection_complete': status_data.get('data_collection_complete', False),
                }
            
            logger.info(f"Fetched autoscaler metrics: {len(metrics)} categories")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to fetch autoscaler metrics: {e}")
            return {}
    
    def _extract_iteration_results(self, iteration_dir, iteration_num, start_time, end_time):
        """Extract and analyze results from a completed test iteration."""
        logger.info(f"Extracting results for iteration {iteration_num}...")
        
        results = {
            "iteration": iteration_num,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "warmup_seconds": self.warmup_duration,
            "data_collection_seconds": self.data_collection_duration,
        }
        
        # Find the actual test run directory (has timestamp)
        test_dirs = [d for d in Path(iteration_dir).iterdir() if d.is_dir() and d.name.startswith("test_run_")]
        if not test_dirs:
            logger.warning(f"No test run directory found in {iteration_dir}")
            return results
        
        test_run_dir = test_dirs[0]  # Should only be one
        logger.info(f"Found test run directory: {test_run_dir}")
        
        # Load HPA results
        hpa_results_path = test_run_dir / "hpa_results.csv"
        combined_results_path = test_run_dir / "combined_results.csv"
        
        if hpa_results_path.exists():
            hpa_df = pd.read_csv(hpa_results_path)
            results["hpa"] = self._analyze_service_results(hpa_df, "HPA", self.warmup_duration, self.skip_warmup)
        else:
            logger.warning(f"HPA results not found: {hpa_results_path}")
            results["hpa"] = {}
        
        if combined_results_path.exists():
            combined_df = pd.read_csv(combined_results_path)
            results["combined"] = self._analyze_service_results(combined_df, "Combined", self.warmup_duration, self.skip_warmup)
        else:
            logger.warning(f"Combined results not found: {combined_results_path}")
            results["combined"] = {}
        
        # Calculate comparative metrics
        if results.get("hpa") and results.get("combined"):
            results["comparison"] = self._calculate_comparison_metrics(results["hpa"], results["combined"])
        
        # Fetch autoscaler model metrics
        autoscaler_metrics = self.fetch_autoscaler_metrics()
        if autoscaler_metrics:
            results['autoscaler_metrics'] = autoscaler_metrics
            
            # Log model MSE values
            if 'mse_per_model' in autoscaler_metrics:
                logger.info("Model MSE Values:")
                for model, mse_data in autoscaler_metrics['mse_per_model'].items():
                    logger.info(f"  {model}: {mse_data['current_mse']:.4f} (avg: {mse_data['avg_mse']:.4f})")
            
            # Log selected model
            if 'selected_model' in autoscaler_metrics:
                logger.info(f"Selected Model: {autoscaler_metrics['selected_model']}")
        
        # Save iteration summary
        summary_path = os.path.join(iteration_dir, "iteration_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved iteration summary to {summary_path}")
        
        return results
    
    def _analyze_service_results(self, df, service_name, warmup_seconds, skip_warmup):
        """Analyze results for a single service (HPA or Combined)."""
        logger.info(f"Analyzing {service_name} results...")
        
        # Filter out warmup period if not skipping
        if not skip_warmup and warmup_seconds > 0:
            original_count = len(df)
            df = df[df['elapsed_seconds'] >= warmup_seconds].copy()
            logger.info(f"{service_name}: Filtered warmup period, {original_count} -> {len(df)} requests")
        
        if df.empty:
            logger.warning(f"{service_name}: No data after warmup filtering!")
            return {}
        
        # Overall metrics
        total_requests = len(df)
        functional_df = df[df['endpoint_type'] != 'health']
        total_functional = len(functional_df)
        
        # DEBUG: Show CSV columns and sample data
        logger.info(f"{service_name} CSV Columns: {df.columns.tolist()}")
        logger.info(f"{service_name} Sample status codes (first 10): {df['status_code'].head(10).tolist()}")
        logger.info(f"{service_name} Status code dtype: {df['status_code'].dtype}")
        
        # DEBUG: Show status code distribution
        status_counts = functional_df['status_code'].value_counts().to_dict()
        logger.info(f"{service_name} Status Code Distribution: {status_counts}")
        
        # Convert status_code to int if it's a string
        if df['status_code'].dtype == 'object':
            logger.warning(f"{service_name}: Converting status_code from string to int")
            functional_df = functional_df.copy()
            functional_df['status_code'] = pd.to_numeric(functional_df['status_code'], errors='coerce').fillna(-1).astype(int)
        
        # Success/Error analysis
        successful = functional_df['status_code'].between(200, 399).sum()
        errors = total_functional - successful
        success_rate = (successful / total_functional * 100) if total_functional > 0 else 0
        error_rate = (errors / total_functional * 100) if total_functional > 0 else 0
        
        # Response time analysis (functional requests only)
        valid_response_times = functional_df[functional_df['response_time_ms'] >= 0]['response_time_ms']
        
        # Per-endpoint breakdown
        endpoint_stats = []
        for endpoint_type in functional_df['endpoint_type'].unique():
            endpoint_df = functional_df[functional_df['endpoint_type'] == endpoint_type]
            endpoint_total = len(endpoint_df)
            endpoint_success = endpoint_df['status_code'].between(200, 399).sum()
            endpoint_errors = endpoint_total - endpoint_success
            
            valid_times = endpoint_df[endpoint_df['response_time_ms'] >= 0]['response_time_ms']
            
            endpoint_stats.append({
                "endpoint_type": endpoint_type,
                "total_requests": int(endpoint_total),
                "successful": int(endpoint_success),
                "errors": int(endpoint_errors),
                "success_rate": float(endpoint_success / endpoint_total * 100) if endpoint_total > 0 else 0,
                "error_rate": float(endpoint_errors / endpoint_total * 100) if endpoint_total > 0 else 0,
                "avg_response_ms": float(valid_times.mean()) if len(valid_times) > 0 else 0,
                "p50_response_ms": float(valid_times.quantile(0.5)) if len(valid_times) > 0 else 0,
                "p95_response_ms": float(valid_times.quantile(0.95)) if len(valid_times) > 0 else 0,
                "p99_response_ms": float(valid_times.quantile(0.99)) if len(valid_times) > 0 else 0,
            })
        
        # Time-series analysis (errors over time)
        df['minute'] = df['elapsed_seconds'] // 60
        time_series = df.groupby('minute').agg({
            'status_code': [
                ('total', 'count'),
                ('success', lambda x: (x.between(200, 399)).sum()),
                ('error', lambda x: (~x.between(200, 399)).sum())
            ]
        }).reset_index()
        time_series.columns = ['minute', 'total', 'success', 'error']
        time_series['error_rate'] = (time_series['error'] / time_series['total'] * 100)
        
        max_error_minute = time_series.loc[time_series['error_rate'].idxmax()] if len(time_series) > 0 else None
        
        analysis = {
            "service": service_name,
            "total_requests": int(total_requests),
            "total_functional_requests": int(total_functional),
            "successful_requests": int(successful),
            "error_requests": int(errors),
            "success_rate_percent": float(success_rate),
            "error_rate_percent": float(error_rate),
            "response_time_ms": {
                "mean": float(valid_response_times.mean()) if len(valid_response_times) > 0 else 0,
                "median": float(valid_response_times.median()) if len(valid_response_times) > 0 else 0,
                "p95": float(valid_response_times.quantile(0.95)) if len(valid_response_times) > 0 else 0,
                "p99": float(valid_response_times.quantile(0.99)) if len(valid_response_times) > 0 else 0,
                "min": float(valid_response_times.min()) if len(valid_response_times) > 0 else 0,
                "max": float(valid_response_times.max()) if len(valid_response_times) > 0 else 0,
            },
            "endpoint_breakdown": endpoint_stats,
            "peak_error_rate": {
                "minute": int(max_error_minute['minute']) if max_error_minute is not None else 0,
                "error_rate_percent": float(max_error_minute['error_rate']) if max_error_minute is not None else 0,
                "total_requests": int(max_error_minute['total']) if max_error_minute is not None else 0,
                "errors": int(max_error_minute['error']) if max_error_minute is not None else 0,
            }
        }
        
        logger.info(f"{service_name} Analysis:")
        logger.info(f"  Total requests: {total_requests}")
        logger.info(f"  Functional requests: {total_functional}")
        logger.info(f"  Success rate: {success_rate:.2f}%")
        logger.info(f"  Error rate: {error_rate:.2f}%")
        logger.info(f"  Avg response time: {analysis['response_time_ms']['mean']:.1f}ms")
        logger.info(f"  P95 response time: {analysis['response_time_ms']['p95']:.1f}ms")
        
        return analysis
    
    def _calculate_comparison_metrics(self, hpa_results, combined_results):
        """Calculate comparative metrics between HPA and Combined."""
        comparison = {}
        
        # Success rate difference
        hpa_success = hpa_results.get('success_rate_percent', 0)
        combined_success = combined_results.get('success_rate_percent', 0)
        comparison['success_rate_diff'] = float(combined_success - hpa_success)
        comparison['success_rate_improvement_percent'] = float(
            ((combined_success - hpa_success) / hpa_success * 100) if hpa_success > 0 else 0
        )
        
        # Error rate difference
        hpa_error = hpa_results.get('error_rate_percent', 0)
        combined_error = combined_results.get('error_rate_percent', 0)
        comparison['error_rate_diff'] = float(combined_error - hpa_error)
        comparison['error_rate_reduction_percent'] = float(
            ((hpa_error - combined_error) / hpa_error * 100) if hpa_error > 0 else 0
        )
        
        # Response time comparison
        hpa_p95 = hpa_results.get('response_time_ms', {}).get('p95', 0)
        combined_p95 = combined_results.get('response_time_ms', {}).get('p95', 0)
        comparison['p95_response_diff_ms'] = float(combined_p95 - hpa_p95)
        comparison['p95_response_improvement_percent'] = float(
            ((hpa_p95 - combined_p95) / hpa_p95 * 100) if hpa_p95 > 0 else 0
        )
        
        # Determine winner
        if combined_success > hpa_success:
            comparison['winner'] = 'Combined'
            comparison['winner_margin_percent'] = comparison['success_rate_improvement_percent']
        elif hpa_success > combined_success:
            comparison['winner'] = 'HPA'
            comparison['winner_margin_percent'] = -comparison['success_rate_improvement_percent']
        else:
            comparison['winner'] = 'Tie'
            comparison['winner_margin_percent'] = 0
        
        logger.info(f"Comparison:")
        logger.info(f"  Success rate diff: {comparison['success_rate_diff']:+.2f}% (Combined vs HPA)")
        logger.info(f"  Error reduction: {comparison['error_rate_reduction_percent']:+.2f}%")
        logger.info(f"  Winner: {comparison['winner']} ({comparison['winner_margin_percent']:+.2f}%)")
        
        return comparison
    
    def wait_between_iterations(self, iteration, cooldown_seconds=60):
        """Wait between test iterations to let the system stabilize."""
        if iteration < self.num_iterations:
            logger.info(f"\nCooldown period: waiting {cooldown_seconds}s before next iteration...")
            time.sleep(cooldown_seconds)
    
    def run_all_tests(self, cooldown_seconds=60):
        """Run all test iterations."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING BATCH TEST RUN")
        logger.info(f"{'='*80}\n")
        
        overall_start = datetime.now()
        
        for i in range(1, self.num_iterations + 1):
            result = self.run_single_test(i)
            
            if result:
                logger.info(f"✅ Iteration {i} completed successfully")
            else:
                logger.error(f"❌ Iteration {i} failed")
            
            # Wait before next iteration (except after the last one)
            if i < self.num_iterations:
                self.wait_between_iterations(i, cooldown_seconds)
        
        overall_end = datetime.now()
        total_duration = (overall_end - overall_start).total_seconds()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"ALL ITERATIONS COMPLETED")
        logger.info(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        logger.info(f"{'='*80}\n")
        
        # Generate final summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report across all iterations."""
        logger.info("Generating summary report...")
        
        if not self.iteration_results:
            logger.warning("No iteration results to summarize")
            return
        
        # Collect metrics across iterations
        hpa_success_rates = []
        combined_success_rates = []
        hpa_error_rates = []
        combined_error_rates = []
        hpa_p95_times = []
        combined_p95_times = []
        success_rate_diffs = []
        error_rate_reductions = []
        winners = []
        
        for result in self.iteration_results:
            if 'hpa' in result and 'combined' in result:
                hpa_success_rates.append(result['hpa'].get('success_rate_percent', 0))
                combined_success_rates.append(result['combined'].get('success_rate_percent', 0))
                hpa_error_rates.append(result['hpa'].get('error_rate_percent', 0))
                combined_error_rates.append(result['combined'].get('error_rate_percent', 0))
                hpa_p95_times.append(result['hpa'].get('response_time_ms', {}).get('p95', 0))
                combined_p95_times.append(result['combined'].get('response_time_ms', {}).get('p95', 0))
                
                if 'comparison' in result:
                    success_rate_diffs.append(result['comparison'].get('success_rate_diff', 0))
                    error_rate_reductions.append(result['comparison'].get('error_rate_reduction_percent', 0))
                    winners.append(result['comparison'].get('winner', 'Unknown'))
        
        # Calculate aggregate statistics
        summary = {
            "test_configuration": {
                "num_iterations": self.num_iterations,
                "test_duration_seconds": self.test_duration,
                "warmup_duration_seconds": self.warmup_duration,
                "data_collection_duration_seconds": self.data_collection_duration,
                "skip_warmup": self.skip_warmup,
            },
            "hpa_aggregate": {
                "success_rate": {
                    "mean": float(np.mean(hpa_success_rates)),
                    "std": float(np.std(hpa_success_rates)),
                    "min": float(np.min(hpa_success_rates)),
                    "max": float(np.max(hpa_success_rates)),
                    "median": float(np.median(hpa_success_rates)),
                },
                "error_rate": {
                    "mean": float(np.mean(hpa_error_rates)),
                    "std": float(np.std(hpa_error_rates)),
                    "min": float(np.min(hpa_error_rates)),
                    "max": float(np.max(hpa_error_rates)),
                    "median": float(np.median(hpa_error_rates)),
                },
                "p95_response_ms": {
                    "mean": float(np.mean(hpa_p95_times)),
                    "std": float(np.std(hpa_p95_times)),
                    "min": float(np.min(hpa_p95_times)),
                    "max": float(np.max(hpa_p95_times)),
                    "median": float(np.median(hpa_p95_times)),
                },
            },
            "combined_aggregate": {
                "success_rate": {
                    "mean": float(np.mean(combined_success_rates)),
                    "std": float(np.std(combined_success_rates)),
                    "min": float(np.min(combined_success_rates)),
                    "max": float(np.max(combined_success_rates)),
                    "median": float(np.median(combined_success_rates)),
                },
                "error_rate": {
                    "mean": float(np.mean(combined_error_rates)),
                    "std": float(np.std(combined_error_rates)),
                    "min": float(np.min(combined_error_rates)),
                    "max": float(np.max(combined_error_rates)),
                    "median": float(np.median(combined_error_rates)),
                },
                "p95_response_ms": {
                    "mean": float(np.mean(combined_p95_times)),
                    "std": float(np.std(combined_p95_times)),
                    "min": float(np.min(combined_p95_times)),
                    "max": float(np.max(combined_p95_times)),
                    "median": float(np.median(combined_p95_times)),
                },
            },
            "comparison_aggregate": {
                "success_rate_improvement": {
                    "mean": float(np.mean(success_rate_diffs)),
                    "std": float(np.std(success_rate_diffs)),
                    "median": float(np.median(success_rate_diffs)),
                },
                "error_rate_reduction": {
                    "mean": float(np.mean(error_rate_reductions)),
                    "std": float(np.std(error_rate_reductions)),
                    "median": float(np.median(error_rate_reductions)),
                },
                "winner_distribution": {
                    "HPA": winners.count("HPA"),
                    "Combined": winners.count("Combined"),
                    "Tie": winners.count("Tie"),
                },
                "overall_winner": max(set(winners), key=winners.count) if winners else "Unknown",
            },
            "detailed_iterations": self.iteration_results,
        }
        
        # Aggregate autoscaler model metrics
        model_metrics = [r.get("autoscaler_metrics", {}) for r in self.iteration_results if r.get("autoscaler_metrics")]
        if model_metrics:
            summary["model_performance"] = {}
            
            # Track MSE history per model
            model_mse_history = {}
            model_selection_count = {}
            
            for iteration_metrics in model_metrics:
                # Aggregate MSE per model
                if 'mse_per_model' in iteration_metrics:
                    for model_name, mse_data in iteration_metrics['mse_per_model'].items():
                        if model_name not in model_mse_history:
                            model_mse_history[model_name] = {
                                'mse_values': [],
                                'min_mse': [],
                                'max_mse': [],
                                'avg_mse': [],
                                'samples': []
                            }
                        model_mse_history[model_name]['mse_values'].append(mse_data['current_mse'])
                        model_mse_history[model_name]['min_mse'].append(mse_data['min_mse'])
                        model_mse_history[model_name]['max_mse'].append(mse_data['max_mse'])
                        model_mse_history[model_name]['avg_mse'].append(mse_data['avg_mse'])
                        model_mse_history[model_name]['samples'].append(mse_data['samples'])
                
                # Count model selections
                if 'selected_model' in iteration_metrics:
                    selected = iteration_metrics['selected_model']
                    model_selection_count[selected] = model_selection_count.get(selected, 0) + 1
            
            # Compute aggregate statistics
            for model_name, history in model_mse_history.items():
                summary["model_performance"][model_name] = {
                    'overall_avg_mse': float(np.mean(history['mse_values'])),
                    'overall_min_mse': float(np.min(history['min_mse'])),
                    'overall_max_mse': float(np.max(history['max_mse'])),
                    'mse_std_dev': float(np.std(history['mse_values'])),
                    'mse_trend': 'improving' if len(history['mse_values']) > 1 and history['mse_values'][-1] < history['mse_values'][0] else 'stable',
                    'iterations_measured': len(history['mse_values']),
                }
            
            # Add selection frequency
            summary["model_selection_frequency"] = model_selection_count
            
            # Determine best model
            if model_mse_history:
                best_model = min(model_mse_history.items(), 
                                key=lambda x: np.mean(x[1]['mse_values']))
                summary["best_model"] = {
                    'name': best_model[0],
                    'avg_mse': float(np.mean(best_model[1]['mse_values'])),
                }
        
        # Save JSON summary
        summary_path = os.path.join(self.output_dir, "FINAL_SUMMARY.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved JSON summary to {summary_path}")
        
        # Generate human-readable report
        report_path = os.path.join(self.output_dir, "FINAL_SUMMARY.txt")
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HPA vs COMBINED - FINAL TEST SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Configuration:\n")
            f.write(f"  Iterations: {self.num_iterations}\n")
            f.write(f"  Test duration: {self.test_duration}s ({self.test_duration/60:.1f} min)\n")
            f.write(f"  Warmup period: {self.warmup_duration}s ({self.warmup_duration/60:.1f} min)\n")
            f.write(f"  Data collection: {self.data_collection_duration}s ({self.data_collection_duration/60:.1f} min)\n\n")
            
            f.write("="*80 + "\n")
            f.write("AGGREGATE RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"HPA (Reactive Scaling):\n")
            f.write(f"  Success Rate: {summary['hpa_aggregate']['success_rate']['mean']:.2f}% ")
            f.write(f"(± {summary['hpa_aggregate']['success_rate']['std']:.2f}%)\n")
            f.write(f"  Error Rate: {summary['hpa_aggregate']['error_rate']['mean']:.2f}% ")
            f.write(f"(± {summary['hpa_aggregate']['error_rate']['std']:.2f}%)\n")
            f.write(f"  P95 Response Time: {summary['hpa_aggregate']['p95_response_ms']['mean']:.1f}ms ")
            f.write(f"(± {summary['hpa_aggregate']['p95_response_ms']['std']:.1f}ms)\n\n")
            
            f.write(f"Combined (Predictive Scaling):\n")
            f.write(f"  Success Rate: {summary['combined_aggregate']['success_rate']['mean']:.2f}% ")
            f.write(f"(± {summary['combined_aggregate']['success_rate']['std']:.2f}%)\n")
            f.write(f"  Error Rate: {summary['combined_aggregate']['error_rate']['mean']:.2f}% ")
            f.write(f"(± {summary['combined_aggregate']['error_rate']['std']:.2f}%)\n")
            f.write(f"  P95 Response Time: {summary['combined_aggregate']['p95_response_ms']['mean']:.1f}ms ")
            f.write(f"(± {summary['combined_aggregate']['p95_response_ms']['std']:.1f}ms)\n\n")
            
            f.write("="*80 + "\n")
            f.write("COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            success_improvement = summary['comparison_aggregate']['success_rate_improvement']['mean']
            error_reduction = summary['comparison_aggregate']['error_rate_reduction']['mean']
            
            f.write(f"Success Rate Improvement (Combined vs HPA): {success_improvement:+.2f}%\n")
            f.write(f"Error Rate Reduction: {error_reduction:+.2f}%\n\n")
            
            f.write(f"Winner Distribution:\n")
            for winner, count in summary['comparison_aggregate']['winner_distribution'].items():
                f.write(f"  {winner}: {count}/{self.num_iterations} ({count/self.num_iterations*100:.1f}%)\n")
            
            f.write(f"\nOverall Winner: {summary['comparison_aggregate']['overall_winner']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("ITERATION DETAILS\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(self.iteration_results, 1):
                f.write(f"Iteration {i}:\n")
                if 'hpa' in result:
                    f.write(f"  HPA:      Success={result['hpa'].get('success_rate_percent', 0):.2f}%, ")
                    f.write(f"Error={result['hpa'].get('error_rate_percent', 0):.2f}%\n")
                if 'combined' in result:
                    f.write(f"  Combined: Success={result['combined'].get('success_rate_percent', 0):.2f}%, ")
                    f.write(f"Error={result['combined'].get('error_rate_percent', 0):.2f}%\n")
                if 'comparison' in result:
                    f.write(f"  Winner: {result['comparison'].get('winner', 'Unknown')}\n")
                f.write("\n")
            
            # Add model performance section
            if "model_performance" in summary:
                f.write("="*80 + "\n")
                f.write("MODEL PERFORMANCE SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                for model_name, perf in summary["model_performance"].items():
                    f.write(f"{model_name.upper()}:\n")
                    f.write(f"  Overall Avg MSE: {perf['overall_avg_mse']:.6f}\n")
                    f.write(f"  Overall Min MSE: {perf['overall_min_mse']:.6f}\n")
                    f.write(f"  Overall Max MSE: {perf['overall_max_mse']:.6f}\n")
                    f.write(f"  MSE Std Dev: {perf['mse_std_dev']:.6f}\n")
                    f.write(f"  Trend: {perf['mse_trend']}\n")
                    f.write(f"  Iterations Measured: {perf['iterations_measured']}\n\n")
                
                if "model_selection_frequency" in summary:
                    f.write("Model Selection Frequency:\n")
                    for model, count in summary["model_selection_frequency"].items():
                        percentage = (count / self.num_iterations) * 100
                        f.write(f"  {model}: {count}/{self.num_iterations} ({percentage:.1f}%)\n")
                    f.write("\n")
                
                if "best_model" in summary:
                    f.write(f"Best Performing Model:\n")
                    f.write(f"  Name: {summary['best_model']['name']}\n")
                    f.write(f"  Avg MSE: {summary['best_model']['avg_mse']:.6f}\n\n")
        
        logger.info(f"Saved text report to {report_path}")
        
        # Generate CSV for easy analysis
        csv_path = os.path.join(self.output_dir, "FINAL_SUMMARY.csv")
        summary_df = pd.DataFrame({
            'iteration': range(1, len(self.iteration_results) + 1),
            'hpa_success_rate': hpa_success_rates,
            'combined_success_rate': combined_success_rates,
            'hpa_error_rate': hpa_error_rates,
            'combined_error_rate': combined_error_rates,
            'hpa_p95_ms': hpa_p95_times,
            'combined_p95_ms': combined_p95_times,
            'success_rate_diff': success_rate_diffs,
            'error_reduction_pct': error_rate_reductions,
            'winner': winners,
        })
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV summary to {csv_path}")
        
        # Save model performance to separate CSV
        if "model_performance" in summary:
            model_csv_path = os.path.join(self.output_dir, "MODEL_PERFORMANCE.csv")
            model_csv_data = []
            
            for model_name, perf in summary["model_performance"].items():
                model_csv_data.append({
                    "model_name": model_name,
                    "overall_avg_mse": perf['overall_avg_mse'],
                    "overall_min_mse": perf['overall_min_mse'],
                    "overall_max_mse": perf['overall_max_mse'],
                    "mse_std_dev": perf['mse_std_dev'],
                    "mse_trend": perf['mse_trend'],
                    "iterations_measured": perf['iterations_measured'],
                    "selection_count": summary.get("model_selection_frequency", {}).get(model_name, 0),
                    "selection_percentage": (summary.get("model_selection_frequency", {}).get(model_name, 0) / self.num_iterations) * 100,
                })
            
            model_df = pd.DataFrame(model_csv_data)
            model_df.to_csv(model_csv_path, index=False)
            logger.info(f"Saved model performance CSV to {model_csv_path}")
        
        # Print summary to console
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        logger.info(f"HPA Success Rate: {summary['hpa_aggregate']['success_rate']['mean']:.2f}% (± {summary['hpa_aggregate']['success_rate']['std']:.2f}%)")
        logger.info(f"Combined Success Rate: {summary['combined_aggregate']['success_rate']['mean']:.2f}% (± {summary['combined_aggregate']['success_rate']['std']:.2f}%)")
        logger.info(f"Improvement: {success_improvement:+.2f}%")
        logger.info(f"Overall Winner: {summary['comparison_aggregate']['overall_winner']}")
        logger.info("="*80 + "\n")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Orchestrate multiple load test runs for HPA vs Combined comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of test iterations to run')
    parser.add_argument('--duration', type=int, default=1800,
                        help='Duration of each test in seconds (default: 1800 = 30 min)')
    parser.add_argument('--warmup', type=int, default=300,
                        help='Warmup period before data collection in seconds (default: 300 = 5 min)')
    parser.add_argument('--cooldown', type=int, default=60,
                        help='Cooldown period between iterations in seconds')
    parser.add_argument('--skip-warmup', action='store_true',
                        help='Skip warmup period and use full duration for data collection')
    parser.add_argument('--output-dir', type=str, default='./multi_run_results',
                        help='Base output directory for all test results')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("TEST ORCHESTRATOR STARTING")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info(f"  Duration per test: {args.duration}s ({args.duration/60:.1f} min)")
    logger.info(f"  Warmup period: {args.warmup}s ({args.warmup/60:.1f} min)")
    logger.info(f"  Cooldown between tests: {args.cooldown}s")
    logger.info(f"  Skip warmup: {args.skip_warmup}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("="*80 + "\n")
    
    orchestrator = TestOrchestrator(
        num_iterations=args.iterations,
        test_duration=args.duration,
        warmup_duration=args.warmup,
        output_base_dir=args.output_dir,
        skip_warmup=args.skip_warmup
    )
    
    try:
        orchestrator.run_all_tests(cooldown_seconds=args.cooldown)
        logger.info("✅ All tests completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Test interrupted by user")
        if orchestrator.iteration_results:
            logger.info("Generating partial summary from completed iterations...")
            orchestrator.generate_summary_report()
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 2

if __name__ == '__main__':
    sys.exit(main())
