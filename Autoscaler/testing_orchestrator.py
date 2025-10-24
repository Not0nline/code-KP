#!/usr/bin/env python3
"""
Multi-Model Testing Orchestration System

This script automates the entire testing workflow:
1. Collect baseline datasets for low/medium/high scenarios (if not already collected)
2. For each model variant (XGBoost, CatBoost, LightGBM, GRU, Holt-Winters, Ensemble):
   - For each traffic scenario (low/medium/high):
     - Load baseline dataset into autoscaler
     - Train the model on baseline data
     - Run load test for specified duration
     - Collect metrics and results
     - Save results to CSV
3. Generate comparison reports

Usage:
    # Collect all baseline datasets first (run once)
    python testing_orchestrator.py collect-baselines --duration 3600
    
    # Run full test suite
    python testing_orchestrator.py run-tests --models all --scenarios all --duration 1800
    
    # Run specific tests
    python testing_orchestrator.py run-tests --models xgboost,catboost --scenarios medium,high
"""

import os
import sys
import json
import time
import requests
import argparse
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
AUTOSCALER_URL = os.getenv('AUTOSCALER_URL', 'http://predictive-scaler.default.svc.cluster.local:5000')
LOAD_TESTER_API = os.getenv('LOAD_TESTER_API', 'http://load-tester.default.svc.cluster.local:8080')
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090')

BASELINE_SCENARIOS = ['low', 'medium', 'high']
MODEL_VARIANTS = ['xgboost', 'catboost', 'lightgbm', 'gru', 'holt_winters']
RESULTS_DIR = './test_results'

class TestingOrchestrator:
    """Orchestrates multi-model, multi-scenario testing"""
    
    def __init__(self, autoscaler_url: str = AUTOSCALER_URL, 
                 load_tester_api: str = LOAD_TESTER_API,
                 results_dir: str = RESULTS_DIR):
        self.autoscaler_url = autoscaler_url
        self.load_tester_api = load_tester_api
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.test_results = []
        
    def check_services(self) -> bool:
        """Check if all required services are accessible"""
        logger.info("🔍 Checking service availability...")
        
        services = {
            'Autoscaler': f"{self.autoscaler_url}/health",
            'Load Tester': f"{self.load_tester_api}/health"
        }
        
        all_ok = True
        for name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"  ✅ {name}: OK")
                else:
                    logger.error(f"  ❌ {name}: HTTP {response.status_code}")
                    all_ok = False
            except Exception as e:
                logger.error(f"  ❌ {name}: {e}")
                all_ok = False
        
        return all_ok
    
    def collect_baseline_dataset(self, scenario: str, duration: int = 3600) -> bool:
        """
        Collect a baseline dataset by running a load test and collecting metrics
        
        Args:
            scenario: Traffic scenario (low/medium/high)
            duration: Collection duration in seconds
        """
        logger.info(f"📊 Collecting baseline dataset for '{scenario}' scenario ({duration}s)...")
        
        try:
            # Start load test with the specific scenario
            load_test_payload = {
                'scenario': scenario,
                'duration': duration,
                'iterations': 1,
                'target': 'combined',
                'collect_baseline': True,  # Special flag to save baseline
                'baseline_name': f'baseline_{scenario}'
            }
            
            logger.info(f"Starting load test: {load_test_payload}")
            response = requests.post(
                f"{self.load_tester_api}/start",
                json=load_test_payload,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to start load test: {response.text}")
                return False
            
            logger.info(f"✅ Load test started. Waiting {duration}s for collection...")
            
            # Wait for collection to complete
            time.sleep(duration + 60)  # Extra 60s for processing
            
            # Trigger baseline save via autoscaler
            save_payload = {
                'scenario': scenario
            }
            
            response = requests.post(
                f"{self.autoscaler_url}/api/baseline/save",
                json=save_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Baseline dataset for '{scenario}' collected successfully")
                return True
            else:
                logger.error(f"Failed to save baseline: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Baseline collection failed: {e}")
            return False
    
    def collect_all_baselines(self, duration: int = 3600):
        """Collect all baseline datasets"""
        logger.info("📊 Collecting all baseline datasets...")
        
        for scenario in BASELINE_SCENARIOS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Collecting baseline: {scenario.upper()}")
            logger.info(f"{'='*60}\n")
            
            success = self.collect_baseline_dataset(scenario, duration)
            if not success:
                logger.error(f"Failed to collect baseline for {scenario}")
                # Continue with other scenarios
            
            # Wait between collections
            if scenario != BASELINE_SCENARIOS[-1]:
                logger.info("Waiting 5 minutes before next collection...")
                time.sleep(300)
        
        logger.info("✅ Baseline collection complete")
    
    def load_baseline_dataset(self, scenario: str) -> bool:
        """Load a baseline dataset into the autoscaler"""
        logger.info(f"📥 Loading baseline dataset: {scenario}")
        
        try:
            response = requests.post(
                f"{self.autoscaler_url}/api/baseline/load",
                json={'scenario': scenario},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Loaded {data['data_points_loaded']} data points")
                return True
            else:
                logger.error(f"Failed to load baseline: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Baseline loading failed: {e}")
            return False
    
    def train_model(self, model_name: str) -> Dict:
        """Train a specific model"""
        logger.info(f"🧠 Training model: {model_name}")
        
        try:
            response = requests.post(
                f"{self.autoscaler_url}/api/models/train/{model_name}",
                timeout=300  # 5 minutes timeout for training
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    logger.info(f"✅ Model trained successfully")
                    return result
                else:
                    logger.error(f"Model training failed: {result.get('error')}")
                    return result
            else:
                logger.error(f"Training request failed: {response.text}")
                return {'success': False, 'error': response.text}
                
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_load_test(self, scenario: str, duration: int, model_name: str) -> Dict:
        """Run a load test"""
        logger.info(f"🚀 Running load test: {scenario} scenario for {duration}s")
        
        try:
            payload = {
                'scenario': scenario,
                'duration': duration,
                'iterations': 1,
                'target': 'combined',
                'test_metadata': {
                    'model': model_name,
                    'baseline_scenario': scenario,
                    'test_type': 'model_comparison'
                }
            }
            
            response = requests.post(
                f"{self.load_tester_api}/start",
                json=payload,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to start load test: {response.text}")
                return {'success': False, 'error': response.text}
            
            logger.info(f"Load test started. Waiting {duration + 120}s for completion...")
            time.sleep(duration + 120)  # Wait for test + processing time
            
            # Get results
            results_response = requests.get(f"{self.load_tester_api}/results", timeout=10)
            if results_response.status_code == 200:
                return results_response.json()
            else:
                return {'success': False, 'error': 'Could not retrieve results'}
                
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_single_test(self, model_name: str, scenario: str, duration: int) -> Dict:
        """
        Run a complete test: load baseline, train model, run load test, collect results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST: Model={model_name}, Scenario={scenario}, Duration={duration}s")
        logger.info(f"{'='*80}\n")
        
        test_start = datetime.now()
        
        result = {
            'model': model_name,
            'scenario': scenario,
            'duration': duration,
            'timestamp': test_start.isoformat(),
            'success': False
        }
        
        # Step 1: Load baseline dataset
        if not self.load_baseline_dataset(scenario):
            result['error'] = 'Failed to load baseline dataset'
            return result
        
        time.sleep(10)  # Let system stabilize
        
        # Step 2: Train model
        training_result = self.train_model(model_name)
        if not training_result.get('success'):
            result['error'] = f"Model training failed: {training_result.get('error')}"
            return result
        
        result['training_time_seconds'] = training_result.get('training_result', {}).get('training_time_seconds')
        result['training_samples'] = training_result.get('training_result', {}).get('samples_used')
        
        time.sleep(30)  # Let model stabilize
        
        # Step 3: Run load test
        load_test_result = self.run_load_test(scenario, duration, model_name)
        if not load_test_result.get('success'):
            result['error'] = f"Load test failed: {load_test_result.get('error')}"
            return result
        
        # Extract metrics from load test results
        result['success'] = True
        result['load_test_results'] = load_test_result
        
        # Calculate test duration
        result['total_test_time_seconds'] = (datetime.now() - test_start).total_seconds()
        
        logger.info(f"✅ Test completed successfully in {result['total_test_time_seconds']:.0f}s")
        
        return result
    
    def run_test_matrix(self, models: List[str], scenarios: List[str], duration: int):
        """
        Run tests for all combinations of models and scenarios
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING TEST MATRIX")
        logger.info(f"Models: {models}")
        logger.info(f"Scenarios: {scenarios}")
        logger.info(f"Duration per test: {duration}s")
        logger.info(f"Total tests: {len(models) * len(scenarios)}")
        logger.info(f"{'='*80}\n")
        
        total_tests = len(models) * len(scenarios)
        completed = 0
        
        for model in models:
            for scenario in scenarios:
                completed += 1
                logger.info(f"\n[TEST {completed}/{total_tests}]")
                
                result = self.run_single_test(model, scenario, duration)
                self.test_results.append(result)
                
                # Save incremental results
                self.save_results()
                
                # Wait between tests
                if completed < total_tests:
                    logger.info("Waiting 2 minutes before next test...")
                    time.sleep(120)
        
        logger.info(f"\n✅ Test matrix complete: {completed} tests")
        self.generate_summary()
    
    def save_results(self):
        """Save test results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = os.path.join(self.results_dir, f'test_results_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"Results saved to: {json_file}")
        
        # Save CSV summary
        if self.test_results:
            df = pd.DataFrame(self.test_results)
            csv_file = os.path.join(self.results_dir, f'test_summary_{timestamp}.csv')
            df.to_csv(csv_file, index=False)
            logger.info(f"Summary saved to: {csv_file}")
    
    def generate_summary(self):
        """Generate a summary report of all tests"""
        if not self.test_results:
            logger.warning("No test results to summarize")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*80}\n")
        
        df = pd.DataFrame(self.test_results)
        
        # Overall statistics
        total_tests = len(df)
        successful_tests = len(df[df['success'] == True])
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        
        # Group by model
        logger.info(f"\nResults by Model:")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            success_rate = (model_df['success'].sum() / len(model_df)) * 100
            logger.info(f"  {model}: {model_df['success'].sum()}/{len(model_df)} successful ({success_rate:.1f}%)")
        
        # Group by scenario
        logger.info(f"\nResults by Scenario:")
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            success_rate = (scenario_df['success'].sum() / len(scenario_df)) * 100
            logger.info(f"  {scenario}: {scenario_df['success'].sum()}/{len(scenario_df)} successful ({success_rate:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Model Testing Orchestration System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect baselines command
    collect_parser = subparsers.add_parser('collect-baselines', help='Collect all baseline datasets')
    collect_parser.add_argument('--duration', type=int, default=3600,
                               help='Duration for each baseline collection (seconds)')
    
    # Run tests command
    test_parser = subparsers.add_parser('run-tests', help='Run test matrix')
    test_parser.add_argument('--models', type=str, default='all',
                            help='Comma-separated list of models or "all"')
    test_parser.add_argument('--scenarios', type=str, default='all',
                            help='Comma-separated list of scenarios or "all"')
    test_parser.add_argument('--duration', type=int, default=1800,
                            help='Duration for each test (seconds)')
    test_parser.add_argument('--autoscaler-url', type=str, default=AUTOSCALER_URL)
    test_parser.add_argument('--load-tester-api', type=str, default=LOAD_TESTER_API)
    
    # Check services command
    check_parser = subparsers.add_parser('check', help='Check service availability')
    
    args = parser.parse_args()
    
    orchestrator = TestingOrchestrator(
        autoscaler_url=getattr(args, 'autoscaler_url', AUTOSCALER_URL),
        load_tester_api=getattr(args, 'load_tester_api', LOAD_TESTER_API)
    )
    
    if args.command == 'collect-baselines':
        orchestrator.collect_all_baselines(args.duration)
    
    elif args.command == 'run-tests':
        # Check services first
        if not orchestrator.check_services():
            logger.error("❌ Some services are not available. Aborting.")
            sys.exit(1)
        
        # Parse models
        if args.models == 'all':
            models = MODEL_VARIANTS
        else:
            models = [m.strip() for m in args.models.split(',')]
        
        # Parse scenarios
        if args.scenarios == 'all':
            scenarios = BASELINE_SCENARIOS
        else:
            scenarios = [s.strip() for s in args.scenarios.split(',')]
        
        # Run tests
        orchestrator.run_test_matrix(models, scenarios, args.duration)
    
    elif args.command == 'check':
        orchestrator.check_services()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
