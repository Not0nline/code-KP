#!/usr/bin/env python3
"""
Baseline Dataset Collection and Management System

This module handles:
1. Collecting baseline datasets for low/medium/high traffic scenarios
2. Loading/saving baseline datasets to disk
3. Initializing the autoscaler with pre-collected data for consistent testing
4. Managing dataset transitions between different traffic scenarios

Usage:
- Collect datasets: python baseline_datasets.py collect --scenario low --duration 3600
- Load dataset: POST /api/load_baseline with {"scenario": "low"}
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Baseline dataset configuration
BASELINE_DIR = "/data/baselines"  # Persistent storage for baseline datasets
BASELINE_SCENARIOS = ['low', 'medium', 'high']

class BaselineDatasetManager:
    """Manages baseline datasets for different traffic scenarios"""
    
    def __init__(self, data_dir: str = BASELINE_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def get_baseline_path(self, scenario: str) -> str:
        """Get the file path for a baseline scenario"""
        if scenario not in BASELINE_SCENARIOS:
            raise ValueError(f"Invalid scenario: {scenario}. Must be one of {BASELINE_SCENARIOS}")
        return os.path.join(self.data_dir, f"baseline_{scenario}.json")
    
    def save_baseline(self, scenario: str, traffic_data: List[Dict], metadata: Optional[Dict] = None):
        """
        Save baseline dataset to disk
        
        Args:
            scenario: Traffic scenario name (low/medium/high)
            traffic_data: List of traffic data points
            metadata: Optional metadata about the collection (duration, timestamp, etc.)
        """
        filepath = self.get_baseline_path(scenario)
        
        baseline_data = {
            'scenario': scenario,
            'collected_at': datetime.now().isoformat(),
            'data_points': len(traffic_data),
            'metadata': metadata or {},
            'traffic_data': traffic_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        logger.info(f"✅ Saved baseline dataset for scenario '{scenario}': {len(traffic_data)} data points to {filepath}")
        return filepath
    
    def load_baseline(self, scenario: str) -> Dict:
        """
        Load baseline dataset from disk
        
        Args:
            scenario: Traffic scenario name (low/medium/high)
            
        Returns:
            Dictionary with baseline data and metadata
        """
        filepath = self.get_baseline_path(scenario)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Baseline dataset not found for scenario '{scenario}' at {filepath}")
        
        with open(filepath, 'r') as f:
            baseline_data = json.load(f)
        
        logger.info(f"✅ Loaded baseline dataset for scenario '{scenario}': {baseline_data['data_points']} data points")
        return baseline_data
    
    def list_available_baselines(self) -> List[Dict]:
        """List all available baseline datasets with their metadata"""
        available = []
        
        for scenario in BASELINE_SCENARIOS:
            filepath = self.get_baseline_path(scenario)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    available.append({
                        'scenario': scenario,
                        'data_points': data['data_points'],
                        'collected_at': data['collected_at'],
                        'filepath': filepath,
                        'metadata': data.get('metadata', {})
                    })
                except Exception as e:
                    logger.error(f"Error reading baseline {scenario}: {e}")
        
        return available
    
    def delete_baseline(self, scenario: str):
        """Delete a baseline dataset"""
        filepath = self.get_baseline_path(scenario)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️ Deleted baseline dataset for scenario '{scenario}'")
        else:
            logger.warning(f"Baseline dataset not found for scenario '{scenario}'")
    
    def validate_baseline(self, scenario: str) -> Dict:
        """
        Validate a baseline dataset
        
        Returns:
            Dictionary with validation results
        """
        try:
            baseline_data = self.load_baseline(scenario)
            traffic_data = baseline_data['traffic_data']
            
            validation = {
                'valid': True,
                'scenario': scenario,
                'data_points': len(traffic_data),
                'issues': []
            }
            
            # Check minimum data points
            if len(traffic_data) < 100:
                validation['issues'].append(f"Insufficient data points: {len(traffic_data)} < 100")
                validation['valid'] = False
            
            # Check data structure
            required_fields = ['timestamp', 'cpu', 'traffic', 'replicas']
            for i, point in enumerate(traffic_data[:10]):  # Check first 10 points
                missing = [field for field in required_fields if field not in point]
                if missing:
                    validation['issues'].append(f"Missing fields in point {i}: {missing}")
                    validation['valid'] = False
                    break
            
            # Check timestamp ordering
            timestamps = [point.get('timestamp') for point in traffic_data if 'timestamp' in point]
            if timestamps and timestamps != sorted(timestamps):
                validation['issues'].append("Timestamps are not in chronological order")
                validation['valid'] = False
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'scenario': scenario,
                'issues': [f"Validation error: {str(e)}"]
            }


def collect_baseline_dataset(scenario: str, duration_seconds: int = 14400, 
                             prometheus_url: str = None) -> str:
    """
    Collect a baseline dataset by running a traffic scenario
    
    This function should be called while a load test is running with the specified scenario.
    It collects metrics from Prometheus for the specified duration.
    
    Default duration is 14400 seconds (4 hours) = 240 data points at 60-second intervals
    
    Args:
        scenario: Traffic scenario (low/medium/high)
        duration_seconds: How long to collect data (default: 14400s = 4 hours = 240 points)
        prometheus_url: Prometheus server URL
        
    Returns:
        Path to saved baseline file
    """
    import time
    import requests
    from prometheus_api_client import PrometheusConnect
    
    if scenario not in BASELINE_SCENARIOS:
        raise ValueError(f"Invalid scenario: {scenario}. Must be one of {BASELINE_SCENARIOS}")
    
    prometheus_url = prometheus_url or os.getenv('PROMETHEUS_SERVER', 
        'http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090')
    
    logger.info(f"📊 Starting baseline collection for scenario '{scenario}'")
    logger.info(f"Duration: {duration_seconds}s ({duration_seconds/3600:.1f} hours)")
    logger.info(f"Expected data points: {duration_seconds//60} (at 60-second intervals)")
    logger.info(f"Prometheus: {prometheus_url}")
    
    prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
    traffic_data = []
    start_time = datetime.now()
    collection_interval = 60  # Collect every 60 seconds
    
    try:
        for elapsed in range(0, duration_seconds, collection_interval):
            try:
                # Query Prometheus for current metrics
                cpu_query = 'avg(rate(container_cpu_usage_seconds_total{pod=~"product-app-combined-.*"}[2m])) * 100'
                traffic_query = 'sum(rate(load_tester_requests_total{exported_service="Combined"}[1m]))'
                replicas_query = 'count(kube_pod_info{pod=~"product-app-combined-.*"})'
                
                cpu_result = prom.custom_query(query=cpu_query)
                traffic_result = prom.custom_query(query=traffic_query)
                replicas_result = prom.custom_query(query=replicas_query)
                
                cpu = float(cpu_result[0]['value'][1]) if cpu_result else 0
                traffic = float(traffic_result[0]['value'][1]) if traffic_result else 0
                replicas = int(replicas_result[0]['value'][1]) if replicas_result else 1
                
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_seconds': elapsed,
                    'cpu': cpu,
                    'traffic': traffic,
                    'replicas': replicas,
                    'scenario': scenario
                }
                
                traffic_data.append(data_point)
                
                # Log progress every 10 points
                if len(traffic_data) % 10 == 0:
                    logger.info(f"Progress: {len(traffic_data)}/~{duration_seconds//60} points - CPU={cpu:.2f}%, Traffic={traffic:.2f} req/s, Replicas={replicas}")
                
            except Exception as e:
                logger.error(f"Error collecting data point: {e}")
            
            # Sleep until next collection interval
            time.sleep(collection_interval)
            
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    
    # Save the collected data
    manager = BaselineDatasetManager()
    metadata = {
        'duration_seconds': duration_seconds,
        'collection_interval': collection_interval,
        'actual_duration': (datetime.now() - start_time).total_seconds(),
        'prometheus_url': prometheus_url,
        'expected_points': duration_seconds // collection_interval,
        'actual_points': len(traffic_data)
    }
    
    filepath = manager.save_baseline(scenario, traffic_data, metadata)
    logger.info(f"✅ Baseline collection complete: {len(traffic_data)} points saved to {filepath}")
    
    return filepath


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Baseline Dataset Management')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect a baseline dataset')
    collect_parser.add_argument('--scenario', required=True, choices=BASELINE_SCENARIOS,
                               help='Traffic scenario to collect')
    collect_parser.add_argument('--duration', type=int, default=3600,
                               help='Collection duration in seconds (default: 3600 = 1 hour)')
    collect_parser.add_argument('--prometheus-url', type=str,
                               help='Prometheus server URL')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available baseline datasets')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a baseline dataset')
    validate_parser.add_argument('--scenario', required=True, choices=BASELINE_SCENARIOS,
                                help='Scenario to validate')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a baseline dataset')
    delete_parser.add_argument('--scenario', required=True, choices=BASELINE_SCENARIOS,
                              help='Scenario to delete')
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        collect_baseline_dataset(args.scenario, args.duration, args.prometheus_url)
    
    elif args.command == 'list':
        manager = BaselineDatasetManager()
        baselines = manager.list_available_baselines()
        if baselines:
            print("\n📊 Available Baseline Datasets:")
            for b in baselines:
                print(f"\n  Scenario: {b['scenario']}")
                print(f"  Data Points: {b['data_points']}")
                print(f"  Collected: {b['collected_at']}")
                print(f"  File: {b['filepath']}")
        else:
            print("\n⚠️ No baseline datasets found")
    
    elif args.command == 'validate':
        manager = BaselineDatasetManager()
        result = manager.validate_baseline(args.scenario)
        print(f"\n{'✅' if result['valid'] else '❌'} Validation Result for '{args.scenario}':")
        print(f"  Data Points: {result.get('data_points', 'N/A')}")
        if result['issues']:
            print(f"  Issues:")
            for issue in result['issues']:
                print(f"    - {issue}")
        else:
            print(f"  Status: Valid ✅")
    
    elif args.command == 'delete':
        manager = BaselineDatasetManager()
        manager.delete_baseline(args.scenario)
    
    else:
        parser.print_help()
