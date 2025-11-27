#!/usr/bin/env python3
"""
Comprehensive test results analyzer
Analyzes all iterations to extract:
- Success rate, failure rate
- Total requests, successful requests, failed requests
- Average pods across the test
- Response time statistics
"""

import pandas as pd
import os
import json
from pathlib import Path

def analyze_iteration(iteration_path):
    """Analyze a single iteration directory"""
    results = {
        'iteration': iteration_path.name,
        'combined': None,
        'hpa': None
    }
    
    # Check if files are directly in iteration directory or in test_run subdirectories
    combined_csv = iteration_path / 'combined_results.csv'
    hpa_csv = iteration_path / 'hpa_results.csv'
    combined_pods_csv = iteration_path / 'combined_pod_counts.csv'
    hpa_pods_csv = iteration_path / 'hpa_pod_counts.csv'
    
    # If files are directly in iteration directory
    if combined_csv.exists():
        results['combined'] = analyze_service_data(combined_csv, combined_pods_csv, 'Combined')
    
    if hpa_csv.exists():
        results['hpa'] = analyze_service_data(hpa_csv, hpa_pods_csv, 'HPA')
    
    # Otherwise check test_run subdirectories (for older format)
    if not results['combined'] and not results['hpa']:
        test_runs = sorted([d for d in iteration_path.iterdir() if d.is_dir() and d.name.startswith('test_run')])
        
        for test_run in test_runs:
            combined_csv = test_run / 'combined_results.csv'
            hpa_csv = test_run / 'hpa_results.csv'
            
            if combined_csv.exists() and not results['combined']:
                results['combined'] = analyze_service_data(combined_csv, None, 'Combined')
            
            if hpa_csv.exists() and not results['hpa']:
                results['hpa'] = analyze_service_data(hpa_csv, None, 'HPA')
    
    return results

def analyze_service_data(results_csv, pods_csv, service_name):
    """Analyze data for a specific service (Combined or HPA)"""
    if not results_csv.exists():
        return None
    
    try:
        # Read results
        df = pd.read_csv(results_csv)
        
        # Calculate metrics
        total_requests = len(df)
        successful_requests = len(df[df['status_code'].between(200, 299)])
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        failure_rate = 100 - success_rate
        
        # Response time stats
        avg_response_time = df['response_time_ms'].mean()
        p50_response_time = df['response_time_ms'].median()
        p95_response_time = df['response_time_ms'].quantile(0.95)
        p99_response_time = df['response_time_ms'].quantile(0.99)
        
        # Try to get pod count from pod counts CSV
        avg_pods = None
        min_pods = None
        max_pods = None
        if pods_csv and pods_csv.exists():
            try:
                df_pods = pd.read_csv(pods_csv)
                # Check for various column names that might contain pod counts
                for col_name in ['pod_count', 'replica_count', 'current_replicas', 'replicas']:
                    if col_name in df_pods.columns:
                        avg_pods = df_pods[col_name].mean()
                        min_pods = int(df_pods[col_name].min())
                        max_pods = int(df_pods[col_name].max())
                        break
            except Exception as e:
                # Silently fail if pod counts can't be read
                pass
        
        result = {
            'service': service_name,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'failure_rate': failure_rate,
            'avg_response_time_ms': avg_response_time,
            'p50_response_time_ms': p50_response_time,
            'p95_response_time_ms': p95_response_time,
            'p99_response_time_ms': p99_response_time,
            'avg_pods': avg_pods,
            'min_pods': min_pods,
            'max_pods': max_pods
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {results_csv}: {e}")
        return None

def main():
    base_path = Path('./saved_tests/Holt_GRU/high')
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return
    
    all_results = []
    
    # Find all iteration directories
    iterations = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('iteration_')])
    
    print(f"Found {len(iterations)} iterations to analyze\n")
    
    for iteration_dir in iterations:
        print(f"Analyzing {iteration_dir.name}...")
        results = analyze_iteration(iteration_dir)
        all_results.append(results)
    
    # Generate summary report
    print("\n" + "="*120)
    print("COMPREHENSIVE TEST RESULTS SUMMARY - HIGH SCENARIO")
    print("="*120)
    
    for iteration_result in all_results:
        if not iteration_result['combined'] and not iteration_result['hpa']:
            continue
            
        print(f"\n{iteration_result['iteration'].upper()}")
        print("-"*120)
        
        # Print Combined results
        if iteration_result['combined']:
            data = iteration_result['combined']
            print(f"\n  [COMBINED SERVICE]:")
            print(f"     Total Requests:      {data['total_requests']:,}")
            print(f"     Successful Requests: {data['successful_requests']:,} ({data['success_rate']:.2f}%)")
            print(f"     Failed Requests:     {data['failed_requests']:,} ({data['failure_rate']:.2f}%)")
            print(f"     Avg Response Time:   {data['avg_response_time_ms']:.2f} ms")
            print(f"     P50 Response Time:   {data['p50_response_time_ms']:.2f} ms")
            print(f"     P95 Response Time:   {data['p95_response_time_ms']:.2f} ms")
            print(f"     P99 Response Time:   {data['p99_response_time_ms']:.2f} ms")
            if data['avg_pods']:
                print(f"     Average Pods:        {data['avg_pods']:.2f} (min: {data['min_pods']}, max: {data['max_pods']})")
        
        # Print HPA results
        if iteration_result['hpa']:
            data = iteration_result['hpa']
            print(f"\n  [HPA SERVICE]:")
            print(f"     Total Requests:      {data['total_requests']:,}")
            print(f"     Successful Requests: {data['successful_requests']:,} ({data['success_rate']:.2f}%)")
            print(f"     Failed Requests:     {data['failed_requests']:,} ({data['failure_rate']:.2f}%)")
            print(f"     Avg Response Time:   {data['avg_response_time_ms']:.2f} ms")
            print(f"     P50 Response Time:   {data['p50_response_time_ms']:.2f} ms")
            print(f"     P95 Response Time:   {data['p95_response_time_ms']:.2f} ms")
            print(f"     P99 Response Time:   {data['p99_response_time_ms']:.2f} ms")
            if data['avg_pods']:
                print(f"     Average Pods:        {data['avg_pods']:.2f} (min: {data['min_pods']}, max: {data['max_pods']})")
        
        print()
    
    # Calculate overall statistics
    print("\n" + "="*120)
    print("OVERALL STATISTICS ACROSS ALL ITERATIONS")
    print("="*120)
    
    combined_stats = []
    hpa_stats = []
    
    for iter_result in all_results:
        if iter_result['combined']:
            combined_stats.append(iter_result['combined'])
        if iter_result['hpa']:
            hpa_stats.append(iter_result['hpa'])
    
    def print_service_stats(stats, service_name):
        if not stats:
            return
        
        total_reqs = sum(t['total_requests'] for t in stats)
        total_success = sum(t['successful_requests'] for t in stats)
        total_failed = sum(t['failed_requests'] for t in stats)
        avg_success_rate = sum(t['success_rate'] for t in stats) / len(stats)
        avg_failure_rate = sum(t['failure_rate'] for t in stats) / len(stats)
        avg_response_time = sum(t['avg_response_time_ms'] for t in stats) / len(stats)
        avg_p95 = sum(t['p95_response_time_ms'] for t in stats) / len(stats)
        avg_p99 = sum(t['p99_response_time_ms'] for t in stats) / len(stats)
        
        pods_data = [t['avg_pods'] for t in stats if t['avg_pods'] is not None]
        avg_pods_overall = sum(pods_data) / len(pods_data) if pods_data else None
        min_pods_overall = min([t['min_pods'] for t in stats if t['min_pods'] is not None]) if any(t['min_pods'] for t in stats) else None
        max_pods_overall = max([t['max_pods'] for t in stats if t['max_pods'] is not None]) if any(t['max_pods'] for t in stats) else None
        
        print(f"\n[{service_name} SERVICE - AGGREGATED STATS]:")
        print(f"   Total Iterations:         {len(stats)}")
        print(f"   Total Requests:           {total_reqs:,}")
        print(f"   Total Successful:         {total_success:,}")
        print(f"   Total Failed:             {total_failed:,}")
        print(f"   Average Success Rate:     {avg_success_rate:.2f}%")
        print(f"   Average Failure Rate:     {avg_failure_rate:.2f}%")
        print(f"   Average Response Time:    {avg_response_time:.2f} ms")
        print(f"   Average P95 Response:     {avg_p95:.2f} ms")
        print(f"   Average P99 Response:     {avg_p99:.2f} ms")
        if avg_pods_overall:
            print(f"   Average Pods Overall:     {avg_pods_overall:.2f}")
            if min_pods_overall and max_pods_overall:
                print(f"   Pod Range:                {min_pods_overall} - {max_pods_overall}")
    
    print_service_stats(combined_stats, "COMBINED (Predictive + HPA)")
    print_service_stats(hpa_stats, "HPA ONLY")
    
    # Save detailed results to JSON
    output_file = 'test_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    main()
