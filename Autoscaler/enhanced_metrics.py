"""
Enhanced Metrics Collection System for Kubernetes Autoscaling
Based on "Time series big data: a survey on data stream frameworks, analysis and algorithms"
Implements comprehensive metrics tracking and analysis
"""

import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance tracking"""
    model_name: str
    
    # Training metrics
    training_time_ms: float = 0.0
    training_success: bool = False
    training_attempts: int = 0
    last_training_time: Optional[datetime] = None
    
    # Prediction metrics
    prediction_time_ms: float = 0.0
    prediction_count: int = 0
    prediction_success_rate: float = 0.0
    last_prediction_time: Optional[datetime] = None
    
    # Accuracy metrics
    mse: float = float('inf')
    mae: float = float('inf')
    rmse: float = float('inf')
    mape: float = float('inf')  # Mean Absolute Percentage Error
    smape: float = float('inf')  # Symmetric Mean Absolute Percentage Error
    r2_score: float = -float('inf')
    
    # Rolling accuracy (last N predictions)
    rolling_mse: deque = field(default_factory=lambda: deque(maxlen=50))
    rolling_mae: deque = field(default_factory=lambda: deque(maxlen=50))
    rolling_accuracy: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Prediction quality metrics
    prediction_variance: float = 0.0
    prediction_stability: float = 0.0  # Low variance = high stability
    overprediction_rate: float = 0.0
    underprediction_rate: float = 0.0
    
    # Resource efficiency metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    model_size_mb: float = 0.0
    
    # Business metrics
    scaling_decisions: int = 0
    successful_scales: int = 0
    scale_up_count: int = 0
    scale_down_count: int = 0
    false_positive_scales: int = 0  # Unnecessary scaling
    false_negative_scales: int = 0  # Missed scaling opportunities
    
    # Reliability metrics
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    uptime_hours: float = 0.0
    
    def update_training_metrics(self, training_time_ms: float, success: bool):
        """Update training-related metrics"""
        self.training_attempts += 1
        self.training_time_ms = training_time_ms
        self.training_success = success
        self.last_training_time = datetime.now()
        
        if not success:
            self.consecutive_failures += 1
            self.max_consecutive_failures = max(self.max_consecutive_failures, self.consecutive_failures)
        else:
            self.consecutive_failures = 0
    
    def update_prediction_metrics(self, prediction_time_ms: float, success: bool):
        """Update prediction-related metrics"""
        self.prediction_count += 1
        self.prediction_time_ms = prediction_time_ms
        self.last_prediction_time = datetime.now()
        
        # Update success rate
        if success:
            self.prediction_success_rate = (self.prediction_success_rate * (self.prediction_count - 1) + 1.0) / self.prediction_count
        else:
            self.prediction_success_rate = (self.prediction_success_rate * (self.prediction_count - 1)) / self.prediction_count
    
    def update_accuracy_metrics(self, actual: float, predicted: float):
        """Update accuracy metrics with new prediction vs actual data"""
        if actual == 0:
            actual = 0.001  # Avoid division by zero
        
        # Calculate individual errors
        error = actual - predicted
        abs_error = abs(error)
        squared_error = error ** 2
        percentage_error = abs_error / abs(actual) * 100
        symmetric_percentage_error = abs_error / (abs(actual) + abs(predicted)) * 200
        
        # Update rolling metrics
        self.rolling_mse.append(squared_error)
        self.rolling_mae.append(abs_error)
        
        # Calculate accuracy (1 - normalized error)
        normalized_error = min(abs_error / max(abs(actual), 1.0), 1.0)
        accuracy = max(0.0, 1.0 - normalized_error)
        self.rolling_accuracy.append(accuracy)
        
        # Update aggregate metrics
        if len(self.rolling_mse) > 0:
            self.mse = np.mean(self.rolling_mse)
            self.rmse = np.sqrt(self.mse)
            self.mae = np.mean(self.rolling_mae)
        
        if len(self.rolling_accuracy) > 0:
            # Calculate R² score approximation
            variance_actual = np.var([actual] * len(self.rolling_accuracy)) if len(self.rolling_accuracy) > 1 else 1.0
            self.r2_score = max(-1.0, 1.0 - (self.mse / max(variance_actual, 0.001)))
        
        # Update prediction quality metrics
        if predicted > actual:
            self.overprediction_rate += 1
        else:
            self.underprediction_rate += 1
        
        total_predictions = self.overprediction_rate + self.underprediction_rate
        if total_predictions > 0:
            self.overprediction_rate = self.overprediction_rate / total_predictions
            self.underprediction_rate = self.underprediction_rate / total_predictions
    
    def calculate_prediction_stability(self, recent_predictions: List[float]):
        """Calculate prediction stability based on variance"""
        if len(recent_predictions) > 1:
            self.prediction_variance = np.var(recent_predictions)
            self.prediction_stability = 1.0 / (1.0 + self.prediction_variance)  # Higher stability = lower variance
    
    def update_scaling_metrics(self, scale_decision: str, success: bool):
        """Update scaling-related business metrics"""
        self.scaling_decisions += 1
        
        if success:
            self.successful_scales += 1
        
        if scale_decision == 'scale_up':
            self.scale_up_count += 1
        elif scale_decision == 'scale_down':
            self.scale_down_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'model': self.model_name,
            'training': {
                'success': self.training_success,
                'time_ms': self.training_time_ms,
                'attempts': self.training_attempts,
                'last_trained': self.last_training_time.isoformat() if self.last_training_time else None
            },
            'prediction': {
                'count': self.prediction_count,
                'success_rate': round(self.prediction_success_rate, 4),
                'avg_time_ms': round(self.prediction_time_ms, 2),
                'last_predicted': self.last_prediction_time.isoformat() if self.last_prediction_time else None
            },
            'accuracy': {
                'mse': round(self.mse, 6),
                'mae': round(self.mae, 4),
                'rmse': round(self.rmse, 4),
                'r2_score': round(self.r2_score, 4),
                'rolling_samples': len(self.rolling_accuracy)
            },
            'quality': {
                'prediction_variance': round(self.prediction_variance, 4),
                'prediction_stability': round(self.prediction_stability, 4),
                'overprediction_rate': round(self.overprediction_rate, 4),
                'underprediction_rate': round(self.underprediction_rate, 4)
            },
            'scaling': {
                'total_decisions': self.scaling_decisions,
                'success_rate': round(self.successful_scales / max(self.scaling_decisions, 1), 4),
                'scale_up_count': self.scale_up_count,
                'scale_down_count': self.scale_down_count
            },
            'reliability': {
                'consecutive_failures': self.consecutive_failures,
                'max_consecutive_failures': self.max_consecutive_failures,
                'uptime_hours': round(self.uptime_hours, 2)
            }
        }


class MetricsCollector:
    """Enhanced metrics collection and analysis system"""
    
    def __init__(self, max_history_size: int = 1000):
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.max_history_size = max_history_size
        
        # System-wide metrics
        self.system_start_time = datetime.now()
        self.total_predictions = 0
        self.total_scaling_decisions = 0
        self.system_uptime_seconds = 0
        
        # Historical data for analysis
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.accuracy_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.performance_timeline: deque = deque(maxlen=max_history_size)
        
        # Model comparison metrics
        self.model_rankings: Dict[str, float] = {}
        self.selection_frequency: Dict[str, int] = defaultdict(int)
        self.head_to_head_comparisons: Dict[tuple, Dict] = {}
        
        # Advanced metrics
        self.concept_drift_detection: Dict[str, List] = defaultdict(list)
        self.seasonal_patterns: Dict[str, Dict] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        
        logger.info("🎯 Enhanced Metrics Collector initialized")
    
    def register_model(self, model_name: str) -> ModelPerformanceMetrics:
        """Register a new model for tracking"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelPerformanceMetrics(model_name)
            logger.info(f"📊 Registered model for metrics tracking: {model_name}")
        
        return self.model_metrics[model_name]
    
    def record_training(self, model_name: str, training_time_ms: float, success: bool, **kwargs):
        """Record model training metrics"""
        metrics = self.register_model(model_name)
        metrics.update_training_metrics(training_time_ms, success)
        
        # Record additional training info if provided
        if 'memory_usage_mb' in kwargs:
            metrics.memory_usage_mb = kwargs['memory_usage_mb']
        if 'model_size_mb' in kwargs:
            metrics.model_size_mb = kwargs['model_size_mb']
        
        logger.info(f"📈 Training recorded for {model_name}: {training_time_ms:.2f}ms, success={success}")
    
    def record_prediction(self, model_name: str, prediction_time_ms: float, predicted_value: float, 
                         success: bool = True, actual_value: Optional[float] = None):
        """Record prediction metrics"""
        metrics = self.register_model(model_name)
        metrics.update_prediction_metrics(prediction_time_ms, success)
        
        self.total_predictions += 1
        
        # Store prediction history
        self.prediction_history[model_name].append({
            'timestamp': datetime.now(),
            'predicted': predicted_value,
            'actual': actual_value,
            'prediction_time_ms': prediction_time_ms
        })
        
        # If we have actual value, update accuracy metrics
        if actual_value is not None:
            metrics.update_accuracy_metrics(actual_value, predicted_value)
            
            # Update accuracy history
            self.accuracy_history[model_name].append({
                'timestamp': datetime.now(),
                'error': abs(actual_value - predicted_value),
                'percentage_error': abs(actual_value - predicted_value) / max(abs(actual_value), 1.0) * 100
            })
        
        # Update prediction stability
        recent_predictions = [p['predicted'] for p in list(self.prediction_history[model_name])[-10:]]
        metrics.calculate_prediction_stability(recent_predictions)
        
        logger.debug(f"🎯 Prediction recorded for {model_name}: {predicted_value:.3f} in {prediction_time_ms:.2f}ms")
    
    def record_scaling_decision(self, model_name: str, decision: str, success: bool, 
                               actual_replicas: int, target_replicas: int):
        """Record scaling decision metrics"""
        if model_name in self.model_metrics:
            self.model_metrics[model_name].update_scaling_metrics(decision, success)
        
        self.total_scaling_decisions += 1
        self.selection_frequency[model_name] += 1
        
        logger.info(f"⚖️ Scaling recorded for {model_name}: {decision} from {actual_replicas} to {target_replicas}, success={success}")
    
    def update_model_ranking(self, rankings: Dict[str, float]):
        """Update model performance rankings"""
        self.model_rankings = rankings.copy()
        
        # Record ranking timeline
        self.performance_timeline.append({
            'timestamp': datetime.now(),
            'rankings': rankings.copy()
        })
        
        logger.debug(f"🏆 Model rankings updated: {rankings}")
    
    def detect_concept_drift(self, model_name: str, window_size: int = 50, threshold: float = 0.1):
        """Detect concept drift in model performance"""
        if model_name not in self.accuracy_history or len(self.accuracy_history[model_name]) < window_size * 2:
            return False
        
        recent_errors = [h['percentage_error'] for h in list(self.accuracy_history[model_name])[-window_size:]]
        historical_errors = [h['percentage_error'] for h in list(self.accuracy_history[model_name])[:-window_size][-window_size:]]
        
        if len(recent_errors) < window_size or len(historical_errors) < window_size:
            return False
        
        recent_mean = np.mean(recent_errors)
        historical_mean = np.mean(historical_errors)
        
        drift_magnitude = abs(recent_mean - historical_mean) / max(historical_mean, 1.0)
        
        if drift_magnitude > threshold:
            self.concept_drift_detection[model_name].append({
                'timestamp': datetime.now(),
                'drift_magnitude': drift_magnitude,
                'recent_error': recent_mean,
                'historical_error': historical_mean
            })
            
            logger.warning(f"🚨 Concept drift detected for {model_name}: {drift_magnitude:.3f} (threshold: {threshold})")
            return True
        
        return False
    
    def analyze_seasonal_patterns(self, model_name: str, period_minutes: int = 60):
        """Analyze seasonal patterns in model performance"""
        if model_name not in self.prediction_history:
            return {}
        
        history = list(self.prediction_history[model_name])
        if len(history) < period_minutes:
            return {}
        
        # Group by time periods
        period_performance = defaultdict(list)
        
        for entry in history:
            if entry['actual'] is not None:
                time_bucket = entry['timestamp'].minute // (period_minutes // 60) if period_minutes < 60 else entry['timestamp'].hour
                error = abs(entry['actual'] - entry['predicted'])
                period_performance[time_bucket].append(error)
        
        # Calculate average performance by period
        seasonal_stats = {}
        for period, errors in period_performance.items():
            if len(errors) > 0:
                seasonal_stats[period] = {
                    'avg_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'sample_count': len(errors)
                }
        
        self.seasonal_patterns[model_name] = seasonal_stats
        return seasonal_stats
    
    def calculate_model_correlations(self):
        """Calculate correlation matrix between model predictions"""
        model_names = list(self.prediction_history.keys())
        if len(model_names) < 2:
            return None
        
        # Get aligned predictions (same timestamps)
        aligned_data = {}
        min_length = min(len(self.prediction_history[name]) for name in model_names)
        
        for name in model_names:
            recent_predictions = [p['predicted'] for p in list(self.prediction_history[name])[-min_length:]]
            aligned_data[name] = recent_predictions
        
        if min_length < 10:  # Need minimum data for correlation
            return None
        
        # Create correlation matrix
        data_matrix = np.array([aligned_data[name] for name in model_names])
        self.correlation_matrix = np.corrcoef(data_matrix)
        
        return {
            'model_names': model_names,
            'correlation_matrix': self.correlation_matrix.tolist()
        }
    
    def get_model_comparison(self, model1: str, model2: str):
        """Get head-to-head comparison between two models"""
        if model1 not in self.model_metrics or model2 not in self.model_metrics:
            return {}
        
        m1_metrics = self.model_metrics[model1]
        m2_metrics = self.model_metrics[model2]
        
        comparison = {
            'models': [model1, model2],
            'mse_comparison': {
                model1: m1_metrics.mse,
                model2: m2_metrics.mse,
                'winner': model1 if m1_metrics.mse < m2_metrics.mse else model2
            },
            'prediction_speed': {
                model1: m1_metrics.prediction_time_ms,
                model2: m2_metrics.prediction_time_ms,
                'winner': model1 if m1_metrics.prediction_time_ms < m2_metrics.prediction_time_ms else model2
            },
            'reliability': {
                model1: m1_metrics.prediction_success_rate,
                model2: m2_metrics.prediction_success_rate,
                'winner': model1 if m1_metrics.prediction_success_rate > m2_metrics.prediction_success_rate else model2
            },
            'stability': {
                model1: m1_metrics.prediction_stability,
                model2: m2_metrics.prediction_stability,
                'winner': model1 if m1_metrics.prediction_stability > m2_metrics.prediction_stability else model2
            }
        }
        
        return comparison
    
    def get_system_summary(self):
        """Get comprehensive system metrics summary"""
        current_time = datetime.now()
        uptime = (current_time - self.system_start_time).total_seconds() / 3600  # hours
        
        # Calculate aggregate statistics
        all_models = list(self.model_metrics.values())
        
        if len(all_models) > 0:
            avg_mse = np.mean([m.mse for m in all_models if m.mse != float('inf')])
            avg_prediction_time = np.mean([m.prediction_time_ms for m in all_models])
            total_successful_scales = sum(m.successful_scales for m in all_models)
            total_scale_decisions = sum(m.scaling_decisions for m in all_models)
        else:
            avg_mse = 0
            avg_prediction_time = 0
            total_successful_scales = 0
            total_scale_decisions = 0
        
        return {
            'system': {
                'uptime_hours': round(uptime, 2),
                'start_time': self.system_start_time.isoformat(),
                'total_predictions': self.total_predictions,
                'registered_models': len(self.model_metrics)
            },
            'performance': {
                'average_mse': round(avg_mse, 6),
                'average_prediction_time_ms': round(avg_prediction_time, 2),
                'scaling_success_rate': round(total_successful_scales / max(total_scale_decisions, 1), 4),
                'model_rankings': self.model_rankings
            },
            'usage': {
                'selection_frequency': dict(self.selection_frequency),
                'total_scaling_decisions': total_scale_decisions,
                'concept_drifts_detected': sum(len(drifts) for drifts in self.concept_drift_detection.values())
            },
            'models': {name: metrics.get_summary() for name, metrics in self.model_metrics.items()}
        }
    
    def export_metrics_csv(self, filepath: str):
        """Export metrics to CSV for analysis"""
        try:
            rows = []
            
            for model_name, metrics in self.model_metrics.items():
                base_row = {
                    'model': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'mse': metrics.mse,
                    'mae': metrics.mae,
                    'rmse': metrics.rmse,
                    'r2_score': metrics.r2_score,
                    'prediction_time_ms': metrics.prediction_time_ms,
                    'training_time_ms': metrics.training_time_ms,
                    'prediction_count': metrics.prediction_count,
                    'success_rate': metrics.prediction_success_rate,
                    'stability': metrics.prediction_stability,
                    'scaling_decisions': metrics.scaling_decisions,
                    'successful_scales': metrics.successful_scales
                }
                rows.append(base_row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            
            logger.info(f"📊 Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export metrics: {e}")
            return False
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_summary': self.get_system_summary(),
            'concept_drift_summary': {
                model: len(drifts) for model, drifts in self.concept_drift_detection.items()
            },
            'seasonal_analysis': self.seasonal_patterns,
            'model_correlations': self.calculate_model_correlations()
        }
        
        # Add top performers
        if self.model_rankings:
            sorted_models = sorted(self.model_rankings.items(), key=lambda x: x[1], reverse=True)
            report['top_performers'] = {
                'best_model': sorted_models[0][0] if sorted_models else None,
                'worst_model': sorted_models[-1][0] if sorted_models else None,
                'rankings': sorted_models
            }
        
        return report


# Global metrics collector instance
global_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    return global_metrics_collector


# Convenience functions for integration
def record_model_training(model_name: str, training_time_ms: float, success: bool, **kwargs):
    """Record model training metrics"""
    global_metrics_collector.record_training(model_name, training_time_ms, success, **kwargs)


def record_model_prediction(model_name: str, prediction_time_ms: float, predicted_value: float, 
                           actual_value: Optional[float] = None, success: bool = True):
    """Record model prediction metrics"""
    global_metrics_collector.record_prediction(model_name, prediction_time_ms, predicted_value, success, actual_value)


def record_scaling_decision(model_name: str, decision: str, success: bool, actual_replicas: int, target_replicas: int):
    """Record scaling decision metrics"""
    global_metrics_collector.record_scaling_decision(model_name, decision, success, actual_replicas, target_replicas)


def update_model_rankings(rankings: Dict[str, float]):
    """Update model performance rankings"""
    global_metrics_collector.update_model_ranking(rankings)


def get_system_metrics():
    """Get comprehensive system metrics"""
    return global_metrics_collector.get_system_summary()


def export_system_metrics(filepath: str):
    """Export system metrics to CSV"""
    return global_metrics_collector.export_metrics_csv(filepath)


def generate_system_report():
    """Generate comprehensive performance report"""
    return global_metrics_collector.generate_performance_report()


if __name__ == "__main__":
    # Test the metrics system
    collector = MetricsCollector()
    
    # Simulate model training and predictions
    models = ['gru', 'lstm', 'arima', 'prophet']
    
    print("🧪 Testing Enhanced Metrics System...")
    
    for model in models:
        # Record training
        collector.record_training(model, np.random.uniform(1000, 5000), True)
        
        # Record predictions with actual values
        for i in range(20):
            predicted = np.random.uniform(1, 10)
            actual = predicted + np.random.normal(0, 0.5)  # Add some noise
            collector.record_prediction(model, np.random.uniform(10, 100), predicted, True, actual)
        
        # Record scaling decisions
        for i in range(5):
            decision = np.random.choice(['scale_up', 'scale_down', 'maintain'])
            success = np.random.choice([True, False], p=[0.8, 0.2])
            collector.record_scaling_decision(model, decision, success, 3, 4)
    
    # Update rankings
    rankings = {model: np.random.uniform(0.1, 0.9) for model in models}
    collector.update_model_ranking(rankings)
    
    # Generate report
    report = collector.generate_performance_report()
    print("\n📊 System Summary:")
    print(f"Models tracked: {report['system_summary']['system']['registered_models']}")
    print(f"Total predictions: {report['system_summary']['system']['total_predictions']}")
    print(f"Average MSE: {report['system_summary']['performance']['average_mse']:.6f}")
    
    print("\n🏆 Model Rankings:")
    for model, score in report['top_performers']['rankings']:
        print(f"  {model}: {score:.4f}")
    
    print("\n✅ Enhanced metrics system testing complete!")