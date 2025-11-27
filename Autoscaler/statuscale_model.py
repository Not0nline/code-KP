"""
StatuScale Algorithm - Lecturer's Custom Scaling Algorithm
Rule-based adaptive autoscaling with trend detection
"""

import numpy as np
import pandas as pd
import logging
from collections import deque

logger = logging.getLogger(__name__)


class StatuScalePredictor:
    """
    StatuScale: Status-based Predictive Scaling
    
    Algorithm characteristics:
    - Rule-based with trend analysis
    - Uses rolling window for CPU/traffic patterns
    - Adaptive thresholds based on historical variance
    - Fast decision-making (no training required)
    """
    
    def __init__(self, config):
        """
        Initialize StatuScale
        
        Args:
            config: Dictionary with parameters
                - threshold_high: CPU threshold for scale up (default: 70%)
                - threshold_low: CPU threshold for scale down (default: 30%)
                - window_size: Rolling window for trend detection (default: 10)
                - look_back: Historical data to consider (default: 30)
                - look_forward: Prediction horizon (default: 24)
        """
        self.config = config
        self.is_trained = True  # Rule-based, always ready
        
        self.threshold_high = config.get('threshold_high', 70)
        self.threshold_low = config.get('threshold_low', 30)
        self.window_size = config.get('window_size', 10)
        self.look_back = config.get('look_back', 30)
        
        # Adaptive thresholds (learned from data)
        self.adaptive_high = self.threshold_high
        self.adaptive_low = self.threshold_low
        
        logger.info(f"StatuScale initialized: high={self.threshold_high}%, low={self.threshold_low}%, window={self.window_size}")
    
    def calculate_trend(self, values):
        """Calculate trend direction and magnitude"""
        if len(values) < 2:
            return 0, 0
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Handle edge cases
        if np.std(y) == 0:
            return 0, np.mean(y)
        
        # Slope and direction
        slope = np.polyfit(x, y, 1)[0]
        direction = 1 if slope > 0 else (-1 if slope < 0 else 0)
        
        return direction, slope
    
    def detect_spike(self, recent_data, threshold_multiplier=1.5):
        """Detect sudden spikes in CPU or traffic"""
        if len(recent_data) < 5:
            return False
        
        recent_mean = np.mean(recent_data[:-1])
        recent_std = np.std(recent_data[:-1])
        current = recent_data[-1]
        
        # Spike if current value exceeds mean + threshold * std
        threshold = recent_mean + (threshold_multiplier * recent_std)
        
        return current > threshold
    
    def calculate_volatility(self, values):
        """Calculate volatility (coefficient of variation)"""
        if len(values) < 2:
            return 0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if mean == 0:
            return 0
        
        return std / mean
    
    def update_adaptive_thresholds(self, data):
        """Update thresholds based on historical data variance"""
        try:
            df = pd.DataFrame(data)
            
            if len(df) < 20:
                return
            
            # Analyze CPU distribution
            cpu_values = df['cpu_utilization'].values
            cpu_mean = np.mean(cpu_values)
            cpu_std = np.std(cpu_values)
            
            # Adaptive thresholds based on variance
            # High variance = wider thresholds to avoid oscillation
            volatility = self.calculate_volatility(cpu_values)
            
            if volatility > 0.3:  # High volatility
                margin = 15
            else:  # Low volatility
                margin = 10
            
            self.adaptive_high = min(90, cpu_mean + margin)
            self.adaptive_low = max(20, cpu_mean - margin)
            
            logger.debug(f"StatuScale adaptive thresholds: high={self.adaptive_high:.1f}%, low={self.adaptive_low:.1f}%")
            
        except Exception as e:
            logger.error(f"StatuScale threshold update error: {e}")
    
    def train(self, data):
        """
        'Training' for StatuScale = learning adaptive thresholds
        This is lightweight and always succeeds
        """
        try:
            if len(data) >= 20:
                self.update_adaptive_thresholds(data)
            
            logger.info(f"StatuScale 'trained' with {len(data)} data points (adaptive thresholds updated)")
            return True
            
        except Exception as e:
            logger.error(f"StatuScale training error: {e}")
            return False
    
    def predict(self, data, steps=24):
        """
        Predict future replicas based on current status and trends
        
        StatuScale Decision Logic:
        1. Analyze recent trend (increasing/decreasing/stable)
        2. Detect spikes or anomalies
        3. Calculate volatility for confidence
        4. Apply rule-based scaling decision
        """
        try:
            if len(data) < self.window_size:
                logger.warning(f"StatuScale: Need {self.window_size} data points, got {len(data)}")
                return None
            
            df = pd.DataFrame(data)
            
            # Get recent window
            recent_data = df.tail(self.window_size)
            current = df.tail(1).iloc[0]
            
            current_cpu = current['cpu_utilization']
            current_traffic = current['traffic']
            current_replicas = current['replicas']
            
            # Extract recent metrics
            recent_cpu = recent_data['cpu_utilization'].values
            recent_traffic = recent_data['traffic'].values
            recent_replicas = recent_data['replicas'].values
            
            # Analyze trends
            cpu_trend_dir, cpu_trend_slope = self.calculate_trend(recent_cpu)
            traffic_trend_dir, traffic_trend_slope = self.calculate_trend(recent_traffic)
            
            # Detect spikes
            cpu_spike = self.detect_spike(recent_cpu)
            traffic_spike = self.detect_spike(recent_traffic)
            
            # Calculate volatility
            cpu_volatility = self.calculate_volatility(recent_cpu)
            
            # Decision logic
            predicted_replicas = current_replicas
            
            # Rule 1: Spike detection -> immediate scale up
            if cpu_spike or traffic_spike:
                predicted_replicas = min(10, current_replicas + 2)
                logger.debug(f"StatuScale: Spike detected, scaling to {predicted_replicas}")
            
            # Rule 2: High CPU with upward trend -> scale up
            elif current_cpu > self.adaptive_high and cpu_trend_dir > 0:
                scale_factor = 1 if cpu_volatility < 0.2 else 2
                predicted_replicas = min(10, current_replicas + scale_factor)
                logger.debug(f"StatuScale: High CPU + upward trend, scaling to {predicted_replicas}")
            
            # Rule 3: Low CPU with downward trend -> scale down
            elif current_cpu < self.adaptive_low and cpu_trend_dir < 0:
                predicted_replicas = max(1, current_replicas - 1)
                logger.debug(f"StatuScale: Low CPU + downward trend, scaling to {predicted_replicas}")
            
            # Rule 4: Moderate CPU but strong upward trend -> proactive scale up
            elif current_cpu > 50 and cpu_trend_slope > 2 and traffic_trend_dir > 0:
                predicted_replicas = min(10, current_replicas + 1)
                logger.debug(f"StatuScale: Strong upward trend, proactive scale to {predicted_replicas}")
            
            # Rule 5: Stable low load -> gradual scale down
            elif current_cpu < 40 and cpu_volatility < 0.1:
                predicted_replicas = max(1, current_replicas - 1)
                logger.debug(f"StatuScale: Stable low load, scaling down to {predicted_replicas}")
            
            # Rule 6: Maintain current state
            else:
                logger.debug(f"StatuScale: Maintaining {predicted_replicas} replicas")
            
            return int(predicted_replicas)
            
        except Exception as e:
            logger.error(f"StatuScale prediction error: {e}")
            return None
    
    def save(self, filepath):
        """StatuScale has no model to save (rule-based)"""
        logger.info("StatuScale is rule-based, no model file to save")
        pass
    
    def load(self, filepath):
        """StatuScale has no model to load"""
        logger.info("StatuScale is rule-based, no model file to load")
        return True
    
    def get_status(self):
        """Get current StatuScale configuration"""
        return {
            'algorithm': 'StatuScale',
            'type': 'rule-based',
            'threshold_high': self.threshold_high,
            'threshold_low': self.threshold_low,
            'adaptive_high': self.adaptive_high,
            'adaptive_low': self.adaptive_low,
            'window_size': self.window_size,
            'is_trained': self.is_trained
        }
