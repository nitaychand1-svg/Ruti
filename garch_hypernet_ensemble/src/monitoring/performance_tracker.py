"""Online performance monitoring."""
import numpy as np
from collections import deque
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class OnlinePerformanceTracker:
    """[FIX-5,16,19] Real-time performance tracking."""
    
    def __init__(self, config):
        self.config = config
        self.window_size = config.window_size
        
        # Core metrics
        self.accuracy_window = deque(maxlen=self.window_size)
        self.diversity_window = deque(maxlen=self.window_size)
        self.correlation_window = deque(maxlen=self.window_size)
        
        # Risk metrics
        self.cvar_window = deque(maxlen=self.window_size)
        self.volatility_window = deque(maxlen=20)
        
        # [FIX-19] Effective bets
        self.effective_bets_window = deque(maxlen=self.window_size)
        
        # [FIX-5] Live vs backtest gap
        self.live_vs_backtest_gap = deque(maxlen=self.window_size)
        self.backtest_score = 0.55
        
        # Regime tracking
        self.regime_trackers = {
            'low_volatility_bull': {
                'accuracy': deque(maxlen=self.window_size // 2),
                'correlation': deque(maxlen=self.window_size // 2)
            },
            'high_volatility': {
                'accuracy': deque(maxlen=self.window_size // 2),
                'correlation': deque(maxlen=self.window_size // 2)
            },
            'chaotic': {
                'accuracy': deque(maxlen=self.window_size // 4),
                'correlation': deque(maxlen=self.window_size // 4)
            }
        }
        
        # [FIX-5] Dynamic thresholds
        self.dynamic_thresholds = {
            'max_correlation': config.max_correlation_threshold,
            'min_confidence': config.min_confidence_threshold,
            'min_diversity': config.min_diversity_threshold
        }
    
    def update_metrics(self, prediction: np.ndarray, 
                      ground_truth: int,
                      regime: str,
                      correlation_matrix: np.ndarray):
        """[FIX-5] Update all tracking metrics."""
        is_correct = int(np.argmax(prediction) == ground_truth)
        self.accuracy_window.append(is_correct)
        
        # Diversity from correlation
        if correlation_matrix.size > 1:
            avg_corr = np.mean(np.abs(correlation_matrix))
            self.correlation_window.append(avg_corr)
            diversity = 1.0 - avg_corr
            self.diversity_window.append(diversity)
        else:
            diversity = 0.0
        
        # [FIX-16] CVaR tracking
        tail_loss = np.percentile(prediction, 5)
        self.cvar_window.append(tail_loss)
        
        # Regime-specific tracking
        if regime in self.regime_trackers:
            self.regime_trackers[regime]['accuracy'].append(is_correct)
            if correlation_matrix.size > 1:
                self.regime_trackers[regime]['correlation'].append(avg_corr)
        
        # Volatility proxy
        vol_proxy = np.std(prediction) if len(prediction) > 1 else 0.0
        self.volatility_window.append(vol_proxy)
        
        # [FIX-5] Gap monitoring
        if hasattr(self, 'backtest_score'):
            gap = abs(np.mean(self.accuracy_window) - self.backtest_score)
            self.live_vs_backtest_gap.append(gap)
        
        # [FIX-19] Effective bets
        if hasattr(prediction, 'weights'):
            effective = self._calculate_effective_bets(
                prediction.weights, correlation_matrix
            )
            self.effective_bets_window.append(effective)
        
        # [FIX-5] Auto-tune thresholds
        self._auto_tune_thresholds(regime, avg_corr, diversity)
    
    def _auto_tune_thresholds(self, regime: str, 
                             current_corr: float, 
                             current_diversity: float):
        """[FIX-5] Adaptive thresholds by regime."""
        alpha = self.config.adaptation_rate
        
        if regime == 'high_volatility':
            if current_diversity < 0.25:
                self.dynamic_thresholds['max_correlation'] *= (1 - alpha * 2)
                self.dynamic_thresholds['min_confidence'] *= (1 + alpha)
            
            self.dynamic_thresholds['max_correlation'] = max(
                0.60, self.dynamic_thresholds['max_correlation']
            )
            self.dynamic_thresholds['min_confidence'] = min(
                0.70, self.dynamic_thresholds['min_confidence']
            )
        
        elif regime == 'chaotic':
            self.dynamic_thresholds['max_correlation'] = 0.50
            self.dynamic_thresholds['min_confidence'] = 0.65
        
        elif regime == 'low_volatility_bull':
            self.dynamic_thresholds['max_correlation'] = min(
                0.85,
                self.dynamic_thresholds['max_correlation'] * (1 + alpha * 0.5)
            )
            self.dynamic_thresholds['min_confidence'] = max(
                0.52,
                self.dynamic_thresholds['min_confidence'] * (1 - alpha * 0.3)
            )
        
        target_diversity = 0.15 if regime in ['high_volatility', 'chaotic'] else 0.30
        self.dynamic_thresholds['min_diversity'] = (
            (1 - alpha) * self.dynamic_thresholds['min_diversity'] + 
            alpha * target_diversity
        )
    
    def should_trigger_retraining(self) -> bool:
        """[FIX-5] Check if retraining needed."""
        if len(self.diversity_window) < 10:
            return False
        
        avg_diversity = np.mean(self.diversity_window)
        avg_vol = np.mean(self.volatility_window) if self.volatility_window else 0.0
        
        # [FIX-5] Gap-based trigger
        if len(self.live_vs_backtest_gap) > 10:
            gap = np.mean(self.live_vs_backtest_gap)
            if gap > 0.10:
                logger.critical(f"🚨 Large gap: {gap:.3f}")
                return True
        
        # [FIX-19] Low effective bets
        if len(self.effective_bets_window) > 10:
            bets = np.mean(self.effective_bets_window)
            if bets < 3.0:
                logger.critical(f"🚨 Low effective bets: {bets:.2f}")
                return True
        
        # High volatility + low diversity
        if (avg_diversity < self.dynamic_thresholds['min_diversity'] and 
            avg_vol > 0.02):
            logger.critical(
                f"🚨 Retrain trigger: diversity={avg_diversity:.3f}, vol={avg_vol:.4f}"
            )
            return True
        
        return False
    
    def _calculate_effective_bets(self, weights: np.ndarray,
                                corr_matrix: np.ndarray) -> float:
        """[FIX-19] Effective number of independent bets."""
        div_ratio = np.sqrt(np.dot(weights.T, np.dot(corr_matrix, weights)))
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        return np.exp(entropy) / (div_ratio + 1e-6)
