"""Online performance monitoring."""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class OnlinePerformanceTracker:
    """[FIX-5,16,19] Real-time performance tracking."""

    def __init__(self, config):
        self.config = config
        self.window_size = config.window_size

        self.accuracy_window: Deque[int] = deque(maxlen=self.window_size)
        self.diversity_window: Deque[float] = deque(maxlen=self.window_size)
        self.correlation_window: Deque[float] = deque(maxlen=self.window_size)

        self.cvar_window: Deque[float] = deque(maxlen=self.window_size)
        self.volatility_window: Deque[float] = deque(maxlen=20)

        self.effective_bets_window: Deque[float] = deque(maxlen=self.window_size)

        self.live_vs_backtest_gap: Deque[float] = deque(maxlen=self.window_size)
        self.backtest_score = 0.55

        self.regime_trackers: Dict[str, Dict[str, Deque[float]]] = {
            "low_volatility_bull": {
                "accuracy": deque(maxlen=self.window_size // 2),
                "correlation": deque(maxlen=self.window_size // 2),
            },
            "high_volatility": {
                "accuracy": deque(maxlen=self.window_size // 2),
                "correlation": deque(maxlen=self.window_size // 2),
            },
            "chaotic": {
                "accuracy": deque(maxlen=self.window_size // 4),
                "correlation": deque(maxlen=self.window_size // 4),
            },
        }

        self.dynamic_thresholds = {
            "max_correlation": config.max_correlation_threshold,
            "min_confidence": config.min_confidence_threshold,
            "min_diversity": config.min_diversity_threshold,
        }

    def update_metrics(
        self,
        prediction: np.ndarray,
        ground_truth: int,
        regime: str,
        correlation_matrix: np.ndarray,
    ) -> None:
        """[FIX-5] Update all tracking metrics."""
        is_correct = int(np.argmax(prediction) == ground_truth)
        self.accuracy_window.append(is_correct)

        avg_corr = 0.0
        diversity = 0.0
        if correlation_matrix.size > 1:
            avg_corr = float(np.mean(np.abs(correlation_matrix)))
            self.correlation_window.append(avg_corr)
            diversity = float(1.0 - avg_corr)
            self.diversity_window.append(diversity)

        tail_loss = float(np.percentile(prediction, 5))
        self.cvar_window.append(tail_loss)

        if regime in self.regime_trackers:
            self.regime_trackers[regime]["accuracy"].append(is_correct)
            if correlation_matrix.size > 1:
                self.regime_trackers[regime]["correlation"].append(avg_corr)

        vol_proxy = float(np.std(prediction)) if prediction.size > 1 else 0.0
        self.volatility_window.append(vol_proxy)

        if self.accuracy_window:
            gap = abs(np.mean(self.accuracy_window) - self.backtest_score)
            self.live_vs_backtest_gap.append(float(gap))

        self._auto_tune_thresholds(regime, avg_corr, diversity)

    def _auto_tune_thresholds(
        self,
        regime: str,
        current_corr: float,
        current_diversity: float,
    ) -> None:
        """[FIX-5] Adaptive thresholds by regime."""
        alpha = self.config.adaptation_rate

        if regime == "high_volatility":
            if current_diversity < 0.25:
                self.dynamic_thresholds["max_correlation"] *= 1 - alpha * 2
                self.dynamic_thresholds["min_confidence"] *= 1 + alpha

            self.dynamic_thresholds["max_correlation"] = max(
                0.60, self.dynamic_thresholds["max_correlation"]
            )
            self.dynamic_thresholds["min_confidence"] = min(
                0.70, self.dynamic_thresholds["min_confidence"]
            )

        elif regime == "chaotic":
            self.dynamic_thresholds["max_correlation"] = 0.50
            self.dynamic_thresholds["min_confidence"] = 0.65

        elif regime == "low_volatility_bull":
            self.dynamic_thresholds["max_correlation"] = min(
                0.85,
                self.dynamic_thresholds["max_correlation"] * (1 + alpha * 0.5),
            )
            self.dynamic_thresholds["min_confidence"] = max(
                0.52,
                self.dynamic_thresholds["min_confidence"] * (1 - alpha * 0.3),
            )

        target_diversity = 0.15 if regime in {"high_volatility", "chaotic"} else 0.30
        self.dynamic_thresholds["min_diversity"] = (
            (1 - alpha) * self.dynamic_thresholds["min_diversity"] + alpha * target_diversity
        )

    def should_trigger_retraining(self) -> bool:
        """[FIX-5] Check if retraining needed."""
        if len(self.diversity_window) < 10:
            return False

        avg_diversity = float(np.mean(self.diversity_window))
        avg_vol = float(np.mean(self.volatility_window)) if self.volatility_window else 0.0

        if len(self.live_vs_backtest_gap) > 10:
            gap = float(np.mean(self.live_vs_backtest_gap))
            if gap > 0.10:
                logger.critical("🚨 Large gap: %.3f", gap)
                return True

        if len(self.effective_bets_window) > 10:
            bets = float(np.mean(self.effective_bets_window))
            if bets < 3.0:
                logger.critical("🚨 Low effective bets: %.2f", bets)
                return True

        if (
            avg_diversity < self.dynamic_thresholds["min_diversity"]
            and avg_vol > 0.02
        ):
            logger.critical(
                "🚨 Retrain trigger: diversity=%.3f, vol=%.4f",
                avg_diversity,
                avg_vol,
            )
            return True

        return False

    def _calculate_effective_bets(
        self, weights: np.ndarray, corr_matrix: np.ndarray
    ) -> float:
        norm_weights = weights / (np.sum(weights) + 1e-12)
        div_ratio = float(np.sqrt(norm_weights.T @ corr_matrix @ norm_weights))
        entropy = -float(np.sum(norm_weights * np.log(norm_weights + 1e-8)))
        return float(np.exp(entropy) / (div_ratio + 1e-6))
