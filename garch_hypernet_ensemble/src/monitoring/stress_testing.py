"""[FIX-2] Monte Carlo stress testing."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np


class StressTestingModule:
    """Comprehensive stress testing."""

    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.logger = logging.getLogger(__name__)

    async def run_monte_carlo_stress(
        self, X: np.ndarray, y: np.ndarray, n_scenarios: int = 1000
    ) -> Dict[str, Any]:
        """[FIX-2] Monte Carlo with GARCH simulations."""
        results: Dict[str, Any] = {
            "correlation_spikes": [],
            "diversity_scores": [],
            "accuracy_degradation": [],
            "worst_case_diversity": 1.0,
            "garch_spikes": [],
            "effective_bets": [],
            "black_swan_scenarios": [],
        }

        scenarios = self._generate_stress_scenarios(X, y)

        for X_stress, y_stress in scenarios[: min(100, len(scenarios))]:
            pred_result = await self.ensemble.predict(
                X_stress, market_regime="high_volatility"
            )

            results["correlation_spikes"].append(1.0 - pred_result.get("diversity", 0.0))
            results["diversity_scores"].append(pred_result.get("diversity", 0.0))
            results["garch_spikes"].append(
                self.ensemble.predictor._get_current_garch_volatility()
                if self.ensemble.predictor
                else 0.0
            )

            if "effective_bets" in pred_result:
                results["effective_bets"].append(pred_result["effective_bets"])

            if pred_result.get("diversity", 1.0) < results["worst_case_diversity"]:
                results["worst_case_diversity"] = pred_result["diversity"]

        black_swan_results = await self._run_black_swan_scenarios()
        results["black_swan_scenarios"].extend(black_swan_results)

        results["mean_correlation"] = float(np.mean(results["correlation_spikes"]))
        results["diversity_at_risk"] = float(np.percentile(results["diversity_scores"], 5))
        results["mean_garch_vol"] = float(np.mean(results["garch_spikes"]))
        results["black_swan_diversity"] = float(np.min(results["black_swan_scenarios"]))

        min_diversity = self.ensemble.config.monitoring.min_diversity_threshold
        if results["diversity_at_risk"] < min_diversity:
            self.logger.critical(
                "❌ Stress test failed: diversity at risk = %.3f",
                results["diversity_at_risk"],
            )

        return results

    def _generate_stress_scenarios(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic stress scenarios."""
        scenarios: List[Tuple[np.ndarray, np.ndarray]] = []
        n_scenarios = 50

        for _ in range(n_scenarios):
            shock_mask = np.random.random(len(X)) < 0.3
            X_stress = X.copy()
            shock_factor = np.random.normal(3.0, 1.0, size=X[shock_mask].shape)
            X_stress[shock_mask] *= shock_factor

            y_stress = (np.diff(X_stress[:, 0]) > 0).astype(int)
            y_stress = np.pad(y_stress, (1, 0), mode="constant")

            scenarios.append((X_stress, y_stress))

        return scenarios

    async def _run_black_swan_scenarios(self) -> List[float]:
        """[FIX-2] Extreme black swan scenarios."""
        results: List[float] = []

        for _ in range(100):
            X_swan = self._generate_black_swan_data(500, spike_factor=15.0)
            pred_swan = await self.ensemble.predict(X_swan, market_regime="chaotic")
            results.append(pred_swan.get("diversity", 0.0))

        return results

    def _generate_black_swan_data(self, n_samples: int, spike_factor: float) -> np.ndarray:
        """Generate extreme scenario data."""
        base = np.random.randn(n_samples, 20)
        shock = (
            np.random.normal(0, spike_factor, base.shape)
            * np.random.binomial(1, 0.3, base.shape)
        )
        return base + shock
