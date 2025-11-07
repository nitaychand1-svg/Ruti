"""
Risk management components for ACTS v6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ImportanceSampler:
    def __init__(self, target_dist: np.ndarray, proposal_dist: np.ndarray) -> None:
        if target_dist.shape != proposal_dist.shape:
            raise ValueError("target and proposal distributions must match")
        if np.any(proposal_dist <= 0):
            raise ValueError("proposal distribution must be strictly positive")
        self.target_dist = target_dist / np.sum(target_dist)
        self.proposal_dist = proposal_dist / np.sum(proposal_dist)
        self.weights = self.target_dist / self.proposal_dist
        self.weights /= np.sum(self.weights)

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        samples = np.random.choice(len(self.proposal_dist), size=n_samples, p=self.proposal_dist)
        weights = self.weights[samples]
        return samples, weights

    def estimate_expectation(self, values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        weighted_sum = np.sum(values * weights)
        norm = np.sum(weights)
        mean = weighted_sum / (norm + 1e-8)
        ess = (np.sum(weights) ** 2) / (np.sum(weights ** 2) + 1e-8)
        variance = np.sum(weights * (values - mean) ** 2) / (norm + 1e-8)
        std_error = float(np.sqrt(variance / (ess + 1e-8)))
        return float(mean), std_error


@dataclass
class ScenarioOutcome:
    scenario: str
    expected_loss: float
    std_error: float
    var_95: float
    survival_probability: float
    recommended_hedges: List[Dict[str, float]]


class ExistentialRiskSimulator:
    def __init__(self) -> None:
        self.scenarios = self._default_scenarios()
        target_probs = np.array([cfg["probability"] for cfg in self.scenarios.values()], dtype=float)
        proposal_probs = np.power(target_probs, 0.7)
        proposal_probs /= np.sum(proposal_probs)
        self.importance_sampler = ImportanceSampler(target_probs, proposal_probs)

    def simulate(self, scenario_name: str, portfolio: Dict[str, float], n_samples: int = 10_000) -> ScenarioOutcome:
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario {scenario_name}")
        scenario = self.scenarios[scenario_name]

        base_loss = sum(portfolio.get(asset, 0.0) * impact for asset, impact in scenario["impact"].items())
        sample_indices, sample_weights = self.importance_sampler.sample(n_samples)

        simulated_losses = []
        for idx in sample_indices:
            sampled_scenario = list(self.scenarios.values())[idx]
            impact = sum(portfolio.get(asset, 0.0) * sampled_scenario["impact"].get(asset, 0.0) for asset in portfolio)
            noisy = impact * (1 + np.random.normal(0, 0.1))
            simulated_losses.append(noisy)
        simulated_losses = np.asarray(simulated_losses)

        expected_loss, std_error = self.importance_sampler.estimate_expectation(simulated_losses, sample_weights)
        var_95 = float(np.percentile(simulated_losses, 5))
        survival_prob = float(np.mean(simulated_losses > -0.5 * sum(portfolio.values())))
        hedges = self._suggest_hedges(scenario, portfolio)

        return ScenarioOutcome(
            scenario=scenario_name,
            expected_loss=expected_loss,
            std_error=std_error,
            var_95=var_95,
            survival_probability=survival_prob,
            recommended_hedges=hedges,
        )

    def _suggest_hedges(self, scenario: Dict[str, any], portfolio: Dict[str, float]) -> List[Dict[str, float]]:
        total_exposure = sum(abs(v) for v in portfolio.values()) + 1e-8
        hedges = []
        for asset, impact in scenario["impact"].items():
            if impact > 0:
                hedges.append({"asset": asset, "size": 0.1 * total_exposure, "reason": "Positive carry under scenario"})
        return hedges

    def _default_scenarios(self) -> Dict[str, Dict[str, any]]:
        return {
            "solar_flare": {
                "description": "Carrington-class solar flare",
                "probability": 0.001,
                "impact": {"equities": -0.70, "crypto": -0.90, "gold": 0.30},
            },
            "cyber_attack": {
                "description": "Coordinated exchange attack",
                "probability": 0.01,
                "impact": {"equities": -0.40, "crypto": -0.60},
            },
            "quantum_break": {
                "description": "Quantum computer breaks cryptography",
                "probability": 0.005,
                "impact": {"crypto": -0.99, "tech": -0.50},
            },
            "agi_disruption": {
                "description": "AGI disrupts labour markets",
                "probability": 0.02,
                "impact": {"tech": 0.50, "traditional": -0.40},
            },
            "hyperinflation": {
                "description": "USD hyperinflation event",
                "probability": 0.001,
                "impact": {"usd": -0.80, "gold": 1.50, "crypto": 0.80},
            },
        }
