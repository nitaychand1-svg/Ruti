"""
Federated learning coordinator for ACTS v6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParticipantUpdate:
    participant_id: str
    gradients: np.ndarray
    samples_seen: int


class FederatedTrainingCoordinator:
    def __init__(self, n_participants: int = 5, noise_scale: float = 0.1) -> None:
        self.n_participants = n_participants
        self.noise_scale = noise_scale
        self.privacy_budget = 10.0

    async def training_round(self) -> Dict[str, any]:
        updates = [self._simulate_participant(i) for i in range(self.n_participants)]
        aggregated = self._fed_avg(updates)
        private_gradients = aggregated + np.random.laplace(0, self.noise_scale, size=aggregated.shape)
        self.privacy_budget -= 1.0
        return {
            "aggregated_update": private_gradients,
            "participants": updates,
            "privacy_budget_remaining": self.privacy_budget,
        }

    def _simulate_participant(self, idx: int) -> ParticipantUpdate:
        rng = np.random.default_rng(idx)
        gradients = rng.normal(0, 1, size=128)
        samples_seen = rng.integers(500, 5_000)
        return ParticipantUpdate(participant_id=f"client_{idx}", gradients=gradients, samples_seen=int(samples_seen))

    def _fed_avg(self, updates: List[ParticipantUpdate]) -> np.ndarray:
        total_samples = sum(update.samples_seen for update in updates)
        weighted = sum(update.gradients * (update.samples_seen / total_samples) for update in updates)
        return weighted
