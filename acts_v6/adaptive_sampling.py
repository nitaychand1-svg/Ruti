"""
Adaptive Monte Carlo sampler for ACTS v6.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict


@dataclass
class SamplingDecision:
    entropy: float
    budget_ms: float
    n_samples: int


class AdaptiveMCSampler:
    def __init__(self, min_samples: int = 1_000, max_samples: int = 50_000) -> None:
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.history: Deque[SamplingDecision] = deque(maxlen=128)

    def determine_n_samples(self, entropy: float, budget_ms: float = 500.0) -> int:
        if entropy <= 0.05:
            requested = self.min_samples
        elif entropy >= 0.15:
            requested = self.max_samples
        else:
            ratio = (entropy - 0.05) / 0.10
            requested = int(self.min_samples + ratio * (self.max_samples - self.min_samples))

        budget_cap = int(budget_ms / 0.01)
        n_samples = max(self.min_samples, min(requested, budget_cap, self.max_samples))

        self.history.append(SamplingDecision(entropy=entropy, budget_ms=budget_ms, n_samples=n_samples))
        return n_samples
