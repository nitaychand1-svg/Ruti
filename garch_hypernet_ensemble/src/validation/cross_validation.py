"""Time series cross-validation utilities."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import TimeSeriesSplit


@dataclass
class WalkForwardValidator:
    """[FIX-14] Walk-forward time series validation."""

    n_splits: int = 5
    purge_gap: int = 10

    def split(self, X, y=None):
        """Generate train/validation splits."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return tscv.split(X)

    def get_oos_holdout(self, n_samples: int):
        """Get out-of-sample holdout indices."""
        holdout_size = max(1, n_samples // 5)
        return slice(-holdout_size, None)
