"""[FIX-12] Comprehensive data validation."""
from __future__ import annotations

from typing import Optional

import numpy as np


class DataValidationLayer:
    """Centralized data validation with context awareness."""

    @staticmethod
    def validate(
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        context: str = "unknown",
    ) -> np.ndarray:
        """Validate input data."""
        if X.ndim not in {1, 2}:
            raise ValueError(f"[FIX-12] Invalid X dimensions: {X.ndim}")

        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if X.ndim == 2:
            stds = np.std(X, axis=0)
            constant_mask = stds < 1e-8
            if np.any(constant_mask):
                noise = np.random.normal(0, 1e-6, (X.shape[0], constant_mask.sum()))
                X[:, constant_mask] += noise

        X = np.clip(X, -1e6, 1e6)

        if y is not None:
            if len(y) != len(X):
                raise ValueError("[FIX-12] X and y shape mismatch")

            y = np.nan_to_num(y, nan=0)
            y = np.clip(y, 0, 1).astype(int)

        return X

    @staticmethod
    def validate_returns(returns: np.ndarray, context: str = "garch") -> np.ndarray:
        """Special validation for returns series."""
        returns = np.nan_to_num(returns, 0.0)
        return np.clip(returns, -10.0, 10.0)
