"""Time series cross-validation utilities."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class WalkForwardValidator:
    """[FIX-14] Walk-forward time series validation."""
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 10):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X, y=None):
        """Generate train/validation splits."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return tscv.split(X)
    
    def get_oos_holdout(self, n_samples: int):
        """Get out-of-sample holdout indices."""
        holdout_size = n_samples // 5  # 20% OOS
        return slice(-holdout_size, None)
