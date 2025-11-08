"""[FIX-12] Comprehensive data validation."""
import numpy as np
import pandas as pd
from typing import Optional

class DataValidationLayer:
    """Centralized data validation with context awareness."""
    
    @staticmethod
    def validate(X: np.ndarray, 
                y: Optional[np.ndarray] = None,
                context: str = "unknown") -> np.ndarray:
        """Validate input data."""
        # Check shapes
        if X.ndim not in [1, 2]:
            raise ValueError(f"[FIX-12] Invalid X dimensions: {X.ndim}")
        
        # Check for NaN/Inf
        if not np.isfinite(X).all():
            # Replace invalid values
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Check for constant features
        if X.ndim == 2:
            stds = np.std(X, axis=0)
            constant_mask = stds < 1e-8
            if constant_mask.any():
                # Add small noise to constant features
                X[:, constant_mask] += np.random.normal(0, 1e-6, 
                                                       (X.shape[0], 
                                                        constant_mask.sum()))
        
        # Clip extreme values
        X = np.clip(X, -1e6, 1e6)
        
        # Validate labels if provided
        if y is not None:
            if len(y) != len(X):
                raise ValueError("[FIX-12] X and y shape mismatch")
            
            y = np.nan_to_num(y, nan=0)
            y = np.clip(y, 0, 1).astype(int)
        
        return X
    
    @staticmethod
    def validate_returns(returns: np.ndarray, 
                        context: str = "garch") -> np.ndarray:
        """Special validation for returns series."""
        returns = np.nan_to_num(returns, 0.0)
        returns = np.clip(returns, -10.0, 10.0)  # Clip extreme returns
        return returns
