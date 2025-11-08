"""Multi-feature GARCH tracker."""
import asyncio
import time
import logging
from collections import defaultdict, deque
from typing import Dict, Optional
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from .model import GARCH11

logger = logging.getLogger(__name__)

class GARCHModelTracker:
    """[FIX-11] Online GARCH tracker for multiple features."""
    
    def __init__(self, window: int = 252, min_obs: int = 100):
        self.window = window
        self.min_obs = min_obs
        self.models: Dict[str, GARCH11] = {}
        self.returns_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self.last_fit_timestamp: Dict[str, float] = defaultdict(float)
        self.fit_interval = 3600  # 1 hour
        self.volatility_history = deque(maxlen=100)
        self.logger = logger
    
    def add_feature(self, name: str, returns_series: np.ndarray):
        """Add feature for GARCH tracking."""
        self.returns_buffer[name].extend(returns_series)
        self.models[name] = None
    
    def update_returns(self, name: str, new_return: float):
        """[FIX-11] Online update."""
        if not np.isfinite(new_return):
            self.logger.warning(f"Invalid return {new_return} for {name}")
            return
        
        self.returns_buffer[name].append(new_return)
        
        # Async refit if interval passed
        current_time = time.time()
        if current_time - self.last_fit_timestamp[name] > self.fit_interval:
            asyncio.create_task(self._async_refit_garch(name))
            self.last_fit_timestamp[name] = current_time
    
    async def _async_refit_garch(self, name: str):
        """Background refit."""
        try:
            returns = np.array(list(self.returns_buffer[name])[-1000:])
            if len(returns) < self.min_obs:
                return
            
            model = GARCH11(returns)
            self.models[name] = model.fit()
            
            self.logger.info(
                f"[FIX-11] Refit GARCH {name}: alpha={model.params[1]:.4f}"
            )
        except Exception as e:
            self.logger.error(f"[FIX-11] Refit failed {name}: {e}")
    
    def fit_garch(self, name: str) -> Optional[float]:
        """Fit GARCH model for feature."""
        if name not in self.returns_buffer:
            return None
        
        returns = np.array(list(self.returns_buffer[name]))
        if len(returns) < self.min_obs:
            return np.std(returns)
        
        try:
            model = GARCH11(returns)
            res = model.fit(maxiter=1000, disp=False)
            self.models[name] = res
            return res.sigma2[-1] if res.sigma2 is not None else np.var(returns)
        except Exception as e:
            self.logger.error(f"[FIX-12] GARCH fit error: {e}")
            return np.var(returns)
    
    def fit_all_garch_parallel(self, max_workers: int = None):
        """[FIX-14] Parallel GARCH fitting."""
        if max_workers is None:
            max_workers = min(4, cpu_count() // 2)
        
        self.logger.info(f"[FIX-14] Parallel GARCH fit: {max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(self._fit_worker, name): name
                for name in self.returns_buffer.keys()
            }
            
            for future in executor.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    model = future.result(timeout=30)
                    self.models[name] = model
                    self.logger.info(f"[FIX-14] Fitted {name}: success")
                except Exception as e:
                    self.logger.error(f"[FIX-14] Fit failed {name}: {e}")
    
    def _fit_worker(self, name: str) -> GARCH11:
        """Worker function for parallel fitting."""
        returns = np.array(list(self.returns_buffer[name]))
        if len(returns) < self.min_obs:
            raise ValueError(f"Insufficient data for {name}")
        
        model = GARCH11(returns)
        return model.fit()
    
    def get_all_volatilities(self) -> float:
        """[FIX-6] Weighted volatility with EWMA smoothing."""
        vols_dict = {}
        
        for name in self.returns_buffer.keys():
            vol = (self.models[name].forecast_vol(1) 
                   if name in self.models and self.models[name] 
                   else self.fit_garch(name))
            vols_dict[name] = vol
        
        # [FIX-6] Feature weights
        feature_weights = {
            'price': 0.4, 'returns': 0.3, 'volume': 0.1,
            'rsi': 0.1, 'macd': 0.1, 'default': 0.05
        }
        
        weighted_vols = []
        for name, vol in vols_dict.items():
            if vol is None or not np.isfinite(vol):
                continue
            
            prefix = name.split('_')[0]
            w = feature_weights.get(prefix, feature_weights['default'])
            weighted_vols.append(vol * w)
        
        if not weighted_vols:
            return 0.01
        
        # [FIX-6] EWMA temporal smoothing
        mean_weighted = np.mean(weighted_vols)
        self.volatility_history.append(mean_weighted)
        
        if len(self.volatility_history) > 1:
            ewma_vol = pd.Series(self.volatility_history).ewm(alpha=0.05).mean().iloc[-1]
        else:
            ewma_vol = mean_weighted
        
        return ewma_vol
    
    def forecast_volatility(self, name: str, steps: int = 5) -> Optional[float]:
        """Forecast volatility for feature."""
        if name not in self.models or self.models[name] is None:
            return self.fit_garch(name)
        
        return self.models[name].forecast_vol(steps)
