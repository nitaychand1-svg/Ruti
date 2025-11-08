"""Multi-feature GARCH tracker."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .model import GARCH11

logger = logging.getLogger(__name__)


class GARCHModelTracker:
    """[FIX-11] Online GARCH tracker for multiple features."""

    def __init__(self, window: int = 252, min_obs: int = 100):
        self.window = window
        self.min_obs = min_obs
        self.models: Dict[str, Optional[GARCH11]] = {}
        self.returns_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.last_fit_timestamp: Dict[str, float] = defaultdict(float)
        self.fit_interval = 3600  # 1 hour
        self.volatility_history = deque(maxlen=100)
        self.logger = logger

    def add_feature(self, name: str, returns_series: np.ndarray) -> None:
        """Add feature for GARCH tracking."""
        validated = np.asarray(returns_series, dtype=float)
        self.returns_buffer[name].extend(validated)
        self.models[name] = None

    def update_returns(self, name: str, new_return: float) -> None:
        """[FIX-11] Online update."""
        if not np.isfinite(new_return):
            self.logger.warning("Invalid return %s for %s", new_return, name)
            return

        self.returns_buffer[name].append(float(new_return))

        current_time = time.time()
        if current_time - self.last_fit_timestamp[name] > self.fit_interval:
            self._schedule_async_refit(name)
            self.last_fit_timestamp[name] = current_time

    def _schedule_async_refit(self, name: str) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_refit_garch(name))
        except RuntimeError:
            asyncio.run(self._async_refit_garch(name))

    async def _async_refit_garch(self, name: str) -> None:
        """Background refit."""
        try:
            returns = np.array(list(self.returns_buffer[name])[-1000:], dtype=float)
            if len(returns) < self.min_obs:
                return

            model = GARCH11(returns)
            result = model.fit()
            self.models[name] = result

            self.logger.info("[FIX-11] Refit GARCH %s: alpha=%.4f", name, result.params[1])
        except Exception as exc:  # pragma: no cover - logged for observability
            self.logger.error("[FIX-11] Refit failed %s: %s", name, exc)

    def fit_garch(self, name: str) -> Optional[float]:
        """Fit GARCH model for feature."""
        if name not in self.returns_buffer:
            return None

        returns = np.array(list(self.returns_buffer[name]), dtype=float)
        if len(returns) < self.min_obs:
            return float(np.std(returns)) if returns.size else None

        try:
            model = GARCH11(returns)
            res = model.fit(maxiter=1000, disp=False)
            self.models[name] = res
            if res.sigma2 is not None:
                return float(res.sigma2[-1])
            return float(np.var(returns))
        except Exception as exc:  # pragma: no cover
            self.logger.error("[FIX-12] GARCH fit error: %s", exc)
            return float(np.var(returns))

    def fit_all_garch_parallel(self, max_workers: Optional[int] = None) -> None:
        """[FIX-14] Parallel GARCH fitting."""
        if max_workers is None:
            max_workers = max(1, min(4, cpu_count() // 2))

        self.logger.info("[FIX-14] Parallel GARCH fit: %d workers", max_workers)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(self._fit_worker, name): name
                for name in self.returns_buffer.keys()
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    model = future.result(timeout=60)
                    self.models[name] = model
                    self.logger.info("[FIX-14] Fitted %s: success", name)
                except Exception as exc:  # pragma: no cover
                    self.logger.error("[FIX-14] Fit failed %s: %s", name, exc)

    def _fit_worker(self, name: str) -> GARCH11:
        """Worker function for parallel fitting."""
        returns = np.array(list(self.returns_buffer[name]), dtype=float)
        if len(returns) < self.min_obs:
            raise ValueError(f"Insufficient data for {name}")

        model = GARCH11(returns)
        return model.fit()

    def get_all_volatilities(self) -> float:
        """[FIX-6] Weighted volatility with EWMA smoothing."""
        vols_dict: Dict[str, float] = {}

        for name in self.returns_buffer.keys():
            if name in self.models and self.models[name] is not None:
                try:
                    vol = float(self.models[name].forecast_vol(1))
                except Exception:  # pragma: no cover
                    vol = None
            else:
                vol = self.fit_garch(name)
            if vol is not None and np.isfinite(vol):
                vols_dict[name] = vol

        feature_weights = {
            "price": 0.4,
            "returns": 0.3,
            "volume": 0.1,
            "rsi": 0.1,
            "macd": 0.1,
            "default": 0.05,
        }

        weighted_vols = []
        for name, vol in vols_dict.items():
            prefix = name.split("_")[0]
            weight = feature_weights.get(prefix, feature_weights["default"])
            weighted_vols.append(vol * weight)

        if not weighted_vols:
            return 0.01

        mean_weighted = float(np.mean(weighted_vols))
        self.volatility_history.append(mean_weighted)

        series = pd.Series(self.volatility_history)
        ewma_vol = float(series.ewm(alpha=0.05).mean().iloc[-1])
        return ewma_vol

    def forecast_volatility(self, name: str, steps: int = 5) -> Optional[float]:
        """Forecast volatility for feature."""
        if name not in self.models or self.models[name] is None:
            return self.fit_garch(name)

        try:
            return float(self.models[name].forecast_vol(steps))
        except Exception:  # pragma: no cover
            return self.fit_garch(name)
