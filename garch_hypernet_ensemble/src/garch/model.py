"""GARCH(1,1) model implementation."""
from __future__ import annotations

import numpy as np
from statsmodels.base.model import GenericLikelihoodModel


class GARCH11(GenericLikelihoodModel):
    """Custom GARCH(1,1) model."""

    def __init__(self, endog: np.ndarray):
        self.start_params = np.array([0.01, 0.05, 0.9])
        self.sigma2 = None
        super().__init__(endog, None)

    def loglike(self, params: np.ndarray) -> float:
        """[FIX-7] Stationarity enforcement."""
        omega, alpha, beta = params

        # [FIX-12] Validate inputs
        if not np.isfinite(params).all():
            return -np.inf

        # Stationarity constraint
        if alpha + beta >= 0.98 or alpha + beta < 0.1:
            return -np.inf

        nobs = len(self.endog)
        self.sigma2 = np.ones(nobs) * np.var(self.endog)
        self.sigma2[0] = np.var(self.endog)

        for t in range(1, nobs):
            self.sigma2[t] = omega + alpha * (self.endog[t - 1] ** 2) + beta * self.sigma2[t - 1]

        logl = -0.5 * (
            np.log(2 * np.pi)
            + np.log(self.sigma2)
            + (self.endog ** 2 / (self.sigma2 + 1e-12))
        )
        finite_mask = np.isfinite(logl)
        return float(np.sum(logl[finite_mask]))

    def forecast_vol(self, steps: int = 1) -> float:
        """[FIX-11] Forecast volatility."""
        if self.sigma2 is None:
            raise ValueError("GARCH not fitted. Call .fit() first.")

        omega, alpha, beta = self.params
        current_sigma2 = self.sigma2[-1]
        forecasted = []

        for _ in range(steps):
            current_sigma2 = omega + alpha * (self.endog[-1] ** 2) + beta * current_sigma2
            forecasted.append(np.sqrt(max(current_sigma2, 1e-12)))

        return float(np.mean(forecasted))
