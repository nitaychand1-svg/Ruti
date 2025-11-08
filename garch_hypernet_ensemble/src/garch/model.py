"""GARCH(1,1) model implementation."""
import numpy as np
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

class GARCH11:
    """Custom GARCH(1,1) model."""
    
    def __init__(self, endog: np.ndarray):
        self.endog = endog
        self.params = None
        self.sigma2 = None
        self.fitted = False
    
    def fit(self, maxiter: int = 1000, disp: bool = False):
        """[FIX-7] Fit GARCH model with stationarity enforcement."""
        # Initial parameters [omega, alpha, beta]
        start_params = np.array([0.01, 0.05, 0.9])
        
        # Bounds for stationarity
        bounds = [(1e-6, 1.0), (0.01, 0.3), (0.5, 0.97)]
        
        # Constraint: alpha + beta < 0.98
        constraints = {'type': 'ineq', 'fun': lambda x: 0.98 - (x[1] + x[2])}
        
        try:
            result = minimize(
                self._negative_loglike,
                start_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'disp': disp}
            )
            
            self.params = result.x
            self.fitted = True
            
            # Calculate final sigma2
            self._calculate_sigma2(self.params)
            
            logger.info(f"GARCH fitted: omega={self.params[0]:.6f}, alpha={self.params[1]:.4f}, beta={self.params[2]:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"[FIX-12] GARCH fit error: {e}")
            # Fallback to simple variance
            self.params = np.array([np.var(self.endog), 0.05, 0.9])
            self.sigma2 = np.ones(len(self.endog)) * np.var(self.endog)
            self.fitted = False
            return self
    
    def _negative_loglike(self, params: np.ndarray) -> float:
        """Negative log-likelihood for minimization."""
        omega, alpha, beta = params
        
        # [FIX-12] Validate inputs
        if not np.isfinite(params).all():
            return 1e10
        
        # Stationarity constraint
        if alpha + beta >= 0.98 or alpha + beta < 0.1:
            return 1e10
        
        nobs = len(self.endog)
        sigma2 = np.ones(nobs) * np.var(self.endog)
        sigma2[0] = np.var(self.endog)
        
        for t in range(1, nobs):
            sigma2[t] = omega + alpha * (self.endog[t-1]**2) + beta * sigma2[t-1]
            
            # Prevent numerical issues
            if sigma2[t] <= 0:
                sigma2[t] = 1e-6
        
        logl = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + 
                       (self.endog**2 / sigma2))
        
        valid_logl = logl[np.isfinite(logl)]
        
        if len(valid_logl) == 0:
            return 1e10
        
        return -np.sum(valid_logl)
    
    def _calculate_sigma2(self, params: np.ndarray):
        """Calculate conditional variance series."""
        omega, alpha, beta = params
        
        nobs = len(self.endog)
        self.sigma2 = np.ones(nobs) * np.var(self.endog)
        self.sigma2[0] = np.var(self.endog)
        
        for t in range(1, nobs):
            self.sigma2[t] = omega + alpha * (self.endog[t-1]**2) + beta * self.sigma2[t-1]
            
            # Prevent numerical issues
            if self.sigma2[t] <= 0:
                self.sigma2[t] = 1e-6
    
    def forecast_vol(self, steps: int = 1) -> float:
        """[FIX-11] Forecast volatility."""
        if not self.fitted or self.sigma2 is None:
            return float(np.std(self.endog))
        
        omega, alpha, beta = self.params
        current_sigma2 = self.sigma2[-1]
        forecasted = []
        
        for _ in range(steps):
            current_sigma2 = omega + alpha * (self.endog[-1]**2) + beta * current_sigma2
            forecasted.append(np.sqrt(current_sigma2))
        
        return float(np.mean(forecasted))
