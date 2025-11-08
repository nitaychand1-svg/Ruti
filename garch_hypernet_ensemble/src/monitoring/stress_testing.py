"""[FIX-2] Monte Carlo stress testing."""
import numpy as np
from typing import Dict, Any, List, Tuple
import asyncio
import logging

logger = logging.getLogger(__name__)

class StressTestingModule:
    """Comprehensive stress testing."""
    
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.logger = logger
    
    def run_monte_carlo_stress(self, X: np.ndarray, y: np.ndarray,
                              n_scenarios: int = 1000) -> Dict[str, Any]:
        """[FIX-2] Monte Carlo with GARCH simulations."""
        results = {
            'correlation_spikes': [],
            'diversity_scores': [],
            'accuracy_degradation': [],
            'worst_case_diversity': 1.0,
            'garch_spikes': [],
            'effective_bets': [],
            'black_swan_scenarios': []
        }
        
        scenarios = self._generate_stress_scenarios(X, y)
        
        for X_stress, y_stress in scenarios[:100]:
            try:
                pred_result = asyncio.run(
                    self.ensemble.predict(X_stress, market_regime='high_volatility')
                )
            except Exception as e:
                self.logger.error(f"Stress test prediction failed: {e}")
                continue
            
            results['correlation_spikes'].append(1.0 - pred_result.get('diversity', 0.0))
            results['diversity_scores'].append(pred_result.get('diversity', 0.0))
            
            if self.ensemble.predictor:
                results['garch_spikes'].append(
                    self.ensemble.predictor._get_current_garch_volatility()
                )
            else:
                results['garch_spikes'].append(0.01)
            
            if 'effective_bets' in pred_result:
                results['effective_bets'].append(pred_result['effective_bets'])
            
            diversity = pred_result.get('diversity', 0.0)
            if diversity < results['worst_case_diversity']:
                results['worst_case_diversity'] = diversity
        
        # [FIX-2] Black swan scenarios
        black_swan_results = self._run_black_swan_scenarios()
        results['black_swan_scenarios'].extend(black_swan_results)
        
        # Summary statistics
        results['mean_correlation'] = np.mean(results['correlation_spikes'])
        results['diversity_at_risk'] = np.percentile(results['diversity_scores'], 5)
        results['mean_garch_vol'] = np.mean(results['garch_spikes'])
        results['black_swan_diversity'] = np.min(results['black_swan_scenarios']) if results['black_swan_scenarios'] else 1.0
        
        # [FIX-5] Threshold check
        min_diversity = self.ensemble.config.monitoring.min_diversity_threshold
        if results['diversity_at_risk'] < min_diversity:
            self.logger.critical(
                f"❌ Stress test failed: diversity at risk = {results['diversity_at_risk']:.3f}"
            )
        
        return results
    
    def _generate_stress_scenarios(self, X: np.ndarray, 
                                 y: np.ndarray) -> List[Tuple]:
        """Generate synthetic stress scenarios."""
        scenarios = []
        n_scenarios = 50
        
        for _ in range(n_scenarios):
            shock_mask = np.random.random(len(X)) < 0.3
            X_stress = X.copy()
            shock_factor = np.random.normal(3.0, 1.0, size=X[shock_mask].shape)
            X_stress[shock_mask] *= shock_factor
            
            # Generate labels without leakage
            y_stress = (np.diff(X_stress[:, 0]) > 0).astype(int)
            y_stress = np.pad(y_stress, (1, 0), mode='constant')
            
            scenarios.append((X_stress, y_stress))
        
        return scenarios
    
    def _run_black_swan_scenarios(self) -> List[float]:
        """[FIX-2] Extreme black swan scenarios."""
        results = []
        
        for _ in range(100):
            try:
                X_swan = self._generate_black_swan_data(500, spike_factor=15.0)
                pred_swan = asyncio.run(
                    self.ensemble.predict(X_swan, market_regime='chaotic')
                )
                results.append(pred_swan.get('diversity', 0.0))
            except Exception as e:
                self.logger.error(f"Black swan scenario failed: {e}")
                results.append(0.0)
        
        return results
    
    def _generate_black_swan_data(self, n_samples: int, spike_factor: float):
        """Generate extreme scenario data."""
        base = np.random.randn(n_samples, 20)
        shock = np.random.normal(0, spike_factor, base.shape) * \
                np.random.binomial(1, 0.3, base.shape)
        return base + shock
