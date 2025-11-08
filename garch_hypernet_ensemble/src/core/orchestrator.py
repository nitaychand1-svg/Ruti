"""Main orchestrator for GARCH-HyperNetwork Ensemble."""
import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from ..core.config import EnsembleConfig
from ..garch.tracker import GARCHModelTracker
from ..hypernetwork.model import AdaptiveHyperNetwork
from ..hypernetwork.trainer import GARCHHyperNetTrainer
from ..monitoring.performance_tracker import OnlinePerformanceTracker
from ..monitoring.stress_testing import StressTestingModule
from ..utils.version_control import ModelVersionControl
from .trainer import EnsembleTrainer
from .predictor import EnsemblePredictor

logger = logging.getLogger(__name__)

class GARCHHyperAdaptiveEnsemble:
    """Production-ready ensemble orchestrator."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.logger = logger
        
        # Core components
        self.trainer = EnsembleTrainer(config)
        self.predictor: Optional[EnsemblePredictor] = None
        
        # Monitoring
        self.performance_tracker = OnlinePerformanceTracker(config.monitoring)
        self.stress_tester = StressTestingModule(self)
        
        # Version control
        self.version_control = ModelVersionControl()
        
        self.logger.info("✅ Ensemble orchestrator initialized")
    
    async def train(self, X: np.ndarray, y: np.ndarray, 
                   feature_names: List[str]) -> Dict[str, Any]:
        """Train full ensemble pipeline."""
        self.logger.info("Starting training pipeline...")
        
        # Train components
        components = await self.trainer.train(X, y, feature_names)
        
        # Initialize predictor
        self.predictor = EnsemblePredictor(components, self.config)
        self.predictor.performance_tracker = self.performance_tracker
        
        # Save model
        metadata = self.version_control.save_model(
            components,
            f"ensemble_v{self.config.model_version}.pt",
            X_sample=X[:1000],
            config=self.config.__dict__
        )
        
        self.logger.info("✅ Training completed successfully")
        return {"status": "success", "version": metadata["version"]}
    
    async def predict(self, X: np.ndarray, 
                     market_regime: str = 'normal',
                     ground_truth: Optional[int] = None,
                     vix: Optional[float] = None) -> Dict[str, Any]:
        """Make prediction with full monitoring."""
        if not self.predictor:
            raise ValueError("Model not trained. Call train() first.")
        
        return await self.predictor.predict(
            X, market_regime, ground_truth, vix
        )
    
    async def run_stress_tests(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive stress testing."""
        return await self.stress_tester.run_monte_carlo_stress(X, y)
    
    async def online_update(self, X: np.ndarray, y: int, 
                           regime: str) -> None:
        """Update models in real-time."""
        if not self.predictor:
            return
        
        # Update GARCH tracking
        for i, name in enumerate(self.predictor.components.feature_names):
            return_val = X[0, i] - X[-1, i] if len(X) > 1 else 0.0
            self.predictor.components.garch_tracker.update_returns(name, return_val)
        
        # Trigger retraining if needed
        if self.performance_tracker.should_trigger_retraining():
            self.logger.warning("🔄 Triggering auto-retrain...")
            await self.train(X, np.array([y]), self.predictor.components.feature_names)
