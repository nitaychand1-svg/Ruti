import pytest
import numpy as np
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.config import load_config

@pytest.mark.asyncio
async def test_full_ensemble_pipeline():
    """End-to-end ensemble test."""
    config = load_config("configs/test.yaml")
    ensemble = GARCHHyperAdaptiveEnsemble(config)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = np.random.randint(0, 2, 500)
    features = [f'f{i}' for i in range(10)]
    
    # Train
    result = await ensemble.train(X, y, features)
    assert result['status'] == 'success'
    
    # Predict
    prediction = await ensemble.predict(X[:1], 'normal', y[0])
    assert 'prediction' in prediction
    assert len(prediction['prediction']) == 2
    
    # Stress test
    stress_results = await ensemble.run_stress_tests(X[:100], y[:100])
    assert 'diversity_at_risk' in stress_results

@pytest.mark.asyncio
async def test_data_validation():
    """Test data validation layer."""
    from src.validation.data_validation import DataValidationLayer
    
    # Test normal data
    X = np.random.randn(100, 10)
    X_valid = DataValidationLayer.validate(X)
    assert X_valid.shape == X.shape
    
    # Test with NaN
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    X_valid = DataValidationLayer.validate(X_nan)
    assert np.isfinite(X_valid).all()

@pytest.mark.asyncio
async def test_garch_tracker():
    """Test GARCH tracker."""
    from src.garch.tracker import GARCHModelTracker
    
    tracker = GARCHModelTracker(window=100, min_obs=50)
    returns = np.random.randn(200)
    tracker.add_feature('test_feature', returns)
    
    vol = tracker.get_all_volatilities()
    assert vol > 0
