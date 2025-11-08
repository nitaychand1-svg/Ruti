"""Ensemble integration tests."""
import pytest
import numpy as np
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    assert 'version' in result
    
    # Predict
    prediction = await ensemble.predict(X[:1], 'normal', y[0])
    assert 'prediction' in prediction
    assert len(prediction['prediction']) == 2
    assert 'confidence' in prediction
    assert 'diversity' in prediction
    
    # Stress test
    stress_results = await ensemble.run_stress_tests(X[:100], y[:100])
    assert 'diversity_at_risk' in stress_results
    assert 'mean_correlation' in stress_results


@pytest.mark.asyncio
async def test_prediction_caching():
    """Test prediction caching."""
    config = load_config("configs/test.yaml")
    ensemble = GARCHHyperAdaptiveEnsemble(config)
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    features = [f'f{i}' for i in range(10)]
    
    await ensemble.train(X, y, features)
    
    # First prediction
    pred1 = await ensemble.predict(X[:1], 'normal')
    
    # Second prediction (should use cache)
    pred2 = await ensemble.predict(X[:1], 'normal')
    
    assert pred1['prediction'][0] == pred2['prediction'][0]


def test_garch_model():
    """Test GARCH model fitting."""
    from src.garch.model import GARCH11
    
    np.random.seed(42)
    returns = np.random.randn(500) * 0.01
    
    model = GARCH11(returns)
    fitted = model.fit()
    
    assert fitted.params is not None
    assert len(fitted.params) == 3
    
    # Test forecast
    forecast = fitted.forecast_vol(steps=5)
    assert forecast > 0


def test_data_validation():
    """Test data validation layer."""
    from src.validation.data_validation import DataValidationLayer
    
    # Test with NaN values
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0])
    
    X_clean = DataValidationLayer.validate(X, y, context="test")
    
    assert not np.isnan(X_clean).any()
    assert X_clean.shape == X.shape


def test_config_loading():
    """Test configuration loading."""
    config = load_config("configs/test.yaml")
    
    assert config.n_base_models == 16
    assert config.garch.window == 50
    assert config.hypernet.meta_dim == 8
    assert config.risk.max_position_size == 0.25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
