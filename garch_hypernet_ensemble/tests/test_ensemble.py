import pytest
import numpy as np
import asyncio
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
