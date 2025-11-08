from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.config import load_config
from src.core.orchestrator import GARCHHyperAdaptiveEnsemble


@pytest.mark.asyncio
async def test_full_ensemble_pipeline():
    config_path = PROJECT_ROOT / "configs" / "test.yaml"
    config = load_config(str(config_path))
    ensemble = GARCHHyperAdaptiveEnsemble(config)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=200)
    features = [f"f{i}" for i in range(10)]

    result = await ensemble.train(X, y, features)
    assert result["status"] == "success"

    prediction = await ensemble.predict(X[:1], "normal", int(y[0]))
    assert "prediction" in prediction
    assert len(prediction["prediction"]) == 2

    stress_results = await ensemble.run_stress_tests(X[:50], y[:50])
    assert "diversity_at_risk" in stress_results
