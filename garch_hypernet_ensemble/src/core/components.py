"""Component dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TrainedComponents:
    """Container for all trained models."""

    base_models: List[Dict[str, Any]]
    blender_model: Any
    meta_regulator: Any
    garch_tracker: Any
    feature_names: List[str]
    correlation_matrix: Optional[np.ndarray] = None
    oos_score: Optional[float] = None
    hypernetwork: Optional[Any] = None
