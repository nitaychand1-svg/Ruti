#!/usr/bin/env python3
"""Production training script."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config import load_config
from src.core.orchestrator import GARCHHyperAdaptiveEnsemble


async def async_main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensemble = GARCHHyperAdaptiveEnsemble(config)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column")

    X = df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=int)
    features = df.drop(columns=["target"]).columns.tolist()

    result = await ensemble.train(X, y, features)
    print(f"✅ Training completed: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GARCH HyperNetwork Ensemble")
    parser.add_argument("--config", default="configs/production.yaml", help="Config path")
    parser.add_argument("--data-path", required=True, help="Path to parquet dataset")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
