#!/usr/bin/env python3
"""Async inference service for the GARCH-HyperNetwork ensemble."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from aiohttp import web

from src.core.config import load_config
from src.core.orchestrator import GARCHHyperAdaptiveEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("garch_ensemble.api")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the GARCH HyperNetwork ensemble API")
    parser.add_argument(
        "--config",
        default=os.getenv("ENSEMBLE_CONFIG_PATH", "configs/production.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for the web server")
    parser.add_argument("--port", type=int, default=8080, help="Port for the web server")
    return parser.parse_args()


async def create_app(config_path: str) -> web.Application:
    config = load_config(config_path)
    ensemble = GARCHHyperAdaptiveEnsemble(config)

    app = web.Application()
    app["ensemble"] = ensemble
    app["train_lock"] = asyncio.Lock()

    app.router.add_get("/health", health)
    app.router.add_post("/predict", predict)
    app.router.add_post("/train", train)

    return app


async def health(request: web.Request) -> web.Response:
    ensemble: GARCHHyperAdaptiveEnsemble = request.app["ensemble"]
    status = {
        "status": "ok",
        "trained": ensemble.predictor is not None,
        "version": ensemble.config.model_version,
    }
    return web.json_response(status)


async def predict(request: web.Request) -> web.Response:
    ensemble: GARCHHyperAdaptiveEnsemble = request.app["ensemble"]

    if ensemble.predictor is None:
        return web.json_response({"error": "model_not_trained"}, status=400)

    payload = await request.json()
    features = payload.get("features")
    if features is None:
        return web.json_response({"error": "missing_features"}, status=400)

    regime = payload.get("regime", "normal")
    ground_truth = payload.get("ground_truth")

    try:
        X = np.asarray(features, dtype=float)
    except Exception as exc:  # pragma: no cover - input validation guard
        return web.json_response({"error": f"invalid_features: {exc}"}, status=400)

    prediction = await ensemble.predict(
        X,
        market_regime=regime,
        ground_truth=int(ground_truth) if ground_truth is not None else None,
    )

    def to_serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        return value

    response = {key: to_serializable(val) for key, val in prediction.items()}
    return web.json_response(response)


async def train(request: web.Request) -> web.Response:
    ensemble: GARCHHyperAdaptiveEnsemble = request.app["ensemble"]
    train_lock: asyncio.Lock = request.app["train_lock"]

    payload = await request.json()
    data_path = payload.get("data_path")
    if not data_path:
        return web.json_response({"error": "missing_data_path"}, status=400)

    path = Path(data_path)
    if not path.exists():
        return web.json_response({"error": f"data_path_not_found: {path}"}, status=400)

    async with train_lock:
        import pandas as pd

        df = pd.read_parquet(path)
        if "target" not in df.columns:
            return web.json_response({"error": "dataset_missing_target"}, status=400)

        X = df.drop(columns=["target"]).to_numpy(dtype=np.float32)
        y = df["target"].to_numpy(dtype=int)
        features = df.drop(columns=["target"]).columns.tolist()

        result = await ensemble.train(X, y, features)

    return web.json_response(result)


def main() -> None:
    args = parse_args()
    app = asyncio.run(create_app(args.config))
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
