#!/usr/bin/env python3
"""Aiohttp inference service for the GARCH-HyperNetwork ensemble."""
import asyncio
import json
import logging
import os
from typing import Optional

import numpy as np
from aiohttp import web

from src.core.config import load_config, EnsembleConfig
from src.core.orchestrator import GARCHHyperAdaptiveEnsemble
from src.core.predictor import EnsemblePredictor
from src.monitoring.performance_tracker import OnlinePerformanceTracker
from src.utils.version_control import ModelVersionControl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class EnsembleService:
    """Wraps the ensemble orchestrator for online inference."""

    def __init__(self, config_path: str = "configs/production.yaml"):
        self.config_path = config_path
        self.config: Optional[EnsembleConfig] = None
        self.ensemble: Optional[GARCHHyperAdaptiveEnsemble] = None
        self.version_control = ModelVersionControl()

    async def initialize(self):
        self.config = load_config(self.config_path)
        self.ensemble = GARCHHyperAdaptiveEnsemble(self.config)
        await self._load_latest_model()

    async def _load_latest_model(self):
        models_dir = self.version_control.storage_path
        if not os.path.isdir(models_dir):
            logger.warning("Models directory does not exist; starting without trained model")
            return

        metadata_files = [
            os.path.join(models_dir, f)
            for f in os.listdir(models_dir)
            if f.endswith("_metadata.json")
        ]
        if not metadata_files:
            logger.warning("No model metadata files found; predictor is unavailable")
            return

        # Sort metadata by creation timestamp descending
        metadata_files.sort(key=os.path.getmtime, reverse=True)
        latest_meta = metadata_files[0]
        model_filename = os.path.basename(latest_meta).replace("_metadata.json", ".pt")

        try:
            loaded = self.version_control.load_model(model_filename)
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_filename, exc)
            return

        components = loaded['components']
        self.ensemble.predictor = EnsemblePredictor(components, self.config)
        self.ensemble.predictor.performance_tracker = OnlinePerformanceTracker(self.config.monitoring)
        logger.info("Loaded model version %s", loaded['metadata']['version'])

    async def predict(self, request: web.Request) -> web.Response:
        if self.ensemble is None or self.ensemble.predictor is None:
            return web.json_response({
                "status": "error",
                "message": "Model not initialized"
            }, status=503)

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"status": "error", "message": "Invalid JSON"}, status=400)

        features = payload.get("features")
        if features is None:
            return web.json_response({"status": "error", "message": "Missing features"}, status=400)

        market_regime = payload.get("market_regime", "normal")
        ground_truth = payload.get("ground_truth")
        regime = str(market_regime)
        
        np_features = np.asarray(features, dtype=float)
        if np_features.ndim == 1:
            np_features = np_features.reshape(1, -1)

        try:
            result = await self.ensemble.predict(np_features, market_regime=regime, ground_truth=ground_truth)
        except Exception as exc:
            logger.exception("Prediction failed")
            return web.json_response({
                "status": "error",
                "message": f"prediction_failed: {exc}"
            }, status=500)

        response = {
            "status": "ok",
            "prediction": result.get("prediction", []).tolist(),
            "confidence": float(result.get("confidence", 0.0)),
            "position_size": float(result.get("position_size", 0.0)),
            "emergency_mode": bool(result.get("emergency_mode", False)),
            "metadata": {
                "active_models": int(result.get("active_models", 0)),
                "diversity": float(result.get("diversity", 0.0)),
                "garch_volatility": float(result.get("garch_volatility", 0.0)),
                "effective_bets": float(result.get("effective_bets", 0.0)) if result.get("effective_bets") is not None else None
            }
        }
        return web.json_response(response)

    async def health(self, _: web.Request) -> web.Response:
        status = "ok" if self.ensemble and self.ensemble.predictor else "initializing"
        return web.json_response({"status": status})

    async def reload(self, _: web.Request) -> web.Response:
        await self._load_latest_model()
        status = "ok" if self.ensemble and self.ensemble.predictor else "initializing"
        return web.json_response({"status": status})


async def create_app() -> web.Application:
    service = EnsembleService()
    await service.initialize()

    app = web.Application()
    app["service"] = service

    app.router.add_get("/health", service.health)
    app.router.add_post("/predict", service.predict)
    app.router.add_post("/reload", service.reload)

    return app


def main():
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(create_app())
    web.run_app(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
