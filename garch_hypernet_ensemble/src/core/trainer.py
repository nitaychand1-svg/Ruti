"""Training logic for ensemble components."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import TimeSeriesSplit

from ..core.config import EnsembleConfig
from ..garch.tracker import GARCHModelTracker
from ..validation.cross_validation import WalkForwardValidator
from ..validation.data_validation import DataValidationLayer
from .components import TrainedComponents

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Handles async training of base models, blender, and meta-regulator."""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.logger = logger
        self.garch_tracker = GARCHModelTracker(
            window=config.garch.window,
            min_obs=config.garch.min_obs,
        )

    async def train(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> TrainedComponents:
        """Full async training pipeline."""
        self.logger.info("Starting async training pipeline...")

        # [FIX-12] Validate inputs
        X = DataValidationLayer.validate(X, y, context="train")

        # [FIX-1] Setup GARCH tracking
        await self._setup_garch_features(X, feature_names)

        # [FIX-14] Walk-forward cross-validation
        wfo = WalkForwardValidator(n_splits=5, purge_gap=10)
        oos_holdout = wfo.get_oos_holdout(len(X))

        X_train_full, y_train_full = X[: oos_holdout.start], y[: oos_holdout.start]
        X_oos, y_oos = X[oos_holdout], y[oos_holdout]

        # [FIX-2] Generate stress scenarios
        stress_scenarios = self._generate_stress_scenarios(X_train_full, y_train_full)

        # [FIX-3] Train base models asynchronously
        base_models = await self._train_base_models_async(
            X_train_full, y_train_full, wfo, stress_scenarios
        )

        # [FIX-5] Prune underperformers
        base_models = self._prune_models(base_models)

        # Train blender
        blender, blender_training_input = self._train_blender(
            base_models, X_train_full, y_train_full
        )

        # Train meta-regulator
        meta = self._train_meta_regulator(
            blender, blender_training_input, y_train_full
        )

        # [FIX-10] OOS evaluation
        oos_score = self._evaluate_oos(
            X_oos, y_oos, base_models, blender, meta
        )
        if oos_score < self.config.min_oos_score:
            raise RuntimeError(f"[FIX-5] OOS validation failed: {oos_score:.3f}")

        # [FIX-19] Calculate correlation matrix
        corr_matrix = self._calculate_correlation_matrix(base_models, X)

        return TrainedComponents(
            base_models=base_models,
            blender_model=blender,
            meta_regulator=meta,
            garch_tracker=self.garch_tracker,
            feature_names=feature_names,
            correlation_matrix=corr_matrix,
            oos_score=oos_score,
        )

    async def _setup_garch_features(self, X: np.ndarray, feature_names: List[str]):
        """Setup GARCH tracking for each feature."""
        self.logger.info("Setting up GARCH features...")

        for i, name in enumerate(feature_names):
            returns_series = np.diff(X[:, i]) / (X[:-1, i] + 1e-6)
            returns_series = np.nan_to_num(returns_series, 0.0)
            returns_series_padded = np.pad(returns_series, (1, 0), mode="constant")
            self.garch_tracker.add_feature(name, returns_series_padded)

        if self.config.garch.enable_forecast:
            self.logger.info("Fitting GARCH models...")
            self.garch_tracker.fit_all_garch_parallel()

    def _generate_stress_scenarios(
        self, X: np.ndarray, y: np.ndarray, n_scenarios: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """[FIX-2] Generate synthetic stress scenarios."""
        scenarios: List[Tuple[np.ndarray, np.ndarray]] = []
        base_returns = np.diff(X[:, 0])
        base_mean = np.mean(base_returns)

        for _ in range(n_scenarios):
            shock_mask = np.random.random(len(X)) < 0.3
            X_stress = X.copy()
            shock_factor = np.random.normal(3.0, 1.0, size=X[shock_mask].shape)
            X_stress[shock_mask] *= shock_factor

            # Generate labels without leakage
            stress_returns = np.diff(X_stress[:, 0])
            y_stress = (stress_returns > base_mean).astype(int)
            y_stress = np.pad(y_stress, (1, 0), mode="constant")

            scenarios.append((X_stress, y_stress))

        return scenarios

    async def _train_base_models_async(
        self,
        X: np.ndarray,
        y: np.ndarray,
        wfo: WalkForwardValidator,
        stress_scenarios: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """[FIX-3] Async training pool with resource isolation."""
        models_to_train = self._initialize_base_models()
        tasks = [
            self._train_single_model_worker(model_info, X, y, wfo, stress_scenarios)
            for model_info in models_to_train
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        trained_models = [r for r in results if isinstance(r, dict) and "model_id" in r]

        self.logger.info("✅ Trained %d models successfully", len(trained_models))
        return trained_models

    async def _train_single_model_worker(
        self,
        model_info: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        wfo: WalkForwardValidator,
        stress_scenarios: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """Worker for single model training."""
        try:
            X = DataValidationLayer.validate(X, y, context="train_full")

            splits = list(wfo.split(X, y))
            val_scores: List[float] = []
            stress_scores: List[float] = []

            for split_idx, (train_idx, val_idx) in enumerate(splits):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                X_stress, y_stress = stress_scenarios[split_idx % len(stress_scenarios)]

                X_train = DataValidationLayer.validate(X_train, y_train, context="train_split")
                X_val = DataValidationLayer.validate(X_val, y_val, context="val_split")

                model = deepcopy(model_info["model"])
                model.fit(X_train, y_train)

                val_scores.append(self._evaluate_model(model, X_val, y_val))
                stress_scores.append(self._evaluate_stress(model, X_stress, y_stress))

            avg_val = float(np.mean(val_scores)) if val_scores else 0.5
            avg_stress = float(np.mean(stress_scores)) if stress_scores else 0.5
            composite = 0.7 * avg_val + 0.3 * avg_stress

            final_model = deepcopy(model_info["model"])
            final_model.fit(X, y)

            return {
                "model_id": model_info["id"],
                "id": model_info["id"],
                "model": final_model,
                "performance": {"accuracy": avg_val},
                "stress_performance": {"accuracy": avg_stress},
                "composite_score": composite,
                "confidence": max(avg_val, avg_stress),
            }
        except Exception as exc:  # pragma: no cover - logged for observability
            self.logger.error("Training failed for %s: %s", model_info["id"], exc)
            return {"error": str(exc)}

    def _initialize_base_models(self) -> List[Dict[str, Any]]:
        """[FIX-5] Initialize diverse base models."""
        from sklearn.ensemble import (
            AdaBoostClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC

        algos = [
            (RandomForestClassifier, {"n_estimators": 200, "max_depth": 8}),
            (
                GradientBoostingClassifier,
                {"n_estimators": 150, "learning_rate": 0.05},
            ),
            (ExtraTreesClassifier, {"n_estimators": 250, "max_depth": 10}),
            (AdaBoostClassifier, {"n_estimators": 100}),
            (SVC, {"kernel": "rbf", "probability": True, "C": 1.0}),
            (MLPClassifier, {"hidden_layer_sizes": (128, 64), "activation": "tanh"}),
            (LogisticRegression, {"penalty": "l1", "C": 0.5, "solver": "liblinear"}),
        ]

        models: List[Dict[str, Any]] = []
        rng = np.random.default_rng(42)
        for i in range(self.config.n_base_models):
            algo_class, params = algos[i % len(algos)]
            modified_params = deepcopy(params)

            # [FIX-5] Add random diversity
            if "learning_rate" in modified_params:
                modified_params["learning_rate"] *= float(0.5 + rng.random())
            if "max_depth" in modified_params:
                modified_params["max_depth"] = max(
                    2, modified_params["max_depth"] + int(rng.integers(-2, 3))
                )

            models.append(
                {
                    "id": f"base_{i:03d}_{algo_class.__name__}",
                    "model": algo_class(**modified_params),
                }
            )

        self.logger.info("✅ Initialized %d base models", len(models))
        return models

    def _prune_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[FIX-5] Remove underperforming models."""
        kept = [m for m in models if m.get("composite_score", 0.0) >= self.config.prune_threshold]
        pruned = len(models) - len(kept)

        if pruned > 0:
            self.logger.warning(
                "[FIX-5] Pruned %d models, %d remaining", pruned, len(kept)
            )

        # Redistribute weights
        for model in kept:
            model["weight"] = 1.0 / max(1, len(kept))

        return kept

    def _train_blender(
        self, base_models: List[Dict[str, Any]], X: np.ndarray, y: np.ndarray
    ):
        """[FIX-2] Train stacking blender with time series CV."""
        blender_input = self._generate_oof_predictions(base_models, X, y)

        if not base_models:
            raise RuntimeError("No base models available after pruning")

        blender = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        blender.fit(blender_input, y)
        return blender, blender_input

    def _train_meta_regulator(
        self, blender_model, blender_features: np.ndarray, y: np.ndarray
    ):
        """[FIX-16] Train meta-regulator for risk management."""
        blender_pred = blender_model.predict_proba(blender_features)
        meta_input = np.column_stack(
            [blender_pred, np.std(blender_pred, axis=1), np.var(blender_pred, axis=1)]
        )

        meta = CatBoostClassifier(
            iterations=800,
            depth=6,
            loss_function="Logloss",
            learning_rate=0.01,
            verbose=False,
            eval_metric="AUC",
            l2_leaf_reg=5.0,
            random_strength=1.5,
        )
        return meta.fit(meta_input, y)

    def _generate_oof_predictions(
        self, base_models: List[Dict[str, Any]], X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """[FIX-2] Prepare OOS predictions for blender."""
        if not base_models:
            raise RuntimeError("Base models list is empty")

        n_samples = len(X)
        n_models = len(base_models)
        oos_preds = np.zeros((n_samples, n_models))

        tscv = TimeSeriesSplit(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for i, model_info in enumerate(base_models):
                model_copy = deepcopy(model_info["model"])
                try:
                    model_copy.fit(X_train, y_train)
                    if hasattr(model_copy, "predict_proba"):
                        oos_preds[val_idx, i] = model_copy.predict_proba(X_val)[:, 1]
                    else:
                        preds = model_copy.predict(X_val)
                        oos_preds[val_idx, i] = preds
                except Exception as exc:  # pragma: no cover
                    self.logger.warning(
                        "Blender fold %d model %s failed: %s", fold, model_info["id"], exc
                    )
                    oos_preds[val_idx, i] = 0.5

        return oos_preds

    def _generate_blender_features(
        self, base_models: List[Dict[str, Any]], X: np.ndarray
    ) -> np.ndarray:
        """Generate stacked features from trained base models."""
        if not base_models:
            return np.zeros((len(X), 0))

        features = np.zeros((len(X), len(base_models)))
        for i, model_info in enumerate(base_models):
            model = model_info["model"]
            try:
                if hasattr(model, "predict_proba"):
                    features[:, i] = model.predict_proba(X)[:, 1]
                else:
                    features[:, i] = model.predict(X)
            except Exception:  # pragma: no cover
                features[:, i] = 0.5

        return features

    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model on validation set."""
        if len(X) == 0:
            return 0.5
        try:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
                return float(np.mean(np.argmax(pred, axis=1) == y))
            pred = model.predict(X)
            return float(np.mean(pred == y))
        except Exception:  # pragma: no cover
            return 0.5

    def _evaluate_stress(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """[FIX-2] Stress test evaluation."""
        if len(X) == 0:
            return 0.5
        try:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
                return float(np.mean(np.argmax(pred, axis=1) == y))
            pred = model.predict(X)
            return float(np.mean(pred == y))
        except Exception:  # pragma: no cover
            return 0.5

    def _evaluate_oos(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: List[Dict[str, Any]],
        blender,
        meta,
    ) -> float:
        """Out-of-sample evaluation."""
        if len(X) == 0:
            return 0.5
        blender_features = self._generate_blender_features(base_models, X)
        blender_pred = blender.predict_proba(blender_features)
        meta_input = np.column_stack(
            [blender_pred, np.std(blender_pred, axis=1), np.var(blender_pred, axis=1)]
        )
        final_pred = meta.predict(meta_input)
        return float(np.mean(final_pred == y))

    def _calculate_correlation_matrix(
        self, models: List[Dict[str, Any]], X: np.ndarray
    ) -> np.ndarray:
        """[FIX-19] Calculate model correlation matrix."""
        if len(models) < 2:
            return np.array([[1.0]])

        max_rows = min(1000, len(X))
        predictions = []
        for model_info in models:
            model = model_info["model"]
            try:
                if hasattr(model, "predict_proba"):
                    predictions.append(model.predict_proba(X[:max_rows])[:, 1])
                else:
                    predictions.append(model.predict(X[:max_rows]))
            except Exception:  # pragma: no cover
                predictions.append(np.full(max_rows, 0.5))

        return np.corrcoef(np.column_stack(predictions).T)
