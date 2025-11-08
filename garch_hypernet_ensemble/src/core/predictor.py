"""Prediction logic with all safety mechanisms."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ..core.config import EnsembleConfig
from ..monitoring.performance_tracker import OnlinePerformanceTracker
from ..utils.caching import PredictionCache
from .components import TrainedComponents
from ..validation.data_validation import DataValidationLayer

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Production prediction engine."""

    def __init__(self, components: TrainedComponents, config: EnsembleConfig):
        self.components = components
        self.config = config
        self.logger = logger

        # [FIX-13] Prediction cache
        self.cache = PredictionCache(
            ttl=config.cache.ttl_seconds,
            maxsize=config.cache.maxsize,
        )

        # [FIX-4] HyperNetwork trainer
        if getattr(components, "hypernetwork", None) is not None:
            from ..hypernetwork.trainer import GARCHHyperNetTrainer

            self.hypernet_trainer = GARCHHyperNetTrainer(
                components.hypernetwork, config.hypernet.learning_rate
            )
        else:
            self.hypernet_trainer = None

        # Performance tracking
        self.performance_tracker: Optional[OnlinePerformanceTracker] = None

    async def predict(
        self,
        X: np.ndarray,
        market_regime: str = "normal",
        ground_truth: Optional[int] = None,
        vix: Optional[float] = None,
    ) -> Dict[str, Any]:
        """[FIX-17] Main prediction method with full error handling."""
        try:
            # [FIX-12] Input validation
            X = DataValidationLayer.validate(X, context="predict_input")
            X = self._ensure_2d(X)

            # [FIX-5] Circuit breaker
            current_vol = self._get_current_garch_volatility()
            if current_vol > 0.05 or market_regime == "chaotic":
                self.logger.warning(
                    "[FIX-5] Circuit breaker: vol=%.4f, regime=%s",
                    current_vol,
                    market_regime,
                )
                return self._emergency_fallback("extreme_volatility")

            # [FIX-5] Check active models
            active_models, active_count = self._get_active_models()
            if active_count < 5:
                self.logger.warning("[FIX-5] Low active models: %d", active_count)
                return self._fallback_rule_based(X)

            # Get base predictions
            base_predictions, base_confidences, model_ids = await self._get_base_predictions(
                active_models, X
            )
            active_mask = np.array(
                [1.0 if m.get("is_active") else 0.0 for m in active_models], dtype=float
            )

            # [FIX-19] Diversity metrics
            diversity = self._calculate_ensemble_diversity(base_predictions)
            corr_matrix, avg_correlation = self._calculate_correlation(base_predictions)

            # [FIX-8] Encode regime
            regime_features = self._encode_regime_continuous(X, current_vol)

            # [FIX-6] GARCH features
            garch_vol = current_vol
            garch_forecast = self._get_garch_forecast()

            # [FIX-4] HyperNetwork meta-features
            meta_features = self._prepare_meta_features(
                diversity,
                avg_correlation,
                regime_features,
                garch_vol,
                garch_forecast,
                len(active_models),
                active_count,
            )

            # HyperNetwork prediction
            if getattr(self.components, "hypernetwork", None) is not None:
                tf_meta = self._to_tensor(meta_features)
                dynamic_weights, dynamic_thresholds = self.components.hypernetwork(
                    tf_meta, training=False
                )

                # Update config dynamically
                self.config.monitoring.max_correlation_threshold = float(
                    dynamic_thresholds[0, 0]
                )
                self.config.monitoring.min_confidence_threshold = float(
                    dynamic_thresholds[0, 1]
                )

                model_weights = dynamic_weights.numpy()[0]
                model_weights = self._apply_active_mask(model_weights, active_mask)

                # [FIX-4] Update correlation matrix
                self.components.hypernetwork.update_correlation_matrix(corr_matrix)
            else:
                model_weights = self._uniform_weights(active_mask)
                dynamic_thresholds = np.array([[0.85, 0.55, 0.10]])

            # Blender prediction
            blender_pred = self.components.blender_model.predict_proba(base_predictions)[0]

            # Meta-regulator
            meta_input = self._prepare_meta_input(blender_pred)
            final_pred = self.components.meta_regulator.predict_proba(meta_input)[0]

            # [FIX-8] Regime calibration
            calibrated_pred = self._regime_specific_calibration(final_pred, market_regime)

            # [FIX-16] Position sizing
            position_size = self._calculate_position_size(
                calibrated_pred, base_predictions, active_mask
            )

            # [FIX-19] Effective bets
            effective_bets = self._calculate_effective_bets(model_weights, corr_matrix)

            # Build result
            result: Dict[str, Any] = {
                "prediction": calibrated_pred,
                "confidence": float(np.max(calibrated_pred)),
                "base_predictions": base_predictions,
                "model_weights": model_weights,
                "diversity": diversity,
                "garch_volatility": garch_vol,
                "regime": market_regime,
                "active_models": int(active_count),
                "position_size": position_size,
                "cvar_95": self._calculate_cvar(base_predictions, active_mask),
                "effective_bets": effective_bets,
                "emergency_mode": False,
                "dynamic_thresholds": dynamic_thresholds[0].tolist(),
                "model_ids": model_ids,
            }

            # [FIX-5] Online update
            if ground_truth is not None and self.hypernet_trainer:
                reward = float(np.argmax(calibrated_pred) == ground_truth)
                self.hypernet_trainer.update_online(meta_features, reward, avg_correlation)

            # Cache result
            cache_key = str(joblib.hash(X.tobytes() + market_regime.encode()))
            self.cache.set_sync(cache_key, result)

            # Update monitoring
            if self.performance_tracker and ground_truth is not None:
                self.performance_tracker.update_metrics(
                    calibrated_pred,
                    ground_truth,
                    market_regime,
                    corr_matrix,
                )

            return result

        except Exception as exc:  # pragma: no cover - catastrophic guard
            self.logger.critical("[FIX-17] Prediction failed: %s", exc, exc_info=True)
            return self._emergency_fallback(str(exc))

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return X.reshape(1, -1)
        return X

    def _get_active_models(self) -> Tuple[List[Dict[str, Any]], int]:
        """Get models with activity flags based on confidence threshold."""
        threshold = self.config.monitoring.min_confidence_threshold
        models: List[Dict[str, Any]] = []
        active_count = 0
        for model_info in self.components.base_models:
            model_copy = dict(model_info)
            is_active = model_copy.get("confidence", 0.0) >= threshold
            model_copy["is_active"] = is_active
            if is_active:
                active_count += 1
            models.append(model_copy)
        return models, active_count

    async def _get_base_predictions(
        self, models: List[Dict[str, Any]], X: np.ndarray
    ) -> Tuple[np.ndarray, List[float], List[str]]:
        """Get predictions from all active models."""
        predictions: List[float] = []
        confidences: List[float] = []
        model_ids: List[str] = []

        for model_info in models:
            cache_key = f"{model_info['id']}_{joblib.hash(X.tobytes())}"
            cached = self.cache.get_sync(cache_key)

            if cached:
                pred = float(cached["prediction"])
                conf = float(cached["confidence"])
            else:
                result = await self._predict_single_model(model_info, X)
                pred = float(result["prediction"])
                conf = float(result["confidence"])
                self.cache.set_sync(cache_key, result)

            predictions.append(pred)
            confidences.append(conf)
            model_ids.append(model_info["id"])

        return np.array(predictions, dtype=float).reshape(1, -1), confidences, model_ids

    async def _predict_single_model(
        self, model_info: Dict[str, Any], X: np.ndarray
    ) -> Dict[str, Any]:
        """Predict with single model."""
        model = model_info["model"]

        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0]
                prediction = float(prob[1])
                confidence = float(np.max(prob))
            else:
                pred = float(model.predict(X)[0])
                prediction = pred
                confidence = float(abs(pred - 0.5) * 2)

            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_id": model_info["id"],
            }
        except Exception as exc:  # pragma: no cover
            self.logger.error("[FIX-17] Model %s failed: %s", model_info["id"], exc)
            return {
                "prediction": 0.5,
                "confidence": 0.0,
                "model_id": model_info["id"],
            }

    def _emergency_fallback(self, reason: str) -> Dict[str, Any]:
        """[FIX-17] Safe fallback response."""
        return {
            "prediction": np.array([0.5, 0.5]),
            "confidence": 0.0,
            "emergency_mode": True,
            "reason": f"exception_{reason[:50]}",
            "position_size": 0.0,
            "active_models": 0,
        }

    def _fallback_rule_based(self, X: np.ndarray) -> Dict[str, Any]:
        """[FIX-5] Rule-based fallback."""
        prices = X[:, 0]
        window_short = min(5, len(prices))
        window_long = min(20, len(prices))
        ma_short = np.mean(prices[-window_short:])
        ma_long = np.mean(prices[-window_long:])
        signal = 1 if ma_short > ma_long else 0

        return {
            "prediction": np.array([1 - signal, signal]),
            "confidence": 0.3,
            "emergency_mode": True,
            "reason": "low_active_models",
            "position_size": 0.1,
            "active_models": 0,
        }

    def _prepare_meta_features(
        self,
        diversity: float,
        avg_correlation: float,
        regime_features: np.ndarray,
        garch_vol: float,
        garch_forecast: float,
        total_models: int,
        active_count: int,
    ) -> np.ndarray:
        """[FIX-4] Prepare meta-features for HyperNetwork."""
        recent_accuracy = 0.5
        if self.performance_tracker and self.performance_tracker.accuracy_window:
            recent_accuracy = float(
                np.mean(list(self.performance_tracker.accuracy_window)) or 0.5
            )

        return np.array(
            [
                diversity,
                avg_correlation,
                regime_features[2],  # High volatility membership
                regime_features[3],  # Chaotic membership
                garch_vol,
                garch_forecast,
                active_count / max(total_models, 1),
                recent_accuracy,
            ]
        ).reshape(1, -1)

    def _encode_regime_continuous(self, X: np.ndarray, vol: float) -> np.ndarray:
        """[FIX-8] Continuous regime encoding."""
        prices = X[:, 0]
        returns = np.diff(prices) if len(prices) > 1 else np.array([0.0])

        percentile = np.percentile(np.abs(returns), 95) if returns.size else 1.0
        vol_percentile = np.clip(vol / (percentile + 1e-6), 0, 1)
        trend_strength = float(
            np.abs(np.mean(returns[-20:])) / (np.std(returns[-20:]) + 1e-6)
            if returns.size >= 20
            else 0.0
        )
        momentum = float(np.mean(prices[-10:]) - np.mean(prices[-min(len(prices), 50) :]))

        low_vol = max(0.0, 1 - vol_percentile * 2)
        transition = max(0.0, 1 - abs(vol_percentile - 0.5) * 2)
        high_vol = max(0.0, vol_percentile - 0.7)
        chaotic = max(0.0, vol_percentile - 0.9) * trend_strength

        memberships = np.array([low_vol, transition, high_vol, chaotic], dtype=float)
        memberships /= np.sum(memberships) + 1e-8

        return np.concatenate(
            [memberships, [trend_strength, momentum, vol_percentile]], dtype=float
        )

    def _regime_specific_calibration(
        self, prediction: np.ndarray, regime: str
    ) -> np.ndarray:
        """[FIX-8] Calibrate based on market regime."""
        calibrators = {
            "low_volatility_bull": lambda p: np.array(
                [min(p[0] * 1.15, 0.95), max(p[1] * 0.85, 0.05)]
            ),
            "low_volatility_bear": lambda p: np.array(
                [max(p[0] * 0.85, 0.05), min(p[1] * 1.15, 0.95)]
            ),
            "high_volatility": lambda p: np.array([p[0] * 0.9 + 0.05, p[1] * 0.9 + 0.05]),
            "transition": lambda p: np.array([0.45, 0.45]),
        }

        calibrated = calibrators.get(regime, lambda p: p)(prediction)
        # Re-normalize
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        calibrated /= np.sum(calibrated)
        return calibrated

    def _calculate_position_size(
        self,
        ensemble_pred: np.ndarray,
        base_predictions: np.ndarray,
        active_mask: np.ndarray,
    ) -> float:
        """[FIX-16] Kelly criterion + VaR position sizing."""
        p_up = float(ensemble_pred[1])
        edge = abs(p_up - 0.5) * 2

        odds = (p_up / (1 - p_up + 1e-6)) if p_up > 0.5 else ((1 - p_up) / (p_up + 1e-6))
        kelly_fraction = (
            (edge * odds - (1 - edge)) / (odds + 1e-8) if odds > 0 else 0.0
        )
        kelly_fraction = float(
            np.clip(kelly_fraction, 0.0, self.config.risk.max_position_size)
        )

        cvar = abs(self._calculate_cvar(base_predictions, active_mask)) + 1e-6
        var_limit = self.config.risk.max_risk_per_trade / cvar
        return float(min(kelly_fraction, var_limit))

    def _calculate_cvar(
        self, predictions: np.ndarray, active_mask: Optional[np.ndarray] = None
    ) -> float:
        """[FIX-16] Conditional Value at Risk (95%)."""
        if (
            active_mask is not None
            and active_mask.size == predictions.shape[1]
            and np.sum(active_mask) > 0
        ):
            filtered = predictions[:, active_mask > 0.0]
            flat_preds = filtered.flatten() if filtered.size else predictions.flatten()
        else:
            flat_preds = predictions.flatten()
        return float(np.percentile(flat_preds, self.config.risk.cvar_percentile))

    def _calculate_effective_bets(
        self, weights: np.ndarray, corr_matrix: np.ndarray
    ) -> float:
        """[FIX-19] Effective number of independent bets."""
        norm_weights = weights / (np.sum(weights) + 1e-12)
        div_ratio = float(np.sqrt(norm_weights.T @ corr_matrix @ norm_weights))
        entropy = -float(np.sum(norm_weights * np.log(norm_weights + 1e-8)))
        return float(np.exp(entropy) / (div_ratio + 1e-6))

    def _apply_active_mask(self, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = weights * mask
        total = masked.sum()
        if total <= 0:
            return self._uniform_weights(mask)
        return masked / total

    def _uniform_weights(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask
        if mask.sum() > 0:
            return mask / mask.sum()
        return np.ones_like(mask) / len(mask)

    def _get_current_garch_volatility(self) -> float:
        """Get current GARCH volatility estimate."""
        return float(self.components.garch_tracker.get_all_volatilities())

    def _get_garch_forecast(self) -> float:
        """Get GARCH volatility forecast."""
        if not self.components.feature_names:
            return self._get_current_garch_volatility()

        feature = self.components.feature_names[0]
        forecast = self.components.garch_tracker.forecast_volatility(feature, steps=5)
        if forecast is None:
            return self._get_current_garch_volatility()
        return float(forecast)

    def _calculate_ensemble_diversity(self, predictions: np.ndarray) -> float:
        """[FIX-5] Spearman correlation-based diversity."""
        from scipy.stats import spearmanr

        if predictions.shape[1] < 2 or predictions.shape[0] < 2:
            return 0.0
        correlations = []
        n_models = predictions.shape[1]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                try:
                    corr, _ = spearmanr(predictions[:, i], predictions[:, j])
                    correlations.append(abs(corr))
                except Exception:  # pragma: no cover
                    correlations.append(0.0)
        return float(1.0 - np.mean(correlations) if correlations else 0.0)

    def _calculate_correlation(self, predictions: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate correlation matrix from predictions."""
        if predictions.shape[1] < 2 or predictions.shape[0] < 2:
            return np.array([[1.0]]), 0.0

        corr_matrix = np.corrcoef(predictions, rowvar=False)
        triu_indices = np.triu_indices_from(corr_matrix, k=1)
        if len(triu_indices[0]) > 0:
            avg_correlation = float(np.mean(np.abs(corr_matrix[triu_indices])))
        else:
            avg_correlation = 0.0

        return corr_matrix, avg_correlation

    def _prepare_meta_input(self, blender_pred: np.ndarray) -> np.ndarray:
        """Prepare input for meta-regulator."""
        std = float(np.std(blender_pred))
        var = float(np.var(blender_pred))
        return np.concatenate([blender_pred, [std, var]]).reshape(1, -1)

    def _to_tensor(self, array: np.ndarray):  # pragma: no cover - simple wrapper
        """Convert numpy array to tensorflow tensor."""
        import tensorflow as tf

        return tf.constant(array, dtype=tf.float32)
