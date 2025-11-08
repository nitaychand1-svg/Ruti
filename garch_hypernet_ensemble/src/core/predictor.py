"""Prediction logic with all safety mechanisms."""
import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from cachetools import TTLCache
import joblib

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
            maxsize=config.cache.maxsize
        )
        
        # [FIX-4] HyperNetwork trainer
        if hasattr(components, 'hypernetwork') and components.hypernetwork:
            from ..hypernetwork.trainer import GARCHHyperNetTrainer
            self.hypernet_trainer = GARCHHyperNetTrainer(
                components.hypernetwork,
                config.hypernet.learning_rate
            )
        else:
            self.hypernet_trainer = None
        
        # Performance tracking
        self.performance_tracker = None
    
    async def predict(self, X: np.ndarray,
                     market_regime: str = 'normal',
                     ground_truth: Optional[int] = None,
                     vix: Optional[float] = None) -> Dict[str, Any]:
        """[FIX-17] Main prediction method with full error handling."""
        try:
            # [FIX-12] Input validation
            X = DataValidationLayer.validate(X, context="predict_input")
            
            # [FIX-5] Circuit breaker
            current_vol = self._get_current_garch_volatility()
            if current_vol > 0.05 or market_regime == 'chaotic':
                self.logger.warning(
                    f"[FIX-5] Circuit breaker: vol={current_vol:.4f}, regime={market_regime}"
                )
                return self._emergency_fallback('extreme_volatility')
            
            # [FIX-5] Check active models
            active_models = self._get_active_models()
            if len(active_models) < 5:
                self.logger.warning(f"[FIX-5] Low active models: {len(active_models)}")
                return self._fallback_rule_based(X)
            
            # Get base predictions
            base_predictions, base_confidences, model_ids = await self._get_base_predictions(
                active_models, X
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
                diversity, avg_correlation, regime_features,
                garch_vol, garch_forecast, len(active_models)
            )
            
            # HyperNetwork prediction
            if hasattr(self.components, 'hypernetwork') and self.components.hypernetwork:
                tf_meta = self._to_tensor(meta_features)
                dynamic_weights, dynamic_thresholds = self.components.hypernetwork(
                    tf_meta, training=False
                )
                
                # Update config dynamically
                self.config.monitoring.max_correlation_threshold = float(dynamic_thresholds[0, 0])
                self.config.monitoring.min_confidence_threshold = float(dynamic_thresholds[0, 1])
                
                model_weights = dynamic_weights.numpy()[0]
                
                # [FIX-4] Update correlation matrix
                self.components.hypernetwork.update_correlation_matrix(corr_matrix)
            else:
                model_weights = np.ones(len(active_models)) / len(active_models)
                dynamic_thresholds = np.array([[0.85, 0.55, 0.10]])
            
            # Blender prediction
            blender_input = self._prepare_blender_input(
                base_predictions, X, current_vol
            )
            blender_pred = self.components.blender_model.predict_proba(blender_input)[0]
            
            # Meta-regulator
            meta_input = self._prepare_meta_input(
                blender_pred, regime_features, diversity,
                base_predictions, base_confidences
            )
            final_pred = self.components.meta_regulator.predict_proba(meta_input)[0]
            
            # [FIX-8] Regime calibration
            calibrated_pred = self._regime_specific_calibration(
                final_pred, market_regime
            )
            
            # [FIX-16] Position sizing
            position_size = self._calculate_position_size(
                calibrated_pred, base_predictions
            )
            
            # [FIX-19] Effective bets
            effective_bets = self._calculate_effective_bets(
                model_weights, corr_matrix
            )
            
            # Build result
            result = {
                'prediction': calibrated_pred,
                'confidence': max(calibrated_pred),
                'base_predictions': base_predictions.tolist() if hasattr(base_predictions, 'tolist') else base_predictions,
                'model_weights': model_weights.tolist() if hasattr(model_weights, 'tolist') else model_weights,
                'diversity': diversity,
                'garch_volatility': garch_vol,
                'regime': market_regime,
                'active_models': len(active_models),
                'position_size': position_size,
                'cvar_95': self._calculate_cvar(base_predictions),
                'effective_bets': effective_bets,
                'emergency_mode': False
            }
            
            # [FIX-5] Online update
            if ground_truth is not None and self.hypernet_trainer:
                reward = float(np.argmax(calibrated_pred) == ground_truth)
                self.hypernet_trainer.update_online(
                    meta_features, reward, avg_correlation
                )
            
            # Cache result
            cache_key = joblib.hash(X.tobytes() + market_regime.encode())
            self.cache.set_sync(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.critical(f"[FIX-17] Prediction failed: {e}", exc_info=True)
            return self._emergency_fallback(str(e))
    
    def _get_active_models(self):
        """Get models above confidence threshold."""
        return [
            m for m in self.components.base_models
            if m.get('confidence', 0.0) >= self.config.monitoring.min_confidence_threshold
        ]
    
    async def _get_base_predictions(self, models: list, X: np.ndarray):
        """Get predictions from all active models."""
        predictions = []
        confidences = []
        model_ids = []
        
        for model_info in models:
            # [FIX-13] Check cache
            cache_key = f"{model_info['id']}_{joblib.hash(X.tobytes())}"
            cached = self.cache.get_sync(cache_key)
            
            if cached:
                pred = cached['prediction']
                conf = cached['confidence']
            else:
                result = await self._predict_single_model(model_info, X)
                pred = result['prediction']
                conf = result['confidence']
                
                # Cache for future use
                self.cache.set_sync(cache_key, result)
            
            predictions.append(pred)
            confidences.append(conf)
            model_ids.append(model_info['id'])
        
        return np.array(predictions), confidences, model_ids
    
    async def _predict_single_model(self, model_info: dict, X: np.ndarray):
        """Predict with single model."""
        model = model_info['model']
        
        try:
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[0]
                prediction = prob
                confidence = max(prob)
            else:
                pred = model.predict(X)[0]
                prediction = np.array([1-pred, pred])
                confidence = abs(prediction[0] - 0.5) * 2
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'model_id': model_info['id']
            }
        except Exception as e:
            self.logger.error(f"[FIX-17] Model {model_info['id']} failed: {e}")
            return {
                'prediction': np.array([0.5, 0.5]),
                'confidence': 0.0,
                'model_id': model_info['id']
            }
    
    def _emergency_fallback(self, reason: str) -> Dict[str, Any]:
        """[FIX-17] Safe fallback response."""
        return {
            'prediction': np.array([0.5, 0.5]),
            'confidence': 0.0,
            'emergency_mode': True,
            'reason': f'exception_{reason[:50]}',
            'position_size': 0.0,
            'active_models': 0
        }
    
    def _fallback_rule_based(self, X: np.ndarray) -> Dict[str, Any]:
        """[FIX-5] Rule-based fallback."""
        ma_short = np.mean(X[-5:, 0]) if X.ndim > 1 else np.mean(X[-5:])
        ma_long = np.mean(X[-20:, 0]) if X.ndim > 1 else np.mean(X[-20:])
        signal = 1 if ma_short > ma_long else 0
        
        return {
            'prediction': np.array([1-signal, signal]),
            'confidence': 0.3,
            'emergency_mode': True,
            'reason': 'low_active_models',
            'position_size': 0.1,
            'active_models': 0
        }
    
    def _prepare_meta_features(self, diversity: float, avg_correlation: float,
                             regime_features: np.ndarray, garch_vol: float,
                             garch_forecast: float, n_active: int):
        """[FIX-4] Prepare meta-features for HyperNetwork."""
        recent_accuracy = 0.5
        if self.performance_tracker:
            recent_accuracy = np.mean(list(self.performance_tracker.accuracy_window)) or 0.5
        
        return np.array([
            diversity,
            avg_correlation,
            regime_features[2],  # High volatility membership
            regime_features[4] if len(regime_features) > 4 else 0.0,  # Chaotic membership
            garch_vol,
            garch_forecast,
            n_active / len(self.components.base_models),
            recent_accuracy
        ]).reshape(1, -1)
    
    def _encode_regime_continuous(self, X: np.ndarray, vol: float):
        """[FIX-8] Continuous regime encoding."""
        prices = X[:, 0] if X.ndim > 1 else X
        returns = np.diff(prices)
        
        if len(returns) == 0:
            returns = np.array([0.0])
        
        vol_percentile = np.clip(
            vol / (np.percentile(np.abs(returns), 95) + 1e-8), 0, 1
        )
        trend_strength = np.abs(np.mean(returns[-20:])) / (np.std(returns) + 1e-8)
        momentum = np.mean(prices[-10:]) - np.mean(prices[-50:])
        
        # Fuzzy membership functions
        low_vol = max(0, 1 - vol_percentile * 2)
        transition = max(0, 1 - abs(vol_percentile - 0.5) * 2)
        high_vol = max(0, vol_percentile - 0.7)
        chaotic = max(0, vol_percentile - 0.9) * trend_strength
        
        memberships = np.array([low_vol, transition, high_vol, chaotic])
        memberships /= (memberships.sum() + 1e-8)
        
        return np.concatenate([memberships, [trend_strength, momentum, vol_percentile]])
    
    def _regime_specific_calibration(self, prediction: np.ndarray, regime: str):
        """[FIX-8] Calibrate based on market regime."""
        calibrators = {
            'low_volatility_bull': lambda p: np.array([
                min(p[0] * 1.15, 0.95), max(p[1] * 0.85, 0.05)
            ]),
            'low_volatility_bear': lambda p: np.array([
                max(p[0] * 0.85, 0.05), min(p[1] * 1.15, 0.95)
            ]),
            'high_volatility': lambda p: np.array([
                p[0] * 0.9 + 0.05, p[1] * 0.9 + 0.05
            ]),
            'transition': lambda p: np.array([0.45, 0.45])
        }
        
        return calibrators.get(regime, lambda p: p)(prediction)
    
    def _calculate_position_size(self, ensemble_pred: np.ndarray, 
                               base_predictions: np.ndarray) -> float:
        """[FIX-16] Kelly criterion + VaR position sizing."""
        p_up = ensemble_pred[1]
        edge = abs(p_up - 0.5) * 2
        
        # Kelly
        odds = (p_up / (1 - p_up + 1e-8)) if p_up > 0.5 else ((1 - p_up) / (p_up + 1e-8))
        kelly_fraction = (edge * odds - (1 - edge)) / (odds + 1e-8) if odds > 0 else 0
        kelly_fraction = np.clip(kelly_fraction, 0, self.config.risk.max_position_size)
        
        # VaR limit
        cvar = self._calculate_cvar(base_predictions)
        var_limit = self.config.risk.max_risk_per_trade / (abs(cvar) + 1e-6)
        
        return float(min(kelly_fraction, var_limit))
    
    def _calculate_cvar(self, predictions: np.ndarray) -> float:
        """[FIX-16] Conditional Value at Risk (95%)."""
        flat_preds = predictions.flatten()
        return float(np.percentile(flat_preds, self.config.risk.cvar_percentile))
    
    def _calculate_effective_bets(self, weights: np.ndarray, 
                                corr_matrix: np.ndarray) -> float:
        """[FIX-19] Effective number of independent bets."""
        div_ratio = np.sqrt(np.dot(weights.T, np.dot(corr_matrix, weights)))
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        return float(np.exp(entropy) / (div_ratio + 1e-6))
    
    def _get_current_garch_volatility(self) -> float:
        """Get current GARCH volatility estimate."""
        return self.components.garch_tracker.get_all_volatilities()
    
    def _get_garch_forecast(self) -> float:
        """Get GARCH volatility forecast."""
        if not self.components.feature_names:
            return self._get_current_garch_volatility()
        
        forecast = self.components.garch_tracker.forecast_volatility(
            self.components.feature_names[0], steps=5
        )
        return forecast if forecast else self._get_current_garch_volatility()
    
    def _calculate_ensemble_diversity(self, predictions: np.ndarray) -> float:
        """[FIX-5] Spearman correlation-based diversity."""
        from scipy.stats import spearmanr
        
        if predictions.ndim == 1:
            return 0.0
            
        n_models = predictions.shape[0]
        if n_models < 2:
            return 0.0
        
        correlations = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                try:
                    corr, _ = spearmanr(predictions[i], predictions[j])
                    correlations.append(abs(corr))
                except:
                    correlations.append(0.0)
        
        return 1.0 - np.mean(correlations) if correlations else 0.0
    
    def _calculate_correlation(self, predictions: np.ndarray):
        """Calculate correlation matrix from predictions."""
        if predictions.ndim == 1 or predictions.shape[0] < 2:
            return np.array([[1.0]]), 0.0
        
        corr_matrix = np.corrcoef(predictions)
        
        # Upper triangle indices
        triu_indices = np.triu_indices_from(corr_matrix, k=1)
        if len(triu_indices[0]) > 0:
            avg_correlation = np.mean(np.abs(corr_matrix[triu_indices]))
        else:
            avg_correlation = 0.0
        
        return corr_matrix, avg_correlation
    
    def _prepare_blender_input(self, base_preds: np.ndarray, 
                             X: np.ndarray, vol: float):
        """[FIX-2] Prepare blender input with statistics."""
        if base_preds.ndim == 1:
            base_preds = base_preds.reshape(-1, 1)
        
        # Ensure we have at least 2D predictions
        if base_preds.shape[0] == 1:
            base_preds = base_preds.T
            
        stats = np.array([
            np.mean(base_preds),
            np.median(base_preds),
            np.std(base_preds),
            np.percentile(base_preds, 25),
            np.percentile(base_preds, 75),
            np.percentile(base_preds, 95),
            np.percentile(base_preds, 5),
        ]).reshape(1, -1)
        
        features = X[-1].reshape(1, -1) if X.ndim > 1 else X.reshape(1, -1)
        
        # Take first 10 predictions or pad if less
        first_preds = base_preds[:10].flatten()
        if len(first_preds) < 10:
            first_preds = np.pad(first_preds, (0, 10 - len(first_preds)), mode='constant')
        first_preds = first_preds.reshape(1, -1)
        
        return np.column_stack([features, stats, first_preds, [[vol]]])
    
    def _prepare_meta_input(self, blender_pred: np.ndarray,
                          regime_features: np.ndarray,
                          diversity: float,
                          base_predictions: np.ndarray,
                          confidences: list):
        """Prepare input for meta-regulator."""
        tail_risk = self._calculate_cvar(base_predictions)
        
        # Take first 4 regime features
        regime_feat = regime_features[:4]
        
        # Pad confidences to fixed size
        conf_array = np.array(confidences[:10])
        if len(conf_array) < 10:
            conf_array = np.pad(conf_array, (0, 10 - len(conf_array)), mode='constant')
        
        return np.concatenate([
            blender_pred,
            regime_feat,
            [diversity, tail_risk],
            conf_array
        ]).reshape(1, -1)
    
    def _to_tensor(self, array: np.ndarray):
        """Convert numpy array to tensorflow tensor."""
        import tensorflow as tf
        return tf.constant(array, dtype=tf.float32)
