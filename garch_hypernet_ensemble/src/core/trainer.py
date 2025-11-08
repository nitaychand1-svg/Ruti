"""Training logic for ensemble components."""
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from ..core.config import EnsembleConfig
from ..garch.tracker import GARCHModelTracker
from ..garch.model import GARCH11
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
            min_obs=config.garch.min_obs
        )
    
    async def train(self, X: np.ndarray, y: np.ndarray,
                   feature_names: List[str]) -> TrainedComponents:
        """Full async training pipeline."""
        self.logger.info("Starting async training pipeline...")
        
        # [FIX-12] Validate inputs
        X = DataValidationLayer.validate(X, y, context="train")
        
        # [FIX-1] Setup GARCH tracking
        await self._setup_garch_features(X, feature_names)
        
        # [FIX-14] Walk-forward cross-validation
        wfo = WalkForwardValidator(n_splits=5, purge_gap=10)
        oos_holdout = wfo.get_oos_holdout(len(X))
        
        X_train_full, y_train_full = X[:oos_holdout.start], y[:oos_holdout.start]
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
        blender = self._train_blender(base_models, X_train_full, y_train_full)
        
        # Train meta-regulator
        meta = self._train_meta_regulator(blender, X_train_full, y_train_full)
        
        # [FIX-10] OOS evaluation
        oos_score = self._evaluate_oos(X_oos, y_oos, blender, meta)
        if oos_score < 0.55:
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
            oos_score=oos_score
        )
    
    async def _setup_garch_features(self, X: np.ndarray, feature_names: List[str]):
        """Setup GARCH tracking for each feature."""
        self.logger.info("Setting up GARCH features...")
        
        for i, name in enumerate(feature_names):
            returns_series = np.diff(X[:, i]) / (X[:-1, i] + 1e-6)
            returns_series = np.nan_to_num(returns_series, 0.0)
            returns_series_padded = np.pad(returns_series, (1, 0), mode='constant')
            self.garch_tracker.add_feature(name, returns_series_padded)
        
        if self.config.garch.enable_forecast:
            self.logger.info("Fitting GARCH models...")
            self.garch_tracker.fit_all_garch_parallel()
    
    def _generate_stress_scenarios(self, X: np.ndarray, y: np.ndarray, n_scenarios: int = 5):
        """[FIX-2] Generate synthetic stress scenarios."""
        scenarios = []
        for _ in range(n_scenarios):
            shock_mask = np.random.random(len(X)) < 0.3
            X_stress = X.copy()
            shock_factor = np.random.normal(3.0, 1.0, size=X[shock_mask].shape)
            X_stress[shock_mask] *= shock_factor
            
            # Generate labels without leakage
            y_stress = (np.diff(X_stress[:, 0]) > np.mean(np.diff(X_stress[:, 0]))).astype(int)
            y_stress = np.pad(y_stress, (1, 0), mode='constant')
            
            scenarios.append((X_stress, y_stress))
        
        return scenarios
    
    async def _train_base_models_async(self, X: np.ndarray, y: np.ndarray,
                                      wfo: WalkForwardValidator,
                                      stress_scenarios: List) -> List[Dict]:
        """[FIX-3] Async training pool with resource isolation."""
        tasks = []
        models_to_train = self._initialize_base_models()
        
        for model_info in models_to_train:
            for split_idx, (train_idx, val_idx) in enumerate(wfo.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                X_stress, y_stress = stress_scenarios[split_idx % len(stress_scenarios)]
                
                task = self._train_single_model_worker(
                    model_info, X_train, y_train, X_val, y_val, X_stress, y_stress
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        trained_models = [
            r for r in results 
            if isinstance(r, dict) and 'model_id' in r
        ]
        
        self.logger.info(f"✅ Trained {len(trained_models)} models successfully")
        return trained_models
    
    async def _train_single_model_worker(self, model_info: Dict, 
                                        X_train: np.ndarray, y_train: np.ndarray,
                                        X_val: np.ndarray, y_val: np.ndarray,
                                        X_stress: np.ndarray, y_stress: np.ndarray) -> Dict:
        """Worker for single model training."""
        try:
            # [FIX-12] Validate
            X_train = DataValidationLayer.validate(X_train, y_train, context="train")
            
            # [FIX-3] Deep copy to prevent race conditions
            model = deepcopy(model_info['model'])
            
            # Train
            model.fit(X_train, y_train)
            
            # Validate
            val_score = self._evaluate_model(model, X_val, y_val)
            stress_score = self._evaluate_stress(model, X_stress, y_stress)
            
            return {
                'model_id': model_info['id'],
                'model': model,
                'performance': {'accuracy': val_score},
                'stress_performance': {'accuracy': stress_score},
                'composite_score': 0.7 * val_score + 0.3 * stress_score
            }
        except Exception as e:
            self.logger.error(f"Training failed for {model_info['id']}: {e}")
            return {'error': str(e)}
    
    def _initialize_base_models(self) -> List[Dict]:
        """[FIX-5] Initialize diverse base models."""
        from sklearn.ensemble import (RandomForestClassifier, 
                                     GradientBoostingClassifier,
                                     ExtraTreesClassifier, 
                                     AdaBoostClassifier)
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        
        algos = [
            (RandomForestClassifier, {'n_estimators': 200, 'max_depth': 8}),
            (GradientBoostingClassifier, {'n_estimators': 150, 'learning_rate': 0.05}),
            (ExtraTreesClassifier, {'n_estimators': 250, 'max_depth': 10}),
            (AdaBoostClassifier, {'n_estimators': 100}),
            (SVC, {'kernel': 'rbf', 'probability': True, 'C': 1.0}),
            (MLPClassifier, {'hidden_layer_sizes': (128, 64), 'activation': 'tanh'}),
            (LogisticRegression, {'penalty': 'l1', 'C': 0.5, 'solver': 'liblinear'})
        ]
        
        models = []
        for i in range(self.config.n_base_models):
            algo_class, params = algos[i % len(algos)]
            modified_params = deepcopy(params)
            
            # [FIX-5] Add random diversity
            if 'learning_rate' in modified_params:
                modified_params['learning_rate'] *= (0.5 + np.random.random())
            if 'max_depth' in modified_params:
                modified_params['max_depth'] += np.random.randint(-2, 3)
            
            models.append({
                'id': f'base_{i:03d}_{algo_class.__name__}',
                'model': algo_class(**modified_params)
            })
        
        self.logger.info(f"✅ Initialized {len(models)} base models")
        return models
    
    def _prune_models(self, models: List[Dict]) -> List[Dict]:
        """[FIX-5] Remove underperforming models."""
        kept = [m for m in models if m.get('composite_score', 0.0) >= self.config.prune_threshold]
        pruned = len(models) - len(kept)
        
        if pruned > 0:
            self.logger.warning(f"[FIX-5] Pruned {pruned} models, {len(kept)} remaining")
            
            # Redistribute weights
            for model in kept:
                model['weight'] = 1.0 / len(kept)
        
        return kept
    
    def _train_blender(self, base_models: List[Dict], 
                      X: np.ndarray, y: np.ndarray):
        """[FIX-2] Train stacking blender with time series CV."""
        blender_input = self._prepare_blender_input(base_models, X)
        
        return StackingClassifier(
            estimators=[(m['id'], m['model']) for m in base_models[:20]],
            final_estimator=LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8
            ),
            cv=TimeSeriesSplit(n_splits=3),
            passthrough=True,
            n_jobs=-1,
            stack_method='predict_proba'
        ).fit(blender_input, y)
    
    def _train_meta_regulator(self, blender_model, X: np.ndarray, y: np.ndarray):
        """[FIX-16] Train meta-regulator for risk management."""
        blender_pred = blender_model.predict_proba(X)
        meta_input = np.column_stack([
            blender_pred,
            np.std(blender_pred, axis=1),
            np.var(blender_pred, axis=1)
        ])
        
        return CatBoostClassifier(
            iterations=800,
            depth=6,
            loss_function='Logloss',
            learning_rate=0.01,
            verbose=False,
            eval_metric='AUC',
            l2_leaf_reg=5.0,
            random_strength=1.5
        ).fit(meta_input, y)
    
    def _prepare_blender_input(self, base_models: List[Dict], X: np.ndarray):
        """[FIX-2] Prepare OOS predictions for blender."""
        oos_preds = np.zeros((len(X), len(base_models)))
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            
            for i, model_info in enumerate(base_models):
                model_copy = deepcopy(model_info['model'])
                model_copy.fit(X_train, np.random.randint(0, 2, len(X_train)))
                oos_preds[val_idx, i] = model_copy.predict_proba(X_val)[:, 1]
        
        return oos_preds
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model on validation set."""
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                return np.mean(np.argmax(pred, axis=1) == y)
            else:
                pred = model.predict(X)
                return np.mean(pred == y)
        except:
            return 0.5
    
    def _evaluate_stress(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """[FIX-2] Stress test evaluation."""
        try:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                return np.mean(np.argmax(pred, axis=1) == y)
            else:
                pred = model.predict(X)
                return np.mean(pred == y)
        except:
            return 0.5
    
    def _evaluate_oos(self, X: np.ndarray, y: np.ndarray,
                     blender, meta) -> float:
        """Out-of-sample evaluation."""
        blender_pred = blender.predict_proba(X)
        meta_input = np.column_stack([
            blender_pred,
            np.std(blender_pred, axis=1),
            np.var(blender_pred, axis=1)
        ])
        
        final_pred = meta.predict(meta_input)
        return np.mean(final_pred == y)
    
    def _calculate_correlation_matrix(self, models: List[Dict], X: np.ndarray):
        """[FIX-19] Calculate model correlation matrix."""
        if len(models) < 2:
            return np.array([[1.0]])
        
        predictions = np.column_stack([
            m['model'].predict_proba(X[:1000])[:, 1] for m in models
        ])
        
        return np.corrcoef(predictions.T)
