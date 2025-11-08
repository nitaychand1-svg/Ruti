"""HyperNetwork training logic."""
from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np
import tensorflow as tf


class GARCHHyperNetTrainer:
    """[FIX-4] Online trainer for HyperNetwork."""

    def __init__(self, hypernetwork: "AdaptiveHyperNetwork", learning_rate: float = 0.001):
        self.hypernetwork = hypernetwork
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.buffer_meta: Deque[np.ndarray] = deque(maxlen=1000)
        self.buffer_reward: Deque[float] = deque(maxlen=1000)
        self.buffer_correlation: Deque[float] = deque(maxlen=1000)

    def update_online(
        self, meta_features: np.ndarray, reward: float, correlation_spike: float
    ) -> None:
        """[FIX-4] Online update with correlation penalty."""
        self.buffer_meta.append(meta_features.flatten())
        self.buffer_reward.append(float(reward))
        self.buffer_correlation.append(float(correlation_spike))

        if len(self.buffer_meta) >= 20:
            self._train_batch()

    def _train_batch(self) -> None:
        """Train on accumulated batch."""
        X_batch = np.array(self.buffer_meta, dtype=float)
        y_batch = np.array(self.buffer_reward, dtype=float)

        if not np.isfinite(X_batch).all() or not np.isfinite(y_batch).all():
            return

        with tf.GradientTape() as tape:
            pred_weights, _ = self.hypernetwork(X_batch, training=True)
            reward_pred = tf.reduce_mean(pred_weights, axis=1, keepdims=True)
            loss = self.loss_fn(y_batch.reshape(-1, 1), reward_pred)
            loss += sum(self.hypernetwork.losses)

            avg_correlation = float(np.mean(self.buffer_correlation)) if self.buffer_correlation else 0.0
            if avg_correlation > 0.8:
                loss += 0.1 * avg_correlation

        grads = tape.gradient(loss, self.hypernetwork.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        self.optimizer.apply_gradients(
            [(g, v) for g, v in zip(grads, self.hypernetwork.trainable_variables) if g is not None]
        )

        self.buffer_meta.clear()
        self.buffer_reward.clear()
        self.buffer_correlation.clear()
