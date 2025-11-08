"""Adaptive HyperNetwork implementation."""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.regularizers import l2


class AdaptiveHyperNetwork(Model):
    """[FIX-4] HyperNetwork with GARCH meta-features."""

    def __init__(self, meta_dim: int, n_models: int, hidden_dims: list, dropout_rate: float):
        super().__init__()
        self.meta_dim = meta_dim
        self.n_models = n_models
        self.correlation_matrix = np.eye(n_models)

        self.hidden_blocks = []
        for i, dim in enumerate(hidden_dims):
            self.hidden_blocks.extend(
                [
                    layers.Dense(dim, kernel_regularizer=l2(0.01), name=f"hyper_dense_{i}"),
                    layers.BatchNormalization(name=f"hyper_bn_{i}"),
                    layers.LeakyReLU(alpha=0.1, name=f"hyper_leaky_{i}"),
                    layers.Dropout(dropout_rate, name=f"hyper_dropout_{i}"),
                ]
            )

        self.weight_generator = layers.Dense(n_models, activation=None, name="weight_logits")
        self.threshold_generator = layers.Dense(3, activation="sigmoid", name="threshold_gen")

    def call(self, meta_features, training: bool = False):  # type: ignore[override]
        """Forward pass with regularization."""
        x = meta_features
        for layer in self.hidden_blocks:
            x = layer(x, training=training)

        weight_logits = self.weight_generator(x)
        weights = tf.nn.softmax(weight_logits, axis=-1)

        if training:
            entropy = -tf.reduce_sum(weights * tf.math.log(weights + 1e-8), axis=-1)
            self.add_loss(-0.01 * tf.reduce_mean(entropy))

            corr_matrix = tf.constant(self.correlation_matrix, dtype=tf.float32)
            weighted_corr = tf.matmul(weights, tf.matmul(corr_matrix, tf.transpose(weights)))
            corr_loss = tf.reduce_mean(tf.linalg.trace(weighted_corr))
            self.add_loss(0.05 * corr_loss)

        thresholds_norm = self.threshold_generator(x)
        thresholds = tf.stack(
            [
                0.50 + thresholds_norm[:, 0] * 0.35,
                0.50 + thresholds_norm[:, 1] * 0.20,
                0.01 + thresholds_norm[:, 2] * 0.09,
            ],
            axis=-1,
        )

        return weights, thresholds

    def update_correlation_matrix(self, corr_matrix: np.ndarray) -> None:
        """Update correlation matrix for regularization."""
        self.correlation_matrix = corr_matrix
