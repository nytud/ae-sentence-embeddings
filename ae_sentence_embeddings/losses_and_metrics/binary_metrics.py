# -*- coding: utf-8 -*-

"""Add metrics for binary classification tasks"""

from typing import Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient as MCCoefficient


class BinaryLogitAccuracy(BinaryAccuracy):
    """A binary accuracy metric implementation for the cases
    when inputs are logits.
    """

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update the metric state.

        Args:
            y_true: The ground truth labels, an integer tensor of shape `(batch_size, 1)`.
            y_pred: The predictions, a float tensor of shape `(batch_size, 1)`.
            sample_weight: Optional. Weights of the data points.
        """
        y_pred = tf.nn.sigmoid(y_pred)
        super().update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class BinaryMCC(MCCoefficient):
    """A class that implements the Matthews Correlation Coefficient
    with a single binary classification `y_true` label.
    """

    def __init__(self, name: str = "binary_mcc", **kwargs) -> None:
        """Initialize the metric.

        Args:
            name: Name of the metric instance. Defaults to `'binary_mcc'`.
            **kwargs: Parent class keyword arguments.
        """
        super().__init__(num_classes=2, name=name, **kwargs)

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update the metric state.

        Args:
            y_true: The ground truth labels, an integer tensor of shape `(batch_size, 1)`.
            y_pred: The predictions, a float tensor of shape `(batch_size, 1)`.
            sample_weight: Optional. Weights of the data points.
        """
        y_true = tf.one_hot(tf.squeeze(y_true), depth=self.num_classes, dtype=y_pred.dtype)
        y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.concat([1-y_pred, y_pred], axis=-1)
        super().update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:
        """Return a config dictionary for serialization."""
        base_config = super(BinaryMCC, self).get_config()
        base_config.pop("num_classes")
        return base_config
