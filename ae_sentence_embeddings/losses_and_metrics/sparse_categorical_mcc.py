# -*- coding: utf-8 -*-

"""A Matthews Correlation Coefficient implementation with scalar `y_true` labels"""

from typing import Optional

import tensorflow as tf
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient as MCCoefficient


class SparseCategoricalMCC(MCCoefficient):
    """A class that implements the Matthews Correlation Coefficient
    with scalar `y_true` labels.
    """

    def __init__(self, num_classes: int, name: str = "sparse_categorical_mcc", **kwargs) -> None:
        """Initialize the metric.

        Args:
            num_classes: Number of classes (or labels).
            name: Name of the metric instance. Defaults to `'sparse_categorical_mcc'`.
            **kwargs: Parent class keyword arguments.
        """
        if num_classes <= 0:
            raise ValueError(f"`num_classes` must be a positive integer, got {num_classes}")
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update the metric state.

        Args:
            y_true: The ground truth labels, an integer tensor of shape `(batch_size, 1)`.
            y_pred: The predictions, a float tensor of shape `(batch_size, num_classes)`.
            sample_weight: Optional. Weights of the data points.
        """
        y_true = tf.squeeze(y_true)
        y_true = tf.ensure_shape(y_true, (None,))
        y_true = tf.one_hot(y_true, depth=self.num_classes, dtype=y_pred.dtype)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)
