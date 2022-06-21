# -*- coding: utf-8 -*-

"""Define loss functions that can ignore a specific label"""

from typing import Dict, Any, Literal, Optional

import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.losses import Reduction
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.utils import register_keras_serializable

ReductionMode = Literal["sum", "sum_over_batch_size", "average"]


def _average_and_sum(loss_tensor: tf.Tensor) -> tf.Tensor:
    """A helper function for implementing averaging over `batch_size` and summing over the `sequence_length` dimension

    Args:
        loss_tensor: A tensor of loss values

    Returns:
        A scalar tensor obtained by averaging over the 0th (batch) axis of `loss_tensor`
        and summing over the last (sequence length) axis
    """
    return tf.reduce_mean(tf.reduce_sum(loss_tensor, axis=-1), axis=0)


@register_keras_serializable(package="ae_sentence_embeddings.losses_and_metrics")
class IgnorantSparseCatCrossentropy(tf.keras.losses.Loss):
    """Sparse categorical crossentropy that can handle mask values"""

    def __init__(
            self,
            mask_label: int = -1,
            reduction_mode: ReductionMode = "sum_over_batch_size",
            from_logits: bool = False,
            factor: float = 1.0,
            **kwargs
    ) -> None:
        """Initialize the loss object

        Args:
            mask_label: The label that will be ignored. Defaults to `-1`.
            reduction: Reduction mode, `"sum"` or `"sum_over_batch_size"`. Please set this argument
                instead of the parent class `reduction` argument. An attempt to set `reduction` will
                lead to an exception. Defaults to `"sum_over_batch_size"`, which indicates averaging
                over the batch dimension.
            from_logits: Specifies whether the inputs are logits. Defaults to `False`.
            factor: A normalizing constant by which the loss value will be multiplied. Defaults to `1.0`.
            **kwargs: Additional keyword arguments for the parent class.
        """
        if reduction_mode == "sum":
            self._reduction_func = tf.reduce_sum
        elif reduction_mode == "sum_over_batch_size":
            self._reduction_func = _average_and_sum
        else:
            raise ValueError(f"Unknown reduction mode: {reduction_mode}")
        self._reduction_mode = reduction_mode
        if kwargs.get("reduction") is not None:
            raise TypeError(f"IgnorantSparseCatCrossentropy got an "
                            f"unexpected keyword argument 'reduction'")
        super().__init__(**kwargs, reduction=Reduction.NONE)
        self.mask_label = mask_label
        self._from_logits = from_logits
        self.factor = factor

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_mod = tf.where(y_true == self.mask_label, 0, y_true)
        cross_entropy = sparse_categorical_crossentropy(y_mod, y_pred, from_logits=self._from_logits)
        masked_cross_entropy = tf.where(y_true == self.mask_label, 0., cross_entropy)
        return tf.multiply(self.factor, self._reduction_func(masked_cross_entropy))

    @property
    def reduction_mode(self) -> str:
        return self._reduction_mode

    @property
    def from_logits(self) -> bool:
        return self._from_logits

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        base_config.pop("reduction")
        return {
            **base_config,
            "mask_label": self.mask_label,
            "reduction_mode": self.reduction_mode,
            "from_logits": self.from_logits,
            "factor": self.factor
        }


@register_keras_serializable(package="ae_sentence_embeddings.losses_and_metrics")
class IgnorantSparseCatAccuracy(SparseCategoricalAccuracy):
    """Sparse categorical accuracy that can handle mask labels"""

    def __init__(self, mask_label: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_label = mask_label

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> None:
        """Use sample weight as mask"""
        sample_weight = tf.where(y_true == self.mask_label, 0., 1.)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "mask_label": self.mask_label}
