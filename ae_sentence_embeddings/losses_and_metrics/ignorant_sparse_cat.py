"""Define loss functions that can ignore a specific label"""

from typing import Dict, Any, Union

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import Reduction
from tensorflow.keras.metrics import SparseCategoricalAccuracy


class IgnorantSparseCatCrossentropy(tf.keras.losses.Loss):

    def __init__(
            self,
            mask_label: int = -1,
            reduction: Union[Reduction.SUM, Reduction.SUM_OVER_BATCH_SIZE] = Reduction.SUM_OVER_BATCH_SIZE,
            from_logits: bool = False,
            **kwargs
    ) -> None:
        """Initialize the loss object

        Args:
            mask_label: The label that will be ignored. Defaults to -1
            reduction: Reduction type, `SUM` or `SUM_OVER_BATCH_SIZE`.
                       Defaults to `SUM_OVER_BATCH_SIZE`, which means averaging over the batch
            from_logits: Specifies whether the inputs are logits. Defaults to `False`
            **kwargs: Additional keyword arguments for the parent class
        """
        if reduction not in {Reduction.SUM, Reduction.SUM_OVER_BATCH_SIZE}:
            raise NotImplementedError("Reduction should be either Reduction.SUM or Reduction.SUM_OVER_BATCH_SIZE")
        self.mask_label = mask_label
        self._base_loss = SparseCategoricalCrossentropy(reduction=Reduction.NONE, from_logits=from_logits)
        super().__init__(**kwargs, reduction=reduction)
        self._reduction_func = tf.reduce_sum if self.reduction == Reduction.SUM else tf.reduce_mean

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_mod = tf.where(y_true == self.mask_label, 0, y_true)
        cross_entropy = self._base_loss(y_mod, y_pred)
        masked_cross_entropy = tf.where(y_true == self.mask_label, 0.,
                                        cross_entropy)
        return self._reduction_func(masked_cross_entropy)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "mask_label": self.mask_label, "reduction": self.reduction}


class IgnorantSparseCatAccuracy(SparseCategoricalAccuracy):

    def __init__(self, mask_label: int = -1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mask_label = mask_label

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> None:
        """Use sample weight as mask"""
        sample_weight = tf.where(y_true == self.mask_label, 0., 1.)
        super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "mask_label": self.mask_label}
