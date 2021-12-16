"""Define loss functions that can ignore a specific label"""

from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


class IgnorantSparseCatCrossentropy(SparseCategoricalCrossentropy):

    def __init__(self, mask_label: int = -1, **kwargs) -> None:
        self.mask_label = mask_label
        super().__init__(**kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_mod = tf.where(y_true == self.mask_label, 0, y_true)
        cross_entropy = super().call(y_mod, y_pred)
        masked_cross_entropy = tf.where(y_true == self.mask_label, 0.,
                                        cross_entropy)
        return tf.reduce_mean(masked_cross_entropy)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "mask_label": self.mask_label}


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
