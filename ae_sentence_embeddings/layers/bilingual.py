"""Define layers that support bilingual training"""

from typing import Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.backend import random_bernoulli


class RandomSwapLayer(tfl.Layer):
    """This layer randomly swaps two inputs or leaves their order unaltered"""

    def __init__(self, p: float = 0.5, **kwargs) -> None:
        """Initialize an input swapping layer

        Args:
            p: The probability that the inputs will be swapped. Defaults to 0.5
            **kwargs: Parent class keyword arguments
        """
        super().__init__(**kwargs)
        if p > 1 or p < 0:
            raise ValueError("`p` should be a floating point number in the interval `[0, 1]`")
        self.p = p

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Call the layer

        Args:
            inputs: A tuple of two tensors `t1` and `t2`

        Returns:
            A tuple of two tensors, either `(t1, t2)` or `(t2, t1)`
        """
        t1, t2 = inputs
        outputs = (t2, t1) if random_bernoulli(shape=(), p=self.p) > 0 else inputs
        return outputs

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "p": self.p
        }
