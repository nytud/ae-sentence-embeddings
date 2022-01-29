"""Define layers that support bilingual training"""

from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.backend import random_bernoulli

TensorPairs = Tuple[Tuple[tf.Tensor, tf.Tensor], ...]


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

    def call(self, inputs: TensorPairs, training: Optional[bool] = None) -> TensorPairs:
        """Call the layer

        Args:
            inputs: A tuple of tensor pairs, where the pairs are tuples.
                    The order of the tensors in will either be interchanged in all pairs
                    or not interchanged in any of the pairs
            training: Specify whether the layer is being used in training mode. If not,
                      the inputs will not be swapped

        Returns:
            The same tensors as the ones in the input but their order may be interchanged in the pairs
        """
        if training and random_bernoulli(shape=(), p=self.p) > 0:
            outputs = []
            for pair in inputs:
                outputs.append(pair[::-1])
            outputs = tuple(outputs)
        else:
            outputs = inputs
        return outputs

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "p": self.p
        }
