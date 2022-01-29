"""Define layers that support bilingual training"""

from typing import Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.backend import random_bernoulli

TensorQuad = Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]


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

    def call(self, inputs: TensorQuad) -> TensorQuad:
        """Call the layer

        Args:
            inputs: A nested structure of 4 tensors: two tuples, each of which consists of two tensors.
                    The order of the tensors in the first pair is interchanged in the output if and only
                    if the tensors of the second pair are swapped as well

        Returns:
            The same tensors as the ones in the input but their order may be interchanged in the nested pairs
        """
        pair1, pair2 = inputs
        if random_bernoulli(shape=(), p=self.p) > 0:
            pair1 = pair1[::-1]
            pair2 = pair2[::-1]
            outputs = (pair1, pair2)
        else:
            outputs = inputs
        return outputs

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "p": self.p
        }
