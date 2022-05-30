# -*- coding: utf-8 -*-

"""Define a KL regularizer with beta annealing:
the KL beta will be annealed linearly between
arbitrarily adjustable steps `k` and `l`.
"""

from typing import Tuple, Dict, Union

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.python.keras.regularizers import Regularizer


@register_keras_serializable(package="ae_sentence_embeddings")
class KLDivergenceRegularizer(Regularizer):
    """A KL regularizer with beta annealing. Based on
    https://stackoverflow.com/questions/62211309
    """
    def __init__(self, iters: Union[int, tf.Variable], start: int, warmup_iters: int) -> None:
        """Initialize the regularizer.

        Args:
            iters: The actual iteration. It should be set to `optimizer.iterations`.
                When the model is serialized, the constant value `0` will be passed!
            start: The iteration when the warmup starts.
            warmup_iters: The iteration when beta equals to `1` and the warmup stops.
        """
        assert start < warmup_iters, f"`start` must be smaller than `warmup_iters`."
        super(KLDivergenceRegularizer, self).__init__()
        self._iters = iters
        self._start = start
        self._warmup_iters = warmup_iters

    def __call__(self, activation: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Calculate the actual KL loss."""
        mu, log_var = activation
        iteration = tf.cond(self._iters >= self._start, lambda: self._iters, lambda: 0.)
        beta = tf.minimum(iteration / self._warmup_iters, 1.)
        # noinspection PyTypeChecker
        return -0.5 * beta * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var))

    def get_config(self) -> Dict[str, int]:
        """Serialize the regularizer.
        Loading a serialized model will be possible, but the regularization effect
        will be lost. The reason for this is that the `__init__` parameter `iters`
        will be filled in with the constant value `0`.
        """
        base_config = super(KLDivergenceRegularizer, self).get_config()
        return {
            **base_config,
            "iters": 0,
            "start": self._start,
            "warmup_iters": self._warmup_iters
        }
