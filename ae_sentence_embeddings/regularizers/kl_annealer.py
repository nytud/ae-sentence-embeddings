# -*- coding: utf-8 -*-

"""Define a KL regularizer with beta annealing:
the KL beta will be annealed linearly between
arbitrarily adjustable steps `k` and `l`.
"""

from typing import Dict, Union

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.python.keras.regularizers import Regularizer


def calculate_beta(
        iters: Union[int, tf.Variable],
        warmup_iters: int,
        start: int
) -> tf.Tensor:
    """Calculate the actual KL loss beta.

    Args:
        iters: The actual iteration.
        warmup_iters: The number of warmup steps.
        start: The iteration when the warmup starts.

    Returns:
        The actual beta value.
    """
    iteration = tf.cond(iters > start, lambda: tf.cast(iters - start, tf.float32), lambda: 0.)
    return tf.minimum(tf.divide(iteration, tf.cast(warmup_iters, tf.float32)), 1.)


@register_keras_serializable(package="ae_sentence_embeddings")
class KLDivergenceRegularizer(Regularizer):
    """A KL regularizer with beta annealing. Based on
    https://stackoverflow.com/questions/62211309
    """
    def __init__(
            self,
            iters: Union[int, tf.Variable],
            warmup_iters: int,
            start: int = 0,
            min_kl: float = 0.,
    ) -> None:
        """Initialize the regularizer.

        Args:
            iters: The actual iteration. It should be set to `optimizer.iterations`.
                When the model is serialized, the constant value `0` will be passed!
            warmup_iters: The number of warmup steps.
            start: The iteration when the warmup starts. Defaults to `0`.
            min_kl: Minimal KL loss value per dimension. It takes effect only when `beta == 1`.
                Defaults to `0.0`.
        """
        super(KLDivergenceRegularizer, self).__init__()
        self._iters = iters
        self._warmup_iters = warmup_iters
        self._start = start
        self._min_kl = min_kl

    def __call__(self, activation: tf.Tensor) -> tf.Tensor:
        """Calculate the actual KL loss."""
        mu, log_var = tf.split(activation, 2, axis=-1)
        beta = tf.cast(
            calculate_beta(iters=self._iters, start=self._start, warmup_iters=self._warmup_iters),
            dtype=mu.dtype
        )
        min_kl = tf.cond(beta == 1., lambda: self._min_kl, lambda: 0.)
        # noinspection PyTypeChecker
        kl_loss = -0.5 * beta * (1 + log_var - tf.square(mu) - tf.exp(log_var))
        return tf.reduce_sum(tf.maximum(min_kl, kl_loss))

    def get_config(self) -> Dict[str, int]:
        """Serialize the regularizer.
        Loading a serialized model will be possible, but the regularization effect
        will be lost. The reason for this is that the `__init__` parameter `iters`
        will be filled in with the constant value `0`.
        """
        return {
            "iters": 0,
            "warmup_iters": self._warmup_iters,
            "start": self._start,
            "min_kl": self._min_kl
        }
