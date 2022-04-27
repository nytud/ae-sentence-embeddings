# -*- coding: utf-8 -*-

"""A module for regularization losses such as KL-divergence"""

import tensorflow as tf


def kl_loss_func(mean: tf.Tensor, log_var: tf.Tensor, min_kl: float = 0.0) -> tf.Tensor:
    """Calculate VAE latent loss.

    Args:
        mean: The Gaussian mean vector.
        log_var: The Gaussian variance vector.
        min_kl: Minimal KL loss value per dimension. Defaults to `0.0`.
    """
    min_kl = -2 * min_kl
    latent_loss_value = -0.5 * tf.reduce_sum(
        tf.minimum(min_kl, 1 + (log_var - tf.square(mean) - tf.exp(log_var))), axis=-1)
    return tf.reduce_mean(latent_loss_value)
