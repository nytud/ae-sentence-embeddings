"""A module for regularization losses such as KL-divergence"""
import tensorflow as tf


def kl_loss_func(mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
    """Calculate VAE latent loss

    Args:
        mean: The Gaussian mean vector
        log_var: The Gaussian variance vector
    """
    latent_loss_value = -0.5 * tf.reduce_sum(1 + (log_var - tf.square(mean) - tf.exp(log_var)), axis=-1)
    return tf.reduce_mean(latent_loss_value)