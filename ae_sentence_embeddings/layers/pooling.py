"""Define layer s for pooling from Transformer outputs"""

from typing import Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers as tfl


def _filter_mask_vectors(token_embeddings: tf.Tensor,
                         attn_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Helper function to zero all mask token vectors.

    Args:
        token_embeddings: The token embedding vector of shape
            `(batch_size, sequence_length, hidden_size)`.
        attn_mask: The attention mask vector of shape
            `(batch_size, sequence_length)`.

    Returns:
        The token embedding tensor with the mask token vectors set to zero and
        a tensor that indicates the number of non-mask tokens in the input.
    """
    expanded_mask = tf.tile(tf.expand_dims(attn_mask, -1),
                            tf.constant([1, 1, token_embeddings.shape[-1]]))
    masked_token_embeddings = tf.where(expanded_mask == 1, token_embeddings, 0.)
    attn_mask = tf.expand_dims(tf.reduce_sum(attn_mask, axis=1), axis=1)
    return masked_token_embeddings, tf.cast(attn_mask, masked_token_embeddings.dtype)


class AveragePoolingLayer(tfl.Layer):
    """A layer for average pooling that takes padding token embedding into account"""

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Do average pooling

        Args:
            inputs: A tuple of two tensors. The first is a 3D hidden state tensor,
                    the second is a 2D attention mask tensor

        Returns:
            A 2D pooled tensor
        """
        hidden_state, attention_mask = inputs
        masked_hidden_state, attention_mask = _filter_mask_vectors(
            hidden_state, attention_mask)
        return tf.divide(tf.reduce_sum(masked_hidden_state, axis=1), attention_mask)


class CLSPlusSEPPooling(tfl.Layer):
    """A layer for CLS + SEP pooling"""

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Do CLS + SEP pooling

        Args:
            inputs: A tuple of two tensors. The first is a 3D hidden state tensor of shape
                `(batch_size, sequence_length, hidden_size)` and the second is a 2D attention
                mask tensor of shape `(batch_size, sequence_length)`.

        Returns:
            A 2D pooled tensor of shape `(batch_size, 2*hidden_size)`
        """
        hidden_state, attn_mask = inputs
        sent_indices = tf.range(tf.shape(attn_mask)[0])
        sep_indices = tf.reduce_sum(attn_mask, axis=-1) - 1
        indices = tf.stack([sent_indices, sep_indices], axis=1)
        sep_rows = tf.gather_nd(hidden_state, indices)
        pooled_tensor = tf.concat([hidden_state[:, 0, :], sep_rows], axis=1)
        return pooled_tensor


class PMeansPooling(tfl.Layer):
    """A layer that implements p means pooling."""

    def __init__(self, n_means: int = 2, **kwargs) -> None:
        """Initialize the layer.

        Args:
            n_means: The highest power to calculate, `n_means` >= 1. Defaults to `2`.
            **kwargs: Parent class initializer arguments.
        """
        if n_means < 1:
            raise ValueError(f"`n_means` must be a positive integer, got {n_means}")
        super().__init__(**kwargs)
        self._n_means = n_means

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Call the layer.

        Args:
            inputs: A tuple of two tensors. The first is a 3D hidden state tensor of shape
                `(batch_size, sequence_length, hidden_size)` and the second is a 2D attention
                mask tensor of shape `(batch_size, sequence_length)`.

        Returns:
            A pooled power means tensor of shape `(batch_size, n_means * hidden_size)`.
        """
        hidden_state, attention_mask = inputs
        masked_hidden_state, attention_mask = _filter_mask_vectors(
            hidden_state, attention_mask)
        p_means = []
        for i in range(1, self._n_means+1):
            ith_mean = tf.divide(
                tf.reduce_sum(tf.pow(masked_hidden_state, i), axis=1),
                attention_mask)
            p_means.append(tf.pow(ith_mean, 1/i))
        return tf.concat(p_means, axis=1)

    @property
    def n_means(self) -> int:
        return self._n_means

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "n_means": self._n_means}
