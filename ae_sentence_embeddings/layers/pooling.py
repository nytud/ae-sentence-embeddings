"""Define layer s for pooling from Transformer outputs"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers as tfl


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
        expanded_mask = tf.tile(tf.expand_dims(attention_mask, -1),
                                tf.constant([1, 1, hidden_state.shape[-1]]))
        masked_hidden_state = tf.where(expanded_mask == 1, hidden_state, 0.)
        attention_mask = tf.expand_dims(tf.reduce_sum(attention_mask, axis=1), axis=1)
        return tf.divide(tf.reduce_sum(masked_hidden_state, axis=1),
                         tf.cast(attention_mask, tf.float32))


class CLSPlusSEPPooling(tfl.Layer):
    """A layer for CLS + SEP pooling"""

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Do CLS + SEP pooling

        Args:
            inputs: A tuple of two tensors. The first is a 3D hidden state tensor of shape
                    `(batch_size, sequence_length, hidden_size)` and the second is a 2D attention mask tensor
                    of shape `(batch_size, sequence_length)`

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
