"""A module for RNN decoder layers as an alternative of a GPT decoder"""

from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras import layers as tfl


class AeGruDecoder(tfl.Layer):
    """A GRU-based decoder"""

    def __init__(
            self,
            num_rnn_layers: int = 2,
            hidden_size: int = 768,
            layernorm_eps: float = 1e-12,
            dropout_rate: float = 0.1,
            **kwargs) -> None:
        """

        Args:
            num_rnn_layers: Number of RNN layers. Defaults to 2
            hidden_size: Hidden size in the RNN and dense layers. Defaults to 768
            layernorm_eps: Layer normalization epsilon parameter. Defaults to 1e-12
            dropout_rate: A dropout rate between 0 and 1. Defaults to 0.1
            **kwargs: Parent class keyword arguments
        """
        super().__init__(**kwargs)
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.layernorm_eps = layernorm_eps
        self.dropout_rate = dropout_rate
        self.rnn = [tfl.GRU(
            units=self.hidden_size,
            recurrent_dropout=self.dropout_rate,
            return_sequences=True,
            return_state=True
        ) for _ in range(self.num_rnn_layers)]
        self.intermediate_dense = tfl.Dense(
            units=self.hidden_size*2,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            input_shape=(None, None, self.hidden_size)
        )
        self.out_dense = tfl.Dense(
            units=self.hidden_size,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            input_shape=(None, None, self.hidden_size*2)
        )
        self.dropout = tfl.Dropout(self.dropout_rate)
        self.layernorm = tfl.LayerNormalization(epsilon=self.layernorm_eps)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the model

        Args:
            inputs: A tuple of three tensors: the initial hidden state tensor of size `batch_size, hidden_size`,
                    an embedded input tensor of shape `batch_size, sequence_length, hidden_size` and
                    an attention mask tensor of shape `batch_size, sequence_length, hidden_size`
            training: Specify whether the layer is being used in training mode

        Returns:
            A tensor of shape `batch_size, sequence_length, hidden_size`

        """
        hidden_state, embeddings, attention_mask = inputs
        attention_mask = tf.cast(attention_mask, 'bool')
        for rnn_layer in self.rnn:
            embeddings, hidden_state = rnn_layer(
                inputs=embeddings,
                initial_state=hidden_state,
                mask=attention_mask,
                training=training
            )
        embeddings = self.dropout(embeddings, training=training)
        embeddings = self.layernorm(embeddings)
        embeddings = self.intermediate_dense(embeddings)
        embeddings = self.out_dense(embeddings)
        return self.layernorm(embeddings)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "num_rnn_layers": self.num_rnn_layers,
            "hidden_size": self.hidden_size,
            "layernorm_eps": self.layernorm_eps,
            "dropout_rate": self.dropout_rate
        }
