"""A module for RNN decoder layers as an alternative of a GPT decoder"""

from typing import Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.initializers import TruncatedNormal


class AeGruDecoder(tfl.Layer):
    """A GRU-based decoder"""

    def __init__(
            self,
            num_rnn_layers: int = 2,
            hidden_size: int = 768,
            initializer_dev: float = 0.02,
            layernorm_eps: float = 1e-12,
            **kwargs) -> None:
        """

        Args:
            num_rnn_layers: Number of RNN layers. Defaults to 2
            hidden_size: Hidden size in the RNN and dense layers. Defaults to 768
            initializer_dev: `TruncatedNormal` kernel initializer deviation. Defaults to 0.02
            layernorm_eps: Layer normalization epsilon parameter. Defaults to 1e-12
            **kwargs: Parent class keyword arguments
        """
        super().__init__(**kwargs)
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.initializer_dev = initializer_dev
        self.layernorm_eps = layernorm_eps
        self.rnn = [tfl.GRU(
            units=self.num_rnn_layers,
            activation='gelu',
            recurrent_activation='gelu',
            kernel_initializer=TruncatedNormal(stddev=self.initializer_dev),
            recurrent_initializer=TruncatedNormal(stddev=self.initializer_dev),
            recurrent_dropout=0.1,
            return_sequences=True,
            return_state=True
        ) for _ in range(self.num_rnn_layers)]
        self.intermediate_dense = tfl.Dense(
            units=self.hidden_size*4,
            activation='gelu',
            kernel_initializer=TruncatedNormal(stddev=self.initializer_dev),
            input_shape=(None, None, self.hidden_size)
        )
        self.out_dense = tfl.Dense(
            units=self.hidden_size,
            activation='gelu',
            kernel_initializer=TruncatedNormal(stddev=self.initializer_dev),
            input_shape=(None, None, self.hidden_size*4)
        )
        self.dropout = tfl.Dropout(0.1)
        self.layernorm = tfl.LayerNormalization(epsilon=self.layernorm_eps)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """Call the model

        Args:
            inputs: A tuple of three tensors: the initial hidden state tensor of size `batch_size, hidden_size`,
                    an embedded input tensor of shape `batch_size, sequence_length, hidden_size` and
                    an attention mask tensor of shape `batch_size, sequence_length, hidden_size`

        Returns:
            A tensor of shape `batch_size, sequence_length, hidden_size`

        """
        hidden_state, embeddings, attention_mask = inputs
        attention_mask = tf.cast(attention_mask, 'bool')
        for rnn_layer in self.rnn:
            embeddings = rnn_layer(
                inputs=embeddings,
                initial_state=hidden_state,
                mask=attention_mask
            )
        embeddings = self.dropout(embeddings)
        embeddings = self.layernorm(embeddings)
        embeddings = self.intermediate_dense(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.out_dense(embeddings)
        embeddings = self.dropout(embeddings)
        return self.layernorm(embeddings)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "num_rnn_layers": self.num_rnn_layers,
            "hidden_size": self.hidden_size,
            "initializer_dev": self.initializer_dev,
            "layernorm_eps": self.layernorm_eps
        }
