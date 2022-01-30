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
        self._num_rnn_layers = num_rnn_layers
        self._hidden_size = hidden_size
        self.layernorm_eps = layernorm_eps
        self.dropout_rate = dropout_rate
        self.rnn = [tfl.GRU(
            units=self._hidden_size,
            recurrent_dropout=self.dropout_rate,
            return_sequences=True,
            return_state=True
        ) for _ in range(self._num_rnn_layers)]
        self.intermediate_dense = tfl.Dense(
            units=self._hidden_size*2,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            input_shape=(None, None, self._hidden_size)
        )
        self.out_dense = tfl.Dense(
            units=self._hidden_size,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            input_shape=(None, None, self._hidden_size*2)
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
            "num_rnn_layers": self._num_rnn_layers,
            "hidden_size": self._hidden_size,
            "layernorm_eps": self.layernorm_eps,
            "dropout_rate": self.dropout_rate
        }

    @property
    def hidden_size(self) -> int:
        return self._hidden_size


class AeGRUCellDecoder(tfl.Layer):
    """An GRU-based AE decoder that does not use teacher forcing"""

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
            dropout_rate: A dropout rate between 0 and 1. that will be applied to the outputs of
                          the dense layer. Defaults to 0.1
            **kwargs: Parent class keyword arguments
        """
        super().__init__(**kwargs)
        if num_rnn_layers < 1:
            raise ValueError("At least 1 RNN layer must be used!")
        self._num_rnn_layers = num_rnn_layers
        self._hidden_size = hidden_size
        self.layernorm_eps = layernorm_eps
        self.dropout_rate = dropout_rate
        self.rnn_layers = [tfl.GRUCell(self._hidden_size) for _ in range(self._num_rnn_layers)]
        self.dense_layers = [tfl.Dense(
            units=self._hidden_size,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            input_shape=(None, None, self._hidden_size)
        ) for _ in range(self._num_rnn_layers)]
        self.dropout = tfl.Dropout(self.dropout_rate)
        self.layernorm = tfl.LayerNormalization(epsilon=self.layernorm_eps)

    def _process_sequence(
            self,
            rnn_cell: tfl.GRUCell,
            dense: tfl.Dense,
            pre_inputs: tf.Tensor,
            hidden_states: tf.Tensor,
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Helper function: process a time series with an RNN cell without teacher forcing

        Args:
            rnn_cell: An RNN cell
            dense: A dense layer
            pre_inputs: A tensor of shape `(batch_size, timesteps, hidden_size)`. These inputs will be added
                        to the RNN cell output of time step `t-1` to get the RNN input of time step `t`
            hidden_states: Hidden state tensor of shape `(batch_size, hidden_size)`
            training: Specify whether dropout is being used in training mode. Defaults to `None`

        Returns:
            RNN output tensor of shape `(batch_size, timesteps, hidden_size)` and a hidden state tensor
            of shape `(batch_size, hidden_size)`
        """
        num_timesteps = tf.shape(pre_inputs)[1]
        outputs = []
        step_output = self.dropout(dense(hidden_states), training=training)
        for i in tf.range(num_timesteps):
            step_input = tf.add(step_output, pre_inputs[i])
            pre_step_output, hidden_states = rnn_cell(step_input, hidden_states)
            step_output = self.dropout(dense(pre_step_output), training=training)
            outputs.append(step_output)
        outputs = self.layernorm(tf.transpose(outputs, [1, 0, 2]))
        return outputs, hidden_states

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the layer

        Args:
            inputs: An input embedding tensor of shape `(batch_size, hidden_size)` and
                    a tensor of shape `(batch_size, timesteps, hidden_size)` which will be used
                    as initial RNN input tensor. This can be a zero tensor.
            training: Specify whether the model is being used in training mode

        Returns:
            Activations of shape `(batch_size, hidden_size)`
        """
        hidden_states, rnn_x = inputs
        for rnn, dense in zip(self.rnn_layers, self.dense_layers):
            rnn_x, hidden_states = self._process_sequence(rnn, dense, rnn_x, hidden_states,
                                                          training=training)
        return rnn_x

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "num_rnn_layers": self._num_rnn_layers,
            "hidden_size": self._hidden_size,
            "layernorm_eps": self.layernorm_eps,
            "dropout_rate": self.dropout_rate
        }
