"""A module for Transformer-based VAE layers"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.initializers import TruncatedNormal
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_tf_bert import TFBertEncoder
from transformers.models.openai.configuration_openai import OpenAIGPTConfig
from transformers.models.openai.modeling_tf_openai import TFBlock
from transformers.modeling_tf_utils import keras_serializable


@keras_serializable
class AeTransformerEncoder(TFBertEncoder):
    """Make TFBertEncoder serializable"""
    config_class = BertConfig


@keras_serializable
class AeTransformerDecoder(tfl.Layer):
    """Define a GPT decoder for an autoencoder"""
    config_class = OpenAIGPTConfig

    def __init__(self, config: OpenAIGPTConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: GPT configuration object. `config.n_layer` contains information
                    about the number of Transformer decoder layers
            **kwargs: Keyword arguments for the superclass initializer
        """
        super().__init__(**kwargs)
        self.hidden = [TFBlock(config, scale=True, name=f"decoder_hidden_._{i}") for i in range(config.n_layer)]

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Call the layer. Itt will not be able to output attentions or use head masks

        Args:
            inputs: Two tensors, the input hidden state and the attention mask
            training: Specifies whether the model is being used in ae_training mode

        Returns:
            The output hidden state
        """
        hidden_state, attn_mask = inputs
        for transformer_block in self.hidden:
            hidden_state = transformer_block(hidden_state, attn_mask, head_mask=None,
                                             output_attentions=False, training=training)[0]
        return hidden_state


class VaeSampling(tfl.Layer):
    """VAE sampling layer as suggested by A. Gérond in his book
    \"Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow\" (second edition, 2019), p. 588
    """

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """Sample from a Gaussian"""
        mean, log_var = inputs
        return tf.keras.backend.random_normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean


@keras_serializable
class PostPoolingLayer(tfl.Layer):
    """A layer applied after pooling from the encoder to obtain Gaussian mean and variance values for a VAE"""
    config_class = BertConfig

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        dense_params = {
            "units": self.config.hidden_size,
            "activation": self.config.hidden_act,
            "kernel_initializer": TruncatedNormal(stddev=self.config.initializer_range),
            "input_shape": (None, self.config.hidden_size)
        }
        self.post_pool_mean_dense = tfl.Dense(**dense_params, name="post_pool_mean_dense")
        self.post_pool_logvar_dense = tfl.Dense(**dense_params, name="post_pool_logvar_dense")
        self.post_pool_dropout = tfl.Dropout(rate=config.hidden_dropout_prob)
        self.post_pool_layernorm = tfl.LayerNormalization(epsilon=config.layer_norm_eps)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean_tensor = self.post_pool_mean_dense(inputs)
        logvar_tensor = self.post_pool_logvar_dense(inputs)
        mean_tensor = self.post_pool_layernorm(self.post_pool_dropout(mean_tensor))
        logvar_tensor = self.post_pool_layernorm(self.post_pool_dropout(logvar_tensor))
        return mean_tensor, logvar_tensor


class AveragePoolingLayer(tfl.Layer):
    """A layer for average pooling that takes padding token embedding into account"""

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """Do average pooling

        Args:
            inputs: A tuple of two tensors. The first is a 3D hidden state tensor,
                    the second is a 2D attention mask tensor
            kwargs: Additional keyword arguments

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
