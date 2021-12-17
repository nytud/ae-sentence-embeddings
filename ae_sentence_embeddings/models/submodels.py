"""A module for models that are intended to be used as layers in a more complex model during pre-training.
Defining them as a model allows to use them separately after pre-training
"""

from typing import Tuple, Sequence

import tensorflow as tf
from tensorflow.keras import Model as KModel, layers as tfl
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig

from ae_sentence_embeddings.layers import AeTransformerEncoder, AeTransformerDecoder, PostPoolingLayer
from ae_sentence_embeddings.modeling_tools import process_attention_mask


class SentAeEncoder(KModel):
    """The full encoder part"""

    def __init__(self, config: BertConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(**kwargs)
        self.config = config
        self.transformer_encoder = AeTransformerEncoder(self.config)
        self.pooling = tfl.Lambda(lambda x: tf.reduce_mean(x, axis=1))
        self.post_pooling = PostPoolingLayer(self.config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor, Sequence[tf.Tensor]]:
        """Call the encoder

        Args:
            inputs: Embedding tensor with shape `(batch_size, sequence_length, hidden_size)`
                    and attention mask with shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            Two pooled tensors (mean and log variance for VAE sampling) and the Transformer encoder hidden state
        """
        embeddings, attention_mask = inputs
        attention_mask = process_attention_mask(attention_mask, embedding_dtype=embeddings.dtype)
        encoder_outputs = self.transformer_encoder(
            hidden_states=embeddings,
            attention_mask=attention_mask,
            head_mask=[None] * self.config.num_hidden_layers,
            past_key_values=[None] * self.config.num_hidden_layers,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            training=training
        )
        sequence_output = encoder_outputs[0]
        pooling_output = self.pooling(sequence_output)
        post_pooling_mean, post_pooling_logvar = self.post_pooling(pooling_output)
        return post_pooling_mean, post_pooling_logvar, encoder_outputs


class SentAeDecoder(KModel):
    """The full decoder part of the autoencoder"""

    def __init__(self, config: OpenAIGPTConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(**kwargs)
        self.config = config
        self.transformer_decoder = AeTransformerDecoder(self.config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        """Call the decoder

        Args:
            inputs: A hidden state tensor of shape `(batch_size, sequence_length, hidden_size)`
                    and an attention mask of shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            The last hidden state
        """
        hidden_state, attention_mask = inputs
        attention_mask = process_attention_mask(attention_mask, embedding_dtype=hidden_state.dtype)
        hidden_output = self.transformer_decoder((hidden_state, attention_mask), training=training)
        return hidden_output
