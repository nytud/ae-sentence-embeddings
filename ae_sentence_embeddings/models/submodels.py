"""A module for models that are intended to be used as layers in a more complex model during pre-ae_training.
Defining them as a model allows to use them separately after pre-training
"""

from typing import Tuple, Sequence

import tensorflow as tf
from tensorflow.keras import Model as KModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig
from transformers.modeling_tf_utils import TFSharedEmbeddings

from ae_sentence_embeddings.layers import (
    AeTransformerEncoder,
    AeTransformerDecoder,
    PostPoolingLayer,
    AveragePoolingLayer,
    AeGruDecoder
)
from ae_sentence_embeddings.modeling_tools import process_attention_mask, make_decoder_inputs
from ae_sentence_embeddings.argument_handling import RnnArgs


class SentAeEncoder(KModel):
    """The full encoder part of an AE"""

    def __init__(self, config: BertConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(**kwargs)
        self.config = config
        self.embedding_layer = TFSharedEmbeddings(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            initializer_range=self.config.initializer_range
        )
        self.transformer_encoder = AeTransformerEncoder(self.config)
        self.pooling = AveragePoolingLayer()

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training=None, mask=None) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
        """Call the encoder

        Args:
            inputs: Input ID tensor with shape `(batch_size, sequence_length)`
                    and attention mask with shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in ae_training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            A pooled tensor and the Transformer encoder outputs

        """
        input_ids, attention_mask = inputs
        embeddings = self.embedding_layer(input_ids, mode="embedding")
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
        return self.pooling((sequence_output, attention_mask)), encoder_outputs


class SentVaeEncoder(SentAeEncoder):
    """The full encoder part of a VAE"""

    def __init__(self, config: BertConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(config, **kwargs)
        self.post_pooling = PostPoolingLayer(config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor, Sequence[tf.Tensor]]:
        """Call the encoder

        Args:
            inputs: Input IDs tensor with shape `(batch_size, sequence_length)`
                    and attention mask with shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in ae_training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            Two pooled tensors (mean and log variance for VAE sampling) and the Transformer encoder outputs
        """
        pooling_output, encoder_outputs = super().call(inputs, training=training)
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
        self.embedding_layer = TFSharedEmbeddings(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.n_embd,
            initializer_range=self.config.initializer_range
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        """Call the decoder

        Args:
            inputs: An input embedding tensor of shape `(batch_size, hidden_size)`.
                    an input token ID tensor of shape `(batch_size, sequence_length, hidden_size)` and
                    an attention mask tensor of shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in ae_training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            Logits for next token prediction
        """
        sent_embeddings, input_ids, attention_mask = inputs
        token_embeddings = self.embedding_layer(input_ids[:, 1:], mode="embedding")
        attention_mask = process_attention_mask(
            attention_mask=make_decoder_inputs(attention_mask),
            embedding_dtype=token_embeddings.dtype
        )
        encodings = tf.concat([tf.expand_dims(sent_embeddings, axis=1), token_embeddings], axis=1)
        hidden_output = self.transformer_decoder((encodings, attention_mask), training=training)
        logits = self.embedding_layer(hidden_output, mode="linear")
        return logits


class SentAeGRUDecoder(KModel):
    """A GRU-based full decoder"""

    def __init__(self, config: RnnArgs, **kwargs) -> None:
        """Layer initializer

        Args:
            config: An RNN configuration dataclass object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(**kwargs)
        self.config = config
        config_dict = config.to_dict()
        vocab_size = config_dict.pop("vocab_size")
        embedding_init_range = config_dict.pop("initializer_dev")
        self.decoder = AeGruDecoder(**config_dict)
        self.embedding_layer = TFSharedEmbeddings(
            vocab_size=vocab_size,
            hidden_size=self.config.hidden_size,
            initializer_range=embedding_init_range
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        """Call the decoder

        Args:
            inputs: An input embedding tensor of shape `(batch_size, hidden_size)`.
                    an input token ID tensor of shape `(batch_size, sequence_length, hidden_size)` and
                    an attention mask tensor of shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in ae_training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            Logits for next token prediction
        """
        sent_embeddings, input_ids, attention_mask = inputs
        attention_mask = make_decoder_inputs(attention_mask)
        token_embeddings = self.embedding_layer(input_ids, mode="embedding")
        hidden_output = self.decoder((sent_embeddings, token_embeddings, attention_mask),
                                     training=training)
        logits = self.embedding_layer(hidden_output, mode="linear")
        return logits
