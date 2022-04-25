# -*- coding: utf-8 -*-

"""A module for models that are intended to be used as layers in a more complex model during pre-training.
Defining them as a model allows to use them separately after pre-training
"""

# Note that does not often recognize whether that Keras layer or model is callable.
# This is the reason why the corresponding inspection were suppressed for some functions and classes.

from __future__ import annotations
from typing import Tuple, Optional, Literal, Dict, Any
from types import MappingProxyType

import tensorflow as tf
from tensorflow.keras import Model as KModel
from tensorflow.keras import layers as tfl
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.utils import register_keras_serializable
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig
from transformers.modeling_tf_utils import keras_serializable

from ae_sentence_embeddings.layers import (
    AeTransformerEncoder,
    AeTransformerDecoder,
    PostPoolingLayer,
    AveragePoolingLayer,
    PMeansPooling,
    CLSPlusSEPPooling,
    AeGRUDecoder,
    AeTransformerGRUDecoder,
    RegularizedEmbedding,
    SinusoidalEmbedding
)
from ae_sentence_embeddings.modeling_tools import process_attention_mask, make_decoder_inputs
from ae_sentence_embeddings.argument_handling import (
    RnnLayerArgs, RnnArgs,
    PositionalEmbeddingArgs, RegularizedEmbeddingArgs
)


@register_keras_serializable(package="ae_sentence_embeddings.models")
class SentAeEncoder(KModel):
    """The full encoder part of an AE"""

    def __init__(self, config: BertConfig,
                 pooling_type: Literal["average", "cls_sep", "p_means"] = "cls_sep",
                 **kwargs) -> None:
        """Layer initializer.

        Args:
            config: A BERT configuration object.
            pooling_type: Pooling method, `'average'`, `'cls_sep'`
                or `'p_means'`. Defaults to `"cls_sep"`.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(**kwargs)
        self._transformer_config = config
        self._pooling_type = pooling_type.lower()
        if self._pooling_type == "average":
            self._pooling = AveragePoolingLayer()
        elif self._pooling_type == "cls_sep":
            self._pooling = CLSPlusSEPPooling()
        elif self._pooling_type == "p_means":
            self._pooling = PMeansPooling()
        else:
            raise NotImplementedError(f"Unknown pooling type: {pooling_type}")
        self._embedding_layer = SinusoidalEmbedding(
            PositionalEmbeddingArgs.collect_from_dict(config.to_dict()))
        self._transformer_encoder = AeTransformerEncoder(self._transformer_config)

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> Tuple[tf.Tensor, Tuple[tf.Tensor]]:
        """Call the encoder.

        Args:
            inputs: Input ID tensor with shape `(batch_size, sequence_length)`
                    and attention mask with shape `(batch_size, sequence_length)`.
            training: Specifies whether the model is being used in training mode.

        Returns:
            A pooled tensor and the Transformer encoder outputs.

        """
        input_ids, attention_mask = inputs
        embeddings = self._embedding_layer(input_ids, training=training)
        mod_attention_mask = process_attention_mask(attention_mask, embedding_dtype=embeddings.dtype)
        encoder_outputs = self._transformer_encoder(
            hidden_states=embeddings,
            attention_mask=mod_attention_mask,
            head_mask=[None] * self._transformer_config.num_hidden_layers,
            past_key_values=[None] * self._transformer_config.num_hidden_layers,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            training=training
        )
        sequence_output = encoder_outputs[0]
        pooling_result = self._pooling((sequence_output, attention_mask))
        return pooling_result, encoder_outputs + (embeddings,)

    @property
    def pooling_type(self) -> Literal["average", "cls_sep", "p_means"]:
        # noinspection PyTypeChecker
        return self._pooling_type  # The return type is correct, PyCharm may complain because of the `str.lower` call

    def get_config(self) -> Dict[str, Any]:
        return {
            "encoder_config": self._transformer_config.to_dict(),
            "pooling_type": self._pooling_type
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> SentAeEncoder:
        encoder_config = BertConfig(**config.pop("encoder_config"))
        return cls(encoder_config, **config, **kwargs)

    @property
    def transformer_config(self) -> MappingProxyType:
        return MappingProxyType(self._transformer_config.to_dict())


@register_keras_serializable(package="ae_sentence_embeddings.models")
class SentVaeEncoder(SentAeEncoder):
    """The full encoder part of a VAE"""
    _keras_serializable = True

    def __init__(
            self,
            config: BertConfig,
            pooling_type: Literal["average", "cls_sep", "p_means"] = "cls_sep",
            kl_factor: float = 1.0,
            **kwargs
    ) -> None:
        """Initialize the encoder.

        Args:
            config: A BERT configuration object.
            pooling_type: Pooling type, `'average'` or `'cls_sep'`. Defaults to `'cls_sep'`.
            kl_factor: A normalizing constant by which the KL loss will be multiplied. Defaults to `1.0`.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(config, pooling_type, **kwargs)
        hidden_size = 2 * config.hidden_size if self._pooling_type in {"cls_sep", "p_means"} \
            else config.hidden_size
        self._post_pooling = PostPoolingLayer(
            hidden_size=hidden_size,
            layer_norm_eps=config.layer_norm_eps,
            kl_factor=kl_factor,
            initializer_range=config.initializer_range
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, Tuple[tf.Tensor]]:
        """Call the encoder

        Args:
            inputs: Input IDs tensor with shape `(batch_size, sequence_length)`
                    and attention mask with shape `(batch_size, sequence_length)`.
            training: Specifies whether the model is being used in training mode.

        Returns:
            Two pooled tensors (mean and log variance for VAE sampling) and the Transformer encoder outputs.
        """
        pooling_output, encoder_outputs = super().call(inputs, training=training)
        # noinspection PyCallingNonCallable
        post_pooling_mean, post_pooling_logvar = self._post_pooling(pooling_output)
        return post_pooling_mean, post_pooling_logvar, encoder_outputs

    @property
    def kl_factor(self) -> float:
        return self._post_pooling.kl_factor

    def set_kl_factor(self, new_kl_factor: float) -> None:
        """Setter for `kl_factor`"""
        self._post_pooling.kl_factor = new_kl_factor

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "kl_factor": self._post_pooling.kl_factor
        }


# noinspection PyAbstractClass
@keras_serializable  # Note that this decorator adds the `get_config` method
class SentAeDecoder(KModel):
    """The full decoder part of the autoencoder"""
    config_class = OpenAIGPTConfig

    def __init__(self, config: OpenAIGPTConfig, **kwargs) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(**kwargs)
        self._transformer_config = config
        self._transformer_decoder = AeTransformerDecoder(self._transformer_config)
        embedding_args = PositionalEmbeddingArgs(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.n_positions,
            hidden_size=config.n_embd,
            hidden_dropout_prob=config.embd_pdrop,
            layer_norm_eps=config.layer_norm_epsilon,
            initializer_range=config.initializer_range
        )
        self._embedding_layer = SinusoidalEmbedding(embedding_args)
        self._out_dense = tfl.Dense(config.vocab_size,
                                    kernel_initializer=TruncatedNormal(stddev=config.initializer_range))

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the decoder

        Args:
            inputs: An input embedding tensor of shape `(batch_size, hidden_size)`.
                    an input token ID tensor of shape `(batch_size, sequence_length, hidden_size)` and
                    an attention mask tensor of shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in training mode

        Returns:
            Logits for next token prediction
        """
        sent_embeddings, input_ids, attention_mask = inputs
        token_embeddings = self._embedding_layer(input_ids[:, 1:], training=training)
        attention_mask = process_attention_mask(
            attention_mask=make_decoder_inputs(attention_mask),
            embedding_dtype=token_embeddings.dtype
        )
        encodings = tf.concat([tf.expand_dims(sent_embeddings, axis=1), token_embeddings], axis=1)
        hidden_output = self._transformer_decoder((encodings, attention_mask), training=training)
        logits = self._out_dense(hidden_output)
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
        self._gru_config = config
        config_dict = config.to_dict()
        vocab_size = config_dict.pop("vocab_size")
        init_range = config_dict.pop("initializer_dev")
        self._decoder = AeGRUDecoder(**config_dict)
        self._out_dense = tfl.Dense(
            vocab_size, kernel_initializer=TruncatedNormal(stddev=init_range))

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the decoder

        Args:
            inputs: An sentence embedding tensor of shape `(batch_size, hidden_size)` and
                an input token embedding tensor of shape `(batch_size, sequence_length, hidden_size)`.
            training: Specifies whether the model is being used in training mode.

        Returns:
            Logits for next token prediction
        """
        sent_embeddings, token_embeddings = inputs
        hidden_output = self._decoder((sent_embeddings, token_embeddings), training=training)
        logits = self._out_dense(hidden_output)
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {"decoder_config": self._gru_config.to_dict()}

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> SentAeGRUDecoder:
        decoder_config = RnnArgs(**config.pop("decoder_config"))
        return cls(decoder_config, **kwargs)

    @property
    def transformer_config(self) -> MappingProxyType:
        return MappingProxyType(self._gru_config.to_dict())


def parallel_decoders(
        sent_hidden_size: int,
        tok_hidden_size: int,
        vocab_size: int,
        linear_stddev: float,
        decoder_class: type,
        decoder_kwargs: Dict[str, Any],
        name: Optional[str] = None,
) -> KModel:
    """Define parallel decoders with the functional API

    Args:
        sent_hidden_size: Hidden size of the input sentence representations.
        tok_hidden_size: Hidden size of the input token representations.
        vocab_size: Vocabulary size for the output logits.
        linear_stddev: `TruncatedNormal` stddev argument of the final dense layer that outputs logits.
        decoder_class: Decoder class to use.
        name: Optional. Model name as a string.
        decoder_kwargs: Keyword arguments to initialize the decoder instances

    Returns:
        A functional Keras model
    """
    sent_embeddings1 = tfl.Input(shape=(sent_hidden_size,), dtype=tf.float32)
    token_embeddings1 = tfl.Input(shape=(None, tok_hidden_size), dtype=tf.float32)
    sent_embeddings2 = tfl.Input(shape=(sent_hidden_size,), dtype=tf.float32)
    token_embeddings2 = tfl.Input(shape=(None, tok_hidden_size), dtype=tf.float32)

    branch1_out = decoder_class(**decoder_kwargs)((sent_embeddings1, token_embeddings1))
    branch2_out = decoder_class(**decoder_kwargs)((sent_embeddings2, token_embeddings2))
    outputs = tfl.concatenate([branch1_out, branch2_out], axis=0)
    outputs = tfl.Dense(
        vocab_size,
        kernel_initializer=TruncatedNormal(stddev=linear_stddev),
    )(outputs)
    return KModel(inputs=[sent_embeddings1, token_embeddings1, sent_embeddings2, token_embeddings2],
                  outputs=outputs, name=name)


# noinspection PyCallingNonCallable
def ae_double_gru(rnn_config: RnnArgs, tok_hidden_size: int) -> KModel:
    """Define parallel decoders with the functional API.

    Args:
        rnn_config: The RNN configuration dataclass shared by the two decoders.
        tok_hidden_size: Embedding size of the input (sub)word vectors.

    Returns:
        A functional Keras model.
    """
    return parallel_decoders(
        sent_hidden_size=rnn_config.hidden_size,
        tok_hidden_size=tok_hidden_size,
        vocab_size=rnn_config.vocab_size,
        linear_stddev=rnn_config.initializer_dev,
        decoder_class=AeGRUDecoder,
        name="double_gru",
        decoder_kwargs={
            "num_rnn_layers": rnn_config.num_rnn_layers,
            "hidden_size": rnn_config.hidden_size,
            "layernorm_eps": rnn_config.layernorm_eps,
            "dropout_rate": rnn_config.dropout_rate
        }
    )


# noinspection PyCallingNonCallable
def ae_double_transformer_gru(
        transformer_layers_config: OpenAIGPTConfig,
        rnn_layers_config: RnnLayerArgs,
        num_transformer2gru: int
) -> KModel:
    """Define parallel decoders with Transformer + GRU layers

    Args:
        transformer_layers_config: The Transformer configuration data
        rnn_layers_config: The GRU configuration data
        num_transformer2gru: number of dense layers between the Transformer and GRU layers

    Returns:
        A functional Keras model
    """
    return parallel_decoders(
        sent_hidden_size=rnn_layers_config.hidden_size,
        tok_hidden_size=transformer_layers_config.n_embd,
        vocab_size=transformer_layers_config.vocab_size,
        linear_stddev=transformer_layers_config.initializer_range,
        decoder_class=AeTransformerGRUDecoder,
        name="double_transformer_gru",
        decoder_kwargs={
            "transformer_config": transformer_layers_config,
            "gru_config": rnn_layers_config,
            "num_transformer2gru": num_transformer2gru
        }
    )
