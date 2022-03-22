"""A module for models that are intended to be used as layers in a more complex model during pre-training.
Defining them as a model allows to use them separately after pre-training
"""

# Note that does not often recognize whether that Keras layer or model is callable.
# This is the reason why the corresponding inspection were suppressed for some functions and classes.

from typing import Tuple, Sequence, Optional, Literal, Dict, Any

import tensorflow as tf
from tensorflow.keras import Model as KModel
from tensorflow.keras import layers as tfl
from tensorflow.keras.initializers import TruncatedNormal
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig
from transformers.modeling_tf_utils import keras_serializable

from ae_sentence_embeddings.layers import (
    AeTransformerEncoder,
    AeTransformerDecoder,
    PostPoolingLayer,
    AveragePoolingLayer,
    CLSPlusSEPPooling,
    AeGruDecoder,
    RegularizedEmbedding,
    SinusoidalEmbedding
)
from ae_sentence_embeddings.modeling_tools import process_attention_mask, make_decoder_inputs
from ae_sentence_embeddings.argument_handling import RnnArgs, PositionalEmbeddingArgs, RegularizedEmbeddingArgs


class SentAeEncoder(KModel):
    """The full encoder part of an AE"""

    def __init__(self, config: BertConfig,
                 pooling_type: Literal["average", "cls_sep"] = "cls_sep",
                 **kwargs) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            pooling_type: Pooling method`, "average"` or `"cls_sep"`. Defaults to `"cls_sep"`
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(**kwargs)
        self.config = config
        self._pooling_type = pooling_type.lower()
        if self._pooling_type == "average":
            self.pooling = AveragePoolingLayer()
        elif self._pooling_type == "cls_sep":
            self.pooling = CLSPlusSEPPooling()
        else:
            raise NotImplementedError(f"Unknown pooling type: {pooling_type}")
        self.embedding_layer = SinusoidalEmbedding(
            PositionalEmbeddingArgs.collect_from_dict(config.to_dict()))
        self.transformer_encoder = AeTransformerEncoder(self.config)

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
        """Call the encoder

        Args:
            inputs: Input ID tensor with shape `(batch_size, sequence_length)`
                    and attention mask with shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in training mode

        Returns:
            A pooled tensor and the Transformer encoder outputs

        """
        input_ids, attention_mask = inputs
        embeddings = self.embedding_layer(input_ids, training=training)
        mod_attention_mask = process_attention_mask(attention_mask, embedding_dtype=embeddings.dtype)
        encoder_outputs = self.transformer_encoder(
            hidden_states=embeddings,
            attention_mask=mod_attention_mask,
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

    @property
    def pooling_type(self) -> str:
        return self._pooling_type

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "encoder_config": self.config.to_dict(),
            "pooling_type": self.pooling_type
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None):
        encoder_config = BertConfig(**config.pop("encoder_config"))
        return cls(encoder_config, **config)


class SentVaeEncoder(SentAeEncoder):
    """The full encoder part of a VAE"""

    def __init__(
            self,
            config: BertConfig,
            pooling_type: Literal["average", "cls_sep"] = "cls_sep",
            kl_factor: float = 1.0,
            **kwargs
    ) -> None:
        """Layer initializer

        Args:
            config: A BERT configuration object
            pooling_type: Pooling type, `average` or `cls_sep`
            kl_factor: A normalizing constant by which the KL loss will be multiplied. Defaults to `1.0`
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(config, pooling_type, **kwargs)
        hidden_size = 2 * config.hidden_size if self.pooling_type == "cls_sep" else config.hidden_size
        self.post_pooling = PostPoolingLayer(
            hidden_size=hidden_size,
            layer_norm_eps=config.layer_norm_eps,
            kl_factor=kl_factor,
            initializer_range=config.initializer_range
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor, Sequence[tf.Tensor]]:
        """Call the encoder

        Args:
            inputs: Input IDs tensor with shape `(batch_size, sequence_length)`
                    and attention mask with shape `(batch_size, sequence_length)`
            training: Specifies whether the model is being used in training mode

        Returns:
            Two pooled tensors (mean and log variance for VAE sampling) and the Transformer encoder outputs
        """
        pooling_output, encoder_outputs = super().call(inputs, training=training)
        # noinspection PyCallingNonCallable
        post_pooling_mean, post_pooling_logvar = self.post_pooling(pooling_output)
        return post_pooling_mean, post_pooling_logvar, encoder_outputs

    @property
    def kl_factor(self) -> float:
        return self.post_pooling.kl_factor

    def set_kl_factor(self, new_kl_factor: float) -> None:
        """Setter for `kl_factor`"""
        self.post_pooling.kl_factor = new_kl_factor

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "kl_factor": self.kl_factor
        }


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
        self.config = config
        self.transformer_decoder = AeTransformerDecoder(self.config)
        embedding_args = PositionalEmbeddingArgs(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.n_positions,
            hidden_size=config.n_embd,
            hidden_dropout_prob=config.embd_pdrop,
            layer_norm_eps=config.layer_norm_epsilon,
            initializer_range=config.initializer_range
        )
        self.embedding_layer = SinusoidalEmbedding(embedding_args)
        self.out_dense = tfl.Dense(config.vocab_size,
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
        token_embeddings = self.embedding_layer(input_ids[:, 1:], training=training)
        attention_mask = process_attention_mask(
            attention_mask=make_decoder_inputs(attention_mask),
            embedding_dtype=token_embeddings.dtype
        )
        encodings = tf.concat([tf.expand_dims(sent_embeddings, axis=1), token_embeddings], axis=1)
        hidden_output = self.transformer_decoder((encodings, attention_mask), training=training)
        logits = self.out_dense(hidden_output)
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
        embedding_layer_config = RegularizedEmbeddingArgs(
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=embedding_init_range,
            layer_norm_eps=config.layernorm_eps,
            hidden_dropout_prob=config.dropout_rate
        )
        self.embedding_layer = RegularizedEmbedding(embedding_layer_config)
        self.out_dense = tfl.Dense(config.vocab_size,
                                   kernel_initializer=TruncatedNormal(stddev=config.initializer_dev))

    # noinspection PyCallingNonCallable
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the decoder

        Args:
            inputs: An input embedding tensor of shape `(batch_size, hidden_size)` and
                    an input token ID tensor of shape `(batch_size, sequence_length, hidden_size)`
            training: Specifies whether the model is being used in training mode

        Returns:
            Logits for next token prediction
        """
        sent_embeddings, input_ids = inputs
        token_embeddings = self.embedding_layer(input_ids)
        hidden_output = self.decoder((sent_embeddings, token_embeddings), training=training)
        logits = self.out_dense(hidden_output)
        return logits

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "decoder_config": self.config.to_dict(),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None):
        decoder_config = RnnArgs(**config.pop("decoder_config"))
        return cls(decoder_config)


# noinspection PyCallingNonCallable
def ae_double_gru(rnn_config: RnnArgs) -> KModel:
    """Define parallel decoders with the functional API

    Args:
        rnn_config: The RNN configuration dataclass shared by the two decoders

    Returns:
        A functional Keras model
    """
    hidden_states1 = tfl.Input(shape=(rnn_config.hidden_size,), dtype=tf.float32)
    inputs1 = tfl.Input(shape=(None, rnn_config.hidden_size), dtype=tf.float32)
    hidden_states2 = tfl.Input(shape=(rnn_config.hidden_size,), dtype=tf.float32)
    inputs2 = tfl.Input(shape=(None, rnn_config.hidden_size), dtype=tf.float32)

    branch1_out = AeGruDecoder(
        num_rnn_layers=rnn_config.num_rnn_layers,
        hidden_size=rnn_config.hidden_size,
        layernorm_eps=rnn_config.layernorm_eps,
        dropout_rate=rnn_config.dropout_rate
    )((hidden_states1, inputs1))
    branch2_out = AeGruDecoder(
        num_rnn_layers=rnn_config.num_rnn_layers,
        hidden_size=rnn_config.hidden_size,
        layernorm_eps=rnn_config.layernorm_eps,
        dropout_rate=rnn_config.dropout_rate
    )((hidden_states2, inputs2))

    outputs = tfl.concatenate([branch1_out, branch2_out], axis=0)
    outputs = tfl.Dense(
        rnn_config.vocab_size,
        kernel_initializer=TruncatedNormal(stddev=rnn_config.initializer_dev),
    )(outputs)
    return KModel(inputs=[hidden_states1, inputs1, hidden_states2, inputs2],
                  outputs=outputs, name="double_gru")
