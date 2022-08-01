# -*- coding: utf-8 -*-

"""A module for defining Transformer-based AEs."""

# Note that does not often recognize whether that Keras layer or model is callable.
# This is the reason why the corresponding inspection were suppressed for some functions and classes.

from __future__ import annotations

import pickle
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from types import MappingProxyType
from typing import Tuple, Union, Optional, Literal, Dict, Any
from warnings import warn

import tensorflow as tf
from tensorflow.keras import Model as KModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig

from ae_sentence_embeddings.argument_handling import RnnArgs
from ae_sentence_embeddings.layers import VaeSampling
from ae_sentence_embeddings.models import (
    SentVaeEncoder,
    SentAeDecoder,
    SentAeEncoder,
    SentAeGRUDecoder
)


class BaseAe(KModel, metaclass=ABCMeta):
    """Base class for Transformer AEs. Used for subclassing only."""

    def __init__(
            self,
            enc_config: BertConfig,
            dec_config: Union[OpenAIGPTConfig, RnnArgs],
            pooling_type: Literal["average", "cls_sep", "p_means"],
            **kwargs
    ) -> None:
        """Initialize the invariant AE layers that do not depend on the AE architecture choice.

        Args:
            enc_config: The encoder configuration object.
            dec_config: The decoder configuration object.
            pooling_type: Pooling method, `'average'`, `'cls_sep'` or `'p_means'`.
            **kwargs: Keyword arguments for the parent class.
        """
        if enc_config.vocab_size != dec_config.vocab_size:
            raise ValueError("Vocab size and should be the same in the encoder and the decoder")
        super().__init__(**kwargs)
        self._enc_config = deepcopy(enc_config)
        self._dec_config = deepcopy(dec_config)
        self._pooling_type = pooling_type

    def checkpoint(self, weight_path: str, optimizer_path: Optional[str] = None) -> None:
        """Save model and (optionally) optimizer weights.
        This method uses the defaults arguments of the `save_weights` method

        About saving the optimizer state see
        https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        """
        self.save_weights(weight_path)
        if optimizer_path is not None:
            symbolic_weights = getattr(self.optimizer, "weights", None)
            if symbolic_weights is None:
                warn("Optimizer weights were not found, so the optimizer state was not saved.")
            else:
                optimizer_weights = tf.keras.backend.batch_get_value(symbolic_weights)
                with open(optimizer_path, "wb") as optimizer_f:
                    pickle.dump(optimizer_weights, optimizer_f)

    def load_checkpoint(self, weight_path: str, optimizer_path: Optional[str] = None,
                        **kwargs) -> None:
        """Load the model from a checkpoint.

        The optimizer weights are loaded as proposed at `https://stackoverflow.com/a/63089235`.

        Args:
            weight_path: Path to the saved model weights.
            optimizer_path: Optional. Path to saved optimizer state file.
            **kwargs: Further arguments passed to the `load_weights` method.
        """
        # noinspection PyCallingNonCallable
        self((tf.keras.Input(shape=(None,), dtype=tf.int32),
              tf.keras.Input(shape=(None,), dtype=tf.int32)))  # Build the model weights
        if optimizer_path is not None:
            if self.optimizer is None:
                warn("Cannot load optimizer as no optimizer has been defined")
            else:
                grad_vars = self.trainable_weights
                zero_grads = [tf.zeros_like(w) for w in grad_vars]
                self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
                with open(optimizer_path, "rb") as f:
                    optimizer_weights = pickle.load(f)
                self.optimizer.set_weights(optimizer_weights)
        self.load_weights(weight_path, **kwargs)

    @abstractmethod
    def call(self, inputs, training=None):
        return

    def get_config(self) -> Dict[str, Any]:
        decoder_config_type = type(self._dec_config).__name__
        return {
            "decoder_config_type": decoder_config_type,
            "encoder_config": self._enc_config.to_dict(),
            "decoder_config": self._dec_config.to_dict(),
            "pooling_type": self._pooling_type
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseAe:
        encoder_config = BertConfig(**config.pop("encoder_config"))
        decoder_config_name = config.pop("decoder_config_type")
        if decoder_config_name == OpenAIGPTConfig.__name__:
            decoder_config_type = OpenAIGPTConfig
        elif decoder_config_name == RnnArgs.__name__:
            decoder_config_type = RnnArgs
        else:
            raise ValueError(f"The decoder config class must be either {OpenAIGPTConfig.__name__} "
                             f"or {RnnArgs.__name__}, got {decoder_config_name}")
        decoder_config = decoder_config_type(**config.pop("decoder_config"))
        return cls(encoder_config, decoder_config, **config)

    @property
    def enc_config(self) -> MappingProxyType:
        return MappingProxyType(self._enc_config.to_dict())

    @property
    def dec_config(self) -> MappingProxyType:
        return MappingProxyType(self._dec_config.to_dict())

    @property
    def pooling_type(self) -> str:
        return self._pooling_type


class BaseVae(BaseAe, metaclass=ABCMeta):
    """Base class for Transformer-based VAE encoders. Use for subclassing only"""

    def __init__(
            self,
            enc_config: BertConfig,
            dec_config: Union[OpenAIGPTConfig, RnnArgs],
            pooling_type: Literal["cls_sep", "average", "p_means"] = "average",
            reg_args: Optional[Dict[str, Union[tf.Variable, int]]] = None,
            **kwargs
    ) -> None:
        """Initialize the VAE.

        Args:
            enc_config: The encoder configuration object.
            dec_config: The decoder configuration object.
            pooling_type: Pooling method`, "average"`, `"p_means"` or `"cls_sep"`.
                Defaults to `"average"`.
            reg_args: KL loss regularization arguments. Optional.
            **kwargs: Keyword arguments for the `keras.Model` class.
        """
        super().__init__(enc_config, dec_config, pooling_type=pooling_type, **kwargs)
        if reg_args is None:
            reg_args = {"warmup_iters": 1, "iters": 0}
        self._encoder = SentVaeEncoder(
            self._enc_config, pooling_type=pooling_type, reg_args=reg_args)
        self._sampler = VaeSampling()

    def get_config(self) -> Dict[str, Any]:
        """Get serialization configuration."""
        base_config = super(BaseVae, self).get_config()
        return {
            **base_config,
            "reg_args": {**self._encoder.reg_args}
        }


# noinspection PyCallingNonCallable
class TransformerAe(BaseAe):
    """A Transformer-based AE"""

    def __init__(
            self, enc_config: BertConfig,
            dec_config: OpenAIGPTConfig,
            pooling_type: Literal["cls_sep", "average", "p_means"],
            **kwargs
    ) -> None:
        """Initialize the AE.

        Args:
            enc_config: The encoder configuration object.
            dec_config: The decoder configuration object.
            pooling_type: Pooling method`, "average"` or `"cls_sep"`.
            **kwargs: Keyword arguments for the `keras.Model` class.
        """
        super().__init__(enc_config, dec_config, pooling_type=pooling_type, **kwargs)
        self._encoder = SentAeEncoder(self._enc_config, pooling_type=pooling_type)
        self._decoder = SentAeDecoder(self._dec_config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in training mode

        Returns:
            The logits of a probability distribution for next token prediction

        """
        input_ids, attn_mask = inputs
        sent_embedding, _ = self._encoder((input_ids, attn_mask), training=training)
        logits = self._decoder(inputs=(sent_embedding, input_ids, attn_mask), training=training)
        return logits


# noinspection PyCallingNonCallable
class TransformerVae(BaseVae):
    """A Transformer-based VAE"""

    def __init__(
            self,
            enc_config: BertConfig,
            dec_config: OpenAIGPTConfig,
            pooling_type: Literal["cls_sep", "average", "p_means"],
            reg_args: Dict[str, Union[tf.Variable, int]],
            **kwargs
    ) -> None:
        """Initialize the VAE.

        Args:
            enc_config: The encoder configuration object.
            dec_config: The decoder configuration object.
            pooling_type: Pooling method`, "average"` or `"cls_sep"`. Defaults to `"cls_sep"`.
            reg_args: KL loss regularization arguments.
            **kwargs: Keyword arguments for the `keras.Model` class.
        """
        super().__init__(enc_config, dec_config, pooling_type=pooling_type,
                         reg_args=reg_args, **kwargs)
        self._decoder = SentAeDecoder(self._dec_config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in training mode

        Returns:
            The logits of a probability distribution for next token prediction
        """
        input_ids, attn_mask = inputs
        mean, log_var, _ = self._encoder((input_ids, attn_mask), training=training)
        sent_embedding = self._sampler((mean, log_var))
        logits = self._decoder(inputs=(sent_embedding, input_ids, attn_mask), training=training)
        return logits


# noinspection PyCallingNonCallable
class BertRnnVae(BaseVae):
    """A VAE with a Bert encoder and an RNN decoder."""

    def __init__(
            self, enc_config: BertConfig,
            dec_config: RnnArgs,
            pooling_type: Literal["average", "cls_sep"],
            reg_args: Dict[str, Union[tf.Variable, int]],
            **kwargs
    ) -> None:
        """Initialize the model."""
        super().__init__(enc_config, dec_config, pooling_type=pooling_type,
                         reg_args=reg_args, **kwargs)
        self._decoder = SentAeGRUDecoder(self._dec_config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Call the full model.

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask.
            training: Specifies whether the model is being used in training mode.

        Returns:
            The logits of a probability distribution for next token prediction.
        """
        input_ids, attn_mask = inputs
        mean, log_var, enc_outputs = self._encoder((input_ids, attn_mask), training=training)
        sent_embedding = self._sampler((mean, log_var))
        logits = self._decoder((sent_embedding, enc_outputs[-1]), training=training)
        return logits
