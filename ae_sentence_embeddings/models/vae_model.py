"""A module for defining a Transformer-based VAE"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model as KModel
from transformers.modeling_tf_utils import TFSharedEmbeddings
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig

from ae_sentence_embeddings.models import SentAeEncoder, SentAeDecoder
from ae_sentence_embeddings.layers import VaeSampling
from ae_sentence_embeddings.modeling_tools import make_decoder_mask


class TransformerVae(KModel):
    """The Transformer-based VAE"""

    def __init__(self, enc_config: BertConfig, dec_config: OpenAIGPTConfig, **kwargs) -> None:
        """Initialize the VAE

        Args:
            enc_config: The encoder configuration object
            dec_config: The decoder configuration object
            **kwargs: Keyword arguments for the parent class
        """
        if enc_config.vocab_size != dec_config.vocab_size or enc_config.hidden_size != dec_config.n_embd:
            raise ValueError("Vocab size and hidden size should be the same in the encoder and the decoder")
        super().__init__(**kwargs)
        self.enc_config = enc_config
        self.dec_config = dec_config
        self.embedding = TFSharedEmbeddings(
            vocab_size=self.enc_config.vocab_size,
            hidden_size=self.enc_config.hidden_size,
            initializer_range=self.enc_config.initializer_range
        )
        self.encoder = SentAeEncoder(self.enc_config)
        self.decoder = SentAeDecoder(self.dec_config)
        self.sampler = VaeSampling()

    @staticmethod
    def _latent_loss(mean: tf.Tensor, log_var: tf.Tensor, attn_mask: tf.Tensor) -> tf.Tensor:
        """Calculate VAE latent loss

        Args:
            mean: The Gaussian mean vector
            log_var: The Gaussian variance vector
            attn_mask: Attention mask, which is necessary to calculate a normalizing constant
        """
        latent_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.exp(log_var) - tf.square(mean), axis=-1)
        norm_nominator = tf.cast(tf.reduce_sum(attn_mask, axis=-1), mean.dtype)
        norm_denominator = tf.cast(tf.shape(mean)[-1], mean.dtype)
        norm_const = tf.divide(norm_nominator, norm_denominator)
        return tf.reduce_mean(tf.multiply(latent_loss, norm_const))

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            The logits of a probability distribution for next token prediction
        """
        input_ids, enc_attn_mask = inputs
        dec_attn_mask = make_decoder_mask(enc_attn_mask)
        enc_embeddings = self.embedding(input_ids, mode="embedding")
        mean, log_var, _ = self.encoder((enc_embeddings, enc_attn_mask), training=training)
        self.add_loss(self._latent_loss(mean, log_var, attn_mask=enc_attn_mask))
        sent_embedding = self.sampler((mean, log_var))
        dec_embeddings = tf.concat([tf.expand_dims(sent_embedding, axis=1), enc_embeddings[:, 1:, :]], axis=1)
        dec_output = self.decoder((dec_embeddings, dec_attn_mask), training=training)
        logits = self.embedding(dec_output, mode="linear")
        return logits
