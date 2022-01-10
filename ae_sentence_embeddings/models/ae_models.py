"""A module for defining Transformer-based AEs"""

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model as KModel
from transformers.modeling_tf_utils import TFSharedEmbeddings
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig

from ae_sentence_embeddings.models import SentVaeEncoder, SentAeDecoder, SentAeEncoder
from ae_sentence_embeddings.layers import VaeSampling, AeGruDecoder
from ae_sentence_embeddings.modeling_tools import make_decoder_inputs
from ae_sentence_embeddings.argument_handling import RnnArgs


@tf.function
def _latent_loss(mean: tf.Tensor, log_var: tf.Tensor, attn_mask: tf.Tensor) -> tf.Tensor:
    """Calculate VAE latent loss

    Args:
        mean: The Gaussian mean vector
        log_var: The Gaussian variance vector
        attn_mask: Attention mask, which is necessary to calculate a normalizing constant
    """
    latent_loss = -0.1 * tf.reduce_sum(1 + log_var - tf.exp(log_var) - tf.square(mean), axis=-1)
    norm_nominator = tf.cast(tf.reduce_sum(attn_mask, axis=-1), mean.dtype)
    norm_denominator = tf.cast(tf.shape(mean)[-1], mean.dtype)
    norm_const = tf.divide(norm_nominator, norm_denominator)
    return tf.reduce_mean(tf.multiply(latent_loss, norm_const))


class BaseAe(KModel):
    """Base class for Transformer AEs. Used for subclassing only"""

    def __init__(self, enc_config: BertConfig, dec_config: OpenAIGPTConfig, **kwargs) -> None:
        """Initialize the invariant AE layers that do not depend on the AE architecture choice

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
        self.enc_embedding = TFSharedEmbeddings(
            vocab_size=self.enc_config.vocab_size,
            hidden_size=self.enc_config.hidden_size,
            initializer_range=self.enc_config.initializer_range
        )
        self.dec_embedding = TFSharedEmbeddings(
            vocab_size=self.dec_config.vocab_size,
            hidden_size=self.dec_config.n_embd,
            initializer_range=self.enc_config.initializer_range
        )
        self.decoder = SentAeDecoder(self.dec_config)

    def _call_decoder(self, sent_embedding: tf.Tensor, input_ids: tf.Tensor,
                      dec_attn_mask: tf.Tensor, training: Optional[bool]) -> tf.Tensor:
        """Call the decoder

        Args:
            sent_embedding: A sentence embedding tensor
            input_ids: A tensor of input token IDs
            dec_attn_mask: A tensor of decoder attention mask
            training: The training mode

        Returns:
            A tensor of logits

        """
        dec_embeddings = self.dec_embedding(input_ids[:, 1:], mode="embedding")
        dec_embeddings = tf.concat([tf.expand_dims(sent_embedding, axis=1), dec_embeddings], axis=1)
        dec_output = self.decoder((dec_embeddings, dec_attn_mask), training=training)
        logits = self.dec_embedding(dec_output, mode="linear")
        return logits


class TransformerAe(BaseAe):
    """A Transformer-based AE"""

    def __init__(self, enc_config: BertConfig, dec_config: OpenAIGPTConfig, **kwargs) -> None:
        """Initialize the AE

        Args:
            enc_config: The encoder configuration object
            dec_config: The decoder configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(enc_config, dec_config, **kwargs)
        self.encoder = SentAeEncoder(self.enc_config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in ae_training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            The logits of a probability distribution for next token prediction

        """
        input_ids, enc_attn_mask = inputs
        dec_attn_mask = make_decoder_inputs(enc_attn_mask)
        enc_embeddings = self.enc_embedding(input_ids, mode="embedding")
        sent_embedding, _ = self.encoder((enc_embeddings, enc_attn_mask), training=training)
        logits = self._call_decoder(
            sent_embedding=sent_embedding,
            input_ids=input_ids,
            dec_attn_mask=dec_attn_mask,
            training=training
        )
        return logits


class TransformerVae(BaseAe):
    """A Transformer-based VAE"""

    def __init__(self, enc_config: BertConfig, dec_config: OpenAIGPTConfig, **kwargs) -> None:
        """Initialize the VAE

        Args:
            enc_config: The encoder configuration object
            dec_config: The decoder configuration object
            **kwargs: Keyword arguments for the parent class
        """
        super().__init__(enc_config, dec_config, **kwargs)
        self.encoder = SentVaeEncoder(self.enc_config)
        self.sampler = VaeSampling()

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in ae_training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            The logits of a probability distribution for next token prediction
        """
        input_ids, enc_attn_mask = inputs
        dec_attn_mask = make_decoder_inputs(enc_attn_mask)
        enc_embeddings = self.enc_embedding(input_ids, mode="embedding")
        mean, log_var, _ = self.encoder((enc_embeddings, enc_attn_mask), training=training)
        self.add_loss(_latent_loss(mean, log_var, attn_mask=enc_attn_mask))
        sent_embedding = self.sampler((mean, log_var))
        logits = self._call_decoder(
            sent_embedding=sent_embedding,
            input_ids=input_ids,
            dec_attn_mask=dec_attn_mask,
            training=training
        )
        return logits


class BertRnnVae(KModel):
    """A VAE with a Bert encoder and an RNN decoder"""

    def __init__(self, enc_config: BertConfig, rnn_config: RnnArgs, **kwargs):
        if enc_config.vocab_size != rnn_config.vocab_size or enc_config.hidden_size != rnn_config.hidden_size:
            raise ValueError("Vocab size and hidden size should be the same in the encoder and the decoder")
        super().__init__(**kwargs)
        self.enc_config = enc_config
        self.dec_config = rnn_config
        self.enc_embedding = TFSharedEmbeddings(
            vocab_size=self.enc_config.vocab_size,
            hidden_size=self.enc_config.hidden_size,
            initializer_range=self.enc_config.initializer_range
        )
        self.dec_embedding = TFSharedEmbeddings(
            vocab_size=self.dec_config.vocab_size,
            hidden_size=self.dec_config.hidden_size,
            initializer_range=self.enc_config.initializer_range
        )
        self.encoder = SentVaeEncoder(self.enc_config)
        self.sampler = VaeSampling()
        self.decoder = AeGruDecoder(**self.dec_config.to_dict())

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in ae_training mode
            mask: Additional mask tensor. This will not be used

        Returns:
            The logits of a probability distribution for next token prediction
        """
        input_ids, attn_mask = inputs
        enc_embeddings = self.enc_embedding(input_ids, mode="embedding")
        mean, log_var, _ = self.encoder((enc_embeddings, attn_mask), training=training)
        self.add_loss(_latent_loss(mean, log_var, attn_mask=attn_mask))
        sent_embedding = self.sampler((mean, log_var))
        dec_embeddings = self.dec_embedding(input_ids, mode="embedding")
        dec_hidden_state = self.decoder((sent_embedding, dec_embeddings, attn_mask))
        logits = self.dec_embedding(dec_hidden_state, mode="linear")
        return logits
