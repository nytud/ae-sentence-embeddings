"""A module for defining Transformer-based AEs"""

from typing import Tuple, Union, Optional, Literal, Callable, Dict, Any
import pickle
from warnings import warn

import tensorflow as tf
from tensorflow.keras import Model as KModel, layers as tfl
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.openai.configuration_openai import OpenAIGPTConfig

from ae_sentence_embeddings.argument_handling import RnnArgs, RegularizedEmbeddingArgs
from ae_sentence_embeddings.layers import VaeSampling, RandomSwapLayer, RegularizedEmbedding
from ae_sentence_embeddings.models import (
    SentVaeEncoder,
    SentAeDecoder,
    SentAeEncoder,
    SentAeGRUDecoder,
    ae_double_gru
)


@tf.function
def latent_loss_func(mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
    """Calculate VAE latent loss

    Args:
        mean: The Gaussian mean vector
        log_var: The Gaussian variance vector
    """
    latent_loss_value = -0.5 * tf.reduce_sum(1 + (log_var - tf.square(mean) - tf.exp(log_var)), axis=-1)
    return tf.reduce_mean(latent_loss_value)


class BaseAe(KModel):
    """Base class for Transformer AEs. Used for subclassing only"""

    def __init__(self, enc_config: BertConfig,
                 dec_config: Union[OpenAIGPTConfig, RnnArgs], **kwargs) -> None:
        """Initialize the invariant AE layers that do not depend on the AE architecture choice

        Args:
            enc_config: The encoder configuration object
            dec_config: The decoder configuration object
            **kwargs: Keyword arguments for the parent class
        """
        if enc_config.vocab_size != dec_config.vocab_size:
            raise ValueError("Vocab size and should be the same in the encoder and the decoder")
        super().__init__(**kwargs)
        self.enc_config = enc_config
        self.dec_config = dec_config

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
                with open(optimizer_path, 'wb') as optimizer_f:
                    pickle.dump(optimizer_weights, optimizer_f)

    def load_checkpoint(self, weight_path: str, optimizer_path: Optional[str] = None,
                        **kwargs) -> None:
        """Load model from checkpoint

        Args:
            weight_path: Path to the saved model weights
            optimizer_path: Optional. Path to saved optimizer state file
            **kwargs: Further arguments passed to the `load_weights` method
        """
        self.load_weights(weight_path, **kwargs)
        if optimizer_path is not None:
            if self.optimizer is None:
                warn("Cannot load optimizer as no optimizer has been defined")
            else:
                with open(optimizer_path, 'rb') as f:
                    optimizer_weights = pickle.load(f)
                self.optimizer.set_weights(optimizer_weights)

    def save(
            self,
            filepath: str,
            overwrite: bool = True,
            include_optimizer: bool = True,
            save_format: Optional[Literal["tf", "h5"]] = None,
            signatures: Optional[Union[Callable, Dict]] = None,
            options: Optional[tf.train.CheckpointOptions] = None,
            save_traces: bool = True
    ) -> None:
        """Add a warning to the parent class method"""
        warn("This method is useful for fully Keras serializable models. As it might not work as expected, "
             "consider calling `model.save_weights` or `model.checkpoint`")
        super().save(filepath, overwrite=overwrite, include_optimizer=include_optimizer,
                     save_format=save_format, signatures=signatures,
                     save_traces=save_traces, options=options)


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
        self.decoder = SentAeDecoder(self.dec_config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in training mode

        Returns:
            The logits of a probability distribution for next token prediction

        """
        input_ids, attn_mask = inputs
        sent_embedding, _ = self.encoder((input_ids, attn_mask), training=training)
        logits = self.decoder(inputs=(sent_embedding, input_ids, attn_mask), training=training)
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
        self.decoder = SentAeDecoder(self.dec_config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in training mode

        Returns:
            The logits of a probability distribution for next token prediction
        """
        input_ids, attn_mask = inputs
        mean, log_var, _ = self.encoder((input_ids, attn_mask), training=training)
        self.add_loss(latent_loss_func(mean, log_var))
        sent_embedding = self.sampler((mean, log_var))
        logits = self.decoder(inputs=(sent_embedding, input_ids, attn_mask), training=training)
        return logits


class BertRnnVae(BaseAe):
    """A VAE with a Bert encoder and an RNN decoder"""

    def __init__(self, enc_config: BertConfig, rnn_config: RnnArgs, **kwargs) -> None:
        super().__init__(enc_config, rnn_config, **kwargs)
        self.encoder = SentVaeEncoder(self.enc_config)
        self.sampler = VaeSampling()
        self.decoder = SentAeGRUDecoder(self.dec_config)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: A tensor of input token IDs and a tensor of attention mask
            training: Specifies whether the model is being used in training mode

        Returns:
            The logits of a probability distribution for next token prediction
        """
        input_ids, attn_mask = inputs
        mean, log_var, _ = self.encoder((input_ids, attn_mask), training=training)
        self.add_loss(latent_loss_func(mean, log_var))
        sent_embedding = self.sampler((mean, log_var))
        logits = self.decoder((sent_embedding, input_ids, attn_mask), training=training)
        return logits


class BertBiRnnVae(BaseAe):
    """A Transformer-RNN VAE for bilingual training"""

    def __init__(self, enc_config: BertConfig, rnn_config: RnnArgs, **kwargs) -> None:
        super().__init__(enc_config, rnn_config, **kwargs)
        self.encoder = SentVaeEncoder(self.enc_config)
        self.sampler = VaeSampling()
        self.splitter = tfl.Lambda(lambda x: tf.split(x, 2))
        self.swapper = RandomSwapLayer()
        self.decoder = ae_double_gru(rnn_config)
        dec_embedding_config = RegularizedEmbeddingArgs(
            vocab_size=rnn_config.vocab_size,
            hidden_size=rnn_config.hidden_size,
            initializer_range=rnn_config.initializer_dev,
            layer_norm_eps=rnn_config.layernorm_eps,
            hidden_dropout_prob=max([2*rnn_config.dropout_rate, 0.5])
        )
        self.dec_embedding_layer = RegularizedEmbedding(dec_embedding_config)

    def call(self, inputs: Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the full model

        Args:
            inputs: Two tensor pairs, each of which consists of an input token IDs and
                    a tensor of attention mask, respectively
            training: Specifies whether the model is being used in training mode

        Returns:
            The logits of a probability distribution for next token prediction
        """
        lang1, lang2 = inputs
        input_ids = tf.concat([lang1[0], lang2[0]], axis=0)
        attn_mask = tf.concat([lang1[1], lang2[1]], axis=0)
        mean, log_var, _ = self.encoder((input_ids, attn_mask), training=training)
        latent_loss_val = latent_loss_func(mean, log_var)
        self.add_loss(latent_loss_val)
        self.add_metric(latent_loss_val, name="KL_loss")
        sents1, sents2 = self.splitter(self.sampler((mean, log_var)))
        sents1, sents2 = self.swapper(((sents1, sents2),), training=training)[0]

        dec_inputs1, dec_inputs2 = self.splitter(self.dec_embedding_layer(input_ids))
        logits = self.decoder((sents1, dec_inputs1, sents2, dec_inputs2), training=training)
        return logits

    def train_step(self, data) -> Dict[str, Any]:
        """Override parent class method in order to handle bilingual data correctly"""
        x, y = data
        y = tf.concat(y, axis=0)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data) -> Dict[str, Any]:
        """Override parent class method in order to handle bilingual data correctly"""
        x, y = data
        y = tf.concat(y, axis=0)
        return super().test_step((x, y))
