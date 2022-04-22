# -*- coding: utf-8 -*-

"""A module for autoencoder-based sentence encoders with
a classifier head on top.
"""

from __future__ import annotations
from typing import Literal, Tuple, Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from transformers import BertConfig
from transformers.modeling_tf_utils import get_initializer

from ae_sentence_embeddings.models import SentVaeEncoder


@register_keras_serializable(package="ae_sentence_embeddings.models")
class SentVaeClassifier(SentVaeEncoder):
    """The encoder part of a sentence embedding VAE
    with a classifier head on top.
    """

    def __init__(
            self,
            config: BertConfig,
            num_labels: int,
            pooling_type: Literal["average", "cls_sep"] = "cls_sep",
            kl_factor: float = 0.,
            **kwargs
    ) -> None:
        """Initialize the encoder and the classifier head.
        Original code for defining the classifier head:
        https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/bert/modeling_tf_bert.py#L1595

        Args:
            config: A BERT configuration object.
            num_labels: The number of classification labels. Set it to `1` for binary classification.
            pooling_type: Pooling type, `'average'` or `'cls_sep'`. Defaults to `'cls_sep'`.
            kl_factor: A normalizing constant by which the KL loss will be multiplied.
                Set it to zero if the KL loss should be ignored. Defaults to `0.0`.
            **kwargs: Keyword arguments passed to the `keras.Model` initializer.
        """
        if num_labels <= 0:
            raise ValueError(f"The number of labels must be a positive integer, got {num_labels}")
        super().__init__(config=config, pooling_type=pooling_type,
                         kl_factor=kl_factor, **kwargs)
        self._num_labels = num_labels
        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None \
            else config.hidden_dropout_prob
        self._dropout = tfl.Dropout(rate=classifier_dropout)
        self._classifier = tfl.Dense(
            units=num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier"
        )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the model.

        Args:
            inputs: An input token ID tensor and an attention mask tensor,
                both of shape `(batch_size, sequence_length)`.
            training: Specifies whether the model is being used in training mode.

        Returns:
            A tensor of shape `(batch_size, num_labels)` if `num_labels > 1` or
            a tensor of shape `(batch_size)` otherwise.
        """
        post_pooling_mean, _, _ = super().call(inputs, training=training)
        return self._classifier(self._dropout(post_pooling_mean, training=training))

    @classmethod
    def from_pretrained(cls, ckpt_path: str, num_labels: int = 1,
                        kl_factor: Optional[float] = 0.) -> SentVaeClassifier:
        """Load the encoder weights from a pre-trained model.

        Args:
            ckpt_path: Path to a Keras-serialized model checkpoint.
            num_labels: Number of classification labels. Defaults to `1`.
            kl_factor: A value which will override the KL multiplier of the
                pre-trained model. Set it to `None` to omit this. Defaults to `0.`.

        Returns:
            A model whit pre-trained encoder weights and newly initialized classifier weights.
        """
        if num_labels <= 0:
            raise ValueError(f"The number of labels must be a positive integer, got {num_labels}")

        # load the pre-trained model and get its configuration
        pre_trained_model = load_model(ckpt_path)
        pre_trained_config = pre_trained_model.get_config()
        pre_trained_encoder_config = BertConfig(**pre_trained_config["encoder_config"])
        pre_trained_encoder_config.num_labels = num_labels
        new_kl_factor = kl_factor if kl_factor is not None else pre_trained_config["kl_factor"]

        # create the new model
        new_model = cls(pre_trained_encoder_config, pooling_type=pre_trained_config["pooling_type"],
                        kl_factor=new_kl_factor, num_labels=num_labels)
        # build the model by calling it on dummy inputs
        dummy_inputs = (tf.constant([[0, 1, 2]]), tf.constant([[1, 1, 1]]))
        # noinspection PyCallingNonCallable
        _ = new_model(dummy_inputs, training=False)

        # copy the weights
        for pre_trained_layer, new_layer in zip(pre_trained_model.layers, new_model.layers):
            new_layer.set_weights(pre_trained_layer.get_weights())
        return new_model

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "num_labels": self._num_labels}

    @property
    def num_labels(self) -> int:
        return self._num_labels
