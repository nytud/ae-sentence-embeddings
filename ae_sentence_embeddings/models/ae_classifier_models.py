# -*- coding: utf-8 -*-

"""A module for autoencoder-based sentence encoders with
a classifier head on top.
"""

from __future__ import annotations
from typing import Literal, Tuple, Optional, Dict, Any, Union

import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.utils import register_keras_serializable
from transformers import BertConfig
from transformers.modeling_tf_utils import get_initializer

from ae_sentence_embeddings.models import SentVaeEncoder, BaseAe


@register_keras_serializable(package="ae_sentence_embeddings.models")
class SentVaeClassifier(SentVaeEncoder, BaseAe):
    """The encoder part of a sentence embedding VAE
    with a classifier head on top.
    """

    def __init__(
            self,
            config: BertConfig,
            num_labels: int,
            pooling_type: Literal["average", "cls_sep", "p_means"] = "average",
            reg_args: Optional[Dict[str, Union[tf.Variable, int]]] = None,
            **kwargs
    ) -> None:
        """Initialize the encoder and the classifier head.
        Original code for defining the classifier head:
        https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/bert/modeling_tf_bert.py#L1595

        Args:
            config: A BERT configuration object.
            num_labels: The number of classification labels. Set it to `1` for binary classification.
            pooling_type: Pooling type, `'average'` or `'cls_sep'`. Defaults to `'average'`.
            reg_args: KL loss regularization arguments. If not specified, the KL loss will always be 0.
                Optional.
            **kwargs: Keyword arguments passed to the `keras.Model` initializer.

        Raises:
            `AssertionError` if `num_labels < 0`.
        """
        assert num_labels > 0, f"The number of labels must be a positive integer, got {num_labels}"
        if reg_args is None:
            reg_args = {"iters": 0, "warmup_iters": 1, "start": 0, "min_kl": 0.}
        super().__init__(config=config, pooling_type=pooling_type, reg_args=reg_args, **kwargs)
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

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "num_labels": self._num_labels}

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def enc_config(self) -> None:
        raise NotImplementedError(
            "Please call the `get_config` method to get the model configuration.")

    @property
    def dec_config(self) -> None:
        raise NotImplementedError(
            "This model does not have a decoder part. Please call the `get_config` "
            "method to get the model configuration.")
