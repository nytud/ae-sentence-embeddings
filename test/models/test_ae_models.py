# -*- coding: utf-8 -*-

"""A module for testing the full models."""

import tensorflow as tf
from transformers import BertConfig
import numpy as np

from ae_sentence_embeddings.models import BertRnnVae
from ae_sentence_embeddings.argument_handling import RnnArgs


class TransformerVaeTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Create a dummy model."""
        super().setUpClass()
        enc_config = BertConfig(
            vocab_size=30,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=128,
            intermediate_size=512
        )
        dec_config = RnnArgs(
            num_rnn_layers=2,
            vocab_size=30,
            hidden_size=128
        )
        reg_args = {
            "iters": 0,
            "warmup_iters": 4
        }
        cls._model = BertRnnVae(enc_config, dec_config,
                                reg_args=reg_args, pooling_type="average")

    def setUp(self) -> None:
        """Fixture setup. Create input token IDs and attention mask."""
        super().setUp()
        self._input_ids = tf.constant(np.random.randint(30, size=(2, 8)), dtype=tf.int32)
        self._attn_mask = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

    def test_bert_rnn_vae_call(self) -> None:
        """Call the dummy model and check if the output shape is correct."""
        expected_shape = (2, 8, 30)
        # noinspection PyCallingNonCallable
        logits = self._model((self._input_ids, self._attn_mask), training=False)
        self.assertEqual(expected_shape, logits.shape)


if __name__ == "__main__":
    tf.test.main()
