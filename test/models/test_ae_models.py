"""A module for AE model unittest"""

import tensorflow as tf
from transformers import BertConfig, OpenAIGPTConfig
import numpy as np

from ae_sentence_embeddings.models import TransformerVae


class TransformerVaeTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Create a dummy model"""
        super().setUpClass()
        enc_config = BertConfig(
            vocab_size=30,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=128,
            intermediate_size=512
        )
        dec_config = OpenAIGPTConfig(
            vocab_size=30,
            n_layer=2,
            n_head=2,
            n_embd=128,
        )
        cls.model = TransformerVae(enc_config, dec_config, pooling_type="average")

    def setUp(self) -> None:
        """Fixture setup. Create input token IDs and attention mask"""
        super().setUp()
        self.input_ids = tf.constant(np.random.randint(30, size=(2, 8)), dtype=tf.int32)
        self.attn_mask = tf.constant([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

    def test_model_transformer_vae_call(self) -> None:
        expected_shape = (2, 8, 30)
        logits = self.model((self.input_ids, self.attn_mask), training=False)
        print(f"The resulting logits are: {logits}")
        self.assertEqual(expected_shape, logits.shape)


if __name__ == "__main__":
    tf.test.main()
