"""Test layers connected to a Transformer architecture"""

import tensorflow as tf

from ae_sentence_embeddings.layers import SinusoidalEmbedding, RegularizedEmbedding
from ae_sentence_embeddings.argument_handling import PositionalEmbeddingArgs, RegularizedEmbeddingArgs


class AeTransformerTest(tf.test.TestCase):
    """Test class for Transformer-connected layers"""

    def setUp(self) -> None:
        """Fixture setup. This creates some dummy input"""
        super().setUp()
        self.dummy_input = tf.constant([[1, 26, 15, 30, 0, 0], [1, 8, 19, 10, 21, 0]])

    def test_regularized_embedding(self) -> None:
        """Test regularized embeddings without positional information"""
        embedding_args = RegularizedEmbeddingArgs(vocab_size=32, hidden_size=8)
        embedding_layer = RegularizedEmbedding(embedding_args)
        expected_shape = (2, 6, 8)
        res = embedding_layer(self.dummy_input)
        self.assertAllEqual(expected_shape, tf.shape(res), msg=f"The result is:\n{res}")

    def test_sinusoidal_embedding(self) -> None:
        """Test sinusoidal positional embeddings"""
        embedding_args = PositionalEmbeddingArgs(
            vocab_size=32,
            max_position_embeddings=16,
            hidden_size=8
        )
        embedding_layer = SinusoidalEmbedding(embedding_args)
        expected_shape = (2, 6, 8)
        res = embedding_layer(self.dummy_input)
        self.assertAllEqual(expected_shape, tf.shape(res), msg=f"The result is:\n{res}")


if __name__ == '__main__':
    tf.test.main()
