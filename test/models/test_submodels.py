"""Test encoder or decoder modules"""

import tensorflow as tf
from numpy.random import rand as np_rand, randint as np_randint

from ae_sentence_embeddings.models import SentAeGRUDecoder
from ae_sentence_embeddings.argument_handling import RnnArgs


class SubmodelTest(tf.test.TestCase):
    """Test encoder or decoder submodels"""

    def test_sent_ae_gru_decoder(self) -> None:
        """Test a GRU-based decoder"""
        vocab_size = 128
        hidden_size = 64
        sequence_length = 16
        batch_size = 2
        dummy_sent_embedding = tf.constant(np_rand(batch_size, hidden_size), dtype=tf.float32)
        dummy_ids = tf.constant(np_randint(vocab_size, size=(batch_size, sequence_length)))
        config = RnnArgs(
            num_rnn_layers=2,
            hidden_size=hidden_size,
            vocab_size=vocab_size
        )
        model = SentAeGRUDecoder(config)
        result = model((dummy_sent_embedding, dummy_ids))
        self.assertAllEqual((batch_size, sequence_length, vocab_size), result.shape,
                            msg=f"The RNN output is of shape:\n{result.shape}")


if __name__ == '__main__':
    tf.test.main()
