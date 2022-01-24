"""Test custom pooling layers"""

import tensorflow as tf

from ae_sentence_embeddings.layers import CLSPlusSEPPooling


class PoolingTest(tf.test.TestCase):
    """Test pooling layers"""

    def setUp(self) -> None:
        """Fixture setup. This will create a tensor of 3 axes that will be the test input
        and a tensor of 2 axes that will be used as an attention mask tensor
        """
        hs_matrix = [[n]*6 for n in range(1, 5)]
        self.hidden_states = tf.constant([hs_matrix, hs_matrix], dtype=tf.float32)
        self.attn_mask = tf.constant([[1, 1, 1, 1], [1, 1, 1, 0]])

    def test_cls_plus_sep(self) -> None:
        """Test CLS + SEP pooling"""
        expected_output = tf.constant([[1]*6 + [4]*6, [1]*6 + [3]*6], dtype=tf.float32)
        layer_output = CLSPlusSEPPooling()((self.hidden_states, self.attn_mask))
        print(f"The expected CLS + SEP output is:\n{expected_output}")
        print(f"The evaluated CLS + SEP output is:\n{layer_output}")
        self.assertAllEqual(expected_output, layer_output)


if __name__ == '__main__':
    tf.test.main()
