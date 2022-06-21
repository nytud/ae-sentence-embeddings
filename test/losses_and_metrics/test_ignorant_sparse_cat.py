# -*- coding: utf-8 -*-

"""Test sparse categorical crossentropy loss with mask labels"""

import tensorflow as tf

from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy


class IgnorantSparseCatTest(tf.test.TestCase):

    def setUp(self) -> None:
        """Fixture setup.

        This creates the following:
            a tensor of shape `(2, 2, 3)`, which represents the logits;
            an integer-valued tensor of shape `(2, 2)`, which represent the class labels.
        """
        super().setUp()
        self._logits = tf.convert_to_tensor([[[1.5, -0.3, 0.2], [2.1, -0.7, -1.9]]]*2)
        self._labels = tf.convert_to_tensor([[0, 1], [0, -1]])

    def test_ignorant_sparse_cat_crossentropy(self) -> None:
        """Test IgnorantSparseCatCrossentropy."""
        loss_func = IgnorantSparseCatCrossentropy(from_logits=True)
        expected_result = 1.8012111
        # noinspection PyCallingNonCallable
        loss_val = loss_func(self._labels, self._logits)
        self.assertEqual(expected_result, loss_val)


if __name__ == "__main__":
    tf.test.main()
