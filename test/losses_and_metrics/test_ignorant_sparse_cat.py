"""Test sparse categorical crossentropy loss with mask labels"""

import numpy as np
import tensorflow as tf

from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy


class IgnorantSparseCatTest(tf.test.TestCase):

    def setUp(self) -> None:
        """Fixture setup

        This creates the following:
            a random tensor of shape (2, 4, 5), which represents the logits;
            an integer-valued tensor of shape (2, 4), which represent the class labels
        """
        super().setUp()
        self.logits = tf.keras.backend.random_normal(shape=(2, 4, 5))
        self.labels = tf.constant([[0, 3, 2, -1], [1, 4, -1, -1]])

    def test_ignorant_sparse_cat_crossentropy(self) -> None:
        """Test IgnorantSparseCatCrossentropy"""
        loss_func = IgnorantSparseCatCrossentropy(from_logits=True)
        loss_val = loss_func(self.labels, self.logits)
        self.assertShapeEqual(np.array(1.), loss_val, msg=f"Loss value is {loss_val}")


if __name__ == '__main__':
    tf.test.main()
