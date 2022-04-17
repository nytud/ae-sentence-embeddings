# -*- coding: utf-8 -*-

"""Test binary classification metric with pre-sigmoid inputs"""

import tensorflow as tf

from ae_sentence_embeddings.losses_and_metrics import BinaryMCC, BinaryLogitAccuracy


class TestSparseCategoricalMCC(tf.test.TestCase):
    """Test class for the binary metrics"""

    def setUp(self) -> None:
        """Fixture setup. Get ground truth labels and the predictions."""
        super().setUp()
        self.y_true = tf.constant([[1], [1], [1], [0]])
        self.y_pred = tf.constant([[1.9], [-1.1], [2.05], [1.8]])

    def test_binary_mcc(self) -> None:
        """Test the MCC for binary classification."""
        expected_result = -1/3
        metric = BinaryMCC()
        metric.update_state(self.y_true, self.y_pred)
        result = metric.result().numpy().astype(float)
        self.assertAllClose(expected_result, result)

    def test_binary_accuracy(self) -> None:
        """Test the accuracy metric for binary classification from logits."""
        expected_result = 0.5
        metric = BinaryLogitAccuracy()
        metric.update_state(self.y_true, self.y_pred)
        result = metric.result().numpy().astype(float)
        self.assertAllClose(expected_result, result)


if __name__ == "__main__":
    tf.test.main()
