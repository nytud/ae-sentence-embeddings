# -*- coding: utf-8 -*-

"""Test the Matthews Correlation Coefficient with scalar ground truth labels"""

import tensorflow as tf

from ae_sentence_embeddings.losses_and_metrics import SparseCategoricalMCC


class TestSparseCategoricalMCC(tf.test.TestCase):
    """Test class for the MCC with scalar ground truth labels"""

    def test_sparse_cat_mcc(self) -> None:
        """Test the metric. Use the example given in the MCC documentation:
        https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/MatthewsCorrelationCoefficient
        """
        y_true = tf.constant([1, 1, 1, 0])
        y_pred = tf.constant([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        expected_result = -1/3

        metric = SparseCategoricalMCC(num_classes=2)
        metric.update_state(y_true, y_pred)
        result = metric.result().numpy().astype(float)
        self.assertAllClose(expected_result, result)
