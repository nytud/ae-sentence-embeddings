# -*- coding: utf-8 -*-

"""Test the KL loss regularizer."""

import tensorflow as tf

from ae_sentence_embeddings.regularizers import KLDivergenceRegularizer


class RegularizerTest(tf.test.TestCase):
    """A test case for custom KL loss regularizers."""

    def setUp(self) -> None:
        """Fixture setup. This creates the following data:
            1. A tensor of shape `(2, 4)` that comprises the mean and a variance vectors;
            2. A `tf.Variable` on which the KL loss beta depends.
        """
        super(RegularizerTest, self).setUp()
        mean = tf.convert_to_tensor([[1.5, -0.9], [-0.4, 1.0]])
        logvar = tf.convert_to_tensor([[-1.8, 0.5], [-0.1, 1.2]])
        self._gaussian = tf.concat([mean, logvar], axis=-1)
        self._iteration = tf.Variable(0)

    def test_kl_divergence_regularizer(self) -> None:
        """Test the KL divergence regularizer that implements KL annealing
        and KL thresholding.
        """
        regularizer = KLDivergenceRegularizer(
            iters=self._iteration,
            warmup_iters=2,
            start=2,
            min_kl=0.1
        )
        expected_results = (0., 1.6147435, 3.2470684)
        res1 = regularizer(self._gaussian)
        self._iteration.assign(3)
        res2 = regularizer(self._gaussian)
        self._iteration.assign(6)
        res3 = regularizer(self._gaussian)
        self.assertAllClose(expected_results, (res1, res2, res3))


if __name__ == "__main__":
    tf.test.main()
