"""Test regularization loss functions"""

import tensorflow as tf
import numpy as np

from ae_sentence_embeddings.losses_and_metrics import kl_loss_func


class RegularizationLossTest(tf.test.TestCase):
    """Test case for regularization losses"""

    def test_latent_loss(self) -> None:
        """Test latent loss calculation"""
        mean_vecs = tf.keras.backend.random_normal((2, 64))
        logvar_vecs = tf.keras.backend.random_normal((2, 64))
        latent_loss = kl_loss_func(mean_vecs, logvar_vecs)
        print(f"Latent loss is: {latent_loss}")
        zero_dim_arr = np.array(0)
        self.assertShapeEqual(zero_dim_arr, latent_loss)


if __name__ == '__main__':
    tf.test.main()
