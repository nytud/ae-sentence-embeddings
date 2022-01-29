"""Test layers designed for bilingual training"""

from typing import Tuple

import tensorflow as tf
import numpy as np

from ae_sentence_embeddings.layers import RandomSwapLayer


class BilingLayerTest(tf.test.TestCase):
    """Test lass for layers designed for bilingual training"""

    def setUp(self) -> None:
        """Fixture setup. This creates two tensor pairs (tuples)"""
        super().setUp()
        self.pair1 = (
            tf.reshape(tf.range(12), (2, 2, -1)),
            tf.reshape(tf.range(-11, 1), (2, 2, -1))
        )
        self.pair2 = (
            tf.ones_like(self.pair1[0]),
            tf.zeros_like(self.pair1[1])
        )

    @staticmethod
    def is_swapped(pair1: Tuple[tf.Tensor, tf.Tensor],
                   pair2: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Check if `pair1` and `pair2` contains the same two tensors in different orders
        This method returns a Boolean tensor with a single value
        """
        p11, p12 = pair1
        p21, p22 = pair2
        res = tf.reduce_all(tf.equal(p11, p22)) and tf.reduce_all(tf.equal(p12, p21))
        if not res:
            if not (tf.reduce_all(tf.equal(p11, p21)) and tf.reduce_all(tf.equal(p12, p22))):
                raise ValueError("The tensors in the 2 pairs are not the same!")
        return res

    def test_random_swap_layer(self) -> None:
        """Test the layer that randomly swaps inputs"""
        tf.random.set_seed(42)
        np.random.seed(42)
        swapper = RandomSwapLayer()
        results = []
        for _ in range(1000):
            pair11, pair12 = swapper((self.pair1,), training=True)[0]
            result = self.is_swapped((pair11, pair12), self.pair1)
            results.append(tf.cast(result, tf.int16))
        swapped_times = tf.reduce_sum(results)
        print(f"Swapped tensors this many times:\n{swapped_times}")
        self.assertAllInRange(swapped_times, 450, 550)

        swapper.p = 1.0
        pair1, pair2 = swapper((self.pair1, self.pair2), training=True)
        self.assertAllEqual(pair1[::-1], self.pair1, msg=f"Tensors not swapped!\n{pair1}")
        self.assertAllEqual(pair2[::-1], self.pair2, msg=f"Tensors not swapped!\n{pair2}")

        swapper.p = 0.0
        pair1, pair2 = swapper((self.pair1, self.pair2), training=True)
        self.assertAllEqual(pair1, self.pair1, msg=f"Tensors swapped!\n{pair1}")
        self.assertAllEqual(pair2, self.pair2, msg=f"Tensors swapped!\n{pair2}")


if __name__ == '__main__':
    tf.test.main()
