# -*- coding: utf-8 -*-

"""A module for testing modeling tools"""

import tensorflow as tf

from ae_sentence_embeddings.modeling_tools import make_decoder_inputs, process_attention_mask


class ModelingToolTest(tf.test.TestCase):

    def setUp(self) -> None:
        super().setUp()
        """Fixture setup. Create an attention mask tensor"""
        self.attn_mask = tf.constant([[1, 1, 1, 0], [1, 1, 1, 1]])

    def test_make_decoder_inputs(self) -> None:
        """Test creating a decoder mask from an encoder mask"""
        expected_dec_attn_mask = tf.constant([[1, 1, 0, 0], [1, 1, 1, 0]])
        dec_attn_mask = make_decoder_inputs(self.attn_mask)
        self.assertAllEqual(expected_dec_attn_mask, dec_attn_mask, msg=f"Resulting mask is {dec_attn_mask}")

    def test_process_attention_mask(self) -> None:
        """Test processed attention mask shape"""
        batch_size, seq_len_size = self.attn_mask.shape
        processed_attn_mask = process_attention_mask(self.attn_mask)
        expected_shape = (batch_size, 1, 1, seq_len_size)
        self.assertEqual(expected_shape, processed_attn_mask.shape,
                         msg=f"The resulting shape is {processed_attn_mask.shape}")


if __name__ == "__main__":
    tf.test.main()
