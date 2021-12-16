"""A module for functions that adapt inputs to layers"""

from typing import Dict, List, Union
import tensorflow as tf
from transformers.modeling_tf_utils import shape_list


def process_attention_mask(
        attention_mask: tf.Tensor,
        embedding_dtype: Union[str, tf.float32, tf.float32] = tf.float32,
) -> tf.Tensor:
    """Reshape attention mask and modify its values

    Args:
        attention_mask: A 2D tensor with shape `(batch_size, sequence_length)` that contains attention mask.
                        Possible values of an attention mask are `1` and `0`, where `0` stands for token mask
                        and `1` means no mask.
        embedding_dtype: Expected datatype of the tensors returned by an embedding layer. Defaults to `float32`

    Returns:
       The modified attention mask tensor

    """
    attention_mask_shape = shape_list(attention_mask)
    extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]))
    extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_dtype)
    one_cst = tf.constant(1.0, dtype=embedding_dtype)
    ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_dtype)
    extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
    return extended_attention_mask


def make_decoder_mask(attention_mask: tf.Tensor) -> tf.Tensor:
    """Modify attention mask so that the decoder can use it correctly"""
    zero_col = tf.zeros((tf.shape(attention_mask)[0], 1), dtype=attention_mask.dtype)
    return tf.concat([attention_mask[:, 1:], zero_col], axis=-1)


# This is no longer needed
def make_dummy_bert_inputs(
        input_tensor: tf.Tensor,
        num_hidden_layers: int = 6,
) -> Dict[str, Union[tf.Tensor, List[None]]]:
    """Create dummy BERT inputs so that layers can handle the data. Source:
    https://github.com/huggingface/transformers/blob/546a91abe95117fb06d3ed34edfa0b8010d9b48c/src/transformers/models/
    bert/modeling_tf_bert.py

    Args:
        input_tensor: A 2D tensor with shape `(batch_size, sequence_length)` that contains token IDs or attention mask
        num_hidden_layers: Number of BERT hidden layers. Defaults to 6

    Returns:
        A dictionary with inputs required by HuggingFace BERT layers

    """
    input_shape = shape_list(input_tensor)
    head_mask = [None] * num_hidden_layers
    token_type_ids = tf.fill(dims=input_shape, value=0)
    past_key_values = [None] * num_hidden_layers
    return {
        "head_mask": head_mask,
        "token_type_ids": token_type_ids,
        "past_key_values": past_key_values,
    }
