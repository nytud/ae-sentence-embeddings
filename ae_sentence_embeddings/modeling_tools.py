"""A module for functions that adapt inputs to layers"""

from typing import Dict, List, Union, Any, Optional, Literal
from argparse import ArgumentParser
import json
import logging

import tensorflow as tf
from tensorflow.python.framework.dtypes import DType
from transformers.modeling_tf_utils import shape_list


def process_attention_mask(
        attention_mask: tf.Tensor,
        embedding_dtype: Union[str, DType] = tf.float32,
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


def make_decoder_inputs(input_tensor: tf.Tensor, pad_value: int = 0) -> tf.Tensor:
    """Modify attention mask so that the decoder can use it correctly.
    This means removing the first column and appending a new column filled with `pad_value`.
    This can also be applied to create decoder targets token IDs.

    Args:
        input_tensor: a tensor of shape `(batch_size, sequence_length)`
        pad_value: The value with which the appended column will be filled. Defaults to 0

    Returns:
        The modified tensor of the same shape as `input_tensor`
    """
    pad_col = tf.zeros((tf.shape(input_tensor)[0], 1), dtype=input_tensor.dtype) + pad_value
    return tf.concat([input_tensor[:, 1:], pad_col], axis=-1)


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


def read_json(file_path: str) -> Dict[str, Any]:
    """Read in a `json` configuration file"""
    with open(file_path, 'rb') as f:
        json_dict = json.load(f)
    return json_dict


def get_training_args() -> ArgumentParser:
    """Get an `ArgumentParser` to handle command line arguments"""
    parser = ArgumentParser(description="Command line arguments for model ae_training")
    parser.add_argument("config_file", help="Path to a `json` configuration file")
    return parser


def get_custom_logger(
        log_path: str,
        logger_name: Optional[str] = None,
        level: Optional[int] = None,
        log_format: Optional[str] = None,
        f_mode: Literal['a', 'w'] = 'a',
        encoding: str = "utf-8"
) -> logging.Logger:
    """Get a custom logger instance

    Args:
        log_path: Path to a file where logs can be written
        logger_name: Optional. Logger name. If not specified, it will be set to `UserLogger`
        level: Optional. A valid logging level. If not specified, it will be set to `DEBUG`
        log_format: Optional. A format for logging messages. If not specified, it will be set to
                    `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
        f_mode: Opening mode for the log file. Defaults to `'a'` (append)
        encoding: Log file encoding. Defaults to `"utf-8"`

    Returns:
        A file logger
    """
    if logger_name is None:
        logger_name = "UserLogger"
    logger = logging.getLogger(logger_name)
    if level is None:
        level = logging.DEBUG
    logger.setLevel(level)
    file_handler = logging.FileHandler(
        filename=log_path, mode=f_mode, encoding=encoding)
    file_handler.setLevel(level)
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
