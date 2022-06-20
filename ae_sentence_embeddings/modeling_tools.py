"""A module for functions that adapt inputs to layers
and other utils: logging, I/O handling, n-gram iteration"""

import json
import logging
from os.path import isfile
from contextlib import contextmanager
from time import time
from datetime import timedelta
from typing import Dict, Tuple, Union, Any, Optional, Literal, Iterable, Iterator
from argparse import ArgumentParser, ArgumentTypeError
from itertools import tee, islice

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


def read_json(file_path: str) -> Dict[str, Any]:
    """Read in a `json` configuration file."""
    if not isfile(file_path):
        raise ArgumentTypeError(f"{file_path} is not a path to a file.")
    with open(file_path, "rb") as f:
        json_dict = json.load(f)
    return json_dict


def get_training_args() -> ArgumentParser:
    """Get an `ArgumentParser` to handle command line arguments"""
    parser = ArgumentParser(description="Command line arguments for model ae_training")
    parser.add_argument("config_file", help="Path to a `json` configuration file")
    return parser


def get_custom_logger(
        log_path: Optional[str] = None,
        level: Optional[int] = None,
        log_format: Optional[str] = None,
        logger_name: str = "UserLogger",
        f_mode: Literal["a", "w"] = "a",
        encoding: str = "utf-8"
) -> logging.Logger:
    """Get a custom logger instance

    Args:
        log_path: Path to a file where logs can be written. If not specified, the logger
            will print to `stderr`.
        level: Optional. A valid logging level. If not specified, it will be set to `DEBUG`.
        log_format: Optional. A format for logging messages. If not specified, it will be set to
                    `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`.
        logger_name: The logger name. Defaults to `'UserLogger'`.
        f_mode: Opening mode for the log file. Relevant only if `log_path` is specified.
            Defaults to `'a'` (append).
        encoding: Log file encoding. Relevant only if `log_path` is specified. Defaults to `"utf-8"`.

    Returns:
        A file logger
    """
    logger = logging.getLogger(logger_name)
    if level is None:
        level = logging.DEBUG
    logger.setLevel(level)
    if log_path is not None:
        handler = logging.FileHandler(filename=log_path, mode=f_mode, encoding=encoding)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(level)
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def make_ngram_iter(it: Iterable, ngram_length: int, step_size: int = 1) -> Iterator[Tuple]:
    """Create an n-gram iterator. This is similar to the implementation at
    https://github.com/dlazesz/n-gram-benchmark/blob/master/ngram.py

    Args:
        it: Any iterable object. The n-grams will be created from its elements
        ngram_length: Length of the n-grams; e.g. `ngram_length=2` means using bigrams.
        step_size: Step size between the starting elements of the n-grams. It can be useful
            if non-overlapping n-grams need to be created. For example, `ngram_length=2, step_size=2`
            creates non-overlapping bigrams. Defaults to `1`

    Returns:
        An iterator of tuples where the tuples are n-grams
    """
    return zip(*(islice(tee_it, i, None, step_size) for i, tee_it in enumerate(tee(it, ngram_length))))


@contextmanager
def timing(description: str, time_logger: logging.Logger) -> None:
    """Create a context manager to measure execution time.

    Args:
        description: Short description of the wrapped code.
        time_logger: A logger that will write DEBUG messages.
    """
    time_logger.debug(f"Starting: {description}")
    start = time()
    yield
    elapsed_time = str(timedelta(seconds=time() - start))
    time_logger.debug(f"Done: {description}: {elapsed_time}")
