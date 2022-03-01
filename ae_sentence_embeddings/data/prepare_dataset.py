"""A module for preparing datasets for usage with TensorFlow/Keras"""

from typing import Generator, Tuple, Optional, Iterable, Union
from copy import deepcopy
from functools import partial

import tensorflow as tf
from tensorflow.data import Dataset as TFDataset  # This is a correct import. PyCharm may not find `data`.
from datasets import Dataset as HgfDataset, load_dataset

from ae_sentence_embeddings.argument_handling import DataStreamArgs, DataSplitPathArgs
from ae_sentence_embeddings.modeling_tools import make_ngram_iter

MultiLingTensorStruct = Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]


def convert_to_tf_dataset(dataset: HgfDataset) -> TFDataset:
    """Convert a `datasets.Dataset` object to a TensorFlow dataset using a generator

    Args:
        dataset: The original `datasets.Dataset` object

    Returns:
        A TensorFlow dataset

    """
    target_names = ("target", "label")
    target_cols = sorted((col for col in dataset.features.keys() if col.startswith(target_names)),
                         reverse=True)
    feature_cols = sorted([feature_col for feature_col in dataset.features.keys()
                           if feature_col not in target_cols and not feature_col.startswith("text")],
                          reverse=True)
    feature_spec = (tf.TensorSpec(shape=(None,), dtype=tf.int32),) * len(feature_cols)

    if len(target_cols) == 1:
        target_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32)
        out_spec = (feature_spec, target_spec)

        def data_gen() -> Generator[Tuple[Tuple[tf.Tensor, ...], tf.Tensor], None, None]:
            for example in dataset:
                input_fields = tuple(tf.constant(example[feature]) for feature in feature_cols)
                target_field = tf.constant(example[target_cols[0]])
                yield input_fields, target_field

    else:
        target_spec = (tf.TensorSpec(shape=(None,), dtype=tf.int32),) * len(target_cols)
        out_spec = (feature_spec, target_spec)

        def data_gen() -> Generator[MultiLingTensorStruct, None, None]:
            for example in dataset:
                input_fields = tuple(tf.constant(example[feature]) for feature in feature_cols)
                target_field = tuple(tf.constant(example[target]) for target in target_cols)
                yield input_fields, target_field

    return TFDataset.from_generator(data_gen, output_signature=out_spec)


@tf.function
def pre_pad_multilingual(feature_tensors: Tuple[tf.Tensor, ...],
                         target_tensors: Tuple[tf.Tensor, ...],
                         padding_tuples: Tuple[Iterable[int], Iterable[int]]) -> MultiLingTensorStruct:
    """Pad tensors in a single example in a multilingual dataset so that they have the same length

    Args:
        feature_tensors: A tuple of feature tensors
        target_tensors: A tuple of target tensors
        padding_tuples: Padding values nested with the structure `(feature_tensors, target_tensors)`

    Returns:
        The padded tensors nested with the structure `(feature_tensors, target_tensors)`
    """
    outputs = []
    for tensor_tuple, padding_tuple in zip((feature_tensors, target_tensors), padding_tuples):
        new_tensors = []
        max_len = tf.reduce_max([tf.shape(tensor)[0] for tensor in tensor_tuple])
        for tensor, padding_value in zip(tensor_tuple, padding_tuple):
            new_tensor = tf.pad(tensor, [(0, max_len - tf.shape(tensor)[0])],
                                mode="CONSTANT", constant_values=padding_value)
            new_tensors.append(new_tensor)
        outputs.append(tuple(new_tensors))
    # The output type is correct, PyCharm may detect a mismatch between expected type and returned type
    return tuple(outputs)


def pad_and_batch(
        tf_dataset: TFDataset,
        data_stream_args: DataStreamArgs,
        drop_remainder: bool = True
) -> TFDataset:
    """Pad and batch a TensorFlow dataset

    Variable calculations for bucketing:
        bucket_boundaries: If both the `num_buckets` and `first_bucket_boundary` fields of the input dataclass are
                           specified, the nth bucket boundary will be calculated as
                           `(first_bucket_boundary-1) * 2**n + 1`. This variable will not be calculated otherwise
        batch_sizes: If bucketing is used, the batch size for the nth bucket will be calculated as `batch_size // 2**n`.
                     This variable will not be calculated otherwise.

    Args:
        tf_dataset: A `tensorflow.data.Dataset` object
        data_stream_args: A dataclass that contains data streaming arguments as defined in `argument_handling.py`.
        drop_remainder: Specify whether the last batch should be dropped. Defaults to `True`

    Returns:
        The padded and batched TensorFlow dataset

    """
    if data_stream_args.shuffling_buffer_size is not None:
        tf_dataset = tf_dataset.shuffle(buffer_size=data_stream_args.shuffling_buffer_size)

    if isinstance(input_padding := data_stream_args.input_padding, list):
        input_padding = tuple(input_padding)
    if isinstance(target_padding := data_stream_args.target_padding, list):
        target_padding = tuple(input_padding)
    padding_values = (input_padding, target_padding)
    if isinstance(target_padding, tuple):
        tf_dataset = tf_dataset.map(partial(pre_pad_multilingual, padding_tuples=padding_values))

    if data_stream_args.num_buckets is None or data_stream_args.first_bucket_boundary is None:
        tf_dataset = tf_dataset.padded_batch(data_stream_args.batch_size, padding_values=padding_values)
    else:
        bucket_boundaries = [(data_stream_args.first_bucket_boundary - 1) * 2 ** n + 1
                             for n in range(data_stream_args.num_buckets - 1)]
        bucket_batch_sizes = [data_stream_args.batch_size // 2 ** n
                              for n in range(data_stream_args.num_buckets)]
        tf_dataset = tf_dataset.bucket_by_sequence_length(
            element_length_func=lambda features, targets: tf.shape(features[0])[0],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padding_values=padding_values,
            drop_remainder=drop_remainder
        )
    return tf_dataset


@tf.function
def post_batch_multilingual(
        feature_tensors: Tuple[tf.Tensor, ...],
        target_tensors: Tuple[tf.Tensor, ...]
) -> Union[MultiLingTensorStruct, Tuple[Union[Tuple[tf.Tensor, ...], tf.Tensor], tf.Tensor]]:
    """Reorganize padded batches of feature and target tensors. Every `2k`th and `2k-1`th (k=1,...)
    tensor will be concatenated

    Args:
        feature_tensors: A tuple of feature tensors
        target_tensors: A tuple of target tensors

    Returns:
        The new feature and target tensors. If the original batch consisted of N tensors, the new one will
        consist of N/2 tensors
    """
    outputs = []
    for tensor_tuple in (feature_tensors, target_tensors):
        new_tensors = tuple(tf.concat([left, right], axis=0) for left, right in make_ngram_iter(tensor_tuple, 2, 2))
        if len(new_tensors) == 1:
            new_tensors = new_tensors[0]
        outputs.append(new_tensors)
    # The output type is correct, PyCharm may detect a mismatch between expected type and returned type
    return tuple(outputs)


def post_batch_feature_pair(
        feature_tensors: Tuple[tf.Tensor, ...],
        target_tensors: Tuple[tf.Tensor, ...],
        num_languages: int = 2
) -> Tuple[Tuple[Tuple[tf.Tensor, tf.Tensor], ...], Union[Tuple[tf.Tensor], tf.Tensor]]:
    """Reorganize padded batches of feature tensors into pairs. Leave target tensors unchanged

    Args:
        feature_tensors: A tuple of feature tensors
        target_tensors: A tuple of target tensors
        num_languages: Number of languages. It will be assumed that each feature
                       is repeated `num_languages` times. Defaults to 2

    Returns:
        The new feature tensor tuples and the target tensors
    """
    pairs = []
    num_features = len(feature_tensors) // num_languages
    for i in range(num_features):
        pair = (feature_tensors[i], feature_tensors[i+num_features])
        pairs.append(pair)
    return tuple(pairs), target_tensors


def get_train_and_validation(
        data_split_paths: DataSplitPathArgs,
        train_args: DataStreamArgs,
        cache_dir: Optional[str] = None
) -> Tuple[TFDataset, TFDataset]:
    """Get a training and a validation TensorFlow dataset from files

    Args:
        data_split_paths: A dataclass with paths to `.jsonl` dataset splits
        train_args: Data streaming arguments for ae_training as a dataclass
        cache_dir: Optional. Cache directory for dataset loading

    Returns:
        The ae_training and the validation TensorFlow dataset

    """
    dev_args = deepcopy(train_args)
    dev_args.shuffling_buffer_size = None
    train_dataset = convert_to_tf_dataset(load_dataset('json', split='train',
                                                       data_files=[data_split_paths.train_path], cache_dir=cache_dir))
    dev_dataset = convert_to_tf_dataset(load_dataset('json', split='train',
                                                     data_files=[data_split_paths.dev_path], cache_dir=cache_dir))
    train_dataset = pad_and_batch(train_dataset, data_stream_args=train_args)
    dev_dataset = pad_and_batch(dev_dataset, data_stream_args=dev_args)
    return train_dataset, dev_dataset
