"""A module for preparing datasets for usage with TensorFlow/Keras"""

from typing import Generator, Tuple, Optional
from copy import deepcopy

import tensorflow as tf
from tensorflow.data import Dataset as TFDataset
from datasets import Dataset as HgfDataset, load_dataset

from ae_sentence_embeddings.argument_handling import DataStreamArgs, DataSplitPathArgs


def convert_to_tf_dataset(dataset: HgfDataset) -> TFDataset:
    """Convert a `datasets.Dataset` object to a TensorFlow dataset using a generator

    Args:
        dataset: The original `datasets.Dataset` object

    Returns:
        A TensorFlow dataset

    """
    target_col = {'target', 'targets', 'label', 'labels'}.intersection(dataset.features.keys()).pop()
    feature_cols = [feature_col for feature_col in dataset.features.keys() if feature_col != target_col]
    out_spec = ((tf.TensorSpec(shape=(None,), dtype=tf.int32),) * len(feature_cols),
                tf.TensorSpec(shape=(None,), dtype=tf.int32))

    def data_gen() -> Generator[Tuple[Tuple[tf.Tensor, ...], tf.Tensor], None, None]:
        for example in dataset:
            input_fields = tuple(tf.constant(example[feature]) for feature in feature_cols)
            target_field = tf.constant(example[target_col])
            yield input_fields, target_field

    return TFDataset.from_generator(data_gen, output_signature=out_spec)


def pad_and_batch(
        tf_dataset: TFDataset,
        data_stream_args: DataStreamArgs
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

    Returns:
        The padded and batched TensorFlow dataset

    """
    if data_stream_args.shuffling_buffer_size is not None:
        tf_dataset = tf_dataset.shuffle(buffer_size=data_stream_args.shuffling_buffer_size)

    input_padding = data_stream_args.input_padding
    if not isinstance(data_stream_args.input_padding, (int, tuple)):
        input_padding = tuple(input_padding)
    padding_values = (input_padding, data_stream_args.target_padding)

    if data_stream_args.num_buckets is None or data_stream_args.first_bucket_boundary is None:
        tf_dataset = tf_dataset.padded_batch(data_stream_args.batch_size, padding_values=padding_values)
    else:
        bucket_boundaries = [(data_stream_args.first_bucket_boundary - 1) * 2 ** n + 1
                             for n in range(data_stream_args.num_buckets - 1)]
        bucket_batch_sizes = [data_stream_args.batch_size // 2 ** n for n in range(data_stream_args.num_buckets)]
        tf_dataset = tf_dataset.bucket_by_sequence_length(
            element_length_func=lambda features, targets: tf.shape(targets)[0],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padding_values=padding_values
        )
    return tf_dataset


def get_train_and_validation(
        data_split_paths: DataSplitPathArgs,
        train_args: DataStreamArgs,
        cache_dir: Optional[str] = None
) -> Tuple[TFDataset, TFDataset]:
    """Get a training and a validation TensorFlow dataset from files

    Args:
        data_split_paths: A dataclass with paths to `.jsonl` dataset splits
        train_args: Data streaming arguments for training as a dataclass
        cache_dir: Optional. Cache directory for dataset loading

    Returns:
        The training and the validation TensorFlow dataset

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
