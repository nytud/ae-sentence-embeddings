"""A module for tokenizing a saving a raw text dataset"""

from typing import Generator, Tuple, Optional, Union, Sequence

import tensorflow as tf
from tensorflow.data import Dataset as TFDataset
from transformers import BertTokenizer
from datasets import Dataset as HgfDataset

from ae_sentence_embeddings.modeling_tools import make_decoder_inputs


def tokenize_dataset(dataset: HgfDataset, tokenizer: BertTokenizer, text_col_name: str = "text") -> HgfDataset:
    """Tokenize a dataset

    Args:
        dataset: The input dataset
        tokenizer: The tokenizer model
        text_col_name: Text column name in the input dataset. Defaults to `\"text\"`

    Returns:
        The tokenized dataset with columns `(input_ids, attention_mask, targets)`,
        where `targets` are the token IDs to be generated

    """
    input_ids_name = "input_ids"
    attn_mask_name = "attention_mask"
    target_name = "targets"

    def tok_func(example):
        res = tokenizer(example[text_col_name], return_token_type_ids=False, return_tensors='tf')
        res[input_ids_name] = tf.squeeze(res[input_ids_name])
        res[attn_mask_name] = tf.squeeze(res[attn_mask_name])
        decoder_input_ids = make_decoder_inputs(res[input_ids_name], pad_value=tokenizer.pad_token_id)
        res[target_name] = tf.where(decoder_input_ids == tokenizer.pad_token_id, -1, decoder_input_ids)
        return res

    return dataset.map(tok_func, batched=False)


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
        input_padding: Union[Sequence[int], int] = (0, 0),
        target_padding: int = -1,
        batch_size: int = 32,
        shuffling_buffer_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        first_bucket_boundary: Optional[int] = None
) -> TFDataset:
    """Pad and batch a TensorFlow dataset

    Args:
        tf_dataset: A `tensorflow.data.Dataset` object
        input_padding: The values used for padding the input features. It can be specified as a
                       single integer if there is only one input feature. Defaults to `(0, 0)`
        target_padding: The value used for padding the target token IDs. Defaults to -1
        batch_size: Batch size. If bucketing is used, the batch size for the nth bucket will be calculated as
                    `batch_size // 2**n`. Defaults to 32
        shuffling_buffer_size: Optional. Buffer size for data shuffling before batching. If not specified,
                               shuffling will not be applied
        num_buckets: Optional. Optional. Number of batch buckets. If not specified, bucketing will not be used.
                               It takes effect only if `first_bucket_boundary` is specified as well
        first_bucket_boundary: Optional. Non-inclusive upper boundary of the first batch bucket. The nth boundary will
                               be calculated as `(first_bucket_boundary-1) * 2**n + 1`. If not specified, bucketing
                               will not be used. It takes effect only if `num_buckets` is specified as well
    """
    if shuffling_buffer_size is not None:
        tf_dataset = tf_dataset.shuffle(buffer_size=shuffling_buffer_size)

    if isinstance(input_padding, int):
        padding_values = (input_padding, target_padding)
    else:
        padding_values = (tuple(input_padding), target_padding)

    if num_buckets is None or first_bucket_boundary is None:
        tf_dataset = tf_dataset.padded_batch(batch_size, padding_values=padding_values)
    else:
        tf_dataset = tf_dataset.bucket_by_sequence_length(
            element_length_func=lambda features, targets: tf.shape(targets)[0],
            bucket_boundaries=[(first_bucket_boundary - 1) * 2 ** n + 1 for n in range(num_buckets - 1)],
            bucket_batch_sizes=[batch_size // 2 ** n for n in range(num_buckets)],
            padding_values=padding_values
        )
    return tf_dataset
