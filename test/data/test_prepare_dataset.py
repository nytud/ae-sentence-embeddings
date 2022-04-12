# -*- coding: utf-8 -*-

"""Test TensorFlow dataset creation"""

from typing import Generator, Tuple, List, Union
from copy import deepcopy
from types import MappingProxyType
from functools import partial

import tensorflow as tf
from datasets import Dataset as HgfDataset
from numpy.random import randint
from numpy import ndarray, array as np_array

from ae_sentence_embeddings.data import (
    convert_to_tf_dataset,
    pad_and_batch,
    post_batch_multilingual,
    post_batch_feature_pair
)
from ae_sentence_embeddings.argument_handling import DataStreamArgs


def _create_data_batches(data_size: Union[List[int], Tuple[int, ...]],
                         seq_end_ids: Union[ndarray, tf.Tensor]) -> List[List[List[int]]]:
    """Helper function to create a batch of dummy data (integer vectors)

    Args:
        data_size: Integers indicating the following sizes: number of features, batch size, sequence length
        seq_end_ids: Integers indicating the last token in each sequence of each feature (in order to create
                     sequences of variable lengths). This should be a tensor or array of size
                     `(num_features, batch_size)`

    Returns:
        A list that contains a batch of variable length vectors for each feature
    """
    num_features, batch_size, _ = data_size
    if num_features != seq_end_ids.shape[0] or batch_size != seq_end_ids.shape[1]:
        raise ValueError("`seq_end_ids` should be a tensor or array of size "
                         "`(num_features, batch_size)`")
    outputs = []
    data_block = randint(100, size=data_size, dtype="int32")
    for i, seq_end_vec in enumerate(seq_end_ids):
        batch = [data_block[i, row, :to_col].tolist() for row, to_col in enumerate(seq_end_vec)]
        outputs.append(batch)
    # noinspection PyTypeChecker
    return outputs


class TFDatasetTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Fixture setup. This will create random data"""
        mono_data_size = (3, 4, 15)
        mono_seq_end_ids = np_array([[5, 8, 11, 15]] * 3)
        mono_data = _create_data_batches(mono_data_size, mono_seq_end_ids)
        mono_keys = ("input_ids", "attention_mask", "targets")
        text_key = "text"
        mono_data_dict = {mono_key: batch for mono_key, batch in zip(mono_keys, mono_data)}
        mono_data_dict[text_key] = ["ab", "cd", "ef", "gh"]

        bi_data_size = (6, 4, 15)
        bi_seq_end_ids = np_array([[5, 8, 11, 15]] * 3 + [[3, 7, 12, 9]] * 3)
        bi_data = _create_data_batches(bi_data_size, bi_seq_end_ids)
        bi_keys = mono_keys + tuple(mono_key + "_other" for mono_key in mono_keys)
        bi_data_dict = {bi_key: batch for bi_key, batch in zip(bi_keys, bi_data)}
        bi_data_dict[text_key] = deepcopy(mono_data_dict[text_key])
        bi_data_dict[text_key + "_other"] = [s[::-1] for s in mono_data_dict[text_key]]

        cls._mono_data = MappingProxyType(mono_data_dict)
        cls._bi_data = MappingProxyType(bi_data_dict)
        cls._data_to_reorganize = (
            ([[1, 2], [3, 4]], [[-1, -2], [-3, -4]],
             [[5, 6], [7, 8]], [[-5, -6], [-7, -8]]),
            ([[9, 10], [11, 12]], [[-9, -10], [-11, -12]])
        )

    @property
    def mono_data(self) -> MappingProxyType:
        return self._mono_data

    @property
    def bi_data(self) -> MappingProxyType:
        return self._bi_data

    @property
    def data_to_reorganize(self) -> Tuple[Tuple[List[List[int]], ...], ...]:
        return self._data_to_reorganize

    @staticmethod
    def data_gen(
            data: MappingProxyType,
            is_mono: bool
    ) -> Generator[Tuple[Tuple[tf.Tensor, ...], Union[tf.Tensor, Tuple[tf.Tensor, ...]]], None, None]:
        """Helper method for testing `pad_and_batch`"""
        full_batch_size = len(data["input_ids"])
        target_key_start = "target"
        text_key_start = "text"
        for i in range(full_batch_size):
            feature_vecs = []
            target_vecs = []
            for key in data.keys():
                if key.startswith(target_key_start):
                    target_vecs.append(tf.constant(data[key][i]))
                elif key.startswith(text_key_start):
                    pass
                else:
                    feature_vecs.append(tf.constant(data[key][i]))
            targets = target_vecs[0] if is_mono else tuple(target_vecs)
            yield tuple(feature_vecs), targets

    def get_bilingual_datagen(self) -> tf.data.Dataset:
        """Helper function: Set up a multilingual data generator"""
        out_spec = ((tf.TensorSpec(shape=(None,), dtype=tf.int32),) * 4,
                    (tf.TensorSpec(shape=(None,), dtype=tf.int32),) * 2)
        data_gen = partial(self.data_gen, self.bi_data, False)
        return tf.data.Dataset.from_generator(data_gen, output_signature=out_spec)

    def get_monolingual_datagen(self) -> tf.data.Dataset:
        """Helper function: Set up a monolingual data generator"""
        out_spec = ((tf.TensorSpec(shape=(None,), dtype=tf.int32),) * 2,
                    tf.TensorSpec(shape=(None,), dtype=tf.int32))
        data_gen = partial(self.data_gen, self.mono_data, True)
        return tf.data.Dataset.from_generator(data_gen, output_signature=out_spec)

    def test_convert_to_tf_dataset(self) -> None:
        """Test conversion of a `datasets.Dataset` object to a TensorFlow dataset"""
        for data in (self.mono_data, self.bi_data):
            # noinspection PyTypeChecker
            dataset = HgfDataset.from_dict(data)  # `from_dict` work correctly with `MappingProxyType`
            tf_dataset = convert_to_tf_dataset(dataset)
            example = next(iter(tf_dataset))
            print(f"A data point is: {example}")
            self.assertIsInstance(tf_dataset, tf.data.Dataset)
            self.assertEqual(2, len(example))

    def test_pad_and_batch_mono(self) -> None:
        """Test padding and batching a monolingual TensorFlow dataset"""
        data_stream_args = DataStreamArgs(batch_size=2, num_buckets=2, first_bucket_boundary=9)
        tf_dataset = self.get_monolingual_datagen()
        tf_dataset = pad_and_batch(tf_dataset=tf_dataset, data_stream_args=data_stream_args)
        outputs = list(iter(tf_dataset))
        self.assertEqual(3, len(outputs), msg=f"Output tensors are:\n{outputs}")
        self.assertAllEqual([(2, 8), (1, 11), (1, 15)], [data_tensors[0][0].shape for data_tensors in outputs])

    def test_pad_and_batch_bi(self) -> None:
        """Test padding and batching a bilingual TensorFlow dataset"""
        data_stream_args = DataStreamArgs(
            batch_size=2,
            num_buckets=2,
            target_padding=(-1, -1),
            input_padding=(0, 0, 0, 0),
            first_bucket_boundary=9
        )
        tf_dataset = self.get_bilingual_datagen()
        tf_dataset = pad_and_batch(tf_dataset=tf_dataset, data_stream_args=data_stream_args)
        outputs = list(iter(tf_dataset))
        lang1_shape = outputs[0][0][0].shape
        lang2_shape = outputs[0][1][1].shape
        self.assertEqual(3, len(outputs), msg=f"Output tensors are:\n{outputs}")
        self.assertAllEqual([(2, 8), (2, 8)], [lang1_shape, lang2_shape])
        self.assertAllEqual([(2, 8), (1, 12), (1, 15)], [data_tensors[0][0].shape for data_tensors in outputs])

    def test_post_batch_multilingual(self) -> None:
        """Test batch reorganization for multilingual data"""
        tf_dataset = tf.data.Dataset.from_tensors(self.data_to_reorganize)
        new_dataset = tf_dataset.map(post_batch_multilingual)
        outputs = next(iter(new_dataset))
        expected_features, expected_targets = (
            (tf.constant([[1, 2], [3, 4], [-1, -2], [-3, -4]]),
             tf.constant([[5, 6], [7, 8], [-5, -6], [-7, -8]])),
            tf.constant([[9, 10], [11, 12], [-9, -10], [-11, -12]])
        )
        self.assertAllEqual(expected_targets, outputs[1], msg=f"Targets are:\n{outputs[1]}")
        self.assertAllEqual(expected_features, outputs[0], msg=f"Targets are:\n{outputs[0]}")

    def test_post_batch_feature_pair(self) -> None:
        """Test feature data reorganization into tuples"""
        tf_dataset = tf.data.Dataset.from_tensors(self.data_to_reorganize)
        new_dataset = tf_dataset.map(post_batch_feature_pair)
        outputs = next(iter(new_dataset))
        expected_features, expected_targets = (
            ((tf.constant([[1, 2], [3, 4]]), tf.constant([[5, 6], [7, 8]])),
             (tf.constant([[-1, -2], [-3, -4]]), tf.constant([[-5, -6], [-7, -8]]))),
            (tf.constant([[9, 10], [11, 12]]), tf.constant([[-9, -10], [-11, -12]]))
        )
        self.assertAllEqual(expected_targets, outputs[1], msg=f"Targets are:\n{outputs[1]}")
        self.assertAllEqual(expected_features, outputs[0], msg=f"Targets are:\n{outputs[0]}")


if __name__ == "__main__":
    tf.test.main()
