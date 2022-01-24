"""Test TensorFlow dataset creation"""

from typing import Generator, Tuple

import tensorflow as tf
from datasets import Dataset as HgfDataset
from numpy.random import randint

from ae_sentence_embeddings.data import convert_to_tf_dataset, pad_and_batch
from ae_sentence_embeddings.argument_handling import DataStreamArgs


class TFDatasetTest(tf.test.TestCase):

    def test_convert_to_tf_dataset(self) -> None:
        """Test conversion of a `datasets.Dataset` object to a TensorFlow dataset"""
        data_mono = {
            "text": ["This is a dummy sentence.", "Yet another dummy input."],
            "input_ids": [[101, 42, 14, 678, 6, 1017, 9, 102], [101, 53, 202, 678, 98, 9, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]],
            "targets": [[42, 14, 678, 6, 1017, 9, 102, 0], [53, 202, 678, 98, 9, 102, 0]]
        }
        data_bi = {
            **data_mono,
            "text_other": ["Ez is egy probamondat", "Ahogyan ez is itt."],
            "input_ids_other": [[101, 56, 789, 85, 102], [101, 79, 1022, 77, 9, 102]],
            "attention_mask_other": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
            "targets_other": [[56, 789, 85, 102], [79, 1022, 77, 9, 102]]
        }
        for data in (data_mono, data_bi):
            dataset = HgfDataset.from_dict(data)
            tf_dataset = convert_to_tf_dataset(dataset)
            example = next(iter(tf_dataset))
            print(f"A data point is: {example}")
            self.assertIsInstance(tf_dataset, tf.data.Dataset)
            self.assertEqual(2, len(example))

    def test_pad_and_batch(self) -> None:
        """Test padding and batching a TensorFlow dataset"""
        data_block = randint(100, size=(3, 4, 15), dtype="int32")
        seq_end_ids = [5, 8, 11, 15]
        out_spec = ((tf.TensorSpec(shape=(None,), dtype=tf.int32),) * 2,
                    tf.TensorSpec(shape=(None,), dtype=tf.int32))
        data_stream_args = DataStreamArgs(batch_size=2, num_buckets=2, first_bucket_boundary=9)

        def data_gen() -> Generator[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor], None, None]:
            for row, to_col in enumerate(seq_end_ids):
                input_ids, attn_mask, targets = [tf.constant(data_block[i, row, :to_col]) for i in range(3)]
                yield (input_ids, attn_mask), targets

        tf_dataset = tf.data.Dataset.from_generator(data_gen, output_signature=out_spec)
        tf_dataset = pad_and_batch(tf_dataset=tf_dataset, data_stream_args=data_stream_args)
        outputs = list(iter(tf_dataset))
        self.assertEqual(3, len(outputs), msg=f"Output tensors are:\n{outputs}")
        self.assertAllEqual([(2, 8), (1, 11), (1, 15)], [data_tensors[0][0].shape for data_tensors in outputs])


if __name__ == '__main__':
    tf.test.main()
