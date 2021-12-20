"""Test dataset tokenization and conversion to TensorFlow dataset"""

from typing import Generator, Tuple

import tensorflow as tf
from datasets import Dataset as HgfDataset
from transformers import BertTokenizer
from numpy.random import randint

from ae_sentence_embeddings.data import tokenize_hgf_dataset, convert_to_tf_dataset, pad_and_batch


class DataTest(tf.test.TestCase):

    def test_tokenize_hgf_dataset(self) -> None:
        """Test the tokenization of a `datasets.Dataset`"""
        data = {
            "id": list(range(1, 5)),
            "text": [
                "A rather short sentence.",
                "Yet another text fragment.",
                "This, however, is a longer and more complicated sentence.",
                "And here is the last attempt to make some dummy text!"
            ]
        }
        dataset = HgfDataset.from_dict(data)
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        dataset = tokenize_hgf_dataset(dataset, tokenizer)
        print(f"An example from the tokenized dataset is:\n{dataset[0]}")
        self.assertIsInstance(dataset[1]["input_ids"], list)
        self.assertEqual(len(data["text"]), len(dataset))

    def test_convert_to_tf_dataset(self) -> None:
        """Test conversion of a `datasets.Dataset` object to a TensorFlow dataset"""
        dataset = HgfDataset.from_dict({
            "input_ids": [randint(100, size=32, dtype="int32") for _ in range(4)],
            "targets": [randint(100, size=32, dtype="int32") for _ in range(4)]
        })
        tf_dataset = convert_to_tf_dataset(dataset)
        self.assertIsInstance(tf_dataset, tf.data.Dataset)

    def test_pad_and_batch(self) -> None:
        """Test padding and batching a TensorFlow dataset"""
        data_block = randint(100, size=(3, 4, 15), dtype="int32")
        seq_end_ids = [5, 8, 11, 15]
        out_spec = ((tf.TensorSpec(shape=(None,), dtype=tf.int32),) * 2,
                    tf.TensorSpec(shape=(None,), dtype=tf.int32))

        def data_gen() -> Generator[Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor], None, None]:
            for row, to_col in enumerate(seq_end_ids):
                input_ids, attn_mask, targets = [tf.constant(data_block[i, row, :to_col]) for i in range(3)]
                yield (input_ids, attn_mask), targets

        tf_dataset = tf.data.Dataset.from_generator(data_gen, output_signature=out_spec)
        tf_dataset = pad_and_batch(
            tf_dataset=tf_dataset,
            batch_size=2,
            num_buckets=2,
            first_bucket_boundary=9
        )
        outputs = list(iter(tf_dataset))
        self.assertEqual(3, len(outputs), msg=f"Output tensors are:\n{outputs}")
        self.assertAllEqual([(2, 8), (1, 11), (1, 15)], [data_tensors[0][0].shape for data_tensors in outputs])


if __name__ == '__main__':
    tf.test.main()
