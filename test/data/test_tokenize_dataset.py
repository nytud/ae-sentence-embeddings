# -*- coding: utf-8 -*-

"""Test dataset tokenization"""

import tensorflow as tf
from datasets import Dataset as HgfDataset
from transformers import BertTokenizer

from ae_sentence_embeddings.data import tokenize_hgf_dataset, tokenize_labelled_sequences


class DataTest(tf.test.TestCase):

    def setUp(self) -> None:
        """Fixture setup. This creates a raw text data dictionary and a tokenizer."""
        super().setUp()
        self._data = {
            "id": list(range(1, 5)),
            "text": [
                "A rather short sentence.",
                "Yet another text fragment.",
                "This, however, is a longer and more complicated sentence.",
                "And here is the last attempt to make some dummy text!"
            ]
        }
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def test_tokenize_dataset(self) -> None:
        """Test the tokenization of a monolingual `datasets.Dataset`"""
        dataset = tokenize_hgf_dataset(
            HgfDataset.from_dict(self._data), self._tokenizer, remove_old_cols=True)
        print(f"An example from the tokenized tokenized dataset is:\n{dataset[0]}")
        self.assertIsInstance(dataset[1]["input_ids"], list)
        self.assertIsNotNone(dataset[1].get("attention_mask"))

    def test_tokenize_labelled_sequences(self) -> None:
        """Test the tokenization of labelled sequences."""
        label_col_name = "label"
        expected_cols = (label_col_name, "input_ids", "attention_mask")
        self._data[label_col_name] = [0, 1, 1, 0]
        tokenized_dataset = tokenize_labelled_sequences(
            dataset=HgfDataset.from_dict(self._data),
            tokenizer=self._tokenizer,
            text_col_names=("text",),
            label_col_name=label_col_name,
            max_length=128,
            remove_old_cols=True
        )
        print(f"An example from the dataset:\n{tokenized_dataset[0]}")
        self.assertAllEqual(expected_cols, tuple(tokenized_dataset.features.keys()))


if __name__ == "__main__":
    tf.test.main()
