"""Test dataset tokenization"""

from typing import Dict, Any, Iterable, Union

import tensorflow as tf
from datasets import Dataset as HgfDataset
from transformers import BertTokenizer
from tokenizers import Tokenizer

from ae_sentence_embeddings.data import tokenize_hgf_dataset


class DataTest(tf.test.TestCase):

    def setUp(self) -> None:
        """Fixture setup. This creates a monolingual data dictionary and a tokenizer"""
        super().setUp()
        self.data_mono = {
            "id": list(range(1, 5)),
            "text": [
                "A rather short sentence.",
                "Yet another text fragment.",
                "This, however, is a longer and more complicated sentence.",
                "And here is the last attempt to make some dummy text!"
            ]
        }
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    @staticmethod
    def _prepare_for_test(data_dict: Dict[str, Any], tokenizer: Union[BertTokenizer, Tokenizer],
                          text_col_names: Iterable[str]) -> HgfDataset:
        """Helper function for tokenization testing

        Args:
            data_dict: The input dataset as a dictionary that can be converted to a `datasets.Dataset`
            text_col_names: Names of the text columns

        Returns:
            The tokenized dataset
        """
        dataset = HgfDataset.from_dict(data_dict)
        dataset = tokenize_hgf_dataset(
            dataset, tokenizer, text_col_names=text_col_names, remove_old_cols=True)
        print(f"An example from the tokenized dataset is:\n{dataset[0]}")
        return dataset

    def test_tokenize_hgf_mono(self) -> None:
        """Test the tokenization of a monolingual `datasets.Dataset`"""
        text_col_names = ("text",)
        dataset = self._prepare_for_test(self.data_mono, self.bert_tokenizer, text_col_names)
        self.assertIsInstance(dataset[1]["input_ids"], list)
        self.assertIsNotNone(dataset[1].get("attention_mask"))
        self.assertEqual(len(self.data_mono["text"]), len(dataset))

    def test_tokenize_hgf_bi(self) -> None:
        """Test the tokenization of a bilingual `datasets.Dataset`"""
        data_bi = {
            "id": self.data_mono["id"],
            "text_en": self.data_mono["text"],
            "text_hu": [
                "Ez egy elég rövid mondat.",
                "Még egy szövegtöredék.",
                "Ez azonban egy hosszabb és bonyolultabb mondat.",
                "És itt az utolsó kísérlet próbaszöveg alkotására!"
            ]
        }
        text_col_names = ("text_en", "text_hu")
        dataset = self._prepare_for_test(data_bi, self.bert_tokenizer, text_col_names)
        self.assertIsInstance(dataset[1]["input_ids_en"], list)
        self.assertIsNotNone(dataset[1].get("attention_mask_en"))
        self.assertEqual(len(data_bi["text_en"]), len(dataset))


if __name__ == '__main__':
    tf.test.main()
