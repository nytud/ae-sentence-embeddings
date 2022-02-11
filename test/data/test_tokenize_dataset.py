"""Test dataset tokenization"""

from typing import Dict, Any, Iterable, Union

import tensorflow as tf
from datasets import Dataset as HgfDataset
from transformers import BertTokenizer
from tokenizers import Tokenizer

from ae_sentence_embeddings.data import tokenize_hgf_dataset


class DataTest(tf.test.TestCase):

    def setUp(self) -> None:
        """Fixture setup. This creates monolingual and bilingual data dictionaries and a tokenizer"""
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
        self.data_bi = {
            "id": self.data_mono["id"],
            "text_en": self.data_mono["text"],
            "text_hu": [
                "Ez egy elég rövid mondat.",
                "Még egy szövegtöredék.",
                "Ez azonban egy hosszabb és bonyolultabb mondat.",
                "És itt az utolsó kísérlet próbaszöveg alkotására!"
            ]
        }
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    @staticmethod
    def _prepare_for_test(data_dict: Dict[str, Any], tokenizer: Union[BertTokenizer, Tokenizer],
                          text_col_names: Iterable[str], swap_feature_cols: bool = False) -> HgfDataset:
        """Helper function for tokenization testing

        Args:
            data_dict: The input dataset as a dictionary that can be converted to a `datasets.Dataset`
            text_col_names: Names of the text columns
            swap_feature_cols: Specify whether feature columns should be swapped between languages.
                Defaults to `False`

        Returns:
            The tokenized dataset
        """
        dataset = HgfDataset.from_dict(data_dict)
        dataset = tokenize_hgf_dataset(
            dataset, tokenizer,
            text_col_names=text_col_names,
            remove_old_cols=True,
            swap_feature_cols=swap_feature_cols
        )
        return dataset

    def test_tokenize_hgf_mono(self) -> None:
        """Test the tokenization of a monolingual `datasets.Dataset`"""
        text_col_names = ("text",)
        dataset = self._prepare_for_test(self.data_mono, self.bert_tokenizer, text_col_names)
        print(f"An example from the tokenized monolingual dataset is:\n{dataset[0]}")
        self.assertIsInstance(dataset[1]["input_ids"], list)
        self.assertIsNotNone(dataset[1].get("attention_mask"))
        self.assertEqual(len(self.data_mono["text"]), len(dataset))

    def test_tokenize_hgf_bi(self) -> None:
        """Test the tokenization of a bilingual `datasets.Dataset`"""
        text_col_names = ("text_en", "text_hu")
        dataset = self._prepare_for_test(self.data_bi, self.bert_tokenizer, text_col_names)
        print(f"An example from the tokenized bilingual dataset is:\n{dataset[0]}")
        self.assertIsInstance(dataset[1]["input_ids_en"], list)
        self.assertIsNotNone(dataset[1].get("attention_mask_en"))
        self.assertEqual(len(self.data_bi["text_en"]), len(dataset))

    def test_swapped_bi_tokenization(self) -> None:
        """Test if swapping target columns works as expected"""
        text_col_names = ("text_en", "text_hu")
        translation_data_point = self._prepare_for_test(
            self.data_bi, self.bert_tokenizer, text_col_names, swap_feature_cols=True)[0]
        print(f"An example from the tokenized bilingual translation dataset is:\n{translation_data_point}")
        auto_encoding_data_point = self._prepare_for_test(
            self.data_bi, self.bert_tokenizer, text_col_names)[0]
        print(f"An example from the tokenized bilingual "
              f"auto-encoding dataset is:\n{auto_encoding_data_point}")
        self.assertAllEqual(auto_encoding_data_point["target_en"], translation_data_point["target_en"])
        self.assertAllEqual(auto_encoding_data_point["target_hu"], translation_data_point["target_hu"])
        self.assertAllEqual(auto_encoding_data_point["input_ids_en"], translation_data_point["input_ids_hu"])
        self.assertAllEqual(
            auto_encoding_data_point["attention_mask_en"], translation_data_point["attention_mask_hu"])


if __name__ == '__main__':
    tf.test.main()
