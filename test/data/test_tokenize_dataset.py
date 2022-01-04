"""Test dataset tokenization"""

import tensorflow as tf
from datasets import Dataset as HgfDataset
from transformers import BertTokenizer

from ae_sentence_embeddings.data import tokenize_hgf_dataset


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


if __name__ == '__main__':
    tf.test.main()
