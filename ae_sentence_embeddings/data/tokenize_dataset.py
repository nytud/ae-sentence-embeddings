"""A module for tokenizing a saving a raw text dataset"""

import tensorflow as tf
from transformers import BertTokenizer
from datasets import Dataset as HgfDataset

from ae_sentence_embeddings.modeling_tools import make_decoder_inputs


def tokenize_hgf_dataset(
        dataset: HgfDataset,
        tokenizer: BertTokenizer,
        text_col_name: str = "text",
        target_pad: int = -1
) -> HgfDataset:
    """Tokenize a dataset

    Args:
        dataset: The input dataset
        tokenizer: The tokenizer model
        text_col_name: Text column name in the input dataset. Defaults to `\"text\"`
        target_pad: Padding ID for target token IDs. Defaults to `-1`

    Returns:
        The tokenized dataset with columns `(input_ids, attention_mask, targets)`,
        where `targets` are the token IDs to be generated

    """
    input_ids_name = "input_ids"
    attn_mask_name = "attention_mask"
    target_name = "targets"

    def tok_func(example):
        res = tokenizer(example[text_col_name], return_token_type_ids=False,
                        return_tensors='tf', truncation=True)
        res[target_name] = tf.squeeze(make_decoder_inputs(res[input_ids_name], pad_value=target_pad))
        res[input_ids_name] = tf.squeeze(res[input_ids_name])
        res[attn_mask_name] = tf.squeeze(res[attn_mask_name])
        return res

    return dataset.map(tok_func, batched=False)
