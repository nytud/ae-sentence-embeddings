"""A module for tokenizing a saving a raw text dataset"""

from typing import Iterable, Dict, Any, Mapping, Tuple, Union, List
from functools import partial

import tensorflow as tf
from transformers import BertTokenizer
from tokenizers import Tokenizer
from datasets import Dataset as HgfDataset

from ae_sentence_embeddings.modeling_tools import make_decoder_inputs, make_ngram_iter


def tokenize_hgf_dataset(
        dataset: HgfDataset,
        tokenizer: Union[BertTokenizer, Tokenizer],
        text_col_names: Iterable[str] = ("text",),
        input_ids_name: str = "input_ids",
        target_prefix: str = "target",
        target_pad: int = -1,
        remove_old_cols: bool = False,
        swap_feature_cols: bool = False
) -> HgfDataset:
    """Tokenize a dataset

    Args:
        dataset: The input dataset
        tokenizer: The tokenizer model
        text_col_names: Text column names in the input dataset. Defaults to `("text",)`
        input_ids_name: The name of the column returned by the tokenizer that contains token IDs.
            Defaults to `input_ids`
        target_prefix: Target column prefix. Target columns contain the token IDs that are to be
            generated by a decoder. Defaults to `target`
        target_pad: Padding ID for target token IDs. Defaults to `-1`
        remove_old_cols: Set to `True` if the original columns are to be removed. Defaults to `False`
        swap_feature_cols: Swap all columns between languages, except target input IDs. This can mean turning
            an auto-encoding task into a translation task. If `True`, it works only if there are 2 elements in
            `text_col_names`. Defaults to `False`

    Returns:
        The tokenized dataset with feature and target columns
    """
    if not isinstance(text_col_names, (list, tuple)):
        text_col_names = list(text_col_names)
    if len(text_col_names) != len(set(text_col_names)):
        raise ValueError("All text column names are required to be unique")
    if swap_feature_cols and len(text_col_names) != 2:
        raise ValueError("Swapping columns is only possible if there are exactly 2 text columns")
    cols_to_remove = list(dataset.features.keys()) if remove_old_cols else None

    frozen_kwargs = {
        "tokenizer": tokenizer,
        "text_col_names": text_col_names,
        "input_ids_name": input_ids_name,
        "target_prefix": target_prefix,
        "target_pad": target_pad
    }
    if isinstance(tokenizer, BertTokenizer):
        tok_func = partial(bert_tokenize, **frozen_kwargs)
    elif isinstance(tokenizer, Tokenizer):
        tok_func = partial(tokenize_special, **frozen_kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {type(tokenizer)}")

    dataset = dataset.map(tok_func, batched=False, remove_columns=cols_to_remove)
    if swap_feature_cols:
        feature_col_pairs = _get_feature_pairs(dataset.features.keys(), target_prefix)
        swapping_func = partial(swap_features, feature_col_pairs=feature_col_pairs)
        dataset = dataset.map(swapping_func, batched=False)
    return dataset


def _process_tokenization_features(
        names_and_values: Iterable[Tuple[str, tf.Tensor]],
        input_ids_name: str,
        orig_col_name: str,
        target_prefix: str,
        target_pad: int
) -> Dict[str, tf.Tensor]:
    """A helper function to process a single tokenization example

    Args:
        names_and_values: An iterable of tuples where the first element of each tuple is a key (or feature column name)
            and the second element is a tensor associated with the key
        input_ids_name: The name of the feature column associated with input IDs (e.g. `"input_ids"`)
        orig_col_name: The name of the original text column from which the tokenization features were extracted.
            This can be an empty string or a language identifier prepended with an underscore (e.g. `"_en"`)
        target_prefix: A prefix for the target column. It is usually `"target_"`
        target_pad: Padding value for targets

    Returns:
        A dictionary whose keys are feature and target column names. The values are 1D tensors
    """
    result_dict = {}
    for feature_col_name, feature_value in names_and_values:
        if feature_col_name == input_ids_name:
            target_col_name = ''.join([target_prefix, orig_col_name])
            result_dict[target_col_name] = tf.squeeze(
                make_decoder_inputs(feature_value, pad_value=target_pad))
        new_col_name = ''.join([feature_col_name, orig_col_name])
        result_dict[new_col_name] = tf.squeeze(feature_value)
    return result_dict


def bert_tokenize(
        example: Mapping[str, Any],
        tokenizer: BertTokenizer,
        text_col_names: Iterable[str] = ("text",),
        input_ids_name: str = "input_ids",
        target_prefix: str = "target",
        target_pad: int = -1
) -> Dict[str, Union[tf.Tensor, str]]:
    """Tokenize a single example with a BERT tokenizer

    Args:
        example: A single data point as a mapping between features and their values
        tokenizer: The tokenizer model
        text_col_names: Text column names in the input dataset. Defaults to `("text",)`
        input_ids_name: The name of the column returned by the tokenizer that contains token IDs.
                        Defaults to `input_ids`
        target_prefix: Target column prefix. Target columns contain the token IDs that are to be
                       generated by a decoder. Defaults to `target`
        target_pad: Padding ID for target token IDs. Defaults to `-1`

    Returns:
        The tokenized data point with feature and target columns
    """
    all_cols = {}
    for text_col_name in text_col_names:
        res = tokenizer(example[text_col_name], return_token_type_ids=False,
                        return_tensors='tf', truncation=True)
        text_col_name_stripped = text_col_name.lstrip("text")
        custom_res = _process_tokenization_features(
            res.items(),
            input_ids_name=input_ids_name,
            orig_col_name=text_col_name_stripped,
            target_prefix=target_prefix,
            target_pad=target_pad
        )
        all_cols.update(custom_res)
    return all_cols


def tokenize_special(
    example: Mapping[str, Any],
    tokenizer: Tokenizer,
    text_col_names: Iterable[str] = ("text",),
    input_ids_name: str = "input_ids",
    target_prefix: str = "target",
    target_pad: int = -1
) -> Dict[str, Union[tf.Tensor, str]]:
    """Tokenize a single example with a `tokenizers.Tokenizer` model

    Args:
        example: A single data point as a mapping between features and their values
        tokenizer: The tokenizer model
        text_col_names: Text column names in the input dataset. Defaults to `("text",)`
        input_ids_name: The name of the column returned by the tokenizer that contains token IDs.
                        Defaults to `input_ids`
        target_prefix: Target column prefix. Target columns contain the token IDs that are to be
                       generated by a decoder. Defaults to `target`
        target_pad: Padding ID for target token IDs. Defaults to `-1`

    Returns:
        The tokenized data point with feature and target columns
    """
    all_cols = {}
    for text_col_name in text_col_names:
        res = tokenizer.encode(example[text_col_name])
        text_col_name_stripped = text_col_name.lstrip("text")
        names_and_values = ((input_ids_name, tf.expand_dims(tf.constant(res.ids), axis=0)),
                            ("attention_mask", tf.expand_dims(tf.constant(res.attention_mask), axis=0)))
        custom_res = _process_tokenization_features(
            names_and_values=names_and_values,
            input_ids_name=input_ids_name,
            orig_col_name=text_col_name_stripped,
            target_prefix=target_prefix,
            target_pad=target_pad
        )
        all_cols.update(custom_res)
    return all_cols


def swap_features(
        example: Dict[str, Any],
        feature_col_pairs: Iterable[Tuple[str, str]]
) -> Dict[str, Union[tf.Tensor, str]]:
    """Swap feature columns. This can mean turning an auto-encoding task into a translation task

    Args:
        example: A single data point as a mapping between features and their values
        feature_col_pairs: Tuples of feature column names, e.g.
            `[("input_ids_en", "input_ids_hu"), ("attention_mask_en", "attention_mask_hu")]`

    Returns:
        A data point with swapped target values
    """
    for feature_lang1, feature_lang2 in feature_col_pairs:
        feature_lang1_value = example[feature_lang1]
        example[feature_lang1] = example[feature_lang2]
        example[feature_lang2] = feature_lang1_value
    return example


def _get_feature_pairs(
        dataset_cols: Iterable[str],
        target_prefix: str
) -> List[Tuple[str, str]]:
    """Helper function to obtain tuples of feature column names

    Args:
        dataset_cols: All the column names in a dataset
        target_prefix: Target column prefix

    Returns:
        Tuples of feature column names, e.g. `[("input_ids_en", "input_ids_hu"),
        ("attention_mask_en", "attention_mask_hu")]`
    """
    feature_cols = sorted([col for col in dataset_cols if not col.startswith(target_prefix)])
    return list(make_ngram_iter(feature_cols, 2, 2))
