#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tokenize raw text data"""

from argparse import Namespace, ArgumentParser
from warnings import warn

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from tokenizers import Tokenizer

from ae_sentence_embeddings.data import tokenize_hgf_dataset, tokenize_labelled_sequences
from ae_sentence_embeddings.data.tokenize_dataset import load_tokenizer
from ae_sentence_embeddings.argument_handling import (
    check_if_file,
    check_if_positive_int,
    check_if_output_path
)


def get_tokenization_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(description="Get command line arguments for tokenization")
    parser.add_argument("dataset_path", type=check_if_file,
                        help="Path to the data to be tokenized in `sentence-per-line` or `jsonlines` format.")
    parser.add_argument("tokenizer_output_path", type=check_if_output_path,
                        help="Path to the output `.jsonl` dataset.")
    parser.add_argument("--langs", choices=["mono", "multi"], default="mono",
                        help="Specify if the input data is multilingual (`multi`) or monolingual (`mono`)."
                             "If it is multilingual, the input dataset is expected to be given in `jsonlines` format. "
                             "Otherwise, `sentence-per-line` format is expected. This argument is irrelevant if a "
                             "labelled dataset is to be tokenized (so `--label-column` is specified). In that case, "
                             "`jsonlines` input format is expected. Defaults to `'mono'`.")
    parser.add_argument("--text-columns", dest="text_columns", nargs="+", default=["text"],
                        help="Dataset column (key) names associated with the input texts. Only relevant if the "
                             "input is multilingual (e.g. the text column names can be `['text_en', 'text_hu']`. "
                             "If your data is monolingual, use the default value: `['text',]`.")
    parser.add_argument("--label-column", dest="label_column",
                        help="Optional. The name of the column in the input dataset that contains the text labels.")
    parser.add_argument("--tokenizer", default="SZTAKI-HLT/hubert-base-cc", type=load_tokenizer,
                        help="Tokenizer name or path. Defaults to `SZTAKI-HLT/hubert-base-cc`.")
    parser.add_argument("--max-length", dest="max_length", type=check_if_positive_int, default=128,
                        help="Maximal sequence length in the number of tokens. Defaults to `128`.")
    parser.add_argument("--cache-dir", dest="cache_dir", help="Optional. The cache directory for the dataset.")
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = get_tokenization_args()
    dataset_format = "json" if args.label_column is not None or args.langs == "multi" \
        else "text"
    raw_dataset = load_dataset(dataset_format, data_files=[args.dataset_path], split="train",
                               cache_dir=args.cache_dir)

    if isinstance(tokenizer := args.tokenizer, Tokenizer):
        tokenizer.enable_truncation(max_length=args.max_length)
    elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        tokenizer.model_max_length = args.max_length
    else:
        warn(f"Setting maximum sequence length for {type(tokenizer)} has not been implemented yet!")

    tokenization_kwargs = {
        "dataset": raw_dataset,
        "tokenizer": tokenizer,
    }
    if args.label_column is not None:
        tokenization_func = tokenize_labelled_sequences
        tokenization_kwargs["text_col_names"] = args.text_columns
        tokenization_kwargs["label_col_name"] = args.label_column
        tokenization_kwargs["remove_old_cols"] = True
    else:
        tokenization_kwargs["text_col_name"] = args.text_columns[0]
        tokenization_func = tokenize_hgf_dataset

    tokenized_dataset = tokenization_func(**tokenization_kwargs)
    tokenized_dataset.to_json(args.tokenizer_output_path, force_ascii=False)


if __name__ == "__main__":
    main()
