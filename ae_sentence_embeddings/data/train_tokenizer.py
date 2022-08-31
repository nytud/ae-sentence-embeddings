#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A script for training a BPE tokenizer with the `tokenizers` library"""

from typing import Iterable
from argparse import Namespace, ArgumentParser

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit, Punctuation, ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from ae_sentence_embeddings.argument_handling import (
    check_if_file,
    check_if_positive_int,
    check_if_output_path
)


def get_tokenizer_training_args() -> Namespace:
    """Get command line arguments for tokenizer training"""
    parser = ArgumentParser(description="Arguments for training a BPE tokenizer from scratch")
    parser.add_argument("training_file", type=check_if_file,
                        help="Path to the plain text training corpus.")
    parser.add_argument("save_path", type=check_if_output_path,
                        help="Path to a `.json` file where the trained tokenizer will be saved,")
    parser.add_argument("--vocab-size", dest="vocab_size", default=30000, type=check_if_positive_int,
                        help="Required vocabulary size. Defaults to 30000")
    parser.add_argument("--max-length", dest="max_length", type=check_if_positive_int,
                        help="Optional. Maximal sequence length to which longer sequences will be truncated. "
                             "If not specified, truncation will not be used.")
    parser.add_argument("--avoid-cls-sep-template", dest="use_cls_sep_template", action="store_false",
                        help="Avoid using a CLS+SEP post-processing template. This means not adding a CLS and a SEP "
                             "token to any encoded text.")
    return parser.parse_args()


def train_bpe(
        files: Iterable[str],
        vocab_size: int,
        use_cls_sep_template: bool = True,
        **kwargs
) -> Tokenizer:
    """Train a BPE tokenizer as recommended in the `tokenizers` tutorial:
    https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

    Args:
        files: The text files on which the tokenizer will be trained.
        vocab_size: Maximal size of the vocabulary.
        use_cls_sep_template: Specifies whether `CLS` and `SEP` tokens should be added to encodings.
            Defaults to `True`.
        **kwargs: Keyword arguments passed to `BpeTrainer`.

    Returns:
        The trained tokenizer
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]

    # noinspection PyPropertyAccess
    tokenizer.normalizer = NFKC()
    # noinspection PyPropertyAccess
    tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(), Punctuation(), ByteLevel()])
    # noinspection PyArgumentList
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size, **kwargs)
    tokenizer.train(files, trainer)
    if use_cls_sep_template:
        cls_token, sep_token = special_tokens[1:3]
        # noinspection PyPropertyAccess
        tokenizer.post_processor = TemplateProcessing(
            single=f"{cls_token} $0 {sep_token}",
            pair=f"{cls_token} $A {sep_token} $B:1 {sep_token}:1",
            special_tokens=[
                (cls_token, tokenizer.token_to_id(cls_token)),
                (sep_token, tokenizer.token_to_id(sep_token)),
            ],
        )
    # noinspection PyPropertyAccess
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer_main() -> None:
    """Main function for tokenizer training"""
    args = get_tokenizer_training_args()
    tokenizer = train_bpe(
        files=[args.training_file],
        vocab_size=args.vocab_size,
        use_cls_sep_template=args.use_cls_sep_template
    )
    if args.max_length is not None:
        tokenizer.enable_truncation(max_length=args.max_length)
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        max_len=args.max_length
    )
    wrapped_tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    train_tokenizer_main()
