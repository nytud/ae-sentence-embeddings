"""A script for training a BPE tokenizer with the `tokenizers` library"""

from typing import Iterable, Optional
from argparse import Namespace, ArgumentParser
from os.path import isfile

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from ae_sentence_embeddings.argument_handling import (
    arg_checker,
    check_if_positive,
    check_if_output_path
)


def get_tokenizer_training_args() -> Namespace:
    """Get command line arguments for tokenizer training"""
    parser = ArgumentParser(description="Arguments for training a BPE tokenizer from scratch")
    parser.add_argument("training_file", type=arg_checker(isfile),
                        help="Path to the plain text training corpus")
    parser.add_argument("save_path", type=check_if_output_path,
                        help="Path to a `.json` file where the trained tokenizer will be saved")
    parser.add_argument("--vocab-size", dest="vocab_size", default=30000, type=check_if_positive,
                        help="Required vocabulary size. Defaults to 30000")
    parser.add_argument("--max-length", dest="max_length", type=check_if_positive,
                        help="Optional. Maximal sequence length to which longer sequences will be truncated. "
                             "If not specified, truncation will not be used")
    parser.add_argument("--unk-token", dest="unk_token", default="[UNK]", type=arg_checker(lambda x: len(x) > 0),
                        help="Unknown token. Defaults to `'[UNK]'`")
    parser.add_argument("--special-tokens", nargs='*', dest="special_tokens",
                        help="Optional. A list of special tokens apart from the unknown token. If a CLS+SEP "
                             "post-processing templated is used, it is expected to list special tokens in the "
                             "following order: (CLS token, SEP token, other tokens).")
    parser.add_argument("--avoid-cls-sep-template", dest="use_cls_sep_template", action="store_false",
                        help="Avoid using a CLS+SEP post-processing template. This means not adding a CLS and a SEP "
                             "token to any encoded text")
    return parser.parse_args()


def train_bpe(
        files: Iterable[str],
        vocab_size: int,
        special_tokens: Optional[Iterable[str]] = None,
        unk_token: str = "[UNK]",
        use_cls_sep_template: bool = True,
        **kwargs
) -> Tokenizer:
    """Train a BPE tokenizer as recommended in the `tokenizers` tutorial:
    https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

    Args:
        files: The text files on which the tokenizer will be trained
        vocab_size: Maximal size of the vocabulary
        special_tokens: Optional. Special tokens added to the tokenizer model, apart from the `UNK` token.
            If not specified, it will be set to `["[CLS]", "[SEP]", "[PAD]", "[MASK]"]`
        unk_token: The unknown token as a string. Defaults to `"[UNK]"`
        use_cls_sep_template: Specifies whether `CLS` and `SEP` tokens should be added to encodings. If `True`,
            `special_tokens` (if specified) is expected to list special tokens in the following order:
            (CLS token, SEP token, other tokens). Defaults to `True`
        **kwargs: Keyword arguments passed to `BpeTrainer`

    Returns:
        The trained tokenizer
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    if special_tokens is None:
        special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    elif not isinstance(special_tokens, list):
        special_tokens = list(special_tokens)
    if len(special_tokens) < 4:
        raise ValueError("There should be at least 4 special tokens: cls, sep, pad and mask.")
    if special_tokens[0] != unk_token:
        special_tokens.insert(0, unk_token)
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size, **kwargs)
    tokenizer.pre_tokenizer = Whitespace()  # This is correct, PyCharm may complain about trying to set a property
    tokenizer.train(files, trainer)
    if use_cls_sep_template:
        cls_token, sep_token = special_tokens[1:3]
        # The next line is also correct, no problem caused by setting a property
        tokenizer.post_processor = TemplateProcessing(
            single=f"{cls_token} $A {sep_token}",
            pair=f"{cls_token} $A {sep_token} $B:1 {sep_token}:1",
            special_tokens=[
                (cls_token, tokenizer.token_to_id(cls_token)),
                (sep_token, tokenizer.token_to_id(sep_token)),
            ],
        )
    return tokenizer


def train_tokenizer_main() -> None:
    """Main function for tokenizer training"""
    args = get_tokenizer_training_args()
    tokenizer = train_bpe(
        files=[args.training_file],
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        unk_token=args.unk_token,
        use_cls_sep_template=args.use_cls_sep_template
    )
    if args.max_length is not None:
        tokenizer.enable_truncation(max_length=args.max_length)
    tokenizer.save(args.save_path)


if __name__ == "__main__":
    train_tokenizer_main()
