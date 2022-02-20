"""A script for training a BPE tokenizer with the `tokenizers` library"""

from typing import Iterable, Optional
from argparse import Namespace, ArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def get_tokenizer_training_args() -> Namespace:
    """Get command line arguments for tokenizer training"""
    parser = ArgumentParser(description="Arguments for training a BPE tokenizer from scratch")
    parser.add_argument("training_file", help="Path to train plain text training corpus")
    parser.add_argument("save_path", help="Path to a `.json` file where the trained tokenizer will be saved")
    parser.add_argument("--vocab-size", dest="vocab_size", type=int, default=30000,
                        help="Required vocabulary size. Defaults to 30000")
    parser.add_argument("--max-length", dest="max_length", type=int,
                        help="Optional. Maximal sequence length to which longer sequences will be truncated. "
                             "If not specified, truncation will not be used")
    parser.add_argument("--unk-token", dest="unk_token", default="[UNK]",
                        help="Unknown token. Defaults to `'[UNK]'`")
    parser.add_argument("--special-tokens", nargs='*', dest="special_tokens",
                        help="Optional. A list of special tokens aprt from the unknown token. If a CLS+SEP "
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
        special_tokens = [unk_token, "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    else:
        special_tokens = [unk_token] + list(special_tokens)
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size, **kwargs)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)
    if use_cls_sep_template:
        cls_token = special_tokens[1]
        sep_token = special_tokens[2]
        tokenizer.post_processor = TemplateProcessing(
            single=' '.join([cls_token, "$A", sep_token]),
            pair=' '.join([cls_token, "$A", sep_token, "$B:1", sep_token]) + ":1",
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


if __name__ == '__main__':
    train_tokenizer_main()
