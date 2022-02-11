"""A script for training a BPE tokenizer with the `tokenizers` library"""

from typing import Iterable, Optional
from argparse import Namespace, ArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_tokenizer_training_args() -> Namespace:
    """Get command line arguments for tokenizer training"""
    parser = ArgumentParser(description="Arguments for training a BPE tokenizer from scratch")
    parser.add_argument("training_file", help="Path to train plain text training corpus")
    parser.add_argument("save_path", help="Path where the trained tokenizer will be saved")
    parser.add_argument("--vocab-size", dest="vocab_size", type=int, default=30000,
                        help="Required vocabulary size. Defaults to 30000")
    parser.add_argument("--max-length", dest="max_length", type=int,
                        help="Optional. Maximal sequence length to which longer sequences will be truncated. "
                             "If not specified, truncation will not be used")
    parser.add_argument("--unk-token", dest="unk_token", default="[UNK]",
                        help="Unknown token. Defaults to `'[UNK]'`")
    parser.add_argument("--special-tokens", nargs='*', dest="special_tokens",
                        help="Optional. A list of special tokens")
    return parser.parse_args()


def train_bpe(
        files: Iterable[str],
        vocab_size: int,
        special_tokens: Optional[Iterable[str]] = None,
        unk_token: str = "[UNK]",
        **kwargs
) -> Tokenizer:
    """Train a BPE tokenizer as recommended in the `tokenizers` tutorial:
    https://huggingface.co/docs/tokenizers/python/latest/quicktour.html

    Args:
        files: The text files on which the tokenizer will be trained
        vocab_size: Maximal size of the vocabulary
        special_tokens: Optional. Special tokens added to the tokenizer model. If not specified, it will be
            set to `["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]`
        unk_token: The unknown token as a string. Defaults to `"[UNK]"`
        **kwargs: Keyword arguments passed to `BpeTrainer`

    Returns:
        The trained tokenizer
    """
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    if special_tokens is None:
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size, **kwargs)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)
    return tokenizer


def train_tokenizer_main() -> None:
    """Main function for tokenizer training"""
    args = get_tokenizer_training_args()
    tokenizer = train_bpe(
        files=[args.training_file],
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        unk_token=args.unk_token
    )
    if args.max_length is not None:
        tokenizer.enable_truncation(max_length=args.max_length)
    tokenizer.save(args.save_path)


if __name__ == '__main__':
    train_tokenizer_main()
