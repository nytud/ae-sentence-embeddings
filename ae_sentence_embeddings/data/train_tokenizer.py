"""Train a BPE tokenizer with the `tokenizers` library"""

from typing import Iterable, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


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
