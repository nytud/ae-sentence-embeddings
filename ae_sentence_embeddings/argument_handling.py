"""A module for dataclasses that can help handle training and data arguments.
The module also contains helper tools for argparse
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Dict, Any, Union, Optional, Sequence, Callable, Literal
from inspect import signature
from logging import Logger
import re
from functools import partial

from transformers import BertConfig, OpenAIGPTConfig


def _subst_func(word: str, pattern: re.Pattern, repl: str) -> str:
    """A function for regexp replacements. Input returned in lowercase"""
    return pattern.sub(repl, word).lower()


camel_to_snake = partial(_subst_func, pattern=re.compile(r'(?<!^)([A-Z])'), repl=r'_\1')


class DeeplArgs:
    """Base class for ae_training arguments. Subclass it and add fields to use its methods"""

    @classmethod
    def collect_from_dict(cls, args_dict: Mapping[str, Any], prefix: str = "") -> DeeplArgs:
        """Create a DeeplArgs instance from a dictionary. Ignore irrelevant keys

        Args:
            args_dict: A dictionary from which the data will be collected. If a key is provided that is the snake_case
                class name and its value is a dictionary D, then it will be attempted to collect the data field
                values from D
            prefix: An expected prefix of key names in the dictionary from which the data fields should be collected.
                This can be useful if the dictionary contains arguments of multiple dataclasses with possibly
                overlapping keys. Defaults to the empty string
        """
        subdict_key = camel_to_snake(cls.__name__)
        config_dict = args_dict[subdict_key] if isinstance(args_dict.get(subdict_key), dict) else args_dict
        filtered_args = {stripped_key: val for key, val in config_dict.items()
                         if (stripped_key := key[len(prefix):]) in signature(cls).parameters.keys()}
        result = cls(**filtered_args) if filtered_args else None
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Return self as a dictionary"""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TokenizationArgs(DeeplArgs):
    """A dataclass for tokenization arguments

    Fields:
        text_dataset_path: Path to a text file
        tokenizer: Tokenizer name or path
        tokenized_output_path: Output dataset path
    """
    text_dataset_path: str
    tokenizer: str
    tokenized_output_path: str


@dataclass
class DataStreamArgs(DeeplArgs):
    """A dataclass for data streaming arguments

    Fields:
        input_padding: The values used for padding the input features. It can be specified as a
            single integer if there is only one input feature. Defaults to `(0, 0)`
        target_padding: The value used for padding the target token IDs. Defaults to -1
        batch_size: Batch size. This should be a single integer even if bucketing is used. In this case, bucket batch
            sizes are expected to be derived from the specified value by the function that requires the fields
            of this dataclass as arguments. Defaults to 32
        shuffling_buffer_size: Optional. Buffer size for data shuffling before batching. If not specified,
            shuffling cannot be applied
        num_buckets: Optional. Number of batch buckets. If not specified, bucketing cannot be used. It can take effect
            only if `first_bucket_boundary` is specified as well
        first_bucket_boundary: Optional. Non-inclusive upper boundary of the first batch bucket. If not specified,
            bucketing will not be used. It can take effect only if `num_buckets` is specified as well. Other bucket
            boundaries are expected to be derived from the specified value by the function that requires the fields
            of this dataclass as arguments.
    """
    input_padding: Union[Sequence[int], int] = (0, 0)
    target_padding: Union[Sequence[int], int] = -1
    batch_size: int = 32
    shuffling_buffer_size: Optional[int] = None
    num_buckets: Optional[int] = None
    first_bucket_boundary: Optional[int] = None


@dataclass
class LearningRateArgs(DeeplArgs):
    """A dataclass for the arguments of a one-cycle learning rate scheduler. It can also contain the learning rate
    only if a one-cycle scheduler is not required

    Fields:
        learning_rate: A starting learning rate
        scheduled_iterations: Optional. Number of iterations which allow the scheduler to modify the learning rate. The
            automatic specification of the optional fields below requires this field to be specified
        cycle_end: Optional. The iteration at which the learning rate retakes its starting value. If not specified,
            it will be defined as half the `scheduled iterations`
        max_rate: Optional. Maximal learning rate. If not specified, it will be set to `10 * learning_rate`
        last_rate: Optional. Learning rate when reaching `scheduled_iterations`. If not specified, it will be set to
            `learning_rate / 100`
    """
    learning_rate: float
    scheduled_iterations: Optional[int] = None
    cycle_end: Optional[int] = None
    max_rate: Optional[float] = None
    last_rate: Optional[float] = None

    def __post_init__(self) -> None:
        """Specify field values automatically if possible and necessary"""
        if self.scheduled_iterations is not None:
            if self.cycle_end is None:
                last_iterations = self.scheduled_iterations // 2 + 1
                self.half_cycle = (self.scheduled_iterations - last_iterations) // 2
                self.cycle_end = self.half_cycle * 2
            else:
                self.half_cycle = self.cycle_end // 2
            if self.max_rate is None:
                self.max_rate = self.learning_rate * 10
            if self.last_rate is None:
                self.last_rate = self.learning_rate / 100


@dataclass
class OneCycleArgs(DeeplArgs):
    """A dataclass for the arguments of a one-cycle learning rate scheduler. It can also contain the learning rate
    only if a one-cycle scheduler is not required

    Fields:
        initial_rate: An initial rate (learning rate or beta_1 momentum factor)
        total_steps: Total number of iterations necessary to reach the last iteration beginning from the first
            iteration, i.e. `total_number_of_iterations - 1`
        half_cycle: Number of iterations after which the rate reaches its extremum at mid-cycle beginning from
            the first iteration
        cycle_extremum: Extremum reached at mid-cycle
        end_extremum: Optional. Extremum reached at the training end
    """
    initial_rate: float
    total_steps: int
    half_cycle: int
    cycle_extremum: float
    end_extremum: Optional[float] = None

    def __post_init__(self) -> None:
        """Check argument compatibility"""
        if 2 * self.half_cycle > self.total_steps:
            raise ValueError("`total_steps` must be larger than the number of steps in a single cycle")


@dataclass
class AdamwArgs(DeeplArgs):
    """A dataclass for AdamW optimizer arguments
    For details on the fields, see `https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW`
    """
    weight_decay: Union[float, Callable] = 5e-5
    learning_rate: Union[float, Callable] = 1e-5
    beta_1: Union[float, Callable] = 0.9
    beta_2: Union[float, Callable] = 0.999
    amsgrad: Union[bool, int] = True

    def __post_init__(self) -> None:
        """Convert the `amsgrad` argument to Boolean if it is an integer"""
        self.amsgrad = bool(self.amsgrad)


@dataclass
class SaveAndLogArgs(DeeplArgs):
    """A dataclass for checkpoint and log arguments

    Fields:
        checkpoint_path: Model checkpoint path
        log_tool: Optional. Tensorboard logging path, a `logging.Logger` instance or `"wandb"`
            (if logs are to be passed to WandB)
        save_freq: `"epoch"` or integer. If `"epoch"`, checkpoints will be created after each epoch. If integer,
            the model will be saved after each `save_freq` iterations. Defaults to `"epoch"`
        log_update_freq: `"epoch"` or integer, specifies how often to log. If integer, logs will be updated after
            each `log_update_freq` iterations. Defaults to `"epoch"`
        save_optimizer: Set to `True` if the optimizer state should be saved at each checkpoint.
            Defaults to `True`
    """
    checkpoint_path: str
    log_tool: Optional[Union[str, Logger]] = None
    save_freq: Union[Literal["epoch"], int] = "epoch"
    log_update_freq: Union[Literal["epoch"], int] = "epoch"
    save_optimizer: Union[bool, int] = True

    def __post_init__(self) -> None:
        """If `save_optimizer` is an integer, convert is to Boolean"""
        self.save_optimizer = bool(self.save_optimizer)


@dataclass
class TransformerConfigs(DeeplArgs):
    """A dataclass for Transformer configuration objects

    Fields:
        bert_config: A BERT configuration object
        gpt_config: Optional. A GPT configuration object
    """
    bert_config: BertConfig
    gpt_config: Optional[OpenAIGPTConfig] = None


@dataclass
class RnnArgs(DeeplArgs):
    """A dataclass for RNN decoder arguments

    Fields:
        num_rnn_layers: Number of layers in a deep RNN. Defaults to 2
        hidden_size: RNN hidden size. Defaults to 768
        vocab_size: Number of elements in the vocabulary. Defaults to 32001
        initializer_dev: Stddev in the `TruncatedNormal` initializer for the embedding layer.
            Defaults to 0.02
        layernorm_eps: Epsilon parameter for layer normalization. Defaults to 1e-12
        dropout_rate: A dropout rate between 0 and 1. Defaults to 0.1
    """
    num_rnn_layers: int = 2
    hidden_size: int = 768
    vocab_size: int = 32001
    initializer_dev: float = 0.02
    layernorm_eps: float = 1e-12
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        """Check argument values"""
        if self.dropout_rate > 1 or self.dropout_rate < 0:
            raise ValueError("The dropout rate should be between 0 and 1")


@dataclass
class RegularizedEmbeddingArgs(DeeplArgs):
    """Arguments for creating embedding layers regularized with layer normalization and dropout

    Fields:
        vocab_size: Number of elements in the vocabulary
        hidden_size: Embedding size. Defaults to 768
        initializer_range: Stddev for weight kernel initialization. Defaults to 0.02
        layer_norm_eps: Epsilon parameter for layer normalization. Defaults to 1e-12
        hidden_dropout_prob: Dropout probability. Defaults to 0.1
    """
    vocab_size: int
    hidden_size: int = 768
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.1


@dataclass
class PositionalEmbeddingArgs(RegularizedEmbeddingArgs):
    """Arguments for a positional embedding layer.
    This inherits from `RegularizedEmbeddingArgs`

    Fields:
        vocab_size: Number of elements in the vocabulary
        max_position_embeddings: The maximal sequence length that the model can handle. Defaults to 512
        hidden_size: Embedding size. Defaults to 768
        min_freq: Minimal frequency for the sinusoidal positional encoding. Defaults to 1e-4
        initializer_range: Stddev for weight kernel initialization. Defaults to 0.02
        layer_norm_eps: Epsilon parameter for layer normalization. Defaults to 1e-12
        hidden_dropout_prob: Dropout probability. Defaults to 0.1
    """
    max_position_embeddings: int = 512
    min_freq: float = 1e-4


@dataclass
class DataSplitPathArgs(DeeplArgs):
    """A dataclass for train, dev and test dataset paths

    Fields:
        train_path: Path to the training data file
        dev_path: Path to the validation data file
        test_path: Optional. Path to the test data file
    """
    train_path: str
    dev_path: str
    test_path: Optional[str] = None


def arg_checker(check_function: Callable[[str], bool]) -> Callable[[str], Union[str, int, float]]:
    """A decorator to return a function that checks whether a command line argument is correct

    Args:
        check_function: A function that checks whether its string input satisfies certain criteria

    Returns:
        A function that can be used as a `type` argument in `parser.add_argument()`
    """

    def _check_arg(input_from_cl: str) -> Union[str, int, float]:
        if not check_function(input_from_cl):
            raise ValueError(f"Invalid command line argument: {input_from_cl}")
        return input_from_cl

    return _check_arg
