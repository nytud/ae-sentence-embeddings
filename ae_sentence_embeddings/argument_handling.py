"""A module for dataclasses that can help handle training and data arguments.
The module also contains helper tools for argparse
"""

from __future__ import annotations
from argparse import ArgumentTypeError
from dataclasses import dataclass
from typing import Mapping, Dict, Any, Tuple, Union, Optional, Sequence, Callable, Literal
from inspect import signature
from logging import Logger
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from functools import partial

from transformers import BertConfig, OpenAIGPTConfig


def _subst_func(word: str, pattern: re.Pattern, repl: str) -> str:
    """A function for regexp replacements. Input returned in lowercase"""
    return pattern.sub(repl, word).lower()


camel_to_snake = partial(_subst_func, pattern=re.compile(r'(?<!^)([A-Z])'), repl=r'_\1')


class DeeplArgs(metaclass=ABCMeta):
    """Base class for training arguments. Subclass it and add fields to use its methods"""

    def __init__(self, **kwargs):
        # Making `__init__` an `abstractmethod` would raise a `TypeError` if the child classes are
        # `dataclasses` without an explicit `__init__` method, even though the `dataclass` decorator
        # does add an `__init__` method to the class by default
        raise NotImplementedError("Initialization must be implemented by children classes")

    @abstractmethod
    def __post_init__(self) -> None:
        """Implement this method to check arguments"""
        return

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


@dataclass  # Note that the `@dataclass` decorator does implement the `__init__` method
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

    def __post_init__(self) -> None:
        """Check arguments"""
        self.text_dataset_path = check_if_file(self.text_dataset_path)
        self.tokenized_output_path = check_if_output_path(self.tokenized_output_path)


@dataclass
class DataStreamArgs(DeeplArgs):
    """A dataclass for data streaming arguments

    Fields:
        input_padding: The values used for padding the input features. It can be specified as a
            single integer if there is only one input feature. Defaults to `(0, 0)`
        target_padding: The values used for padding the target token IDs. It can be specified as a
            single integer if there is only one input feature. Defaults to `-1`.
        batch_size: Batch size. This should be a single integer even if bucketing is used. In this case, bucket batch
            sizes are calculated with the formula `batch_size // 2**n` Defaults to `32`.
        shuffling_buffer_size: Optional. Buffer size for data shuffling before batching. If not specified,
            shuffling cannot be applied
        num_buckets: Optional. Number of batch buckets. If not specified, bucketing cannot be used. It can take effect
            only if `first_bucket_boundary` is specified as well.
        first_bucket_boundary: Optional. Non-inclusive upper boundary of the first batch bucket. If not specified,
            bucketing will not be used. It can take effect only if `num_buckets` is specified as well.
            The nth bucket boundary will be calculated as `(first_bucket_boundary-1) * 2**n + 1`
    """
    input_padding: Union[Sequence[int], int] = (0, 0)
    target_padding: Union[Sequence[int], int] = -1
    batch_size: int = 32
    shuffling_buffer_size: Optional[int] = None
    num_buckets: Optional[int] = None
    first_bucket_boundary: Optional[int] = None

    def __post_init__(self) -> None:
        """Check arguments"""
        self.batch_size = check_if_positive_int(self.batch_size)
        if self.shuffling_buffer_size is not None:
            self.shuffling_buffer_size = check_if_positive_int(self.shuffling_buffer_size)
        if self.num_buckets is not None:
            self.num_buckets = check_if_positive_int(self.num_buckets)
            if self.batch_size // 2**(self.num_buckets-1) == 0:
                raise ArgumentTypeError(
                    "Too many batching buckets, the last bucket would contain 0 data points.")
        if self.first_bucket_boundary is not None:
            self.first_bucket_boundary = check_if_positive_int(self.first_bucket_boundary)


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
        """Check arguments and specify values automatically if possible and necessary"""
        self._check_arguments()
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

    def _check_arguments(self) -> None:
        """Helper function to check learning rate arguments"""
        self.learning_rate = check_lr_interval(self.learning_rate)
        if self.scheduled_iterations is not None:
            self.scheduled_iterations = check_if_positive_int(self.scheduled_iterations)
        if self.cycle_end is not None:
            self.cycle_end = check_if_positive_int(self.cycle_end)
            if self.cycle_end >= self.scheduled_iterations:
                raise ArgumentTypeError(
                    f"The cycle end ({self.cycle_end}) should be "
                    f"less than the number of scheduled iterations ({self.scheduled_iterations}).")
        if self.last_rate is not None:
            self.last_rate = check_lr_interval(self.last_rate)
        if self.max_rate is not None:
            self.max_rate = check_lr_interval(self.max_rate)


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
        """Check arguments and their compatibility"""
        self.initial_rate = check_lr_interval(self.initial_rate)
        self.total_steps = check_if_positive_int(self.total_steps)
        self.half_cycle = check_if_positive_int(self.half_cycle)
        self.cycle_extremum = check_lr_interval(self.cycle_extremum)
        if self.end_extremum is not None:
            self.end_extremum = check_lr_interval(self.end_extremum)
        if 2 * self.half_cycle > self.total_steps:
            raise ArgumentTypeError(
                f"`total_steps` ({self.total_steps}) must be larger than "
                f"the number of steps in a single cycle ({2 * self.half_cycle})")


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
        """Check arguments and convert the `amsgrad` argument to Boolean
        if it is an integer
        """
        self.amsgrad = bool(self.amsgrad)
        self.learning_rate = check_lr_interval(self.learning_rate)
        self.weight_decay = check_if_float_in_interval(
            self.weight_decay, interval=(0., 1.), interval_type="closed")
        self.beta_1 = check_lr_interval(self.beta_1)
        self.beta_2 = check_lr_interval(self.beta_2)


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
        """Check arguments. If `save_optimizer` is an integer,
        convert is to Boolean
        """
        self.save_optimizer = bool(self.save_optimizer)
        self.checkpoint_path = check_if_output_path(self.checkpoint_path)
        if isinstance(self.save_freq, (int, float)):
            self.save_freq = check_if_positive_int(self.save_freq)
        elif self.save_freq != "epoch":
            raise ArgumentTypeError(f"Invalid save frequency: {self.save_freq}")
        if isinstance(self.log_update_freq, (int, float)):
            self.log_update_freq = check_if_positive_int(self.log_update_freq)
        elif self.log_update_freq != "epoch":
            raise ArgumentTypeError(f"Invalid log update frequency: {self.log_update_freq}")


@dataclass
class TransformerConfigs(DeeplArgs):
    """A dataclass for Transformer configuration objects

    Fields:
        bert_config: A BERT configuration object
        gpt_config: Optional. A GPT configuration object
    """
    bert_config: BertConfig
    gpt_config: Optional[OpenAIGPTConfig] = None

    def __post_init__(self) -> None:
        """Check arguments"""
        if not isinstance(self.bert_config, BertConfig):
            raise ArgumentTypeError(
                f"Bad encoder configuration object of type {type(self.bert_config)}, "
                f"a `BertConfig` instance is expected.")
        if not isinstance(self.gpt_config, OpenAIGPTConfig):
            raise ArgumentTypeError(
                f"Bad decoder configuration object of type {type(self.gpt_config)}, "
                f"an `OpenAIGPTConfig` instance is expected.")


@dataclass
class RnnLayerArgs(DeeplArgs):
    """A dataclass for basic RNN arguments
    This can be useful when arguments need to be passed to a Transformer + RNN decoder
    Fields:
        num_rnn_layers: Number of layers in a deep RNN. Defaults to 2
        hidden_size: RNN hidden size. Defaults to 768
        layernorm_eps: Epsilon parameter for layer normalization. Defaults to 1e-12
        dropout_rate: A dropout rate between 0 and 1. Defaults to 0.1
    """
    num_rnn_layers: int
    hidden_size: int = 768
    layernorm_eps: float = 1e-12
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        """Check argument values"""
        self.num_rnn_layers = check_if_positive_int(self.num_rnn_layers)
        self.hidden_size = check_if_positive_int(self.hidden_size)
        self.layernorm_eps = check_lr_interval(self.layernorm_eps)
        self.dropout_rate = check_if_float_in_interval(
            self.dropout_rate, (0., 1.), interval_type="left_closed")


@dataclass
class RnnArgs(RnnLayerArgs):
    """A dataclass for RNN decoder arguments

    Fields:
        num_rnn_layers: Number of layers in a deep RNN. Defaults to 2
        hidden_size: RNN hidden size. Defaults to 768
        layernorm_eps: Epsilon parameter for layer normalization. Defaults to 1e-12
        dropout_rate: A dropout rate between 0 and 1. Defaults to 0.1
        vocab_size: Number of elements in the vocabulary. Defaults to 32001
        initializer_dev: Stddev in the `TruncatedNormal` initializer for the embedding layer.
            Defaults to 0.02
    """
    vocab_size: int = 32001
    initializer_dev: float = 0.02

    def __post_init__(self) -> None:
        """Check arguments"""
        super().__post_init__()
        self.vocab_size = check_if_positive_int(self.vocab_size)
        self.initializer_dev = check_if_positive_float(self.initializer_dev)


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

    def __post_init__(self) -> None:
        """Check arguments"""
        self.vocab_size = check_if_positive_int(self.vocab_size)
        self.hidden_size = check_if_positive_int(self.hidden_size)
        self.initializer_range = check_if_positive_float(self.initializer_range)
        self.layer_norm_eps = check_lr_interval(self.layer_norm_eps)
        self.hidden_dropout_prob = check_lr_interval(self.hidden_dropout_prob)


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

    def __post_init__(self) -> None:
        """Check arguments"""
        super().__post_init__()
        self.max_position_embeddings = check_if_positive_int(self.max_position_embeddings)
        self.min_freq = check_if_positive_float(self.min_freq)


@dataclass
class KlArgs(DeeplArgs):
    """A dataclass for handling the KL loss term.

    Fields:
        kl_factor: A constant by which the KL loss will be multiplied. Defaults to `1.0`.
        min_kl: Element-wise minimum of the KL loss vector, often denoted as lambda. Defaults to `0.0`.
        target_kl_factor: Optional. A target KL multiplier if `kl_factor` is to be annealed.
        kl_steps: Optional. The number of steps to make until `target_kl_factor` is reached if
            `kl_factor` is annealed.
    """
    kl_factor: float = 1.0
    min_kl: float = 0.0
    target_kl_factor: Optional[float] = None
    kl_steps: Optional[int] = None

    def __post_init__(self) -> None:
        """Check arguments."""
        self.kl_factor = check_if_nonnegative_float(self.kl_factor)
        self.min_kl = check_if_nonnegative_float(self.min_kl)
        if self.target_kl_factor is not None:
            self.target_kl_factor = check_if_nonnegative_float(self.target_kl_factor)
        if self.kl_steps is not None:
            self.kl_steps = check_if_positive_int(self.kl_steps)


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

    def __post_init__(self) -> None:
        """Check arguments"""
        self.train_path = check_if_file(self.train_path)
        self.dev_path = check_if_file(self.dev_path)
        if self.test_path is not None:
            self.test_path = check_if_file(self.test_path)


def check_if_positive_int(maybe_positive: Union[str, int, float]) -> int:
    """Check if an integer is positive"""
    if not isinstance(maybe_positive, int):
        maybe_positive = int(maybe_positive)
    if maybe_positive <= 0:
        raise ArgumentTypeError(f"A positive integer is expected, got {maybe_positive}")
    return maybe_positive


def check_if_output_path(maybe_output_path: str) -> str:
    """Check if the input can be an output path"""
    if not Path(maybe_output_path).parent.is_dir():
        raise ArgumentTypeError(f"{maybe_output_path} is not a valid path "
                                "as the parent directory does not exist.")
    return maybe_output_path


def check_if_dir(maybe_dir: str) -> str:
    """Check if the input is a path to a directory"""
    if not Path(maybe_dir).is_dir():
        raise ArgumentTypeError(f"{maybe_dir} is not a path to a directory.")
    return maybe_dir


def check_if_file(maybe_file: str) -> str:
    """Check if the input is a path to a file"""
    if not Path(maybe_file).is_file():
        raise ArgumentTypeError(f"{maybe_file} is not a path to a file.")
    return maybe_file


def check_if_nonempty_string(maybe_nonempty: str) -> str:
    """Check if the input is not the empty string"""
    if maybe_nonempty == "":
        raise ArgumentTypeError(f"The empty string is not a valid argument.")
    return maybe_nonempty


def check_if_positive_float(maybe_positive: Union[str, int, float]) -> float:
    """Check if the input `float` is positive"""
    maybe_positive = float(maybe_positive)
    if maybe_positive <= 0:
        raise ArgumentTypeError(
            f"A positive float is expected, got {maybe_positive}")
    return maybe_positive


def check_if_nonnegative_float(maybe_nonnegative: Union[str, int, float]) -> float:
    """Check if the input `float` is non-negative."""
    maybe_nonnegative = float(maybe_nonnegative)
    if maybe_nonnegative < 0:
        raise ArgumentTypeError(
            f"A non-negative float is expected, got {maybe_nonnegative}")
    return maybe_nonnegative


def check_if_float_in_interval(
        number: Union[str, int, float],
        interval: Tuple[float, float],
        interval_type: Literal["open", "closed", "left_closed", "right_closed"]
) -> float:
    """Check if a `float` is in the specified interval"""
    number = float(number)
    lower, upper = interval
    if interval_type == "open":
        is_correct = lower < number < upper
    elif interval_type == "closed":
        is_correct = lower <= number <= upper
    elif interval_type == "left_closed":
        is_correct = lower <= number < upper
    elif interval_type == "right_closed":
        is_correct = lower < number <= upper
    else:
        raise ValueError(f"Unknown interval type: {interval_type}")
    if not is_correct:
        raise ArgumentTypeError(f"{number} is not in the {interval_type} interval "
                                f"between {lower} and {upper}.")
    return number


check_lr_interval = partial(
    check_if_float_in_interval, interval=(0., 1.), interval_type="right_closed")
