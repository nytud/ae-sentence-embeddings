"""A module for dataclasses that can help handle training and data arguments"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Dict, Any, Union, Optional, Sequence
from inspect import signature

from transformers import BertConfig, OpenAIGPTConfig


class DeeplArgs:
    """Base class for training arguments. Subclass it and add fields to use its methods"""

    @classmethod
    def collect_from_dict(cls, args_dict: Mapping[str, Any]) -> DeeplArgs:
        """Create a TrainingArgs instance from a dictionary. Ignore irrelevant keys"""
        filtered_args = {key: val for key, val in args_dict.items()
                         if key in signature(cls).parameters}
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
                               bucketing will not be used. It can take effect only if `num_buckets` is specified
                               as well. Other bucket boundaries are expected to be derived from the specified value
                               by the function that requires the fields of this dataclass as arguments.
    """
    input_padding: Union[Sequence[int], int] = (0, 0)
    target_padding: int = -1
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
class AdamWArgs(DeeplArgs):
    """A dataclass for AdamW optimizer arguments
    For details on the fields, see `https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW`
    """
    weight_decay: float = 5e-5
    learning_rate: float = 1e-5
    beta_1: float = 0.9
    beta_2: float = 0.999


@dataclass
class SaveAndLogArgs(DeeplArgs):
    """A dataclass for checkpoint and log arguments

    Fields:
        checkpoint_path: Model checkpoint path
        log_path: Tensorboard logging path
        save_freq: `'epoch'` or integer. If `'epoch'`, checkpoints will be created after each epoch. If integer,
                   the model will be saved after each `save_freq` iterations. Defaults to `\"epoch\"`
        log_update_freq: `'batch'`, '`epoch`' or integer, `update_freq` parameter of `tf.keras.callbacks.TensorBoard`.
                         Defaults to `\"epoch\"`
    """
    checkpoint_path: str
    log_path: str
    save_freq: Union[str, int] = "epoch"
    log_update_freq: Union[str, int] = "epoch"


@dataclass
class TransformerConfigs(DeeplArgs):
    """A dataclass for Transformer configuration objects

    Fields:
        bert_config: A BERT configuration object
        gpt_config: A GPT configuration object
    """
    bert_config: BertConfig
    gpt_config: OpenAIGPTConfig


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
