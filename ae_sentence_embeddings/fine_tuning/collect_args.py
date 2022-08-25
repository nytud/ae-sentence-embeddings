# -*- coding: utf-8 -*-

"""A module to define tools that help collect
training arguments from a configuration dict.
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Mapping, Any, Optional, Union, Literal, Dict

import wandb
from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    AdamwArgs,
    SaveAndLogArgs,
    DataSplitPathArgs,
    OneCycleArgs,
    camel_to_snake
)
from ae_sentence_embeddings.modeling_tools import get_training_args, read_json

PoolingTypes = Literal["average", "cls_sep", "p_means"]


@dataclass
class GroupedArgs:
    """A dataclass for training dataclasses."""
    data_split_path_args: DataSplitPathArgs
    data_stream_args: DataStreamArgs
    adamw_args: AdamwArgs
    lr_one_cycle_args: Optional[OneCycleArgs]
    momentum_one_cycle_args: Optional[OneCycleArgs]
    save_and_log_args: SaveAndLogArgs


def _underscored_snake_from_camel(word: Union[str, type]) -> str:
    """Convert CamelCase to snake_case and append an underscore.
    If the input is a type, the snake_cased and underscored class name will be returned.
    """
    if isinstance(word, str):
        snake_word = camel_to_snake(word)
    else:
        snake_word = camel_to_snake(word.__name__)
    return snake_word + "_"


def flatten_nested_dict(nested_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a json-style structure (nested dictionary).

    Args:
        nested_data: A json-style data structure. Only the first level of nesting will be flattened.
            The new keys that originally come from a nested dict will be prefixed with the name of the
            key whose value was the nested dict.

    Returns:
        The flattened dictionary.
    """
    flattened_dict = {}
    for key, value in deepcopy(nested_data).items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened_dict["_".join([key, nested_key])] = nested_value
        else:
            flattened_dict[key] = value
    return flattened_dict


def group_train_args_from_flat(args: Mapping[str, Any]) -> GroupedArgs:
    """Get training-related dataclasses from a single flat dictionary.

    Args:
        args: A flat dictionary that is required to prefix dataclass field names
            with the snake_case dataclass names.

    Returns:
        Six specialized dataclasses for the arguments of dataset split preparation,
        data streaming, AdamW optimizer, learning rate scheduling, momentum scheduling
        and model saving/logging, respectively.
    """
    dataset_split_paths = DataSplitPathArgs.collect_from_dict(
        args, prefix=_underscored_snake_from_camel(DataSplitPathArgs))
    data_args = DataStreamArgs.collect_from_dict(args, prefix=_underscored_snake_from_camel(DataStreamArgs))
    adamw_args = AdamwArgs.collect_from_dict(args, prefix=_underscored_snake_from_camel(AdamwArgs))
    if any(key for key in args.keys() if key.startswith(_underscored_snake_from_camel(OneCycleArgs))):
        one_cycle_prefix = _underscored_snake_from_camel(OneCycleArgs)
        lr_args = OneCycleArgs.collect_from_dict(args, prefix=one_cycle_prefix + "lr_")
        momentum_args = OneCycleArgs.collect_from_dict(args, prefix=one_cycle_prefix + "momentum_")
    else:
        lr_args = None
        momentum_args = None
    save_log_args = SaveAndLogArgs.collect_from_dict(args, prefix=_underscored_snake_from_camel(SaveAndLogArgs))
    return GroupedArgs(
        data_split_path_args=dataset_split_paths,
        data_stream_args=data_args,
        adamw_args=adamw_args,
        lr_one_cycle_args=lr_args,
        momentum_one_cycle_args=momentum_args,
        save_and_log_args=save_log_args
    )


def _check_cycling_total_steps(arg_dict: Dict[str, Any]) -> None:
    """Helper function to check that the number of 1cycle scheduler total steps
    is the same for the learning rate and the momentum.

    Args:
        arg_dict: The flat dictionary that contains all hyperparameters.
            The function will add one key-vale pair to the dictionary if only
            `one_cycle_args_lr_total_steps` or `one_cycle_args_momentum_total_steps`
            is specified.
    """
    one_cycle_prefix = _underscored_snake_from_camel(OneCycleArgs)
    lr_total_steps_name = one_cycle_prefix + "lr_total_steps"
    momentum_total_steps_name = one_cycle_prefix + "momentum_total_steps"
    lr_total_steps = arg_dict.get(lr_total_steps_name)
    momentum_total_steps = arg_dict.get(momentum_total_steps_name)
    if all((lr_total_steps, momentum_total_steps)) and lr_total_steps != momentum_total_steps:
        raise ValueError(f"You have specified both learning rate cycling total steps ({lr_total_steps}) "
                         f"and momentum cycling total steps ({momentum_total_steps}) but they are not "
                         f"equal. Please specify only one of them or set them equal.")
    total_steps = lr_total_steps or momentum_total_steps
    arg_dict[lr_total_steps_name] = total_steps
    arg_dict[momentum_total_steps_name] = total_steps


def collect_wandb_args() -> Dict[str, Any]:
    """Create a WandB configuration dict from a configuration file.

    Returns:
        The `WandB` configuration dict.
    """
    parser = get_training_args()
    parser.add_argument("--project", help="Optional. Name of the current WandB project.")
    parser.add_argument("--run-name", dest="run_name", help="Optional. Name of the current run.")
    args = parser.parse_args()
    arg_dict = flatten_nested_dict(read_json(args.config_file))
    _check_cycling_total_steps(arg_dict)
    wandb.init(project=args.project, name=args.run_name, config=arg_dict)
    config = wandb.config
    # `momentum_half_cycle` may be unspecified. If, however, `lr_half_cycle` is specified,
    # they will be set equal. This makes sense as one usually wants to cycle learning rate
    # and momentum through the same number of iterations.
    one_cycle_prefix = _underscored_snake_from_camel(OneCycleArgs)
    lr_half_cycle = config.get(one_cycle_prefix + "lr_half_cycle")
    momentum_half_cycle = config.get(one_cycle_prefix + "momentum_half_cycle")
    if lr_half_cycle is not None and momentum_half_cycle is None:
        config[one_cycle_prefix + "momentum_half_cycle"] = lr_half_cycle
    return config
