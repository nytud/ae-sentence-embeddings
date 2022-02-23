"""A module for pre-training"""

from collections import namedtuple
from copy import deepcopy
from os import environ
from typing import Mapping, Any, Optional, Sequence, Union, Literal, Dict

import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow_addons.optimizers import AdamW
from transformers import BertConfig, OpenAIGPTConfig
from wandb.keras import WandbCallback

from ae_sentence_embeddings.ae_training.model_type_config import (
    multilingual_models,
    model_type_map,
    rnn_decoder_models
)
from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    AdamwArgs,
    SaveAndLogArgs,
    DataSplitPathArgs,
    OneCycleArgs,
    RnnArgs,
    camel_to_snake
)
from ae_sentence_embeddings.callbacks import (
    AeCustomCheckpoint,
    OneCycleScheduler,
    DevEvaluator
)
from ae_sentence_embeddings.data import get_train_and_validation, post_batch_feature_pair
from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy, IgnorantSparseCatAccuracy

GroupedArgs = namedtuple("GroupedArgs", ["data_split_path_args", "data_stream_args", "adamw_args",
                         "lr_one_cycle_args", "momentum_one_cycle_args", "save_and_log_args"])
ModelArgs = namedtuple("ModelArgs", ["model_type_name", "encoder_config", "decoder_config", "pooling_type"])


def _underscored_snake_from_camel(word: Union[str, type]) -> str:
    """Convert CamelCase to snake_case and append an underscore.
    If the input is a type, the snake_cased and underscored class name will be returned"""
    if isinstance(word, str):
        snake_word = camel_to_snake(word)
    else:
        snake_word = camel_to_snake(word.__name__)
    return snake_word + "_"


def devices_setup(devices: Optional[Sequence[str]] = None) -> int:
    """Setup devices for ae_training

    Args:
        devices: Optional. A list of device names

    Returns:
        The number of GPUs to be used

    """
    if devices is None:
        num_gpus = 0
    else:
        gpus = [device for device in devices if device.lower().startswith(("gpu", "/gpu"))]
        num_gpus = len(gpus)
        if num_gpus != 0:
            gpus = ",".join(gpu[-1] for gpu in gpus)
            environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environ["CUDA_VISIBLE_DEVICES"] = gpus
    return num_gpus


def group_train_args_from_structured(args: Mapping[str, Any]) -> GroupedArgs:
    """Get training-related dataclasses from a single json-style data structure

    Args:
        args: A json-style data structure

    Returns:
        Six specialized dataclasses for the arguments of dataset split preparation, data streaming,
        AdamW optimizer, learning rate scheduling, momentum scheduling and model saving/logging, respectively.
        They are returned in a `namedtuple`
    """
    dataset_split_paths = DataSplitPathArgs.collect_from_dict(args)
    data_args = DataStreamArgs.collect_from_dict(args)
    adamw_args = AdamwArgs.collect_from_dict(args)
    if "one_cycle_args" in args.keys():
        lr_args = OneCycleArgs.collect_from_dict(args, prefix="lr_")
        momentum_args = OneCycleArgs.collect_from_dict(args, prefix="momentum_")
    else:
        lr_args = None
        momentum_args = None
    save_log_args = SaveAndLogArgs.collect_from_dict(args)
    grouped_args = GroupedArgs(dataset_split_paths, data_args, adamw_args,
                               lr_args, momentum_args, save_log_args)
    return grouped_args


def group_train_args_from_flat(args: Mapping[str, Any]) -> GroupedArgs:
    """Get training-related dataclasses from a single flat dictionary

    Args:
        args: A flat dictionary that is required to prefix dataclass field names with the snake_case dataclass names

    Returns:
        Six specialized dataclasses for the arguments of dataset split preparation, data streaming,
        AdamW optimizer, learning rate scheduling, momentum scheduling and model saving/logging, respectively.
        They are returned in a `namedtuple`
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
    grouped_args = GroupedArgs(dataset_split_paths, data_args, adamw_args,
                               lr_args, momentum_args, save_log_args)
    return grouped_args


def group_model_args_from_flat(config_dict: Mapping[str, Any]) -> ModelArgs:
    """Get model configuration data structures from a flat dictionary

    Args:
        config_dict: A flat dictionary that is required to prefix dataclass field names with
            the snake_case dataclass names

    Returns:
        The model type name (the class name), the encoder configuration object, the decoder configuration object
            and the pooling type (a string). These are returned as a namedtuple
    """
    model_type_name = config_dict["model_type"]
    encoder_config = BertConfig(**{k.lstrip(bert_config_pref): v for k, v in config_dict.values()
                                   if k.startswith((bert_config_pref := "bert_config_"))})
    if model_type_name in rnn_decoder_models:
        decoder_config = RnnArgs.collect_from_dict(config_dict, prefix=_underscored_snake_from_camel(RnnArgs))
    else:
        decoder_config = OpenAIGPTConfig(**{k.lstrip(gpt_config_pref): v for k, v in config_dict.values()
                                            if k.startswith((gpt_config_pref := "openai_gpt_config_"))})
    pooling_type = config_dict["pooling_type"]
    return ModelArgs(model_type_name, encoder_config, decoder_config, pooling_type)


def flatten_nested_dict(nested_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a json-style structure (nested dictionary)

    Args:
        nested_data: A json-style data structure. Only the first level of nesting will be flattened.
            The new keys that originally come from a nested dict will be prefixed with the name of the
            key whose value was the nested dict

    Returns:
        The flattened dictionary
    """
    flattened_dict = {}
    for key, value in deepcopy(nested_data).items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flattened_dict["_".join([key, nested_key])] = nested_value
        else:
            flattened_dict[key] = value
    return flattened_dict


def pretrain_transformer_ae(
        model_type_name: str,
        dataset_split_paths: DataSplitPathArgs, *,
        data_stream_args: DataStreamArgs,
        adamw_args: AdamwArgs,
        save_and_log_args: SaveAndLogArgs,
        validation_freq: Union[int, Literal["epoch"]],
        encoder_config: BertConfig,
        decoder_config: Union[OpenAIGPTConfig, RnnArgs],
        pooling_type: Literal["average", "cls_sep"] = "cls_sep",
        kl_factor: float = 1.0,
        num_epochs: int = 3,
        lr_args: Optional[OneCycleArgs] = None,
        momentum_args: Optional[OneCycleArgs] = None,
        dataset_cache_dir: Optional[str] = None,
        devices: Optional[Sequence[str]] = None,
        verbose: Literal[0, 1, 2] = 2
) -> History:
    """Do pre-train

    Args:
        model_type_name: model class name as a string
        dataset_split_paths: Paths to the dataset splits as a dataclass
        data_stream_args: Data streaming arguments as a dataclass
        adamw_args: AdamW optimizer arguments as a dataclass
        save_and_log_args: Model checkpoint and logging arguments as a dataclass
        validation_freq: Specify how often to validate: After each epoch (value `"epoch"`)
            or after `validation_freq` iterations (if integer)
        encoder_config: Encoder configuration data
        decoder_config: Decoder configuration data
        pooling_type: Pooling method`, "average"` or `"cls_sep"`. Defaults to `"cls_sep"`
        kl_factor: A normalizing constant by which the KL loss will be multiplied. This has effect
            only if a VAE is to be trained. Defaults to `1.0`
        num_epochs: Number of ae_training epochs. Defaults to 3
        lr_args: Optional. Learning rate scheduler arguments as a dataclass
        momentum_args: Optional. Momentum scheduler arguments as a dataclass
        dataset_cache_dir: Optional. A cache directory for loading the `dataset`
        devices: Optional. GPU devices to use, e.g. `\"GPU:0\", \"GPU:1\"`
        verbose: `verbose` argument for `model.fit`. Defaults to 2

    Returns:
        The training history object
    """
    model_type = model_type_map[model_type_name]
    data_options = tf.data.Options()
    data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset, dev_dataset = get_train_and_validation(
        data_split_paths=dataset_split_paths,
        train_args=data_stream_args,
        cache_dir=dataset_cache_dir
    )
    if model_type_name in multilingual_models:
        train_dataset = train_dataset.map(post_batch_feature_pair)
        dev_dataset = dev_dataset.map(post_batch_feature_pair)
    train_dataset = train_dataset.with_options(data_options).prefetch(2)
    dev_dataset = dev_dataset.with_options(data_options).prefetch(2)

    callbacks = [AeCustomCheckpoint(
        checkpoint_root=save_and_log_args.checkpoint_path,
        save_freq=save_and_log_args.save_freq,
        save_optimizer=save_and_log_args.save_optimizer
    )]
    if validation_freq != "epoch":
        if save_and_log_args.log_tool is None:
            raise ValueError(
                "Please specify a logging tool if validation is to be made more frequently than after each epoch")
        callbacks.append(DevEvaluator(
            dev_data=dev_dataset,
            logger=save_and_log_args.log_tool,
            log_freq=validation_freq,
        ))
        fit_validation_data = None
    else:
        fit_validation_data = dev_dataset
    if lr_args is not None:
        lr_scheduler = OneCycleScheduler(
            schedule_args=lr_args,
            parameter="lr",
            log_tool=save_and_log_args.log_tool,
            log_freq=save_and_log_args.log_update_freq
        )
        callbacks.append(lr_scheduler)
    if momentum_args is not None:
        momentum_scheduler = OneCycleScheduler(
            schedule_args=momentum_args,
            parameter="beta_1",
            log_tool=save_and_log_args.log_tool,
            log_freq=save_and_log_args.log_update_freq
        )
        callbacks.append(momentum_scheduler)
    if isinstance(save_and_log_args.log_tool, str):
        if save_and_log_args.log_tool.lower() == "wandb":
            callbacks.append(WandbCallback(
                monitor="loss",
                save_model=False,
                predictions=100,
                log_batch_frequency=save_and_log_args.log_update_freq
            ))
        else:
            raise NotImplementedError(
                "Please use `WandB` as a logging tool or a custom `Logging.logger` instance "
                "(the latter may only log messages from custom callbacks). Other logging "
                "tools such as Tensorboard are currently not supported.")

    num_gpus = devices_setup(devices)
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        device_to_use = devices[0] if devices is not None else None
        strategy = tf.distribute.OneDeviceStrategy(device=device_to_use)
    with strategy.scope():
        model = model_type(
            enc_config=encoder_config, dec_config=decoder_config, pooling_type=pooling_type)
        if hasattr(model, "kl_factor") and kl_factor != 1.0:
            model.set_kl_factor(kl_factor)
        optimizer = AdamW(**adamw_args.to_dict())
        model.compile(optimizer=optimizer, loss=IgnorantSparseCatCrossentropy(from_logits=True),
                      metrics=[IgnorantSparseCatAccuracy()])
    history = model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        validation_data=fit_validation_data,
        verbose=verbose
    )
    return history
