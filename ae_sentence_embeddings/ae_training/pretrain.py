"""A module for pre-training"""

from dataclasses import dataclass
from copy import deepcopy
from os import environ
from typing import Mapping, Any, Optional, Sequence, Union, Literal, Dict, List
from inspect import signature

import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow_addons.optimizers import AdamW
from transformers import BertConfig, OpenAIGPTConfig
import wandb
from wandb.keras import WandbCallback

from ae_sentence_embeddings.ae_training.model_type_config import (
    multilingual_models,
    model_type_map,
    rnn_only_decoder_models
)
from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    AdamwArgs,
    SaveAndLogArgs,
    DataSplitPathArgs,
    OneCycleArgs,
    RnnLayerArgs,
    RnnArgs,
    KlArgs,
    camel_to_snake
)
from ae_sentence_embeddings.callbacks import (
    AeCustomCheckpoint,
    OneCycleScheduler,
    CyclicScheduler,
    DevEvaluator
)
from ae_sentence_embeddings.data import get_train_and_validation
from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy, IgnorantSparseCatAccuracy
from ae_sentence_embeddings.modeling_tools import get_training_args, read_json

PoolingTypes = Literal["average", "cls_sep", "p_means"]


@dataclass
class GroupedArgs:
    """A dataclass for training dataclasses"""
    data_split_path_args: DataSplitPathArgs
    data_stream_args: DataStreamArgs
    adamw_args: AdamwArgs
    lr_one_cycle_args: Optional[OneCycleArgs]
    momentum_one_cycle_args: Optional[OneCycleArgs]
    save_and_log_args: SaveAndLogArgs


@dataclass
class ModelArgs:
    """A dataclass for model specification data"""
    model_type_name: str
    encoder_config: BertConfig
    decoder_config: Union[OpenAIGPTConfig, RnnArgs]
    pooling_type: PoolingTypes
    kl_args: KlArgs


def _underscored_snake_from_camel(word: Union[str, type]) -> str:
    """Convert CamelCase to snake_case and append an underscore.
    If the input is a type, the snake_cased and underscored class name will be returned
    """
    if isinstance(word, str):
        snake_word = camel_to_snake(word)
    else:
        snake_word = camel_to_snake(word.__name__)
    return snake_word + "_"


def devices_setup(
        devices: Optional[Sequence[str]] = None
) -> Union[tf.distribute.MirroredStrategy, tf.distribute.OneDeviceStrategy]:
    """Setup devices for training.

    Args:
        devices: Optional. A list of device names.

    Returns:
        A distribution strategy: `MirroredStrategy` is multiple GPUs
        are available or `OneDeviceStrategy` otherwise.
    """
    num_gpus = len(tf.config.get_visible_devices("GPU"))
    if devices is not None:
        gpus = [device for device in devices if device.lower().startswith(("gpu", "/gpu"))]
        num_requested_gpus = len(gpus)
        assert num_requested_gpus <= num_gpus, f"{num_gpus} GPUs are available, requested {num_requested_gpus}."
        num_gpus = num_requested_gpus
        if num_gpus != 0:
            gpus = ",".join(gpu[-1] for gpu in gpus)
            environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environ["CUDA_VISIBLE_DEVICES"] = gpus
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        device_to_use = devices[0] if devices is not None else "/cpu:0"
        strategy = tf.distribute.OneDeviceStrategy(device=device_to_use)
    return strategy


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
    return GroupedArgs(
        data_split_path_args=dataset_split_paths,
        data_stream_args=data_args,
        adamw_args=adamw_args,
        lr_one_cycle_args=lr_args,
        momentum_one_cycle_args=momentum_args,
        save_and_log_args=save_log_args
    )


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
    bert_config_pref = "bert_config_"
    encoder_config = BertConfig(**{k[len(bert_config_pref):]: v for k, v in config_dict.items()
                                   if k.startswith(bert_config_pref)})
    if model_type_name in rnn_only_decoder_models:
        decoder_config = RnnArgs.collect_from_dict(config_dict, prefix=_underscored_snake_from_camel(RnnArgs))
    else:
        gpt_config_pref = "openai_gpt_config_"
        decoder_config = OpenAIGPTConfig(**{k[len(gpt_config_pref):]: v for k, v in config_dict.items()
                                            if k.startswith(gpt_config_pref)})
    kl_args = KlArgs.collect_from_dict(config_dict, prefix=_underscored_snake_from_camel(KlArgs))
    return ModelArgs(
        model_type_name=model_type_name,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        kl_args=kl_args,
        pooling_type=config_dict["pooling_type"]
    )


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


def get_model_kwargs(
        model_type: type,
        pooling_type: PoolingTypes,
        kl_factor: float,
        min_kl: float,
) -> Dict[str, Union[PoolingTypes, float, int, RnnLayerArgs, None]]:
    """Helper function to collect keyword arguments for model initialization"""
    pooling_type_name = "pooling_type"
    kl_factor_name, min_kl_name = "kl_factor", "min_kl"
    model_init_kwargs = {pooling_type_name: pooling_type}
    expected_args = signature(model_type.__init__).parameters.keys()
    if kl_factor_name in expected_args:
        model_init_kwargs[kl_factor_name] = kl_factor
    if min_kl_name in expected_args:
        model_init_kwargs[min_kl_name] = min_kl
    return model_init_kwargs


def log_and_schedule_setup(
        dev_dataset: tf.data.Dataset,
        save_and_log_args: SaveAndLogArgs,
        validation_freq: Union[int, Literal["epoch"]],
        lr_args: Optional[OneCycleArgs] = None,
        momentum_args: Optional[OneCycleArgs] = None,
        keep_lr_cycling: bool = False
) -> List[tf.keras.callbacks.Callback]:
    """Logging and learning rate scheduling setup.

    Args:
        dev_dataset: The validation dataset.
        save_and_log_args: Model checkpoint and logging arguments as a dataclass.
        validation_freq: Specify how often to validate: After each epoch (value `"epoch"`).
            or after `validation_freq` iterations (if integer).
        lr_args: Optional. Learning rate scheduler arguments as a dataclass.
        momentum_args: Optional. Momentum scheduler arguments as a dataclass.
        keep_lr_cycling: Keep the learning rate cycling. The `initial_rate`, `total_steps` and `cycle_extremum`
            fields of `lr_args` are required to apply this option. Defaults to `False`.

    Returns:
        A list of callbacks:
            Checkpoints: This is always included.
            Learning rate scheduling: Only included if one-cycle learning rate scheduling was chosen.
            Momentum scheduling: Only included if one-cycle momentum scheduling was chosen.
            Validation: Only included if validation is expected to be done more frequently
                than after each epoch. Otherwise, the Keras `fit` method can handle the validation.
            WandB: Only included if the logging tool is `WandB`.
    """
    assert not (keep_lr_cycling and lr_args is None), "Cycling learning rate was requested but " \
                                                      "no arguments were provided."
    callbacks = [AeCustomCheckpoint(
        checkpoint_root=save_and_log_args.checkpoint_path,
        save_freq=save_and_log_args.save_freq,
        save_optimizer=save_and_log_args.save_optimizer
    )]
    if validation_freq != "epoch":
        assert save_and_log_args.log_tool is not None, "Please specify a logging tool if validation is to be made " \
                                                       "more frequently than after each epoch."
        callbacks.append(DevEvaluator(
            dev_data=dev_dataset,
            logger=save_and_log_args.log_tool,
            log_freq=validation_freq,
        ))
    wandb_log_freq = save_and_log_args.log_update_freq if \
        save_and_log_args.log_tool.lower() == "wandb" else None
    if lr_args is not None:
        if keep_lr_cycling:
            lr_scheduler = CyclicScheduler(
                initial_rate=lr_args.initial_rate,
                cycle_extremum=lr_args.cycle_extremum,
                half_cycle=lr_args.half_cycle,
                log_freq=wandb_log_freq
            )
        else:
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
    return callbacks


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


def pretrain_transformer_ae(
        model_type_name: str,
        dataset_split_paths: DataSplitPathArgs, *,
        data_stream_args: DataStreamArgs,
        adamw_args: AdamwArgs,
        save_and_log_args: SaveAndLogArgs,
        validation_freq: Union[int, Literal["epoch"]],
        encoder_config: BertConfig,
        decoder_config: Union[OpenAIGPTConfig, RnnArgs],
        kl_args: KlArgs,
        model_ckpt: Optional[str] = None,
        pooling_type: PoolingTypes = "cls_sep",
        lr_args: Optional[OneCycleArgs] = None,
        momentum_args: Optional[OneCycleArgs] = None,
        keep_lr_cycling: bool = False,
        dataset_cache_dir: Optional[str] = None,
        devices: Optional[Sequence[str]] = None,
        num_epochs: int = 3,
        prefetch: Optional[int] = 2,
        verbose: Literal[0, 1, 2] = 2
) -> History:
    """Do pre-train.

    Args:
        model_type_name: Model class name as a string.
        dataset_split_paths: Paths to the dataset splits as a dataclass.
        data_stream_args: Data streaming arguments as a dataclass.
        adamw_args: AdamW optimizer arguments as a dataclass. It will be ignored if the model
            is loaded from a checkpoint.
        save_and_log_args: Model checkpoint and logging arguments as a dataclass.
        validation_freq: Specify how often to validate: After each epoch (value `"epoch"`).
            or after `validation_freq` iterations (if integer).
        encoder_config: Encoder configuration data. It will be ignored if the model is loaded from a checkpoint.
        decoder_config: Decoder configuration data. It will be ignored if the model is loaded from a checkpoint.
        kl_args: Arguments for handling the KL multiplier.
        model_ckpt: Optional. Path to a checkpoint to continue training.
        pooling_type: Pooling method, `'average'` or `'cls_sep'` or `'p_means'`. It will be ignored if the model
            is loaded from a checkpoint. Defaults to `'cls_sep'`.
            only if a VAE is to be trained. Defaults to `0.0`.
        num_epochs: Number of ae_training epochs. Defaults to `3`.
        lr_args: Optional. Learning rate scheduler arguments as a dataclass.
        momentum_args: Optional. Momentum scheduler arguments as a dataclass.
        keep_lr_cycling: Keep the learning rate cycling. The `initial_rate`, `total_steps` and `cycle_extremum`
            fields of `lr_args` are required to apply this option. Defaults to `False`.
        dataset_cache_dir: Optional. A cache directory for loading the `dataset`.
        devices: Optional. GPU devices to use, e.g. `\"GPU:0\", \"GPU:1\"`.
        prefetch: Optional. If specified, `prefetch` batches will be prefetched during training.
            Prefetching can be disabled by setting this to `None`. Defaults to `2`.
        verbose: `verbose` argument for `model.fit`. Defaults to `2`.

    Returns:
        The training history object.
    """
    if model_type_name in multilingual_models:
        raise NotImplementedError(
            f"The `pretrain_transformer_ae` function cannot be used to train {model_type_name}.")

    # Configure the model
    model_type = model_type_map[model_type_name]
    model_init_kwargs = get_model_kwargs(
        model_type=model_type,
        pooling_type=pooling_type,
        kl_factor=kl_args.kl_factor,
        min_kl=kl_args.min_kl,
    )

    # Configure the data
    train_dataset, dev_dataset = get_train_and_validation(
        data_split_paths=dataset_split_paths,
        train_args=data_stream_args,
        cache_dir=dataset_cache_dir,
        set_data_shard=True,
        prefetch=prefetch
    )

    # Configure the callbacks
    fit_validation_data = dev_dataset if validation_freq == "epoch" else None
    callbacks = log_and_schedule_setup(
        dev_dataset=dev_dataset,
        save_and_log_args=save_and_log_args,
        validation_freq=validation_freq,
        lr_args=lr_args,
        keep_lr_cycling=keep_lr_cycling,
        momentum_args=momentum_args,
    )

    # Configure the training
    strategy = devices_setup(devices)
    with strategy.scope():
        optimizer = AdamW(**adamw_args.to_dict())
        model = model_type(
            enc_config=encoder_config, dec_config=decoder_config, **model_init_kwargs)
        if model_ckpt is not None:
            _ = model((tf.keras.Input(shape=(None,), dtype=tf.int32),
                       tf.keras.Input(shape=(None,), dtype=tf.int32)))
            model.load_weights(model_ckpt)
        model.compile(optimizer=optimizer, loss=IgnorantSparseCatCrossentropy(from_logits=True),
                      metrics=[IgnorantSparseCatAccuracy()])
    del model_init_kwargs, encoder_config, decoder_config
    history = model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        validation_data=fit_validation_data,
        verbose=verbose
    )
    return history
