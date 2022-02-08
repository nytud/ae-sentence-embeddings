"""A module for pre-training"""

from os import environ
from typing import Mapping, Tuple, Any, Optional, Sequence, Union, Literal

import tensorflow as tf
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow_addons.optimizers import AdamW
from transformers import BertConfig, OpenAIGPTConfig
from wandb.keras import WandbCallback

from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    LearningRateArgs,
    AdamWArgs,
    SaveAndLogArgs,
    TransformerConfigs,
    DataSplitPathArgs,
    OneCycleArgs,
    RnnArgs
)
from ae_sentence_embeddings.callbacks import (
    AeCustomCheckpoint,
    OneCycleScheduler,
    DevEvaluator,
    basic_checkpoint_and_log
)
from ae_sentence_embeddings.data import get_train_and_validation, post_batch_feature_pair
from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy, IgnorantSparseCatAccuracy
from ae_sentence_embeddings.models import TransformerVae
from ae_sentence_embeddings.ae_training.model_type_config import multilingual_models, model_type_map


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
            gpus = ",".join([gpu[-1] for gpu in gpus])
            environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environ["CUDA_VISIBLE_DEVICES"] = gpus
    return num_gpus


def group_arguments(args: Mapping[str, Any]
                    ) -> Tuple[DataSplitPathArgs, DataStreamArgs, LearningRateArgs, AdamWArgs, SaveAndLogArgs]:
    """Get dataclasses from a single mapping of all arguments

    Args:
        args: A mapping between all argument names and values

    Returns:
        Five specialized dataclasses for the arguments of dataset split preparation, data streaming,
        learning rate scheduling, AdamW optimizer and model saving/logging, respectively

    """
    dataset_split_paths = DataSplitPathArgs.collect_from_dict(args)
    data_args = DataStreamArgs.collect_from_dict(args)
    lr_args = LearningRateArgs.collect_from_dict(args)
    adamw_args = AdamWArgs.collect_from_dict(args)
    save_log_args = SaveAndLogArgs.collect_from_dict(args)
    return dataset_split_paths, data_args, lr_args, adamw_args, save_log_args


def get_transformer_configs(config_dict: Mapping[str, Any]) -> TransformerConfigs:
    bert_config = BertConfig(**config_dict["bert_config"])
    gpt_config = OpenAIGPTConfig(**config_dict["gpt_config"])
    return TransformerConfigs(bert_config, gpt_config)


def pretrain_transformer_ae(
        model_type_name: str,
        dataset_split_paths: DataSplitPathArgs, *,
        data_args: DataStreamArgs,
        adamw_args: AdamWArgs,
        save_log_args: SaveAndLogArgs,
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
        data_args: Data streaming arguments as a dataclass
        adamw_args: AdamW optimizer arguments as a dataclass
        save_log_args: Model checkpoint and logging arguments as a dataclass
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
        train_args=data_args,
        cache_dir=dataset_cache_dir
    )
    if model_type_name in multilingual_models:
        train_dataset = train_dataset.map(post_batch_feature_pair)
        dev_dataset = dev_dataset.map(post_batch_feature_pair)
    train_dataset = train_dataset.with_options(data_options).prefetch(2)
    dev_dataset = dev_dataset.with_options(data_options).prefetch(2)

    callbacks = [AeCustomCheckpoint(
        checkpoint_root=save_log_args.checkpoint_path,
        save_freq=save_log_args.save_freq,
        save_optimizer=save_log_args.save_optimizer
    )]
    if validation_freq != "epoch":
        if save_log_args.log_tool is None:
            raise ValueError(
                "Please specify a logging tool if validation is to be made more frequently than after each epoch")
        callbacks.append(DevEvaluator(
            dev_data=dev_dataset,
            logger=save_log_args.log_tool,
            log_freq=validation_freq,
        ))
        fit_validation_data = None
    else:
        fit_validation_data = dev_dataset
    if lr_args is not None:
        lr_scheduler = OneCycleScheduler(
            schedule_args=lr_args,
            parameter="lr",
            log_tool=save_log_args.log_tool,
            log_freq=save_log_args.log_update_freq
        )
        callbacks.append(lr_scheduler)
    if momentum_args is not None:
        momentum_scheduler = OneCycleScheduler(
            schedule_args=momentum_args,
            parameter="beta_1",
            log_tool=save_log_args.log_tool,
            log_freq=save_log_args.log_update_freq
        )
        callbacks.append(momentum_scheduler)
    if isinstance(save_log_args.log_tool, str):
        if save_log_args.log_tool.lower() == "wandb":
            callbacks.append(WandbCallback(
                monitor="loss",
                save_model=False,
                predictions=100,
                log_batch_frequency=save_log_args.log_update_freq
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


def hparam_search(
        train_ds: tf.data.Dataset,
        dev_ds: tf.data.Dataset,
        lr_args: LearningRateArgs,
        adamw_args: AdamWArgs,
        transformer_configs: TransformerConfigs,
        save_log_args: SaveAndLogArgs
) -> float:
    """Training for hyperparameter tuning

    Args:
        train_ds: The ae_training dataset
        dev_ds: The validation dataset
        lr_args: Learning rate scheduler arguments as a dataclass
        adamw_args: AdamW optimizer arguments as a dataclass
        transformer_configs: Transformer configurations as a dataclass
        save_log_args: Model checkpoint and logging arguments as a dataclass

    Returns:
        The validation loss as a single floating point number

    """
    scheduler = OneCycleScheduler(lr_args)
    log_callback = basic_checkpoint_and_log(save_log_args)[0]
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = TransformerVae(enc_config=transformer_configs.bert_config,
                               dec_config=transformer_configs.gpt_config)
        optimizer = AdamW(**adamw_args.to_dict())
        model.compile(optimizer=optimizer, loss=IgnorantSparseCatCrossentropy(from_logits=True))
    history = model.fit(
        x=train_ds,
        callbacks=[scheduler, log_callback, EarlyStopping(patience=2)],
        validation_data=dev_ds
    )
    return history.history["val_loss"]
