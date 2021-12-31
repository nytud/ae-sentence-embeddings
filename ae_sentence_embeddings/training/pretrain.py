"""A module for pre-training"""

from os import environ
from typing import Mapping, Tuple, Any, Optional, Sequence

import tensorflow as tf
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow_addons.optimizers import AdamW
from transformers import BertConfig, OpenAIGPTConfig

from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    LearningRateArgs,
    AdamWArgs,
    SaveAndLogArgs,
    TransformerConfigs,
    DataSplitPathArgs
)
from ae_sentence_embeddings.callbacks import basic_checkpoint_and_log, OneCycleScheduler
from ae_sentence_embeddings.data import get_train_and_validation
from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy
from ae_sentence_embeddings.models import TransformerVae


def devices_setup(devices: Optional[Sequence[str]]) -> int:
    """Setup devices for training

    Args:
        devices: Optional. A list of device names

    Returns:
        The number of GPUs to be used

    """
    if devices is None:
        num_gpus = 0
    else:
        gpus = [device for device in devices if device.startswith(("gpu", "GPU"))]
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
        dataset_split_paths: DataSplitPathArgs,
        data_args: DataStreamArgs,
        lr_args: LearningRateArgs,
        adamw_args: AdamWArgs,
        save_log_args: SaveAndLogArgs,
        transformer_configs: TransformerConfigs,
        num_epochs: int = 3,
        dataset_cache_dir: Optional[str] = None,
        devices: Optional[Sequence[str]] = None
) -> History:
    """Do pre-train

    Args:
        dataset_split_paths: Paths to the dataset splits as a dataclass
        data_args: Data streaming arguments as a dataclass
        lr_args: Learning rate scheduler arguments as a dataclass
        adamw_args: AdamW optimizer arguments as a dataclass
        save_log_args: Model checkpoint and logging arguments as a dataclass
        transformer_configs: Transformer configurations as a dataclass
        num_epochs: Number of training epochs. Defaults to 3
        dataset_cache_dir: Optional. A cache directory for loading the `dataset`
        devices: Optional. GPU devices to use, e.g. `\"GPU:0\", \"GPU:1\"`

    Returns:
        The training history object

    """
    train_dataset, dev_dataset = get_train_and_validation(
        data_split_paths=dataset_split_paths,
        train_args=data_args,
        cache_dir=dataset_cache_dir
    )
    scheduler = OneCycleScheduler(lr_args)
    callbacks = basic_checkpoint_and_log(save_log_args)
    callbacks.append(scheduler)
    num_gpus = devices_setup(devices)
    strategy = tf.distribute.MirroredStrategy() if num_gpus > 1 else tf.distribute.OneDeviceStrategy()
    with strategy.scope():
        model = TransformerVae(enc_config=transformer_configs.bert_config,
                               dec_config=transformer_configs.gpt_config)
        optimizer = AdamW(**adamw_args.to_dict())
        model.compile(optimizer=optimizer, loss=IgnorantSparseCatCrossentropy(from_logits=True))
    history = model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        validation_data=dev_dataset
    )
    return history


def hparam_search(
        train_ds: tf.data.Dataset,
        dev_ds: tf.data.Dataset,
        lr_args: LearningRateArgs,
        adamw_args: AdamWArgs,
        transformer_configs: TransformerConfigs,
) -> float:
    """Training for hyperparameter tuning

    Args:
        train_ds: The training dataset
        dev_ds: The validation dataset
        lr_args: Learning rate scheduler arguments as a dataclass
        adamw_args: AdamW optimizer arguments as a dataclass
        transformer_configs: Transformer configurations as a dataclass

    Returns:
        The validation loss as a single floating point number

    """
    scheduler = OneCycleScheduler(lr_args)
    strategy = tf.distribute.OneDeviceStrategy()
    with strategy.scope():
        model = TransformerVae(enc_config=transformer_configs.bert_config,
                               dec_config=transformer_configs.gpt_config)
        optimizer = AdamW(**adamw_args.to_dict())
        model.compile(optimizer=optimizer, loss=IgnorantSparseCatCrossentropy(from_logits=True))
    history = model.fit(
        x=train_ds,
        callbacks=[scheduler, EarlyStopping(patience=2)],
        validation_data=dev_ds
    )
    return history.history["val_loss"]
