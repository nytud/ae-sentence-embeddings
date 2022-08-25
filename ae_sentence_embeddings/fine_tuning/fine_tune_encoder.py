# -*- coding: utf-8 -*-

"""Fine-tune an encoder model for classification."""

from typing import Optional, Union, Literal, List, Dict, Any

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback
from transformers import (
    TFAutoModel,
    TFAutoModelForTokenClassification,
    TFAutoModelForSequenceClassification
)

from ae_sentence_embeddings.losses_and_metrics import SparseCategoricalMCC, BinaryMCC, BinaryLogitAccuracy
from ae_sentence_embeddings.data import get_train_and_validation
from ae_sentence_embeddings.pretrain_vae import strategy_setup
from ae_sentence_embeddings.models import SentVaeClassifier
from ae_sentence_embeddings.callbacks import (
    AeCustomCheckpoint, DevEvaluator,
    OneCycleScheduler, CyclicScheduler
)
from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs, AdamwArgs, SaveAndLogArgs,
    DataSplitPathArgs, OneCycleArgs
)

transformers_type_dict = {
    "Model": TFAutoModel,
    "SequenceClassification": TFAutoModelForSequenceClassification,
    "TokenClassification": TFAutoModelForTokenClassification,
    "SentVaeClassifier": SentVaeClassifier
}


def lookup_transformer_type(
        transformer_type: Literal["Model", "SequenceClassification", "TokenClassification"]
) -> type:
    """Get the `transformers` architecture indicated by a string."""
    try:
        transformer_auto_type = transformers_type_dict[transformer_type]
    except KeyError:
        raise NotImplementedError(f"Unrecognized model type: {transformer_type}. "
                                  f"Please specify one of {list(transformers_type_dict.keys())}")
    return transformer_auto_type


def log_and_schedule_setup(
        dev_dataset: tf.data.Dataset,
        save_and_log_args: SaveAndLogArgs,
        validation_freq: Union[int, Literal["epoch"]],
        lr_args: Optional[OneCycleArgs] = None,
        momentum_args: Optional[OneCycleArgs] = None,
        keep_lr_cycling: bool = False,
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

    Raises:
        AssertionError if `keep_lr_cycling` is set to `True` but `lr_args` is not specified.
    """
    assert not (keep_lr_cycling and lr_args is None), \
        "Cycling learning rate was requested but no arguments were provided."

    callbacks = [AeCustomCheckpoint(
        checkpoint_root=save_and_log_args.checkpoint_path,
        save_freq=save_and_log_args.save_freq,
        save_optimizer=save_and_log_args.save_optimizer,
        no_serialization=True
    )]

    if validation_freq != "epoch":
        callbacks.append(DevEvaluator(
            dev_data=dev_dataset,
            logger=save_and_log_args.log_tool,
            log_freq=validation_freq,
        ))

    if lr_args is not None:
        if keep_lr_cycling:
            lr_scheduler = CyclicScheduler(
                initial_rate=lr_args.initial_rate,
                cycle_extremum=lr_args.cycle_extremum,
                half_cycle=lr_args.half_cycle,
                log_freq=save_and_log_args.log_update_freq
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

    if str(save_and_log_args.log_tool).lower() == "wandb":
        callbacks.append(WandbCallback(
            monitor="loss",
            save_model=False,
            predictions=100,
            log_batch_frequency=save_and_log_args.log_update_freq
        ))

    return callbacks


def fine_tune(
        model_ckpt: str, *,
        num_labels: int,
        dataset_split_paths: DataSplitPathArgs,
        data_stream_args: DataStreamArgs,
        adamw_args: AdamwArgs,
        save_and_log_args: SaveAndLogArgs,
        validation_freq: Union[int, Literal["epoch"]],
        model_type: type = SentVaeClassifier,
        encoder_config: Optional[Dict[str, Any]] = None,
        lr_args: Optional[OneCycleArgs] = None,
        momentum_args: Optional[OneCycleArgs] = None,
        freeze_encoder: bool = False,
        dataset_cache_dir: Optional[str] = None,
        drop_remainder: bool = True,
        use_mcc: bool = False,
        num_epochs: int = 3,
        prefetch: Optional[int] = 2,
        verbose: Literal[0, 1, 2] = 2
) -> tf.keras.callbacks.History:
    """Do fine-tuning.

    Args:
        model_ckpt: A model checkpoint or a model name. If available, the model will be loaded
            with the `from_pretrained` method (implemented by all `transformers` models).
            Otherwise, it will be initialized using `encoder_config` and the weights will be
            loaded with a custom `build_and_load` method.
        num_labels: Number of classification labels.
        dataset_split_paths: Paths to the dataset splits as a dataclass.
        data_stream_args: Data streaming arguments as a dataclass.
        adamw_args: AdamW optimizer arguments as a dataclass.
        save_and_log_args: Model checkpoint and logging arguments as a dataclass.
        validation_freq: Specify how often to validate: After each epoch (value `"epoch"`).
            or after `validation_freq` iterations (if integer).
        model_type: The model class, i.e. `TFBertModelForSequenceClassification`.
        encoder_config: Optional. The encoder configuration object as a dict. Only relevant
            if the model class does not implement the `from_pretrained` method.
        lr_args: Optional. Learning rate scheduler arguments as a dataclass.
        momentum_args: Optional. Momentum scheduler arguments as a dataclass.
        freeze_encoder: Set it to `True` if only the classifier head is to be trained.
            Defaults to `False`.
        dataset_cache_dir: Optional. A cache directory for loading the `dataset`.
        drop_remainder: Specify whether the last batch should be dropped. Defaults to `True`.
        use_mcc: Use Matthews Correlation Coefficient metric instead of accuracy.
            Defaults to `False`.
        num_epochs: Number of ae_training epochs. Defaults to `3`.
        prefetch: Optional. If specified, `prefetch` batches will be prefetched during
            training. Prefetching can be disabled by setting this to `None`. Defaults to `2`.
        verbose: `verbose` argument for `model.fit`. Defaults to `2`.

    Returns:
        The training history object.

    Raises:
        TypeError if the model class does not implement the `from_pretrained` method
        and `encoder_config` is not specified.
    """
    # Model check
    is_transformers_like = hasattr(model_type, "from_pretrained")
    if not is_transformers_like and encoder_config is None:
        raise TypeError(f"You have specified the model type {model_type.__name__} which does not have "
                        "a `from_pretrained` method. Please specify a model class that has a `build_and_load` "
                        "method and pass an encoder configuration object to `encoder_config`. `model_ckpt` "
                        "should be a pretrained TensorFlow model.")

    # Configure the data
    train_dataset, dev_dataset = get_train_and_validation(
        data_split_paths=dataset_split_paths,
        train_args=data_stream_args,
        cache_dir=dataset_cache_dir,
        set_data_shard=True,
        prefetch=prefetch,
        drop_remainder=drop_remainder
    )

    # Configure the callbacks
    fit_validation_data = dev_dataset if validation_freq == "epoch" else None
    callbacks = log_and_schedule_setup(
        dev_dataset=dev_dataset,
        save_and_log_args=save_and_log_args,
        validation_freq=validation_freq,
        lr_args=lr_args,
        momentum_args=momentum_args
    )

    # Configure the training
    strategy = strategy_setup()
    with strategy.scope():
        if is_transformers_like:
            # PyCharm may regard `model_type` as `None` here
            # noinspection PyUnresolvedReferences
            model = model_type.from_pretrained(model_ckpt, num_labels=num_labels)
            # Call the model with a dummy input to build the layers
            model((tf.keras.Input(shape=(None,), dtype=tf.int32),
                   tf.keras.Input(shape=(None,), dtype=tf.int32)))
        else:
            model = model_type(encoder_config, num_labels=num_labels)
            model.build_and_load(model_ckpt)
        optimizer = AdamW(**adamw_args.to_dict())

        if num_labels > 1:
            metrics = [SparseCategoricalAccuracy()]
            loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        else:
            metrics = [BinaryLogitAccuracy()]
            loss_fn = BinaryCrossentropy(from_logits=True)
        if use_mcc:
            mcc = SparseCategoricalMCC(num_classes=num_labels) if num_labels > 1 \
                else BinaryMCC()
            metrics.append(mcc)

        if freeze_encoder:
            for encoder_layer in (layer for layer in model.layers if
                                  not layer.name.lower().startswith("classifier")):
                encoder_layer.trainable = False
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    history = model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        validation_data=fit_validation_data,
        verbose=verbose
    )
    return history
