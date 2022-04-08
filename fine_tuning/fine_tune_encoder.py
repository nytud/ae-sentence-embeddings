# -*- coding: utf-8 -*-

"""Fine-tune an encoder model for classification."""

from typing import Optional, Union, Sequence, Literal

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient as MCCoefficient

from ae_sentence_embeddings.data import get_train_and_validation
from ae_sentence_embeddings.ae_training import devices_setup, log_and_schedule_setup
from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    AdamwArgs,
    SaveAndLogArgs,
    DataSplitPathArgs,
    OneCycleArgs
)


def fine_tune(
        model_ckpt: str, *,
        num_labels: int,
        dataset_split_paths: DataSplitPathArgs,
        data_stream_args: DataStreamArgs,
        adamw_args: AdamwArgs,
        save_and_log_args: SaveAndLogArgs,
        validation_freq: Union[int, Literal["epoch"]],
        model_type: Optional[type] = None,
        lr_args: Optional[OneCycleArgs] = None,
        momentum_args: Optional[OneCycleArgs] = None,
        dataset_cache_dir: Optional[str] = None,
        devices: Optional[Sequence[str]] = None,
        use_mcc: bool = False,
        num_epochs: int = 3,
        prefetch: Optional[int] = 2,
        verbose: Literal[0, 1, 2] = 2
) -> History:
    """Do pre-train

    Args:
        model_ckpt: A model checkpoint or a model name. If `model_type` is specified, the
            model will be loaded with the `from_pretrained` method (implemented by all
            `transformers` models). Otherwise, the model will be loaded with the Keras
            `load_model` function.
        num_labels: Number of classification labels.
        dataset_split_paths: Paths to the dataset splits as a dataclass.
        data_stream_args: Data streaming arguments as a dataclass.
        adamw_args: AdamW optimizer arguments as a dataclass.
        save_and_log_args: Model checkpoint and logging arguments as a dataclass.
        validation_freq: Specify how often to validate: After each epoch (value `"epoch"`).
            or after `validation_freq` iterations (if integer).
        model_type: Optional. Path to a model checkpoint. If it is specified, the model
            weights will be loaded from the checkpoint before fine-tuning.
        lr_args: Optional. Learning rate scheduler arguments as a dataclass.
        momentum_args: Optional. Momentum scheduler arguments as a dataclass.
        dataset_cache_dir: Optional. A cache directory for loading the `dataset`.
        devices: Optional. GPU devices to use, e.g. `\"GPU:0\", \"GPU:1\"`.
        use_mcc: Use Matthews Correlation Coefficient metric instead of accuracy.
            Defaults to `False`.
        num_epochs: Number of ae_training epochs. Defaults to `3`.
        prefetch: Optional. If specified, `prefetch` batches will be prefetched during
            training. Prefetching can be disabled by setting this to `None`. Defaults to `2`.
        verbose: `verbose` argument for `model.fit`. Defaults to `2`.

    Returns:
        The training history object.
    """
    # Model check
    is_transformers_model = hasattr(model_type, "from_pretrained")
    if not is_transformers_model and model_type is not None:
        raise TypeError(f"You have specified the model type {model_type.__name__} which does not have "
                        "a `from_pretrained` method. If you have such a model class, please leave the "
                        "`model_type` arguments unspecified and pass the checkpoint of a Keras "
                        "serializable model to `model_ckpt`.")

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
        momentum_args=momentum_args
    )

    # Configure the training
    strategy = devices_setup(devices)
    with strategy.scope():
        if is_transformers_model:
            # PyCharm may regard `model_type` as `None` here
            # noinspection PyUnresolvedReferences
            model = model_type.from_pretrained(model_ckpt, num_labels=num_labels)
        else:
            model = load_model(model_ckpt)
        optimizer = AdamW(**adamw_args.to_dict())
        metric = MCCoefficient(num_classes=num_labels) if use_mcc else Accuracy()
        model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[metric])
    history = model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        validation_data=fit_validation_data,
        verbose=verbose
    )
    return history
