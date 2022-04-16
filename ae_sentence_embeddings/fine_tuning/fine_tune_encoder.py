# -*- coding: utf-8 -*-

"""Fine-tune an encoder model for classification."""

from typing import Optional, Union, Sequence, Literal

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow_addons.optimizers import AdamW
from ae_sentence_embeddings.losses_and_metrics import SparseCategoricalMCC
from transformers import (
    TFAutoModel,
    TFAutoModelForTokenClassification,
    TFAutoModelForSequenceClassification
)

from ae_sentence_embeddings.data import get_train_and_validation
from ae_sentence_embeddings.ae_training import devices_setup, log_and_schedule_setup
from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    AdamwArgs,
    SaveAndLogArgs,
    DataSplitPathArgs,
    OneCycleArgs
)

TransformerTypes = Union[TFAutoModel, TFAutoModelForTokenClassification, TFAutoModelForTokenClassification]
transformers_type_dict = {
    "Model": TFAutoModel,
    "SequenceClassification": TFAutoModelForSequenceClassification,
    "TokenClassification": TFAutoModelForTokenClassification
}


def lookup_transformer_type(
        transformer_type: Literal["Model", "SequenceClassification", "TokenClassification"]
) -> TransformerTypes:
    """Get the `transformers` architecture indicated by a string."""
    try:
        transformer_auto_type = transformers_type_dict[transformer_type]
    except KeyError:
        raise NotImplementedError(f"Unrecognized model type: {transformer_type}. "
                                  f"Please specify one of {list(transformers_type_dict.keys())}")
    return transformer_auto_type


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
        model_type: Optional. The name of a `transformers` model type, i.e. `'TFBertModel'`.
            If it is not specified, a Keras-serialized model checkpoint is expected as the
            `model_ckpt` argument.
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
                        "`model_type` argument unspecified and pass the checkpoint of a Keras "
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
        loss_fn = SparseCategoricalCrossentropy(from_logits=True) if num_labels > 1 \
            else BinaryCrossentropy(from_logits=True)
        metrics = [SparseCategoricalAccuracy()]
        if use_mcc:
            metrics.append(SparseCategoricalMCC(num_classes=num_labels))
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    history = model.fit(
        x=train_dataset,
        epochs=num_epochs,
        callbacks=callbacks,
        validation_data=fit_validation_data,
        verbose=verbose
    )
    return history
