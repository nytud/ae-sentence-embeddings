# -*- coding: utf-8 -*-

"""A custom training loop for pre-training."""

from typing import Sequence, List, Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, CallbackList

from ae_sentence_embeddings.scheduling import LinearAnneal


@tf.function
def do_train_step(
        data: Tuple[Sequence[tf.Tensor], tf.Tensor],
        model: tf.keras.Model,
        kl_loss_rate: tf.Tensor
) -> Dict[str, Any]:
    with tf.GradientTape() as tape:
        x, y = data
        y_pred = model(x, training=True)
        loss = model.compiled_loss(
            y, y_pred, regularization_losses=tf.multiply(kl_loss_rate, model.losses))
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        model.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in model.metrics}


@tf.function
def do_test_step(
        data: Tuple[Sequence[tf.Tensor], tf.Tensor],
        model: tf.keras.Model,
        kl_loss_rate: tf.Tensor
) -> Dict[str, Any]:
    x, y = data
    y_pred = model(x, training=False)
    model.compiled_loss(
        y, y_pred, regularization_losses=tf.multiply(kl_loss_rate, model.losses))
    model.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in model.metrics}


def do_validation(
        dev_dataset: tf.data.Dataset,
        model: tf.keras.Model,
        kl_loss_rate: tf.Tensor,
        callback_list: CallbackList
) -> Dict[str, Any]:
    dev_logs = {}
    model.compiled_metrics.reset_states()
    for j, dev_batch in enumerate(dev_dataset):
        callback_list.on_batch_begin(j, logs=dev_logs)
        callback_list.on_test_batch_begin(j, logs=dev_logs)
        dev_logs = do_test_step(dev_batch, model, kl_loss_rate)
        callback_list.on_batch_end(j, logs=dev_logs)
        callback_list.on_test_batch_end(j, logs=dev_logs)
    dev_logs = {"dev_" + key: value for key, value in dev_logs.items()}
    model.compiled_metrics.reset_states()
    return dev_logs


def training_loop(
        train_dataset: tf.data.Dataset,
        dev_dataset: tf.data.Dataset,
        model: tf.keras.Model,
        callbacks: List[Callback],
        kl_anneal: LinearAnneal,
        num_epochs: int,
        validation_freq: int
) -> None:
    callback_list = CallbackList(callbacks, add_history=True, model=model)
    logs = {}
    callback_list.on_train_begin(logs=logs)
    step = 0
    kl_loss_rate = tf.constant(1.)

    for epoch in range(num_epochs):
        callback_list.on_epoch_begin(epoch, logs=logs)

        for i, train_batch in enumerate(train_dataset):
            callback_list.on_batch_begin(i, logs=logs)
            callback_list.on_train_batch_begin(i, logs=logs)
            kl_loss_rate = kl_anneal(step)
            logs = do_train_step(train_batch, model, kl_loss_rate)
            step += 1
            if validation_freq != "epoch" and step % validation_freq == 0:
                dev_logs = do_validation(
                    dev_dataset=dev_dataset,
                    model=model,
                    kl_loss_rate=kl_loss_rate,
                    callback_list=callback_list
                )
                logs.update(dev_logs)
            callback_list.on_batch_end(i, logs=logs)
            callback_list.on_train_batch_end(i, logs=logs)

        if validation_freq == "epoch":
            dev_logs = do_validation(
                dev_dataset=dev_dataset,
                model=model,
                kl_loss_rate=kl_loss_rate,
                callback_list=callback_list
            )
            logs.update(dev_logs)
        callback_list.on_epoch_end(epoch, logs=logs)

    callback_list.on_train_end(logs=logs)

