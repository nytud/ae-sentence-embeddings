# -*- coding: utf-8 -*-

"""A custom training loop for pre-training."""

from typing import Sequence, List, Tuple, Dict, Any

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, CallbackList
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

from ae_sentence_embeddings.scheduling import LinearAnneal


@tf.function(experimental_relax_shapes=True)
def do_train_step(
        data: Tuple[Sequence[tf.Tensor], tf.Tensor],
        model: tf.keras.Model,
        loss_fn: Loss,
        metric: Metric,
        kl_loss_rate: tf.Tensor,
) -> Dict[str, Any]:
    x, y = data
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        kl_loss = tf.multiply(kl_loss_rate, model.losses)
        prediction_loss = loss_fn(y, y_pred)
        loss = tf.add(prediction_loss, kl_loss)
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    model.optimizer.apply_gradients(zip(gradients, trainable_vars))
    metric.update_state(y, y_pred)
    return {
        metric.name: metric.result(),
        loss_fn.name: loss,
        "kl_loss": kl_loss,
        "kl_factor": kl_loss_rate
    }


@tf.function(experimental_relax_shapes=True)
def do_test_step(
        data: Tuple[Sequence[tf.Tensor], tf.Tensor],
        model: tf.keras.Model,
        kl_loss_rate: tf.Tensor,
        loss_fn: Loss,
        metric: Metric
) -> Dict[str, Any]:
    x, y = data
    y_pred = model(x, training=False)
    kl_loss = tf.multiply(kl_loss_rate, model.losses)
    prediction_loss = loss_fn(y, y_pred)
    loss = tf.add(prediction_loss, kl_loss)
    metric.update_state(y, y_pred)
    return {
        metric.name: metric.result(),
        loss_fn.name: loss,
        "kl_loss": kl_loss,
        "kl_factor": kl_loss_rate
    }


def do_validation(
        dev_dataset: tf.data.Dataset,
        model: tf.keras.Model,
        kl_loss_rate: tf.Tensor,
        callback_list: CallbackList,
        loss_fn: Loss,
        metric: Metric
) -> Dict[str, Any]:
    dev_logs, j = {}, 1
    running_loss = 0.
    running_kl_loss = 0.
    dev_prefix = "dev_"
    metric.reset_states()
    for j, dev_batch in enumerate(dev_dataset):
        callback_list.on_test_batch_begin(j, logs=dev_logs)
        dev_logs = do_test_step(dev_batch, model, kl_loss_rate, loss_fn=loss_fn)
        running_loss += dev_logs[loss_fn.name]
        running_kl_loss += dev_logs[loss_fn]
        callback_list.on_test_batch_end(j, logs=dev_logs)
    dev_logs = {dev_prefix + key: value for key, value in dev_logs.items()}
    dev_logs[dev_prefix + loss_fn.name] = running_loss / j
    dev_logs[dev_prefix + "kl_loss"] = running_kl_loss / j
    metric.reset_states()
    return dev_logs


def training_loop(
        train_dataset: tf.data.Dataset,
        dev_dataset: tf.data.Dataset,
        loss_fn: Loss,
        metric: Metric,
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
            callback_list.on_train_batch_begin(i, logs=logs)
            kl_loss_rate = kl_anneal(step)
            logs = do_train_step(
                data=train_batch,
                model=model,
                kl_loss_rate=kl_loss_rate,
                loss_fn=loss_fn,
                metric=metric
            )
            step += 1
            if validation_freq != "epoch" and step % validation_freq == 0:
                dev_logs = do_validation(
                    dev_dataset=dev_dataset,
                    model=model,
                    kl_loss_rate=kl_loss_rate,
                    callback_list=callback_list,
                    loss_fn=loss_fn,
                    metric=metric
                )
                logs.update(dev_logs)
            callback_list.on_train_batch_end(i, logs=logs)

        if validation_freq == "epoch":
            dev_logs = do_validation(
                dev_dataset=dev_dataset,
                model=model,
                kl_loss_rate=kl_loss_rate,
                callback_list=callback_list,
                loss_fn=loss_fn,
                metric=metric
            )
            logs.update(dev_logs)
        callback_list.on_epoch_end(epoch, logs=logs)
    callback_list.on_train_end(logs=logs)
