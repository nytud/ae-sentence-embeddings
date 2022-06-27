# -*- coding: utf-8 -*-

"""A module for checkpoint and logging callbacks"""

from os import mkdir
from os.path import join as os_path_join, exists
from time import strftime
from typing import List, Union, Literal, Dict, Any, Optional
from logging import Logger
from warnings import warn

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback
import wandb

from ae_sentence_embeddings.argument_handling import SaveAndLogArgs


class AeCustomCheckpoint(Callback):
    """A custom class for AE checkpoints"""

    def __init__(
            self,
            checkpoint_root: str,
            save_freq: Union[Literal["epoch"], int] = "epoch",
            save_optimizer: bool = True,
            no_serialization: bool = False
    ) -> None:
        """Initialize the callback.

        Args:
            checkpoint_root: Path to a root directory where all checkpoints will be saved.
                If it does not exist, it will be created.
            save_freq: Saving frequency that specifies how often the model should be saved.
                If `epoch`, the model will be saved after each epoch. Otherwise, it will be
                saved after every `save_freq` iteration.
            save_optimizer: If `True`, not only the model weights but also the optimizer states
                will be saved. Defaults to `True`.
            no_serialization: Do not use the `save` method of the model. This assumes that a `checkpoint`
                method was implemented. Defaults to `False`.
        """
        super().__init__()
        if not exists(checkpoint_root):
            mkdir(checkpoint_root)
        self._checkpoint_root = os_path_join(checkpoint_root, strftime("run_%Y_%m_%d-%H_%M_%S"))
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        self.no_serialization = no_serialization

    def _make_checkpoint(self, subdir_name: Union[int, str]) -> None:
        """Helper function to create checkpoints.

        Args:
            subdir_name: A subdirectory for the checkpoint files
        """
        subdir_path = os_path_join(self._checkpoint_root, subdir_name)
        weight_dir = os_path_join(subdir_path, f"weight_{subdir_name}.ckpt")
        if not self.no_serialization:
            self.model.save(weight_dir, include_optimizer=self.save_optimizer)
        elif hasattr(self.model, "checkpoint"):
            optimizer_path = os_path_join(subdir_path, f"optim_{subdir_name}.pkl") if self.save_optimizer else None
            self.model.checkpoint(weight_dir, optimizer_path=optimizer_path)
        else:
            self.model.save_weights(weight_dir)
            warn("The model does not implement a `checkpoint` method. The optimizer state was not saved.")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save model if `save_freq == "epoch"`"""
        if self.save_freq == "epoch":
            self._make_checkpoint(f"epoch_{epoch+1}")

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Create checkpoint or pass according to `save_freq`.
        Update the global batch ID (number of batch independently of the epoch)
        """
        if isinstance(self.save_freq, int) and self.model.optimizer.iterations % self.save_freq == 0:
            self._make_checkpoint(f"step_{self.model.optimizer.iterations}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save model on training end if not saved yet"""
        if isinstance(self.save_freq, int) and self.model.optimizer.iterations % self.save_freq != 0:
            self._make_checkpoint("train_end")


def basic_checkpoint_and_log(save_log_args: SaveAndLogArgs) -> List[Callback]:
    """Get a checkpoint callback and a logging (Tensorboard) callback.

    Args:
        save_log_args: A dataclass that contains the callback arguments

    Returns:
        A list of two callbacks (`ModelCheckpoint` and `TensorBoard`)

    """
    if not isinstance(save_log_args.log_tool, str) or save_log_args.log_tool.lower() == "wandb":
        raise ValueError("It is required to pass a `Tensorboard` path to `save_log_args.log_tool`")
    log_dir = os_path_join(save_log_args.log_tool, strftime("run_%Y_%m_%d-%H_%M_%S"))
    callbacks = [TensorBoard(log_dir=log_dir, update_freq=save_log_args.log_update_freq),
                 AeCustomCheckpoint(checkpoint_root=save_log_args.checkpoint_path,
                                    save_freq=save_log_args.save_freq,
                                    save_optimizer=save_log_args.save_optimizer)]
    return callbacks


class DevEvaluator(Callback):
    """A callback that runs a full evaluation epoch after specified training batches"""

    def __init__(self, dev_data: tf.data.Dataset, logger: Union[Logger, Literal["wandb", "WandB"]],
                 log_freq: int = 1000) -> None:
        """Initialize the callback

        Args:
            dev_data: A TensorFlow dataset on which the model will be evaluated.
            logger: A custom logger instance or `"wandb"`. In the latter case, logs will be sent directly to WandB.
                If it is a logger, debug messages will be passed to it.
            log_freq: Log every `log_freq` steps (on batch end). A log will also be made on epoch ends.
                Defaults to `1000`.
        """
        super().__init__()
        self.dev_data = dev_data
        self.log_freq = log_freq
        self._logger = None
        self._logging_method = None
        self.set_logger(logger)

    def _simple_log(self, dev_logs: Dict[str, Any]) -> None:
        """Log results to a simple logger"""
        self._logger.debug(f"Evaluation results at iteration {self.model.optimizer.iterations}:\n{dev_logs}")

    @staticmethod
    def _wandb_log(dev_logs: Dict[str, Any]) -> None:
        """Log results to WandB. This will not be committed automatically!"""
        wandb.log({"dev_" + key: value for key, value in dev_logs.items()}, commit=False)

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.model.optimizer.iterations % self.log_freq == 0:
            results = self.model.evaluate(self.dev_data, return_dict=True)
            self._logging_method(results)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.model.optimizer.iterations % self.log_freq != 0:
            results = self.model.evaluate(self.dev_data, return_dict=True)
            self._logging_method(results)

    @property
    def logger(self) -> Logger:
        return self._logger

    def set_logger(self, new_logger: Union[Logger, Literal["wandb", "WandB"]]) -> None:
        if isinstance(new_logger, Logger):
            self._logging_method = self._simple_log
        elif isinstance(new_logger, str) and new_logger.lower() == "wandb":
            self._logging_method = self._wandb_log
        else:
            raise ValueError(f"Unknown logger: {new_logger}")
        self._logger = new_logger


class RegCallback(Callback):
    """A callback to track regularization losses."""

    def __init__(self, reg_loss_name: str, log_update_freq: int) -> None:
        """Initialize the callback.

        Args:
            reg_loss_name: The name of the regularization loss.
            log_update_freq: Specifies how often to log the regularization loss.
        """
        super(RegCallback, self).__init__()
        self._reg_loss_name = reg_loss_name
        self._devel_reg_loss_name = "dev_" + reg_loss_name
        self._log_update_freq = log_update_freq
        self._train_iteration = 0
        self._devel_iteration = 0
        self._running_train_reg_loss = 0
        self._running_devel_reg_loss = 0

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._running_train_reg_loss += logs[self._reg_loss_name]
        if self._train_iteration % self._log_update_freq == 0 and self._train_iteration != 0:
            logs[self._reg_loss_name] = self._running_train_reg_loss / tf.cast(self._train_iteration, tf.float32)
        self._train_iteration += 1

    def on_test_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._running_devel_reg_loss += logs[self._devel_reg_loss_name]
        self._devel_iteration += 1

    def on_test_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        logs[self._devel_reg_loss_name] = self._running_devel_reg_loss / tf.cast(self._devel_iteration, tf.float32)
        self._devel_iteration = 0
        self._train_iteration = 0
        self._running_devel_reg_loss = 0
        self._running_train_reg_loss = 0
