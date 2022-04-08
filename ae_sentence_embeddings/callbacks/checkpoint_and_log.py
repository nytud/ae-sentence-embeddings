"""A module for checkpoint and logging callbacks"""

from os.path import join as os_path_join
from time import strftime
from typing import List, Union, Literal, Dict, Any
from logging import Logger

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback
import wandb

from ae_sentence_embeddings.argument_handling import SaveAndLogArgs


class AeCustomCheckpoint(Callback):
    """A custom class for AE checkpoints"""

    def __init__(self, checkpoint_root: str,
                 save_freq: Union[Literal["epoch"], int] = "epoch",
                 save_optimizer: bool = True) -> None:
        """Initialize the callback.

        Args:
            checkpoint_root: Path to a root directory where all checkpoints will be saved
            save_freq: Saving frequency that specifies how often the model should be saved.
                If `epoch`, the model will be saved after each epoch. Otherwise, it will be
                saved after every `save_freq` iteration
            save_optimizer: If `True`, not only the model weights but also the optimizer states
                will be saved. Defaults to `True`
        """
        super().__init__()
        self.checkpoint_root = checkpoint_root
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        self.batch_id = 0

    def _make_checkpoint(self, subdir_name: Union[int, str]) -> None:
        """Helper function to create checkpoints.

        Args:
            subdir_name: A subdirectory for the checkpoint files
        """
        subdir_path = os_path_join(self.checkpoint_root, subdir_name)
        weight_dir = os_path_join(subdir_path, f"weight_{subdir_name}.ckpt")
        optimizer_dir = os_path_join(subdir_path, f"optim_{subdir_name}.pkl") \
            if self.save_optimizer else None
        self.model.checkpoint(weight_path=weight_dir, optimizer_path=optimizer_dir)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """Save model if `save_freq == "epoch"`"""
        if self.save_freq == "epoch":
            self._make_checkpoint(f"epoch_{epoch+1}")

    def on_batch_end(self, batch: int, logs=None) -> None:
        """Create checkpoint or pass according to `save_freq`.
        Update the global batch ID (number of batch independently of the epoch)
        """
        self.batch_id += 1
        if isinstance(self.save_freq, int) and self.batch_id % self.save_freq == 0:
            self._make_checkpoint(f"step_{self.batch_id}")

    def on_train_end(self, logs=None) -> None:
        """Save model on training end if not saved yet"""
        if isinstance(self.save_freq, int) and self.batch_id % self.save_freq != 0:
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
        self.iteration = 0

    def _simple_log(self, dev_logs: Dict[str, Any]) -> None:
        """Log results to a simple logger"""
        self._logger.debug(f"Evaluation results at iteration {self.iteration}:\n{dev_logs}")

    @staticmethod
    def _wandb_log(dev_logs: Dict[str, Any]) -> None:
        """Log results to WandB. This will not be committed automatically!"""
        wandb.log({"dev_" + key: value for key, value in dev_logs.items()}, commit=False)

    def on_train_batch_end(self, batch: int, logs=None) -> None:
        if (self.iteration + 1) % self.log_freq == 0:
            results = self.model.evaluate(self.dev_data, return_dict=True)
            self._logging_method(results)
        self.iteration += 1

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if self.iteration % self.log_freq != 0:
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
