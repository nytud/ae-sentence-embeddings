"""A module for checkpoint and logging callbacks"""

from os.path import join as os_path_join
from time import strftime
from typing import List, Union, Literal

from tensorflow.keras.callbacks import TensorBoard, Callback

from ae_sentence_embeddings.argument_handling import SaveAndLogArgs


class AeCustomCheckpoint(Callback):
    """A custom class for AE checkpoints"""

    def __init__(self, checkpoint_root: str,
                 save_freq: Union[Literal["epoch"], str] = "epoch",
                 save_optimizer: bool = True) -> None:
        """Initialize the callback

        Args:
            checkpoint_root: Path to a root directory where all checkpoints will be saved
            save_freq: Saving frequency that specifies how often the model should be saved.
                       If `epoch`, the model will be saved after each epoch. Otherwise, it
                       will be saved after every `save_freq` iteration
            save_optimizer: If `True`, not only the model weights but also the optimizer states
                            will be saved. Defaults to `True`
        """
        super().__init__()
        self.checkpoint_root = checkpoint_root
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        self.batch_id = 0

    def _make_checkpoint(self, path_suffix: Union[int, str]) -> None:
        """Helper function to create checkpoints

        Args:
            path_suffix: A suffix to be added to the model/optimizer checkpoint path
        """
        weight_dir = os_path_join(self.checkpoint_root, f"weight_{path_suffix}.ckpt")
        optimizer_dir = os_path_join(self.checkpoint_root, f"optim_{path_suffix}.pkl") \
            if self.save_optimizer else None
        self.model.checkpoint(weight_path=weight_dir, optimizer_path=optimizer_dir)

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        """Save model if `save_freq == "epoch"`"""
        if self.save_freq == "epoch":
            self._make_checkpoint(epoch+1)

    def on_batch_end(self, batch: int, logs=None) -> None:
        """Create checkpoint or pass according to `save_freq`.
        Update the global batch ID (number of batch independently from the epoch)
        """
        self.batch_id += 1
        if isinstance(self.save_freq, int) and self.batch_id % self.save_freq == 0:
            self._make_checkpoint(self.batch_id)

    def on_train_end(self, logs=None) -> None:
        """Save model on training end if not saved yet"""
        if isinstance(self.save_freq, int) and self.batch_id % self.save_freq != 0:
            self._make_checkpoint("train_end")


def basic_checkpoint_and_log(save_log_args: SaveAndLogArgs) -> List[Callback]:
    """Get a checkpoint callback and a logging (Tensorboard) callback

    Args:
        save_log_args: A dataclass that contains the callback arguments

    Returns:
        A list of two callbacks (`ModelCheckpoint` and `TensorBoard`)

    """
    log_dir = os_path_join(save_log_args.log_path, strftime("run_%Y_%m_%d-%H_%M_%S"))
    callbacks = [TensorBoard(log_dir=log_dir, update_freq=save_log_args.log_update_freq),
                 AeCustomCheckpoint(checkpoint_root=save_log_args.checkpoint_path,
                                    save_freq=save_log_args.save_freq,
                                    save_optimizer=save_log_args.save_optimizer)]
    return callbacks
