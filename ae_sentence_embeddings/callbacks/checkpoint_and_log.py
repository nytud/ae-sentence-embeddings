"""A module for checkpoint and logging callbacks"""

from typing import List
from os.path import join as os_path_join
from time import strftime

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback

from ae_sentence_embeddings.argument_handling import SaveAndLogArgs


def basic_checkpoint_and_log(save_log_args: SaveAndLogArgs) -> List[Callback]:
    """Get a checkpoint callback and a logging (Tensorboard) callback

    Args:
        save_log_args: A dataclass that contains the callback arguments

    Returns:
        A list of two callbacks (`ModelCheckpoint` and `TensorBoard`)

    """
    log_dir = os_path_join(save_log_args.log_path, strftime("run_%Y_%m_%d-%H_%M_%S"))
    callbacks = [TensorBoard(log_dir=log_dir, update_freq=save_log_args.log_update_freq),
                 ModelCheckpoint(filepath=save_log_args.checkpoint_path, save_freq=save_log_args.save_freq)]
    return callbacks
