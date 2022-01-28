"""A module for implementing learning rate schedulers"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from ae_sentence_embeddings.argument_handling import LearningRateArgs


class OneCycleScheduler(Callback):
    """
    A modified version of the scheduler by A. Géron form
    https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb
    """

    def __init__(self, lr_args: LearningRateArgs) -> None:
        """
        Initialize the one-cycle scheduler

        Args:
            lr_args: A `LearningRateArgs` object that contains the learning rate arguments
        """
        super().__init__()
        self.lr_args = lr_args
        self.iteration = 0

    def _interpolate(self, iter_start: int, iter_end: int, rate_start: float, rate_end: float) -> float:
        """Calculate the new learning rate"""
        return ((rate_end - rate_start) * (self.iteration - iter_start)
                / (iter_end - iter_start) + rate_start)

    def on_batch_begin(self, batch: int, logs: Optional = None) -> None:
        if self.iteration <= self.lr_args.scheduled_iterations:
            if self.iteration < self.lr_args.half_cycle:
                lr_rate = self._interpolate(0, self.lr_args.half_cycle,
                                            self.lr_args.learning_rate, self.lr_args.max_rate)
            elif self.iteration < self.lr_args.cycle_end:
                lr_rate = self._interpolate(self.lr_args.half_cycle, self.lr_args.cycle_end,
                                            self.lr_args.max_rate, self.lr_args.learning_rate)
            else:
                lr_rate = self._interpolate(self.lr_args.cycle_end, self.lr_args.scheduled_iterations,
                                            self.lr_args.learning_rate, self.lr_args.last_rate)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate,
                                       lr_rate)
            self.iteration += 1
