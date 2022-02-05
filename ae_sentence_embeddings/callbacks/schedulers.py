"""A module for implementing learning rate schedulers"""

from typing import Optional, Literal, Dict, Any, Union
from logging import Logger

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import wandb

from ae_sentence_embeddings.argument_handling import LearningRateArgs, OneCycleArgs
from ae_sentence_embeddings.scheduling import OneCycleSchedule


class OneCycleScheduler(Callback):
    """A 1cycle scheduler for learning rate or momentum
    """

    def __init__(
            self,
            schedule_args: Union[LearningRateArgs, OneCycleArgs],
            parameter: Literal["learning_rate", "lr", "beta_1", "momentum"] = "lr",
            log_tool: Optional[Union[Literal["wandb"], Logger]] = None,
            log_freq: int = 10,
            name: Optional[str] = None
    ) -> None:
        """
        Initialize the one-cycle scheduler

        Args:
            schedule_args: A `LearningRateArgs` or `OneCycleArgs` object that contains the schedule arguments
            parameter: The optimizer parameter to be tuned, learning rate or momentum. Defaults to `lr`,
                which stands for learning rate
            log_tool: Optional. If a logger, the rate will be passed to it as a debug message. The log can
                also be sent directly to `WandB` by specifying `wandb`
            log_freq: If `log_method` is specified, logs will be written after every `log_freq` steps.
                Defaults to 10
            name: Scheduler name. If not specified, it will be set to `parameter` + `"_scheduler"`
        """
        super().__init__()
        if isinstance(schedule_args, LearningRateArgs):
            self.schedule = OneCycleSchedule(
                initial_rate=schedule_args.learning_rate,
                total_steps=schedule_args.scheduled_iterations,
                half_cycle=schedule_args.half_cycle,
                cycle_extremum=schedule_args.max_rate,
                end_extremum=schedule_args.last_rate,
                name=parameter + "_schedule"
            )
        elif isinstance(schedule_args, OneCycleArgs):
            self.schedule = OneCycleSchedule(
                **schedule_args.to_dict(), name=parameter + "_schedule")
        else:
            raise ValueError("Invalid type of `schedule_args`")

        if parameter in {"lr", "learning_rate"}:
            self.set_parameter = self._set_lr
        elif parameter in {"beta_1", "momentum"}:
            self.set_parameter = self._set_momentum
        else:
            raise ValueError(f"Parameter {parameter} cannot be cycled")

        self.log_tool = log_tool
        self.log_freq = log_freq

        if isinstance(self.log_tool, Logger):
            self.log_method = self._simple_log
        elif self.log_tool in {"wandb", "WandB"}:
            self.log_method = self._wandb_log
        else:
            self.log_method = None

        self.name = name if name is not None else parameter + "_scheduler"
        self.iteration = 0

    def _simple_log(self, rate: tf.Tensor) -> None:
        """Log `rate` to a simple logger"""
        self.log_tool.debug(f"Rate of {self.name} at iteration {self.iteration}: {rate}")

    def _wandb_log(self, rate: tf.Tensor) -> None:
        """Log `rate` to WandB. This will not be committed automatically!"""
        wandb.log({self.name: rate}, commit=False)

    def _set_lr(self, rate: tf.Tensor) -> None:
        """Set model learning rate"""
        K.set_value(self.model.optimizer.learning_rate, rate)

    def _set_momentum(self, rate: tf.Tensor) -> None:
        """Set momentum coefficient"""
        K.set_value(self.model.optimizer.beta_1, rate)

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        rate = self.schedule(self.iteration)
        self.set_parameter(rate)
        if self.log_method is not None and self.iteration % self.log_freq == 0:
            self.log_method(rate)
        self.iteration += 1
