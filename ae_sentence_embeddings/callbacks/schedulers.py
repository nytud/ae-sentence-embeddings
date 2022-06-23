# -*- coding: utf-8 -*-

"""A module for implementing learning rate schedulers"""

from typing import Optional, Literal, Dict, Any, Union
from logging import Logger
from warnings import warn

import wandb
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as keras_backend
from tensorflow_addons.optimizers import CyclicalLearningRate

from ae_sentence_embeddings.argument_handling import LearningRateArgs, OneCycleArgs
from ae_sentence_embeddings.scheduling import OneCycleSchedule, LinearAnneal
from ae_sentence_embeddings.regularizers.kl_annealer import calculate_beta


class OneCycleScheduler(Callback):
    """A 1cycle scheduler for learning rate or momentum"""

    def __init__(
            self,
            schedule_args: Union[LearningRateArgs, OneCycleArgs],
            parameter: Literal["learning_rate", "lr", "beta_1", "momentum"] = "lr",
            log_tool: Optional[Union[Literal["wandb"], Logger]] = None,
            log_freq: int = 10,
            name: Optional[str] = None
    ) -> None:
        """Initialize the one-cycle scheduler

        Args:
            schedule_args: A `LearningRateArgs` or `OneCycleArgs` (recommended) object that contains the
                schedule arguments
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

        self.name = name if name is not None else parameter + "_scheduler"
        self._log_tool = log_tool
        self.log_freq = log_freq
        if isinstance(log_tool, Logger):
            self.log_method = self._simple_log
        elif isinstance(log_tool, str) and log_tool.lower() == "wandb":
            self.log_method = self._wandb_log
        else:
            self.log_method = None
            if log_tool is not None:
                warn(f"Unrecognized logging tool: {log_tool}. Logging will not be done for {self.name}")

        self.iteration = 0

    def _simple_log(self, rate: tf.Tensor) -> None:
        """Log `rate` to a simple logger"""
        self._log_tool.debug(f"Rate of {self.name} at iteration {self.iteration}: {rate}")

    def _wandb_log(self, rate: tf.Tensor) -> None:
        """Log `rate` to WandB. This will not be committed automatically!"""
        wandb.log({self.name: rate}, commit=False)

    def _set_lr(self, rate: tf.Tensor) -> None:
        """Set model learning rate"""
        keras_backend.set_value(self.model.optimizer.learning_rate, rate)

    def _set_momentum(self, rate: tf.Tensor) -> None:
        """Set momentum coefficient"""
        keras_backend.set_value(self.model.optimizer.beta_1, rate)

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        rate = self.schedule(self.iteration)
        self.set_parameter(rate)
        if self.log_method is not None and self.iteration % self.log_freq == 0:
            self.log_method(rate)
        self.iteration += 1

    @property
    def log_tool(self) -> Union[Logger, Literal["wandb", "WandB"]]:
        return self._log_tool


class CyclicScheduler(Callback):
    """A cyclic learning rate scheduler as a callback."""

    def __init__(
            self,
            initial_rate: float,
            cycle_extremum: float,
            half_cycle: float,
            log_freq: Optional[int] = None,
            name: str = "cyclic_lr_scheduler"
    ) -> None:
        """Initialize the callback.

        Args:
            initial_rate: A starting learning rate.
            cycle_extremum: The largest learning rate.
            half_cycle: The number of steps to reach the highest rate in each cycle.
            log_freq: Optional. How often to log to WandB. If not specified, no logging
                will be done.
            name: The callback name. Defaults to `cyclic_lr_scheduler`.
        """
        super().__init__()
        self._log_freq = log_freq
        self.name = name
        self._schedule = CyclicalLearningRate(
            initial_learning_rate=initial_rate,
            maximal_learning_rate=cycle_extremum,
            step_size=half_cycle,
            scale_fn=lambda x: 1 / (2. ** (x - 1))
        )
        self._iteration = 0

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        rate = self._schedule(self._iteration)
        keras_backend.set_value(self.model.optimizer.learning_rate, rate)
        if self._log_freq is not None and self._iteration % self._log_freq == 0:
            wandb.log({self.name: rate}, commit=False)
        self._iteration += 1


class KLAnnealer(Callback):
    """A callback to anneal the KL factor during training."""

    def __init__(
            self,
            initial_rate: float,
            target_rate: float,
            total_steps: int,
            log_freq: Optional[int] = None,
            name: str = "kl_annealer"
    ) -> None:
        """Initialize the callback.

        Args:
            initial_rate: The starting KL factor.
            target_rate: The final KL factor.
            total_steps: The number of steps in which the final KL factor
                is to be reached.
            log_freq: Optional. How often to log to WandB. If not specified, no logging
                will be done.
            name: The callback name. Defaults to `'kl_annealer'`.
        """
        super().__init__()
        self._log_freq = log_freq
        self.name = name
        self._schedule = LinearAnneal(
            initial_rate=initial_rate,
            target_rate=target_rate,
            total_steps=total_steps
        )
        self._iteration = 0

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        rate = self._schedule(self._iteration)
        self.model.set_kl_factor(rate)
        if self._log_freq is not None and self._iteration % self._log_freq == 0:
            wandb.log({self.name: self.model.kl_factor}, commit=False)
        self._iteration += 1


class VaeLogger(Callback):
    """A callback that allows to monitor the learning rate
    and the KL loss beta during training.
    """

    def __init__(
            self,
            log_update_freq: int,
            beta_warmup_iters: int,
            beta_warmup_start: int
    ) -> None:
        """Initialize the callback.

        Args:
            log_update_freq: Logging frequency in terms of batches.
            beta_warmup_iters: The number of iterations while the KL loss `beta` will be annealed.
            beta_warmup_start: The iteration after which the KL loss `beta` will start to be annealed.
        """
        super(VaeLogger, self).__init__()
        self.log_update_freq = log_update_freq
        self._beta_warmup_iters = beta_warmup_iters
        self._beta_warmup_start = beta_warmup_start

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Calculate the actual learning rate and KL beta values."""
        iteration = self.model.optimizer.iterations
        if iteration % self.log_update_freq == 0:
            lr = keras_backend.get_value(self.model.optimizer.lr(iteration))
            beta = calculate_beta(iteration, warmup_iters=self._beta_warmup_iters,
                                  start=self._beta_warmup_start)
            wandb.log({"learning_rate": lr, "kl_beta": beta}, commit=False)
