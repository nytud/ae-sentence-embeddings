"""A module for implementing the 1cycle schedule"""

from typing import Optional, Callable

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class OneCycleSchedule(LearningRateSchedule):
    """The 1cycle schedule for learning rate
    This code relies on the `CyclicalLearningRate` implementation:
    https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/optimizers/cyclical_learning_rate.py
    """

    def __init__(
            self,
            initial_rate: float,
            total_steps: int,
            half_cycle: int,
            cycle_extremum: float,
            end_extremum: Optional[float] = None,
            name: str = "OneCycle"
    ) -> None:
        """Initialize the scheduler

        Args:
            initial_rate: An initial rate (learning rate or beta_1 momentum factor)
            total_steps: Total number of iterations necessary to reach the last iteration beginning from the first
                         iteration, i.e. `total_number_of_iterations - 1`
            half_cycle: Number of iterations after which the rate reaches its extremum at mid-cycle beginning from
                        the first iteration
            cycle_extremum: Extremum reached at mid-cycle
            end_extremum: Optional. Extremum reached at the training end
            name: Name of the operation. Defaults to `"OneCycle"`
        """
        super().__init__()
        self.initial_rate = initial_rate
        self.cycle_extremum = cycle_extremum
        self.end_extremum = end_extremum if end_extremum is not None else initial_rate
        self.cycle_step_size = half_cycle
        self.total_steps = total_steps
        self.tail_step_size = total_steps - 2 * half_cycle
        self.name = name

    def __call__(self, step) -> tf.Tensor:
        """Call the schedule at each iteration"""
        with tf.name_scope(self.name) as name:
            initial_rate = tf.convert_to_tensor(self.initial_rate, name="initial_rate")
            dtype = initial_rate.dtype
            cycle_step_size = tf.cast(self.cycle_step_size, dtype)
            full_cycle_size = tf.multiply(tf.constant(2, dtype=dtype), cycle_step_size)
            step_tensor = tf.cast(step, dtype)
            cycle_coef = tf.cast(tf.less_equal(step_tensor, full_cycle_size), dtype)
            end_coef = tf.subtract(tf.constant(1, dtype=dtype), cycle_coef)
            extremum_rate = tf.add(tf.multiply(cycle_coef, tf.cast(self.cycle_extremum, dtype)),
                                   tf.multiply(end_coef, tf.cast(self.end_extremum, dtype)))
            step_size = tf.add(tf.multiply(cycle_coef, tf.cast(self.cycle_step_size, dtype)),
                               tf.multiply(end_coef, tf.cast(self.tail_step_size, dtype)))
            step_tensor = tf.subtract(step_tensor, tf.multiply(end_coef, full_cycle_size))

            state = tf.abs(tf.subtract(tf.divide(step_tensor, step_size), tf.constant(1, dtype=dtype)))
            return tf.add(initial_rate, tf.multiply(tf.subtract(extremum_rate, initial_rate),
                                                    tf.subtract(tf.constant(1, dtype=dtype), state)), name=name)

    def get_config(self):
        return {
            "initial_rate": self.initial_rate,
            "cycle_extremum": self.cycle_extremum,
            "end_extremum": self.end_extremum,
            "cycle_step_size": self.cycle_step_size,
            "total_steps": self.total_steps,
            "name": self.name
        }


def wrap_one_cycle(schedule: OneCycleSchedule) -> Callable[[None], tf.Tensor]:
    """Wrap a 1cycle schedule in a function with no arguments
    This may be necessary for cycling momentum in Adam and AdamW, as their `beta_1` argument
    does not support schedules
    """
    values = (schedule(i) for i in tf.range(schedule.total_steps))

    def one_cycle_fn():
        return next(values)

    return one_cycle_fn
