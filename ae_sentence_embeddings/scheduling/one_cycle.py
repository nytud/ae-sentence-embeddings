"""A module for implementing the 1cycle schedule"""

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from ae_sentence_embeddings.argument_handling import OneCycleArgs


class OneCycleSchedule(LearningRateSchedule):
    """The 1cycle schedule for learning rate
    This code relies on the `CyclicalLearningRate` implementation:
    https://github.com/tensorflow/addons/blob/v0.15.0/tensorflow_addons/optimizers/cyclical_learning_rate.py
    """

    def __init__(self, schedule_args: OneCycleArgs, name: str = "OneCycle") -> None:
        """Initialize the scheduler

        Args:
            schedule_args: one-cycle schedule arguments as a dataclass
            name: Name of the operation. Defaults to `"OneCycle"`
        """
        self.initial_rate = schedule_args.initial_rate
        self.cycle_extremum = schedule_args.cycle_extremum
        self.end_extremum = schedule_args.end_extremum \
            if schedule_args.end_extremum is not None else schedule_args.initial_rate
        self.cycle_step_size = schedule_args.half_cycle
        self.tail_step_size = schedule_args.total_steps - 2 * schedule_args.half_cycle
        self.name = name

    def __call__(self, step) -> tf.Tensor:
        """Call the schedule at each iteration"""
        with tf.name_scope(self.name):
            initial_rate = tf.convert_to_tensor(self.initial_rate, name="initial_rate")
            dtype = initial_rate.dtype
            cycle_step_size = tf.cast(self.cycle_step_size, dtype)
            full_cycle_size = 2 * cycle_step_size
            step_tensor = tf.cast(step, dtype)
            if step_tensor <= full_cycle_size:
                extremum_rate = tf.cast(self.cycle_extremum, dtype)
                step_size = tf.cast(self.cycle_step_size, dtype)
            else:
                extremum_rate = tf.cast(self.end_extremum, dtype)
                step_size = tf.cast(self.tail_step_size, dtype)
                step_tensor = step_tensor - full_cycle_size

            state = tf.abs(step_tensor / step_size - 1)
            return initial_rate + (extremum_rate - initial_rate) * (1 - state)

    def get_config(self):
        return {
            "initial_rate": self.initial_rate,
            "cycle_extremum": self.cycle_extremum,
            "end_extremum": self.end_extremum,
            "cycle_step_size": self.step_size,
            "total_steps": self.tail_step_size + 2 * self.cycle_step_size,
        }

    @classmethod
    def from_config(cls, config):
        args = OneCycleArgs.collect_from_dict(config)
        return cls(args)
