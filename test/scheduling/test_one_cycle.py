"""Test the 1cycle schedule"""

import tensorflow as tf

from ae_sentence_embeddings.scheduling import OneCycleSchedule
from ae_sentence_embeddings.argument_handling import OneCycleArgs


class OneCycleTest(tf.test.TestCase):
    """Test case for 1cycle schedule"""

    def test_one_cycle_lr(self) -> None:
        """Call a schedule object several times for learning rate scheduling"""
        schedule_args = OneCycleArgs(
            initial_rate=1.0,
            total_steps=14,
            half_cycle=5,
            cycle_extremum=2.0,
            end_extremum=0.0
        )
        lr_schedule = OneCycleSchedule(**schedule_args.to_dict())
        step_vals = []
        for step in range(15):
            step_vals.append(lr_schedule(step))
        expected_vals = tf.constant([1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                                     1.8, 1.6, 1.4, 1.2, 1.0,
                                     0.75, 0.5, 0.25, 0.0])
        self.assertArrayNear(expected_vals, step_vals, err=1e-7,
                             msg=f"Returned values are:\n{step_vals}")

    def test_one_cycle_momentum(self) -> None:
        """Call a schedule object several times for momentum scheduling"""
        schedule_args = OneCycleArgs(
            initial_rate=1.0,
            total_steps=14,
            half_cycle=5,
            cycle_extremum=0.0
        )
        momentum_schedule = OneCycleSchedule(**schedule_args.to_dict())
        step_vals = []
        for step in range(15):
            step_vals.append(momentum_schedule(step))
        expected_vals = tf.constant([1.0, 0.8, 0.6, 0.4, 0.2, 0.0,
                                     0.2, 0.4, 0.6, 0.8, 1.0,
                                     1.0, 1.0, 1.0, 1.0])
        self.assertArrayNear(expected_vals, step_vals, err=1e-7,
                             msg=f"Returned values are:\n{step_vals}")


if __name__ == '__main__':
    tf.test.main()
