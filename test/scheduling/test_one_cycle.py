"""Test the 1cycle schedule"""

import tensorflow as tf

from ae_sentence_embeddings.scheduling import OneCycleSchedule
from ae_sentence_embeddings.argument_handling import OneCycleArgs


class OneCycleTest(tf.test.TestCase):
    """Test case for 1cycle schedule"""

    def test_one_cycle_lr(self) -> None:
        """Call a schedule object several times for learning rate scheduling"""
        schedule_args = OneCycleArgs(
            initial_rate=0.1,
            total_steps=14,
            half_cycle=5,
            cycle_extremum=0.2,
            end_extremum=1e-10
        )
        lr_schedule = OneCycleSchedule(**schedule_args.to_dict())
        step_vals = []
        for step in range(15):
            step_vals.append(lr_schedule(step))
        expected_vals = tf.constant([0.1, 0.12, 0.14, 0.16, 0.18, 0.2,
                                     0.18, 0.16, 0.14, 0.12, 0.1,
                                     0.075, 0.05, 0.025, 1e-10])
        self.assertArrayNear(expected_vals, step_vals, err=1e-7,
                             msg=f"Returned values are:\n{step_vals}")

    def test_one_cycle_momentum(self) -> None:
        """Call a schedule object several times for momentum scheduling"""
        schedule_args = OneCycleArgs(
            initial_rate=1.0,
            total_steps=14,
            half_cycle=5,
            cycle_extremum=1e-10
        )
        momentum_schedule = OneCycleSchedule(**schedule_args.to_dict())
        step_vals = []
        for step in range(15):
            step_vals.append(momentum_schedule(step))
        expected_vals = tf.constant([1.0, 0.8, 0.6, 0.4, 0.2, 1e-10,
                                     0.2, 0.4, 0.6, 0.8, 1.0,
                                     1.0, 1.0, 1.0, 1.0])
        self.assertArrayNear(expected_vals, step_vals, err=1e-7,
                             msg=f"Returned values are:\n{step_vals}")


if __name__ == "__main__":
    tf.test.main()
