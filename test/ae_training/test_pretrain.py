"""Test pretraining tools. These tests concern only argument processing.
Training tests are included in integration tests
"""

import unittest

from ae_sentence_embeddings.ae_training import group_train_args_from_flat, flatten_nested_dict
from ae_sentence_embeddings.argument_handling import (
    DataSplitPathArgs,
    DataStreamArgs,
    AdamwArgs,
    OneCycleArgs,
    SaveAndLogArgs
)


class ArgProcessingTest(unittest.TestCase):
    """A test case for processing arguments"""

    def setUp(self) -> None:
        """Fixture setup. This creates a nested and a flat dictionary"""
        super().setUp()
        self.nested_dict = {
            "data_split_path_args": {
                "train_path": "train.jsonl",
                "dev_path": "dev.jsonl"
            },
            "data_stream_args": {
                "input_padding": [0, 0],
            },
            "adamw_args": {
                "weight_decay": 5e-5,
                "amsgrad": 1
            },
            "one_cycle_args": {
                "lr_initial_rate": 1e-5,
                "lr_total_steps": 10000,
                "lr_half_cycle": 500,
                "lr_cycle_extremum": 1e-4,
                "momentum_initial_rate": 0.95,
                "momentum_total_steps": 10000,
                "momentum_half_cycle": 500,
                "momentum_cycle_extremum": 0.85
            },
            "save_and_log_args": {
                "checkpoint_path": "checkpoints",
            },
            "num_epochs": 3,
            "devices": ["GPU:0", "GPU:1"]
        }
        self.flat_dict = {
            "data_split_path_args_train_path": "train.jsonl",
            "data_split_path_args_dev_path": "dev.jsonl",
            "data_stream_args_input_padding": [0, 0],
            "adamw_args_weight_decay": 5e-5,
            "adamw_args_amsgrad": 1,
            "one_cycle_args_lr_initial_rate": 1e-5,
            "one_cycle_args_lr_total_steps": 10000,
            "one_cycle_args_lr_half_cycle": 500,
            "one_cycle_args_lr_cycle_extremum": 1e-4,
            "one_cycle_args_momentum_initial_rate": 0.95,
            "one_cycle_args_momentum_total_steps": 10000,
            "one_cycle_args_momentum_half_cycle": 500,
            "one_cycle_args_momentum_cycle_extremum": 0.85,
            "save_and_log_args_checkpoint_path": "checkpoints",
            "num_epochs": 3,
            "devices": ["GPU:0", "GPU:1"]
        }

    def test_flatten_nested_dict(self) -> None:
        flattened_dict = flatten_nested_dict(self.nested_dict)
        self.assertDictEqual(self.flat_dict, flattened_dict)

    def test_group_train_args_from_flat(self) -> None:
        expected_types = (DataSplitPathArgs, DataStreamArgs, AdamwArgs,
                          OneCycleArgs, OneCycleArgs, SaveAndLogArgs)
        train_arg_groups = group_train_args_from_flat(self.flat_dict)
        for train_arg_group, expected_type in zip(train_arg_groups, expected_types):
            self.assertIsInstance(train_arg_group, expected_type)


if __name__ == "__main__":
    unittest.main()
