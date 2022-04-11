#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tune hyperparameters or run fine-tuning"""

from ae_sentence_embeddings.fine_tuning import fine_tune
from ae_sentence_embeddings.ae_training.pretrain import (
    collect_wandb_args,
    group_train_args_from_flat
)
from ae_sentence_embeddings.argument_handling import check_if_positive_int, check_if_dir


def main() -> None:
    """Main function"""
    config = collect_wandb_args()
    train_args = group_train_args_from_flat(config)
    model_ckpt = check_if_dir(config["model_ckpt"])
    num_labels = check_if_positive_int(config["num_labels"])
    fine_tune(
        model_ckpt=model_ckpt,
        num_labels=num_labels,
        dataset_split_paths=train_args.data_split_path_args,
        data_stream_args=train_args.data_stream_args,
        adamw_args=train_args.adamw_args,
        save_and_log_args=train_args.save_and_log_args,
        lr_args=train_args.lr_one_cycle_args,
        momentum_args=train_args.momentum_one_cycle_args,
        validation_freq=config.get("validation_freq", "epoch"),
        dataset_cache_dir=config.get("dataset_cache_dir"),
        devices=config.get("devices"),
        use_mcc=bool(config.get("use_mcc")),
        num_epochs=config.get("num_epochs", 2)
    )


if __name__ == "__main__":
    main()
