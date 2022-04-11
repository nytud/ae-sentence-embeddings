#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tune hyperparameters or run fine-tuning"""

from ae_sentence_embeddings.fine_tuning import fine_tune
from ae_sentence_embeddings.ae_training.pretrain import (
    collect_wandb_args,
    group_train_args_from_flat
)


def main() -> None:
    """Main function"""
    config = collect_wandb_args()
    train_args = group_train_args_from_flat(config)
    fine_tune(
        model_ckpt=config["model_ckpt"],
        num_labels=config["num_labels"],
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
