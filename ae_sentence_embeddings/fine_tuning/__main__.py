#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tune hyperparameters or run fine-tuning"""

from ae_sentence_embeddings.fine_tuning.fine_tune_encoder import fine_tune, lookup_transformer_type
from ae_sentence_embeddings.ae_training.pretrain import (
    collect_wandb_args,
    group_train_args_from_flat
)
from ae_sentence_embeddings.argument_handling import check_if_positive_int


def main() -> None:
    """Main function"""
    config = collect_wandb_args()
    train_args = group_train_args_from_flat(config)
    model_ckpt = config["model_ckpt"]
    num_labels = check_if_positive_int(config["num_labels"])
    model_type = config.get("model_type")
    if model_type is not None:
        model_type = lookup_transformer_type(model_type)
    fine_tune(
        model_ckpt=model_ckpt,
        num_labels=num_labels,
        model_type=model_type,
        dataset_split_paths=train_args.data_split_path_args,
        data_stream_args=train_args.data_stream_args,
        adamw_args=train_args.adamw_args,
        save_and_log_args=train_args.save_and_log_args,
        lr_args=train_args.lr_one_cycle_args,
        momentum_args=train_args.momentum_one_cycle_args,
        freeze_encoder=bool(config.get("freeze_encoder")),
        validation_freq=config.get("validation_freq", "epoch"),
        dataset_cache_dir=config.get("dataset_cache_dir"),
        devices=config.get("devices"),
        use_mcc=bool(config.get("use_mcc")),
        num_epochs=config.get("num_epochs", 2),
        drop_remainder=bool(config.get("drop_remainder"))
    )


if __name__ == "__main__":
    main()
