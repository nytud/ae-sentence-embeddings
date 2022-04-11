#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Run full pre-training"""

from ae_sentence_embeddings.ae_training.pretrain import (
    pretrain_transformer_ae,
    group_train_args_from_flat,
    group_model_args_from_flat,
    collect_wandb_args
)


def main() -> None:
    """Main function"""
    config = collect_wandb_args()
    train_args = group_train_args_from_flat(config)
    model_configs = group_model_args_from_flat(config)
    pretrain_transformer_ae(
        model_type_name=model_configs.model_type_name,
        dataset_split_paths=train_args.data_split_path_args,
        data_stream_args=train_args.data_stream_args,
        adamw_args=train_args.adamw_args,
        save_and_log_args=train_args.save_and_log_args,
        lr_args=train_args.lr_one_cycle_args,
        momentum_args=train_args.momentum_one_cycle_args,
        encoder_config=model_configs.encoder_config,
        decoder_config=model_configs.decoder_config,
        pooling_type=model_configs.pooling_type,
        kl_factor=model_configs.kl_factor,
        swap_p=model_configs.swap_p,
        top_rnn_args=model_configs.top_rnn_args,
        num_transformer2gru=model_configs.num_transformer2gru,
        validation_freq=config.get("validation_freq", "epoch"),
        num_epochs=config.get("num_epochs", 2),
        dataset_cache_dir=config.get("dataset_cache_dir"),
        devices=config.get("devices")
    )


if __name__ == "__main__":
    main()
