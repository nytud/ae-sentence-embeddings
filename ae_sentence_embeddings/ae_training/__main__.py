"""Tune hyperparameters or run full pre-training"""

import wandb

from ae_sentence_embeddings.modeling_tools import get_training_args, read_json
from ae_sentence_embeddings.ae_training.pretrain import (
    pretrain_transformer_ae,
    group_train_args_from_flat,
    group_model_args_from_flat,
    flatten_nested_dict
)


def main() -> None:
    """Main function"""
    parser = get_training_args()
    parser.add_argument("--project", help="Optional. Name of the current WandB project.")
    args = parser.parse_args()
    arg_dict = flatten_nested_dict(read_json(args.config_file))
    # wandb.init(project=args.project, config=arg_dict)
    # config = wandb.config
    config = arg_dict

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
        top_rnn_args=model_configs.top_rnn_args,
        num_transformer2gru=model_configs.num_transformer2gru,
        validation_freq=config.get("validation_freq", "epoch"),
        num_epochs=config.get("num_epochs", 2),
        dataset_cache_dir=config.get("dataset_cache_dir"),
        devices=config.get("devices")
    )


if __name__ == "__main__":
    main()
