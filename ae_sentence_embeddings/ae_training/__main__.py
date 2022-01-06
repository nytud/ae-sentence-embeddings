"""Tune hyperparameters or run full pre-ae_training"""

from argparse import Namespace

from ae_sentence_embeddings.modeling_tools import get_training_args, read_json
from ae_sentence_embeddings.ae_training.hyperparameter_tuning import search_hparams
from ae_sentence_embeddings.ae_training.pretrain import (
    pretrain_transformer_ae,
    group_arguments,
    get_transformer_configs
)


def get_args() -> Namespace:
    """Get command line arguments"""
    parser = get_training_args()
    parser.add_argument("--search-hparams", dest="search_hparams", action="store_true",
                        help="Specify if the training is aimed at searching hyperparameters")
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = get_args()
    arg_dict = read_json(args.config_file)
    if args.search_hparams:
        search_hparams(arg_dict)
    else:
        dataset_split_paths, data_args, lr_args, adamw_args, save_log_args = group_arguments(arg_dict)
        transformer_configs = get_transformer_configs(arg_dict)
        pretrain_transformer_ae(
            dataset_split_paths=dataset_split_paths,
            data_args=data_args,
            lr_args=lr_args,
            adamw_args=adamw_args,
            save_log_args=save_log_args,
            transformer_configs=transformer_configs,
            num_epochs=arg_dict["num_epochs"],
            dataset_cache_dir=arg_dict["hgf_cache_dir"],
            devices=arg_dict["devices"]
        )


if __name__ == '__main__':
    main()
