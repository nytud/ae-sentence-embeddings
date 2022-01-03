"""Tune hyperparameters or run full pre-ae_training"""

from argparse import Namespace

from ae_sentence_embeddings.modeling_tools import get_training_args, read_json
from ae_sentence_embeddings.ae_training.hyperparameter_tuning import search_hparams


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
        raise NotImplementedError


if __name__ == '__main__':
    main()
