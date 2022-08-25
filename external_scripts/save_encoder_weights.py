#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Load the weights of a full autoencoder then save the encoder weights."""

import json
from argparse import ArgumentParser, Namespace, FileType

from transformers import BertConfig

from ae_sentence_embeddings.argument_handling import RnnArgs, check_if_output_path
from ae_sentence_embeddings.models import BertRnnVae


def get_encoder_save_args() -> Namespace:
    """Get command line arguments."""
    parser = ArgumentParser(
        description="Specify command line arguments to save the encoder part of an autoencoder.")
    parser.add_argument("encoder_config", type=FileType("rb"), help="The encoder configuration file.")
    parser.add_argument("decoder_config", type=FileType("rb"), help="The decoder configuration file.")
    parser.add_argument("weights", type=check_if_output_path, help="The model weights.")
    parser.add_argument("target_path", type=check_if_output_path, help="A path to save the encoder.")
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = get_encoder_save_args()
    encoder_config = BertConfig.from_dict(json.load(args.encoder_config))
    decoder_config = RnnArgs.collect_from_dict(json.load(args.decoder_config))
    model = BertRnnVae(encoder_config, decoder_config)
    model.load_checkpoint(args.weights)
    model.save_encoder_weights(args.target_path)


if __name__ == "__main__":
    main()
