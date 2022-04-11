#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Load a model from a checkpoint then serialize its encoder part with Keras"""

import tensorflow as tf

from argparse import ArgumentParser, Namespace

from ae_sentence_embeddings.models import SentVaeEncoder
from ae_sentence_embeddings.ae_training.model_type_config import model_type_map, multilingual_models
from ae_sentence_embeddings.argument_handling import (
    check_if_output_path,
    check_if_file
)
from ae_sentence_embeddings.ae_training.pretrain import (
    group_model_args_from_flat,
    flatten_nested_dict,
    get_model_kwargs
)
from ae_sentence_embeddings.modeling_tools import read_json


def get_model_save_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(description="Command line arguments for loading and serializing "
                                        "the encoder part of an autoencoder")
    parser.add_argument("model_type", choices=model_type_map.keys(), help="The model architecture name")
    parser.add_argument("checkpoint_path", type=check_if_output_path, help="Path to the weights checkpoint.")
    parser.add_argument("--config-path", dest="config_path", required=True,
                        type=check_if_file, help="Path to the `json` configuration file "
                                                 "that was used for pre-training.")
    parser.add_argument("--save-path", dest="save_path", type=check_if_output_path,
                        help="Path to a directory where the Keras serialized encoder can be saved.")
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = get_model_save_args()
    config = flatten_nested_dict(read_json(args.config_path))
    model_args = group_model_args_from_flat(config)
    model_type = model_type_map[model_args.model_type_name]
    model_init_kwargs = get_model_kwargs(
        model_type=model_type,
        pooling_type=model_args.pooling_type,
        kl_factor=model_args.kl_factor,
        swap_p=model_args.swap_p,
        rnn_args=model_args.top_rnn_args,
        num_transformer2gru=model_args.num_transformer2gru
    )
    model = model_type(
        enc_config=model_args.encoder_config,
        dec_config=model_args.decoder_config,
        **model_init_kwargs
    )
    # Call the model with dummy inputs so that the weights can be loaded
    dummy_inputs1 = (tf.constant([[2, 5, 122, 3]]), tf.constant([[1]*4]))
    if model_args.model_type_name in multilingual_models:
        dummy_inputs2 = (tf.constant([[2, 9, 876, 3]]), tf.constant([[1]*4]))
        _ = model((dummy_inputs1, dummy_inputs2), training=False)
    else:
        _ = model(dummy_inputs1, training=False)
    model.load_weights(args.checkpoint_path).expect_partial()
    print("The full autoencoder:")
    model.summary()

    # Now prepare the encoder
    encoder = SentVaeEncoder(
        config=model_args.encoder_config,
        pooling_type=model_args.pooling_type,
        kl_factor=model_args.kl_factor
    )
    # noinspection PyCallingNonCallable
    _ = encoder(dummy_inputs1, training=False)  # PyCharm may complain, but the model is callable.
    encoder.set_weights(model.layers[0].get_weights())
    print("The encoder part:")
    encoder.summary()
    encoder.save(args.save_path)


if __name__ == "__main__":
    main()
