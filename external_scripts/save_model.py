#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Load a model from a checkpoint then serialize its encoder part with Keras"""

import tensorflow as tf

from argparse import ArgumentParser, Namespace

from ae_sentence_embeddings.models import SentVaeEncoder
from ae_sentence_embeddings.ae_training.model_type_config import model_type_map
from ae_sentence_embeddings.argument_handling import (
    check_if_output_path,
    check_if_file,
    check_if_non_negative_float
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
                                        "the encoder part of an autoencoder or a full model.")
    parser.add_argument("model_type", choices=model_type_map.keys(), help="The model architecture name")
    parser.add_argument("checkpoint_path", type=check_if_output_path, help="Path to the weights checkpoint.")
    parser.add_argument("--config-path", dest="config_path", required=True,
                        type=check_if_file, help="Path to the `json` configuration file "
                                                 "that was used for pre-training.")
    parser.add_argument("--save-path", dest="save_path", type=check_if_output_path,
                        help="Path to a directory where the Keras serialized model can be saved.")
    parser.add_argument("--new-kl-factor", dest="new_kl_factor", type=check_if_non_negative_float,
                        help="Optional. Set a new KL loss factor for a VAE.")
    parser.add_argument("--new-min-kl", dest="new_min_kl", type=check_if_non_negative_float,
                        help="Optional. Set a minimal KL value for a VAE.")
    parser.add_argument("--save-full-model", dest="save_full_model", action="store_true",
                        help="Specify this flag to save the full model, not only the encoder part.")
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = get_model_save_args()
    config = flatten_nested_dict(read_json(args.config_path))
    model_args = group_model_args_from_flat(config)
    model_type = model_type_map[model_args.model_type_name]
    if args.new_kl_factor is not None:
        model_args.kl_args.kl_factor = args.new_kl_factor
    if args.new_min_kl is not None:
        model_args.kl_args.min_kl = args.new_min_kl
    model_init_kwargs = get_model_kwargs(
        model_type=model_type,
        pooling_type=model_args.pooling_type,
        kl_factor=model_args.kl_args.kl_factor,
        min_kl=model_args.kl_args.min_kl
    )
    model = model_type(
        enc_config=model_args.encoder_config,
        dec_config=model_args.decoder_config,
        **model_init_kwargs
    )
    # Call the model with dummy inputs so that the weights can be loaded
    dummy_inputs = (tf.keras.Input(shape=(None,), dtype=tf.int32),
                    tf.keras.Input(shape=(None,), dtype=tf.int32))
    _ = model(dummy_inputs, training=False)
    model.load_weights(args.checkpoint_path).expect_partial()
    print("The full autoencoder:")
    model.summary()
    if args.save_full_model:
        model.save(args.save_path)
    else:
        # Now prepare the encoder
        encoder = SentVaeEncoder(
            config=model_args.encoder_config,
            pooling_type=model_args.pooling_type,
            kl_factor=model_args.kl_args.kl_factor,
            min_kl=model_args.kl_args.min_kl
        )
        # noinspection PyCallingNonCallable
        _ = encoder(dummy_inputs, training=False)  # PyCharm may complain, but the model is callable.
        encoder.set_weights(model.layers[0].get_weights())
        print("The encoder part:")
        encoder.summary()
        encoder.save(args.save_path)


if __name__ == "__main__":
    main()
