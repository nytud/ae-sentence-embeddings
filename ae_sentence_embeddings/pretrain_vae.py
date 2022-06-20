#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pre-train a variational autoencoder with an Adam optimizer
and a cosine learning rate decay with restarts.
"""

from argparse import ArgumentParser, Namespace
from typing import Union, Optional, Tuple
from logging import Logger
from math import ceil

import wandb
import tensorflow as tf
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from datasets import Dataset as HgfDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BertConfig
from tokenizers import Tokenizer

from ae_sentence_embeddings.data import (
    load_hgf_dataset,
    tokenize_hgf_dataset,
    load_tokenizer,
    convert_to_tf_dataset,
    pad_and_batch
)
from ae_sentence_embeddings.argument_handling import (
    check_if_positive_int,
    check_if_positive_float,
    check_if_non_negative_float,
    check_if_output_path,
    DataStreamArgs,
    RnnArgs
)
from ae_sentence_embeddings.modeling_tools import get_custom_logger, read_json, timing
from ae_sentence_embeddings.callbacks import AeCustomCheckpoint, DevEvaluator, VaeLogger
from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy
from ae_sentence_embeddings.models import BertRnnVae


def get_pretrain_args() -> Namespace:
    """Get command line arguments for VAE pretraining."""
    parser = ArgumentParser(description="VAE pretraining arguments")
    parser.add_argument("train_dataset", type=load_hgf_dataset, help="Path to the raw training dataset.")
    parser.add_argument("devel_dataset", type=load_hgf_dataset, help="Path to the raw validation dataset.")
    parser.add_argument("--tokenizer", required=True, type=load_tokenizer,
                        help="Tokenizer name or path.")
    parser.add_argument("--encoder-config", dest="encoder_config", required=True, type=read_json,
                        help="Path to the `json` file that defines the encoder configuration.")
    parser.add_argument("--decoder-config", dest="decoder_config", required=True, type=read_json,
                        help="Path to the `json` file that defines the decoder configuration.")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", type=check_if_output_path,
                        help="The path to the checkpoint root directory.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=check_if_positive_float,
                        default=1e-5, help="The initial learning rate. Defaults to `1e-5`.")
    parser.add_argument("--first-decay-steps", dest="first_decay_steps", type=check_if_positive_int,
                        default=1000, help="The number of steps in the first learning rate decay period. "
                                           "Defaults to `1000`.")
    parser.add_argument("--iter-mul", dest="iter_mul", type=check_if_positive_float, default=2.,
                        help="The multiplier to derive the number of iterations in the `i+1th` learning rate "
                             "decay period from the number of iterations in the `ith` period. Defaults to `2.`.")
    parser.add_argument("--lr-mul", dest="lr_mul", type=check_if_positive_float, default=1.,
                        help="The multiplier to derive the initial learning rate in the `i+1th` learning rate "
                             "decay period from the initial learning rate in the `ith` period. Defaults to `1.`.")
    parser.add_argument("--decay-alpha", dest="decay_alpha", type=check_if_non_negative_float, default=0.,
                        help="Minimum learning rate value as a fraction of the initial learning rate. "
                             "Defaults to `0`.")
    parser.add_argument("--beta1", type=check_if_positive_float, default=0.9,
                        help="`beta_1` parameter of the `Adam` optimizer. Defaults to `0.9`.")
    parser.add_argument("--beta2", type=check_if_positive_float, default=0.999,
                        help="`beta_2` parameter of the `Adam` optimizer. Defaults to `0.999`.")
    parser.add_argument("--batch-size", dest="batch_size", type=check_if_positive_int, default=64,
                        help="Batch size. Defaults to `64`.")
    parser.add_argument("--epochs", type=check_if_positive_int, default=1,
                        help="The number of training epochs. Defaults to `1`.")
    parser.add_argument("--pooling-type", dest="pooling_type", choices=["average", "cls_sep", "p_means"],
                        default="average", help="The pooling type to use. Defaults to `'average'`.")
    parser.add_argument("--beta-warmup-iters", dest="beta_warmup_iters", type=check_if_positive_int, default=1000,
                        help="The number of iterations while the KL loss `beta` will be annealed. "
                             "Defaults to `1000`.")
    parser.add_argument("--beta-warmup-start", dest="beta_warmup_start", type=check_if_positive_int, default=1000,
                        help="The iteration after which the KL loss `beta` will start to be annealed. "
                             "Defaults to `1000`.")
    parser.add_argument("--min-kl", dest="min_kl", type=check_if_non_negative_float, default=0.,
                        help="The minimal KL loss per dimension. Defaults to `0.`.")
    parser.add_argument("--log-update-freq", dest="log_update_freq", type=check_if_positive_int, default=500,
                        help="The number of iterations after which logging will be done. Defaults to `500`.")
    parser.add_argument("--validation-freq", dest="validation_freq", type=check_if_positive_int, default=500,
                        help="The number of iterations after which validation be done. Defaults to `500`.")
    parser.add_argument("--save-freq", dest="save_freq", type=check_if_positive_int, default=1000,
                        help="The number of iterations after which a checkpoint will be created. Defaults to `1000`.")
    parser.add_argument("--project", help="`WandB` project name. Optional.")
    parser.add_argument("--run-name", dest="run_name", help="`WandB` run name. Optional.")
    return parser.parse_args()


def strategy_setup() -> tf.distribute.Strategy:
    """Select a training strategy based on the available devices."""
    gpus = tf.config.get_visible_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device=gpus[0])
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu")
    return strategy


def process_data(
        dataset: HgfDataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, Tokenizer],
        batch_size: int,
        logger: Optional[Logger] = None
) -> Tuple[tf.data.Dataset, int]:
    """Tokenize a raw text dataset and convert it to `TensorFlow` format.

    Args:
        dataset: The input dataset that contains a field `text`.
        tokenizer: A pretrained tokenizer.
        batch_size: The batch size in the output dataset.
        logger: A customized logger to which `DEBUG` messages will be passed.
            If not specified, it will be created. Optional.

    Returns:
        A batched `TensorFlow` dataset of the structure `((input_ids, attention_mask), labels)`
        and the number of batches in that dataset.
    """
    if logger is None:
        logger = get_custom_logger()
    num_data_points = len(dataset)
    with timing(f"Tokenizing {num_data_points} data points", time_logger=logger):
        dataset = tokenize_hgf_dataset(dataset=dataset, tokenizer=tokenizer, remove_old_cols=True)
    num_batches = ceil(num_data_points / batch_size)
    dataset = convert_to_tf_dataset(dataset)
    dataset = pad_and_batch(
        tf_dataset=dataset,
        data_stream_args=DataStreamArgs(batch_size=batch_size),
        drop_remainder=False
    )
    return dataset, num_batches


def main() -> None:
    """Main function."""
    args = vars(get_pretrain_args())
    logger = get_custom_logger()
    data_kwargs = {
        "tokenizer": args.pop("tokenizer"),
        "batch_size": args["batch_size"],
        "logger": logger
    }
    train_dataset, num_batches = process_data(dataset=args.pop("train_dataset"), **data_kwargs)
    devel_dataset, _ = process_data(dataset=args.pop("devel_dataset"), **data_kwargs)
    logger.debug(f"An example from the training dataset:\n{next(iter(train_dataset))}")
    train_dataset = train_dataset.prefetch(1)

    callbacks = [
        AeCustomCheckpoint(args["checkpoint_dir"], save_freq=args["save_freq"]),
        VaeLogger(
            log_update_freq=args["log_update_freq"],
            beta_warmup_iters=args["beta_warmup_iters"],
            beta_warmup_start=args["beta_warmup_start"],
        ),
        DevEvaluator(devel_dataset, logger="WandB", log_freq=args["validation_freq"]),
        WandbCallback(save_model=False, log_batch_frequency=args["log_update_freq"])
    ]
    project, run_name = args.pop("project"), args.pop("run_name")
    wandb.init(config=args, project=project, name=run_name)

    strategy = strategy_setup()
    with strategy.scope():
        scheduler = CosineDecayRestarts(
            initial_learning_rate=args["learning_rate"],
            first_decay_steps=args["first_decay_steps"],
            t_mul=args["iter_mul"],
            m_mul=args["lr_mul"],
            alpha=args["decay_alpha"]
        )
        optimizer = Adam(
            learning_rate=scheduler,
            beta_1=args["beta1"],
            beta_2=args["beta2"]
        )
        model = BertRnnVae(
            enc_config=BertConfig.from_dict(args["encoder_config"]),
            dec_config=RnnArgs.collect_from_dict(args["decoder_config"]),
            pooling_type=args["pooling_type"],
            reg_args={
                "iters": optimizer.iterations,
                "warmup_iters": args["beta_warmup_iters"],
                "start": args["beta_warmup_start"],
                "min_kl": args["min_kl"]
            }
        )
        model.compile(loss=IgnorantSparseCatCrossentropy(), from_logits=True, optimizer=optimizer)
        logger.debug(f"Begin training on {num_batches} batches.")
        _ = model.fit(
            x=train_dataset,
            epochs=args["epochs"],
            callbacks=callbacks,
            verbose=2
        )


if __name__ == "__main__":
    main()
