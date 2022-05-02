#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test a model on a binary classification dataset"""

from argparse import ArgumentParser, Namespace
from os.path import basename
from typing import Literal

from transformers import (
    TFBertForSequenceClassification,
    TFXLMRobertaForSequenceClassification,
    BertConfig,
    XLMRobertaConfig
)
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.losses import BinaryCrossentropy
from datasets import load_dataset

from ae_sentence_embeddings.argument_handling import DataStreamArgs, check_if_file, check_if_positive_int
from ae_sentence_embeddings.data import pad_and_batch, convert_to_tf_dataset
from ae_sentence_embeddings.losses_and_metrics import BinaryMCC, BinaryLogitAccuracy
from ae_sentence_embeddings.models import SentVaeClassifier


MODEL_TYPES = {
    "bert": (BertConfig, TFBertForSequenceClassification),
    "roberta": (XLMRobertaConfig, TFXLMRobertaForSequenceClassification),
    "bivase": (None, SentVaeClassifier)
}


def _reload_model(model_path: str, model_type: Literal["bert", "roberta"]) -> Model:
    config_type, model_type = MODEL_TYPES[model_type]
    if model_type is SentVaeClassifier:
        config = BertConfig(max_position_embeddings=256)
        new_model = SentVaeClassifier(config, num_labels=1, pooling_type="average")
        # noinspection PyCallingNonCallable
        _ = new_model((tf.keras.Input(shape=(None,), dtype=tf.int32),
                       tf.keras.Input(shape=(None,), dtype=tf.int32)))
        new_model.load_weights(model_path).expect_partial()
    else:
        loaded_model = load_model(model_path, compile=False)
        print(loaded_model.layers)
        loaded_model.summary()
        config = config_type(**loaded_model.get_config(), num_labels=1)
        new_model = model_type(config)
        _ = new_model((tf.keras.Input(shape=(None,), dtype=tf.int32),
                       tf.keras.Input(shape=(None,), dtype=tf.int32)))
        print(new_model.layers)
        new_model.summary()
        for old_layer, new_layer in zip(loaded_model.layers, new_model.layers):
            new_layer.set_weights(old_layer.get_weights())
    return new_model


def get_test_args() -> Namespace:
    """Get command line arguments for testing a model on a dataset."""
    parser = ArgumentParser(description="Test a model on a binary classification test set")
    parser.add_argument("test_file", type=check_if_file, help="Path to a tokenized test split file in `jsonlines`.")
    parser.add_argument("model", help="Path to the checkpoint directory of a Keras serialized model")
    parser.add_argument("--model-type", dest="model_type", choices=["bert", "roberta", "bivase"], default="bert",
                        help="Model architecture to use. Defaults to bert.")
    parser.add_argument("--input-padding", dest="input_padding", nargs="+", default=[0, 0],
                        help="Padding token for each input. Defaults to `[0, 0]`.")
    parser.add_argument("--num-buckets", dest="num_buckets", type=check_if_positive_int,
                        default=3, help="Number of buckets for batch bucketing. Defaults to `3`.")
    parser.add_argument("--batch-size", dest="batch_size", type=check_if_positive_int, default=64,
                        help="Maximal batch size. Defaults to `64`.")
    parser.add_argument("--first-bucket-boundary", dest="first_bucket_boundary", type=check_if_positive_int,
                        default=33, help="Non-inclusive sequence length boundary for the first bucket. "
                                         "Defaults to `33`.")
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = get_test_args()
    data_stream_args = DataStreamArgs(
        input_padding=args.input_padding,
        target_padding=-1,
        batch_size=args.batch_size,
        num_buckets=args.num_buckets,
        first_bucket_boundary=args.first_bucket_boundary
    )
    dataset = convert_to_tf_dataset(
        load_dataset("json", data_files=args.test_file, split="train"))
    dataset = pad_and_batch(dataset, data_stream_args=data_stream_args)
    model = _reload_model(args.model, args.model_type)
    model.compile(loss=BinaryCrossentropy(from_logits=True),
                  metrics=[BinaryMCC(), BinaryLogitAccuracy()])
    print(f"Evaluating on {basename(args.test_file)}...")
    res = model.evaluate(dataset)
    print(f"Results:\n{res}")


if __name__ == "__main__":
    main()
