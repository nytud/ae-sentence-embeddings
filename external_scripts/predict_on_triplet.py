#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare the encoder embeddings on a triplet of inputs."""

import json
from typing import Tuple
from argparse import ArgumentParser, Namespace, FileType

import numpy as np
from numpy.linalg import norm as linalg_norm
from transformers import BertConfig, PreTrainedTokenizer

from ae_sentence_embeddings.data import load_tokenizer
from ae_sentence_embeddings.argument_handling import check_if_output_path
from ae_sentence_embeddings.models.ae_classifier_models import SentVaeClassifier


def get_args() -> Namespace:
    """Get command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("encoder_config", type=FileType("rb"), help="The encoder configuration file.")
    parser.add_argument("weights", type=check_if_output_path, help="The model weights.")
    parser.add_argument("tokenizer", type=load_tokenizer, help="Path to a tokenizer checkpoint.")
    return parser.parse_args()


def euclidean_distance(vec1: np.array, vec2: np.array) -> float:
    """Calculate the euclidean distance between two vectors."""
    vec1, vec2 = np.squeeze(vec1), np.squeeze(vec2)
    return linalg_norm(vec1 - vec2)


def measure_distances(
        triplet: Tuple[str, str, str],
        tokenizer: PreTrainedTokenizer,
        model: SentVaeClassifier
) -> Tuple[float, float, float]:
    anc, pos, neg = triplet
    anc_tokens = tuple(tokenizer(anc, return_tensors="tf", return_token_type_ids=False).values())
    pos_tokens = tuple(tokenizer(pos, return_tensors="tf", return_token_type_ids=False).values())
    neg_tokens = tuple(tokenizer(neg, return_tensors="tf", return_token_type_ids=False).values())

    _, anc_encoding = model.predict(anc_tokens)
    _, pos_encoding = model.predict(pos_tokens)
    _, neg_encoding = model.predict(neg_tokens)

    anc_pos = euclidean_distance(anc_encoding, pos_encoding)
    anc_neg = euclidean_distance(anc_encoding, neg_encoding)
    pos_neg = euclidean_distance(pos_encoding, neg_encoding)
    return anc_pos, anc_neg, pos_neg


def main() -> None:
    """Main function."""
    args = get_args()
    tokenizer = args.tokenizer
    anchor = "The weather is great today."
    positive = "It is such a lovely day."
    negative = "I really hate debugging."

    encoder_config = BertConfig.from_dict(json.load(args.encoder_config))
    model = SentVaeClassifier(encoder_config, num_labels=3)
    model.build_and_load(args.weights)

    anc_pos, anc_neg, pos_neg = measure_distances(
        (anchor, positive, negative), tokenizer, model)
    print(f"Anchor - positive distance: {anc_pos}")
    print(f"Anchor - negative distance: {anc_neg}")
    print(f"Positive - negative distance: {pos_neg}")


if __name__ == "__main__":
    main()
