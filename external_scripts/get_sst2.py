#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Save the sentence-level SST2 dataset."""

import pickle
from argparse import ArgumentParser, FileType, Namespace

import jsonlines


def get_sst2_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("id2sent_file", type=FileType("rb"),
                        help="Path to the file that maps IDs to sentences.")
    parser.add_argument("id2label_file", type=FileType("rb"),
                        help="Path to a file that maps IDs to labels.")
    parser.add_argument("output_file", type=FileType("w", encoding="utf-8"),
                        help="Path to the output file that will contains the `jsonlines` dataset.")
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = get_sst2_args()
    cols = ("sentence", "label")
    id2sent = pickle.load(args.id2sent_file)
    id2label = pickle.load(args.id2label_file)
    writer = jsonlines.Writer(args.output_file)
    for idx, sent in id2sent.items():
        label = id2label.get(idx)
        # drop neutral labels
        if label is None or 0.4 < label <= 0.6:
            continue
        sent = " ".join(sent)
        label = round(label)
        writer.write({k: v for k, v in zip(cols, (sent, label))})


if __name__ == "__main__":
    main()
