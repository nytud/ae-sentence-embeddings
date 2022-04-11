#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A script for calculating the length of a batched dataset.
This can be useful when bucketing is applied. In this case,
calculating the number of batches is not trivial.
"""

from argparse import ArgumentParser, Namespace

from datasets import load_dataset

from ae_sentence_embeddings.argument_handling import (
    DataStreamArgs,
    check_if_positive_int,
    check_if_file,
    check_if_output_path
)
from ae_sentence_embeddings.data import pad_and_batch, convert_to_tf_dataset


def get_data_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(description="Command line arguments for batching")
    parser.add_argument("input_path", type=check_if_file, help="Path to the input `jsonlines` dataset file.")
    parser.add_argument("--batch-size", dest="batch_size", type=check_if_positive_int, default=32,
                        help="Batch size of the largest bucket. Defaults to `32`.")
    parser.add_argument("--num-buckets", dest="num_buckets", type=check_if_positive_int, default=2,
                        help="Number of buckets. Defaults to `2`.")
    parser.add_argument("--first-bucket-boundary", dest="first_bucket_boundary", type=check_if_positive_int,
                        default=33, help="Non-inclusive upper boundary of sequence length in the first "
                                         "bucket. Defaults to `33`.")
    parser.add_argument("--input-padding", dest="input_padding", required=True,
                        nargs="+", help="Input padding tokens, one per input feature.")
    parser.add_argument("--target-padding", dest="target_padding", required=True,
                        nargs="+", help="Target padding tokens, one per target feature.")
    parser.add_argument("--cache-dir", dest="cache_dir", type=check_if_output_path,
                        help="Optional. Path to the dataset cache directory.")
    parser.add_argument("--keep-remainder", dest="drop_remainder", action="store_false",
                        help="Specify if you want to keep the last batch even if "
                             "it is smaller than expected.")
    return parser.parse_args()


def main() -> None:
    """Main function. Iterate over the dataset to get its length."""
    args = get_data_args()
    input_padding = [int(pad) for pad in args.input_padding]
    target_padding = [int(pad) for pad in args.target_padding]
    if len(input_padding) == 1:
        input_padding = input_padding[0]
    if len(target_padding) == 1:
        target_padding = target_padding[0]
    data_stream_args = DataStreamArgs(
        input_padding=input_padding,
        target_padding=target_padding,
        batch_size=args.batch_size,
        num_buckets=args.num_buckets,
        first_bucket_boundary=args.first_bucket_boundary
    )
    dataset = convert_to_tf_dataset(
        load_dataset("json", data_files=args.input_path, split="train", cache_dir=args.cache_dir))
    dataset = pad_and_batch(dataset, data_stream_args=data_stream_args,
                            drop_remainder=args.drop_remainder)
    i = 0
    for _ in dataset:
        i += 1
    print(f"Length of the batched dataset: {i}")


if __name__ == "__main__":
    main()
