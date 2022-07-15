#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A script for calculating the length of a batched dataset.
This can be useful when bucketing is applied. In this case,
calculating the number of batches is not trivial.
"""

from argparse import ArgumentParser, Namespace

from ae_sentence_embeddings.argument_handling import check_if_positive_int
from ae_sentence_embeddings.data import load_tokenizer, load_hgf_dataset, tokenize_hgf_dataset


def get_data_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(description="Command line arguments for batching")
    parser.add_argument("dataset", type=load_hgf_dataset,
                        help="Path to the input `jsonlines` dataset file.")
    parser.add_argument("--tokenizer", required=True, type=load_tokenizer,
                        help="Tokenizer name or path.")
    parser.add_argument("--batch-size", dest="batch_size", type=check_if_positive_int, default=32,
                        help="Batch size of the largest bucket. Defaults to `32`.")
    parser.add_argument("--num-buckets", dest="num_buckets", type=check_if_positive_int, default=1,
                        help="Number of buckets. Defaults to `1`.")
    parser.add_argument("--first-bucket-boundary", dest="first_bucket_boundary", type=check_if_positive_int,
                        default=33, help="Non-inclusive upper boundary of sequence length in the first "
                                         "bucket. Defaults to `33`.")
    parser.add_argument("--log-freq", dest="log_freq", type=check_if_positive_int, default=1000,
                        help="Specify how often to report the number of batches (in number of iterations). "
                             "Defaults to `1000`.")
    return parser.parse_args()


def main() -> None:
    """Main function. Iterate over the dataset to get its length."""
    args = get_data_args()
    dataset = tokenize_hgf_dataset(
        args.dataset, tokenizer=args.tokenizer, remove_old_cols=True)
    boundaries = [(args.first_bucket_boundary - 1) * 2**n + 1 for n in range(args.num_buckets)]
    del boundaries[-1]
    sizes = [int(args.batch_size / 2**n) for n in range(args.num_buckets)]
    buckets = [0] * args.num_buckets
    log_freq = args.log_freq
    num_batches, i = 0, 0

    for i, example in enumerate(dataset, start=1):
        input_len = len(example["input_ids"])
        for j, boundary in enumerate(boundaries):
            if input_len < boundary:
                actual_bucket_id = j
                break
        else:
            actual_bucket_id = len(boundaries)
        buckets[actual_bucket_id] += 1
        if buckets[actual_bucket_id] == sizes[actual_bucket_id]:
            num_batches += 1
            buckets[actual_bucket_id] = 0
            if num_batches % log_freq == 0:
                print(f"Iteration {i}, {num_batches} batches so far.")

    num_batches += sum(1 for bucket in buckets if bucket != 0)
    print(f"The dataset contains {i} data points and {num_batches} batches.")


if __name__ == "__main__":
    main()
