
# -*- coding: utf-8 -*-

"""Download and save a dataset from the Hugging Face hub"""

from argparse import ArgumentParser, Namespace
from typing import Iterable, List

from datasets import load_dataset, Dataset

from ae_sentence_embeddings.argument_handling import check_if_output_path


def get_dataset_download_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(
        description="Command line arguments for downloading and saving a dataset")
    parser.add_argument("builder", help="Dataset format (i.e. `csv`, `text`) or benchmark name.")
    parser.add_argument("--save-paths", dest="save_paths", nargs="+",
                        type=check_if_output_path, required=True,
                        help="Path to the files where the dataset splits will be save, "
                             "one for each split.")
    parser.add_argument("--splits", nargs="+", required=True,
                        help="Dataset splits to use, i.e. `['train', 'validation']`")
    parser.add_argument("--sub-dataset", dest="sub_dataset",
                        help="Optional. Dataset name if `builder` refers to a benchmark, "
                             "i.e. `'CoLA'` (a dataset of the `'GLUE'` benchmark).")
    parser.add_argument("--field", help="Optional. The `field` argument of the "
                                        "`load_dataset` function.")
    parser.add_argument("--cache-dir", dest="cache_dir", type=check_if_output_path,
                        help="Optional. Cache directory for the dataset.")
    args = parser.parse_args()
    num_save_paths, num_splits = len(args.save_paths), len(args.splits)
    assert len(args.save_paths) == len(args.splits), "Please specify one save path for each dataset split. " \
                                                     f"{num_save_paths} paths and {num_splits} splits were given."
    return args


def lowercase_columns(dataset: Dataset) -> Dataset:
    """Lowercase all dataset column names"""
    col_name_map = {key: key.lower() for key in dataset.features.keys()
                    if not key.islower()}
    return dataset.rename_columns(col_name_map)


def main() -> None:
    """Main function"""
    args = get_dataset_download_args()
    dataset = load_dataset(
        path=args.builder,
        name=args.sub_dataset,
        split=args.splits,
        cache_dir=args.cache_dir,
        field=args.field
    )
    if isinstance(dataset, Dataset):
        lowercase_columns(dataset).to_json(args.save_paths[0], force_ascii=False)
    elif isinstance(dataset, list):
        for save_path, dataset_split in zip(args.save_paths, dataset):
            lowercase_columns(dataset_split).to_json(save_path, force_ascii=False)
    else:
        raise NotImplementedError(f"Saving of {type(dataset)} is not implemented.")


if __name__ == "__main__":
    main()
