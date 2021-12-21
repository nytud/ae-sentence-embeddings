"""Tokenize raw text data"""

from argparse import Namespace, ArgumentParser

from datasets import load_dataset
from transformers import AutoTokenizer

from ae_sentence_embeddings.data import tokenize_hgf_dataset


def get_tokenization_args() -> Namespace:
    """Get command line arguments"""
    parser = ArgumentParser(description="Get command line arguments for tokenization")
    parser.add_argument("text_dataset_path", help="Path to the text data to be tokenized")
    parser.add_argument("tokenizer_output_path", help="Path to the output `.jsonl` dataset")
    parser.add_argument("--tokenizer", default="SZTAKI-HLT/hubert-base-cc",
                        help="Tokenizer name or path. Defaults to `SZTAKI-HLT/hubert-base-cc`")
    parser.add_argument("--cache-dir", dest="cache_dir", help="Optional. The cache directory for the dataset")
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = vars(get_tokenization_args())
    raw_dataset = load_dataset('text', data_files=[args["text_dataset_path"]], split='train',
                               cache_dir=args["cache_dir"])
    tokenizer = AutoTokenizer.from_pretrained(args["tokenizer"])
    tokenized_dataset = tokenize_hgf_dataset(dataset=raw_dataset, tokenizer=tokenizer)
    tokenized_dataset.to_json(args["tokenizer_output_path"], force_ascii=False)


if __name__ == '__main__':
    main()
