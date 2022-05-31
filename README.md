# ae-sentence-embeddings

A package for implementing sentence encoders with an autoencoder architecture using `TensorFlow/Keras`.


## Installation

With [Poetry](https://python-poetry.org/):
```bash
git clone https://github.com/nyekibence/ae-sentence-embeddings.git
cd ae-sentence-embeddings
poetry install && python3 setup.py develop
```

In order to run on GPU, please refer to the [TensorFlow documentation](https://www.tensorflow.org/install/gpu).
Alternatively, you can build a docker image using `Dockerfile` that installs
the necessary dependencies to run on GPU as well as the `ae_sentence_embeddings` Python library.
The image is based on a `TensorFlow Docker` image, see the [guide](https://www.tensorflow.org/install/docker) for more detailed information. 


## Train a tokenizer

If you would like to train a BPE tokenizer using [Tokenizers](https://huggingface.co/docs/tokenizers/python/latest/), run `ae_sentence_embeddings/data/train_tokenizer.py`.

```text
usage: train_tokenizer.py [-h] [--vocab-size VOCAB_SIZE] [--max-length MAX_LENGTH] [--unk-token UNK_TOKEN] [--special-tokens [SPECIAL_TOKENS ...]]
                          [--avoid-cls-sep-template]
                          training_file save_path

Arguments for training a BPE tokenizer from scratch

positional arguments:
  training_file         Path to train plain text training corpus
  save_path             Path to a `.json` file where the trained tokenizer will be saved

optional arguments:
  -h, --help            show this help message and exit
  --vocab-size VOCAB_SIZE
                        Required vocabulary size. Defaults to 30000
  --max-length MAX_LENGTH
                        Optional. Maximal sequence length to which longer sequences will be truncated. If not specified, truncation will not be used
  --unk-token UNK_TOKEN
                        Unknown token. Defaults to '[UNK]'
  --special-tokens [SPECIAL_TOKENS ...]
                        Optional. A list of special tokens aprt from the unknown token. If a CLS+SEP post-processing templated is used,
                        it is expected to list special tokens in the following order: (CLS token, SEP token, other tokens).
  --avoid-cls-sep-template
                        Avoid using a CLS+SEP post-processing template. This means not adding a CLS and a SEP token to any encoded text
```

## Tokenize a dataset

In order to tokenize a mono- or multilingual dataset, use the `data` subpackage: `python3 -m ae_sentence_embeddings.data`:

```text
usage: __main__.py [-h] [--multilingual] [--text-columns TEXT_COLUMNS [TEXT_COLUMNS ...]] [--tokenizer TOKENIZER] [--max-length MAX_LENGTH] [--cache-dir CACHE_DIR]
                   dataset_path tokenizer_output_path

Get command line arguments for tokenization

positional arguments:
  dataset_path          Path to the data to be tokenized in `sentence-per-line` or `jsonlines` format
  tokenizer_output_path
                        Path to the output `.jsonl` dataset

optional arguments:
  -h, --help            show this help message and exit
  --multilingual        Specify if the input data is multilingual. If it is multilingual, the input dataset is expected to be given in `jsonlines` format.
                        Otherwise, `sentence-per-line` format is expected.
  --text-columns TEXT_COLUMNS [TEXT_COLUMNS ...]
                        Dataset column (key) names associated with the input texts. Only relevant if the input is multilingual (e.g. the text column names can be
                        `('text_en', 'text_hu')`. If your data is monolingual, use the default value: `('text',)`
  --tokenizer TOKENIZER
                        Tokenizer name or path. Defaults to `SZTAKI-HLT/hubert-base-cc`
  --max-length MAX_LENGTH
                        Maximal sequence length in the number of tokens. Defaults to 128
  --cache-dir CACHE_DIR
                        Optional. The cache directory for the dataset
```

Currently, you can use the following tokenizers:

* A `transformers` [BertTokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer) stored locally or downloadable from the [Hugging Face Hub](https://huggingface.co/docs/hub/main)
* A `BPE tokenizer` that you trained as [described above](#train-a-tokenizer) 


## Train a model

In order to train a model, use the `ae_training` subpackage:

```bash
python3 -m ae_sentence_embeddings.ae_training /path/to/config/file.json
```

As model training involves many arguments, they are required to be specified in a configuration file.
An example is provided here: `config_example.json`

These arguments are processed by the dataclasses defined in `argument_handling.py`

The following model architectures are currently available:
* `TransformerAe`: A BERT encoder and a GPT decoder. The pooled BERT encoder output is used as first token embedding input (similar to CLS) to the decoder
* `TransformerVae`: Similar to `TransformerVae` but with variational auto-encoding.
* `BertRnnVae`: A BERT encoder and an RNN (GRU) decoder. The pooled BERT output is passed to the decoder as its initial hidden state. This architecture relies on variational auto-encoding.

## Download

You can download the pre-trained model weigths [here](https://nc.nlp.nytud.hu/s/TpSQ4zKLypD65XE).

## Notes

Only the [AdamW optimizer](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW) is currently supported.

1cyle scheduling is not implemented by TensorFlow. An implementation is given in the `scheduling` and `callbacks` subpackages.

[WandB](https://wandb.ai/site) is used to handle logging during model training. You can create a free account if you do not have one.
Apart from logging, WandB also offers some tools for [hyperparameter tuning](https://docs.wandb.ai/guides/sweeps).
