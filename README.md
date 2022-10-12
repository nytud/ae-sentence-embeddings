# ae-sentence-embeddings

A package for implementing sentence encoders with an autoencoder architecture using `TensorFlow/Keras`.


## Installation

With [Poetry](https://python-poetry.org/):
```bash
git clone https://github.com/nyekibence/ae-sentence-embeddings.git
cd ae-sentence-embeddings
poetry install
```

In order to run on GPU, please refer to the [TensorFlow documentation](https://www.tensorflow.org/install/gpu).
Alternatively, you can build a docker image using `Dockerfile` that installs
the necessary dependencies to run on GPU as well as the `ae_sentence_embeddings` Python library.
The image is based on a `TensorFlow Docker` image, see the [guide](https://www.tensorflow.org/install/docker) for more detailed information. 


## Train a tokenizer

If you would like to train a BPE tokenizer using [Tokenizers](https://huggingface.co/docs/tokenizers/python/latest/), run `ae_sentence_embeddings/data/train_tokenizer.py`.

Currently, you can use the following tokenizers:

* A `transformers` [BertTokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer) stored locally or downloadable from the [Hugging Face Hub](https://huggingface.co/docs/hub/main)
* A `BPE tokenizer` that you trained as described above


## Train a model

In order to pre-train a model, use the `ae_sentence_embeddings/pretrain_vae.py` script. This will create a `BertRnnVae` model. It consists of a BERT encoder and an RNN (GRU) decoder. The pooled BERT output is passed to the decoder as its initial hidden state. This architecture relies on variational auto-encoding.


Fine-tuning with a classifier head on top can be done by runnig the `ae_sentence_embeddings/fine_tuning/__main__.py` script. (Type `python3 -m ae_sentence_embeddings.fine_tuning -h` to see the command line arguments).

## Notes

1cyle scheduling is not implemented by TensorFlow. An implementation is given in the `scheduling` and `callbacks` subpackages.

[WandB](https://wandb.ai/site) is used to handle logging during model training. You can create a free account if you do not have one.
Apart from logging, WandB also offers some tools for [hyperparameter tuning](https://docs.wandb.ai/guides/sweeps).
