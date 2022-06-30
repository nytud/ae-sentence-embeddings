# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

packages = \
["ae_sentence_embeddings",
 "ae_sentence_embeddings.ae_training",
 "ae_sentence_embeddings.callbacks",
 "ae_sentence_embeddings.data",
 "ae_sentence_embeddings.layers",
 "ae_sentence_embeddings.losses_and_metrics",
 "ae_sentence_embeddings.models",
 "ae_sentence_embeddings.scheduling"]

package_data = \
{"": ["*"]}

install_requires = \
["datasets>=2.0.0,<2.3.0",
 "tensorflow-addons>=0.16.1,<0.18.0",
 "tensorflow>=2.8.0,<3.0.0",
 "transformers>=4.17.0,<5.0.0",
 "wandb>=0.12.11,<0.13.0"]

setup_kwargs = {
    "name": "ae-sentence-embeddings",
    "version": "0.1.0",
    "description": "A package for training Transformer-based autoencoders",
    "long_description": long_description,
    "author": "Nyéki Bence",
    "author_email": "nyeki.bence96@gmail.com",
    "maintainer": None,
    "maintainer_email": None,
    "url": "https://github.com/nyekibence/ae-sentence-embeddings",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.8,<3.10",
}


setup(**setup_kwargs)
