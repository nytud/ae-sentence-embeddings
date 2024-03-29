"""A configuration file for easy model type selection"""

from ae_sentence_embeddings.models import (
    TransformerAe,
    BertRnnVae,
)

all_models = {TransformerAe, BertRnnVae}
vae_models = {model_type.__name__ for model_type in
              (BertRnnVae,)}
model_type_map = {model_class.__name__: model_class for model_class in all_models}
rnn_only_decoder_models = {BertRnnVae.__name__}
