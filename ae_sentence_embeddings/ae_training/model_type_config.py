"""A configuration file for easy model type selection"""

from ae_sentence_embeddings.models import TransformerAe, TransformerVae, BertBiRnnVae, BertRnnVae

model_type_map = {model_class.__name__: model_class
                  for model_class in {TransformerVae, TransformerAe, BertBiRnnVae, BertRnnVae}}
multilingual_models = {BertRnnVae.__name__}
rnn_decoder_models = {BertRnnVae.__name__, BertBiRnnVae.__name__}
