"""Test a bilingual Transformer-RNN VAE
This includes testing learning rate schedules, model calling,
loss calculation, logging and model saving
"""

from os import environ

import tensorflow as tf
import numpy as np
from tensorflow_addons.optimizers import AdamW
from transformers.models.bert.configuration_bert import BertConfig

from ae_sentence_embeddings.scheduling import OneCycleSchedule
from ae_sentence_embeddings.argument_handling import RnnArgs
from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy
from ae_sentence_embeddings.models import BertBiRnnVae
from ae_sentence_embeddings.modeling_tools import make_decoder_inputs


def _get_dummy_dataset(
        vocab_size: int,
        max_sequence_length: int,
        batch_size: int,
        num_repeat: int
) -> tf.data.Dataset:
    """Create a dummy dataset from by repeating a single random batch `num_repeat times`"""
    input_ids = [tf.constant(np.random.randint(1, vocab_size, size=(seq_len,)), dtype=tf.int32)
                 for seq_len in np.random.randint(3, max_sequence_length, size=(batch_size,))]
    input_ids = tf.stack([tf.pad(tens, [[0, max_sequence_length - tf.shape(tens)[0]]])
                          for tens in input_ids], axis=0)
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
    targets = tf.where(tf.equal(attention_mask, 1), input_ids, -1)
    targets = make_decoder_inputs(targets, pad_value=-1)
    dataset = tf.data.Dataset.from_tensors((input_ids, attention_mask, targets))
    return dataset.repeat(num_repeat)


class BilingualVaeTest(tf.test.TestCase):
    """Test case for a bilingual Transformer-RNN VAE"""

    @classmethod
    def setUpClass(cls):
        """Fixture setup. This will do the following:
            - Define a vocab size and a maximal sequence length
            - Create a dummy dataset with input IDs, attention mask and targets
            - Get the environment variables `AE_LOG_ROOT` and `AE_SAVE_ROOT`, which should be paths to
              directories for logging and model checkpoints, respectively. If these variables are not defined
              or are paths to files, two directories will be automatically created at `$HOME/tmp/ae_log_root`
              and `$HOME/tmp/ae_save_root`. Any newly created model checkpoint will be automatically deleted
              after the test
            - Get a device (1 GPU if available or CPU otherwise)
        """
        super().setUpClass()
        cls.vocab_size = 256
        cls.max_sequence_length = 32
        cls.dataset = _get_dummy_dataset(
            vocab_size=cls.vocab_size,
            max_sequence_length=cls.max_sequence_length,
            batch_size=16,
            num_repeat=8
        )
        cls.log_root_dir = environ.get("AE_LOG_ROOT")
        cls.save_root_dir = environ.get("AE_SAVE_ROOT")
        if len(gpus := tf.config.list_physical_devices("GPU")) > 0:
            cls.device = gpus[0]
        else:
            cls.device = "CPU"
