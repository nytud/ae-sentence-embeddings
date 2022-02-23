"""Test a bilingual Transformer-RNN VAE
This includes testing model calling, logging and model saving
"""

from typing import Tuple
from os import environ, mkdir, listdir
from os.path import exists, isfile, join as os_path_join
from shutil import rmtree

import tensorflow as tf
import numpy as np
from tensorflow_addons.optimizers import AdamW
from transformers.models.bert.configuration_bert import BertConfig

from ae_sentence_embeddings.modeling_tools import get_custom_logger
from ae_sentence_embeddings.argument_handling import RnnArgs, OneCycleArgs, SaveAndLogArgs
from ae_sentence_embeddings.losses_and_metrics import IgnorantSparseCatCrossentropy, IgnorantSparseCatAccuracy
from ae_sentence_embeddings.models import BertBiRnnVae
from ae_sentence_embeddings.callbacks import basic_checkpoint_and_log, OneCycleScheduler, DevEvaluator
from ae_sentence_embeddings.modeling_tools import make_decoder_inputs
from ae_sentence_embeddings.data import post_batch_feature_pair


def _get_dummy_data(
        vocab_size: int,
        max_sequence_length: int,
        batch_size: int,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Create dummy data from a single random batch"""
    input_ids = [tf.constant(np.random.randint(1, vocab_size, size=(seq_len,)), dtype=tf.int32)
                 for seq_len in np.random.randint(3, max_sequence_length, size=(batch_size,))]
    input_ids = tf.stack([tf.pad(tens, [[0, max_sequence_length - tf.shape(tens)[0]]])
                          for tens in input_ids], axis=0)
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
    targets = tf.where(tf.equal(attention_mask, 1), input_ids, -1)
    targets = make_decoder_inputs(targets, pad_value=-1)
    return input_ids, attention_mask, targets


def _get_root_dirs() -> Tuple[str, str]:
    """Get log and checkpoint root directories"""
    log_root_dir = environ.get("AE_LOG_ROOT")
    save_root_dir = environ.get("AE_SAVE_ROOT")
    is_valid_log = log_root_dir is not None and not isfile(log_root_dir)
    is_valid_save = save_root_dir is not None and not isfile(log_root_dir)
    if not (is_valid_log and is_valid_save):
        home = environ["HOME"]
        if not exists(tmp_home := os_path_join(home, "tmp")):
            mkdir(tmp_home)
        if not is_valid_log:
            log_root_dir = os_path_join(tmp_home, "ae_log_root")
        if not is_valid_save:
            save_root_dir = os_path_join(tmp_home, "ae_save_root")
    for root_dir in (log_root_dir, save_root_dir):
        if not exists(root_dir):
            mkdir(root_dir)
    return log_root_dir, save_root_dir


class BilingualVaeTest(tf.test.TestCase):
    """Test case for a bilingual Transformer-RNN VAE"""

    @classmethod
    def setUpClass(cls):
        """Fixture setup. This will do the following:
            - Define a vocab size, a maximal sequence length and the train and dev epoch lengths
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
        cls.max_sequence_length = 64
        cls.train_epoch_length = 8
        cls.dev_epoch_length = 2
        dummy_data_args = {
            "vocab_size": cls.vocab_size,
            "max_sequence_length": cls.max_sequence_length,
            "batch_size": 16
        }
        data_splits = []
        data_options = tf.data.Options()
        data_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        for epoch_length in (cls.train_epoch_length, cls.dev_epoch_length):
            input_ids1, attn_mask1, targets1 = _get_dummy_data(**dummy_data_args)
            input_ids2, attn_mask2, targets2 = _get_dummy_data(**dummy_data_args)
            dataset = tf.data.Dataset.from_tensors(
                ((input_ids1, input_ids2, attn_mask1, attn_mask2), (targets1, targets2)))
            dataset = dataset.map(post_batch_feature_pair)
            dataset = dataset.repeat(epoch_length)
            dataset = dataset.with_options(data_options).prefetch(1)
            data_splits.append(dataset)
        cls.train_dataset, cls.dev_dataset = data_splits
        cls.log_root_dir, cls.save_root_dir = _get_root_dirs()
        if len(gpus := tf.config.list_physical_devices("GPU")) > 0:
            cls.device = "/gpu:" + gpus[0].name[-1]
        else:
            cls.device = "/cpu:" + tf.config.list_physical_devices("CPU")[0].name[-1]

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete model checkpoint directory after running the tests"""
        if exists(cls.save_root_dir):
            rmtree(cls.save_root_dir)
        super().tearDownClass()

    def _configure_training(self) -> Tuple[int, BertConfig, RnnArgs, OneCycleArgs, OneCycleArgs]:
        """Get configuration objects for training"""
        bert_config = BertConfig(
            vocab_size=self.vocab_size,
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=256,
            num_attention_heads=2,
            max_position_embeddings=self.max_sequence_length
        )
        rnn_config = RnnArgs(
            vocab_size=self.vocab_size,
            hidden_size=128,
            num_rnn_layers=2
        )
        num_epochs = 2
        total_steps = self.train_epoch_length * num_epochs - 1
        half_cycle = total_steps // 4
        lr_args = OneCycleArgs(
            initial_rate=0.01,
            total_steps=total_steps,
            half_cycle=half_cycle,
            cycle_extremum=0.03,
            end_extremum=0.001
        )
        momentum_args = OneCycleArgs(
            initial_rate=0.95,
            total_steps=total_steps,
            half_cycle=half_cycle,
            cycle_extremum=0.85
        )
        return num_epochs, bert_config, rnn_config, lr_args, momentum_args

    def _configure_save_and_log(self) -> SaveAndLogArgs:
        """Get checkpoint and log configuration"""
        save_log_args = SaveAndLogArgs(
            checkpoint_path=self.save_root_dir,
            log_tool=self.log_root_dir,
            log_update_freq=2
        )
        return save_log_args

    def test_bert_birnn(self) -> None:
        """Train the model, save it, then continue training"""
        num_epochs, bert_config, rnn_config, lr_args, momentum_args = self._configure_training()
        logger = get_custom_logger(os_path_join(self.log_root_dir, "integration.log"))
        callback_args = self._configure_save_and_log()
        callbacks = [
            *basic_checkpoint_and_log(callback_args),
            OneCycleScheduler(lr_args, log_freq=1, log_tool=logger),
            OneCycleScheduler(momentum_args, log_tool=logger, log_freq=1, parameter="beta_1"),
            DevEvaluator(self.dev_dataset, logger=logger, log_freq=5)
        ]

        strategy = tf.distribute.OneDeviceStrategy(self.device)
        with strategy.scope():
            model = BertBiRnnVae(bert_config, rnn_config, kl_factor=0.5)
            optimizer = AdamW(
                weight_decay=1e-6,
                learning_rate=lr_args.initial_rate,
                beta_1=momentum_args.initial_rate,
                amsgrad=True
            )
            loss_fn = IgnorantSparseCatCrossentropy(from_logits=True, factor=0.5)
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=[IgnorantSparseCatAccuracy()])
        history = model.fit(x=self.train_dataset, epochs=num_epochs, callbacks=callbacks)

        print(model.summary())
        self.assertIsInstance(history, tf.keras.callbacks.History)
        self.assertGreater(len(listdir(self.save_root_dir)), 0)

    def test_serialization(self) -> None:
        """Test whether the model can be correctly serialized"""
        _, bert_config, rnn_config, *_ = self._configure_training()
        model0 = BertBiRnnVae(bert_config, rnn_config)
        x = next(iter(self.train_dataset))[0]
        pred0 = model0.predict_on_batch(x)
        model0.save((save_path := os_path_join(self.save_root_dir, "whole_model")))
        print("Model0 architecture:")
        model0.summary()
        del model0
        model1 = tf.keras.models.load_model(save_path)
        print("Model1 architecture:")
        model1.summary()
        pred1 = model1.predict_on_batch(x)
        self.assertAllEqual(pred0.shape, pred1.shape)


if __name__ == "__main__":
    tf.test.main()
