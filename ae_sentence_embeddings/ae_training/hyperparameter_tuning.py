"""A module for hyperparameter tuning"""

from typing import Dict, Any
from copy import deepcopy

import tensorflow as tf
import keras_tuner as kt

from ae_sentence_embeddings.ae_training import hparam_search, group_arguments, get_transformer_configs
from ae_sentence_embeddings.data import get_train_and_validation


class AeTuner(kt.Hyperband):
    """A hyperband tuner for the model"""

    @staticmethod
    def _config_hparams(hp: kt.HyperParameters, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameters from the basic configuration dictionary"""
        hparam_types = {"hp_int": hp.Int, "hp_float": hp.Float, "hp_choice": hp.Choice}
        hp_config = deepcopy(config)
        for param_name, param_details in hp_config.items():
            if isinstance(param_details, dict) and (hp_type := param_details.get("type")) in hparam_types.keys():
                new_hp = hp_type(param_name, **param_details["hp_space"])
                hp_config[param_name] = new_hp
        return hp_config

    def run_trial(self, trial, train_ds: tf.data.Dataset,
                  dev_ds: tf.data.Dataset, arg_dict: Dict[str, Any]) -> float:
        hp = trial.hyperparameters
        hp_config = self._config_hparams(hp, arg_dict)
        _, _, lr_args, adamw_args, save_log_args = group_arguments(hp_config)
        transformer_configs = get_transformer_configs(hp_config)
        return hparam_search(
            train_ds=train_ds,
            dev_ds=dev_ds,
            lr_args=lr_args,
            adamw_args=adamw_args,
            transformer_configs=transformer_configs,
            save_log_args=save_log_args
        )


def search_hparams(arg_dict: Dict[str, Any]) -> None:
    """Tune hyperparameters and print the best values"""
    dataset_split_paths, data_args, *_ = group_arguments(arg_dict)
    train_ds, dev_ds = get_train_and_validation(
        data_split_paths=dataset_split_paths,
        train_args=data_args,
        cache_dir=arg_dict["hgf_cache_dir"]
    )
    tuner = AeTuner(max_epochs=arg_dict["num_epochs"], directory=data_args["checkpoint_path"],
                    overwrite=True, project_name="ae_sentence_embeddings")
    tuner.search(train_ds=train_ds, dev_ds=dev_ds, arg_dict=arg_dict)
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp.values)
