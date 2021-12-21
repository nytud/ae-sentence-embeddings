"""A module for hyperparameter tuning"""

from typing import Dict, Any

import keras_tuner as kt

from ae_sentence_embeddings.training import pretrain_transformer_ae, group_arguments


class AeTuner(kt.Hyperband):
    """A hyperband tuner for the model"""

    @staticmethod
    def _config_hparams(hp: kt.HyperParameters, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameters from the basic configuration dictionary"""
        hparam_types = {"hp_int": hp.Int, "hp_float": hp.Float, "hp_choice": hp.Choice}
        for param_name, param_details in config.items():
            if isinstance(param_details, dict) and (hp_type := param_details.get("type")) in hparam_types.keys():
                new_hp = hp_type(**param_details["hp_space"])
                config[param_name] = new_hp
        return config

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters
