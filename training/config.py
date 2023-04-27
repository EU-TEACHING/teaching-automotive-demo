from ray import tune

from typing import Literal

EXP_PATH = "/raid/decaro/experiments/avl_study"


def get_exp_dir(user_id: str, method: str, classification: bool):
    return (
        f"{EXP_PATH}/{user_id}/{'classification' if classification else 'regression'}"
    )


def get_hparams_config(user_id: str, method: str, classification: bool):
    config = {
        "method": method,
        "data": {
            "user_id": user_id,
            "columns": ["EDA_Tonic", "EDA_Phasic"],
            "norm": "baseline",
            "classification": classification,
        },
    }
    models = {
        "reservoir": {
            "input_size": tune.sample_from(
                lambda spec: len(spec.config["data"]["columns"])
            ),
            "activation": "tanh",
            "hidden_size": tune.choice([100, 200, 300, 400, 500, 700]),
            "rho": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
            "leakage": tune.choice([0.1, 0.3, 0.5, 0.7, 0.8, 0.9]),
            "input_scaling": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
            "recurrent_initializer": tune.choice(["uniform", "ring"]),
        },
    }
    models["l2"] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    models["ip"] = {
        "mu": 0,
        "eta": 1e-2,
        "sigma": tune.uniform(0.1, 0.99),
        "epochs": tune.choice([3, 5, 10, 12]),
    }
    models["weights"] = [1, tune.uniform(1, 4)]
    if method in ["ridge", "ip"]:
        config["reservoir"] = models["reservoir"]
        config["l2"] = models["l2"]
        if method == "ip":
            config["reservoir"]["net_gain_and_bias"] = True
            config["ip"] = models["ip"]

    return config


def get_analysis(
    phase: Literal["model_selection", "retraining", "test"],
    exp_dir: str,
    mode: Literal["min", "max"],
):
    return tune.ExperimentAnalysis(
        f"{exp_dir}/{phase}", default_metric="eval_loss", default_mode=mode
    )
