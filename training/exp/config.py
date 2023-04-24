from ray import tune

from typing import Literal

EXP_PATH = "/raid/decaro/experiments/avl"


def get_exp_dir(method: str):
    return f"{EXP_PATH}/{method}"


def get_hparams_config(method: str):
    config = {
        "method": method,
        "data": {
            "user_id": None,
            "scenario_ids": [],
            "maneuvre_ids": [],
        },
    }
    models = {
        "reservoir": {
            "input_size": 2,
            "activation": "tanh",
            "hidden_size": tune.choice([100, 200, 300, 400, 500, 700]),
            "rho": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
            "leakage": tune.choice([0.1, 0.3, 0.5, 0.7, 0.8, 0.9]),
            "input_scaling": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9]),
            "recurrent_initializer": tune.choice(["uniform", "ring"]),
        },
        "gru": {
            "input_size": tune.sample_from(
                lambda spec: 4 - len(spec.config["data"]["drop_features"])
            ),
            "hidden_size": tune.choice([10, 20, 50, 100, 150]),
            "num_layers": tune.choice([1, 2, 3]),
        },
        "dt": {
            "max_depth": tune.choice([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
            "min_samples_split": tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "min_samples_leaf": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "criterion": tune.choice(["gini", "entropy"]),
            "splitter": tune.choice(["best", "random"]),
        },
    }
    models["l2"] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    models["ip"] = {
        "mu": 0,
        "eta": 1e-2,
        "sigma": tune.uniform(0.1, 0.99),
        "epochs": tune.choice([10, 12, 15]),
    }
    models["weights"] = [1, tune.uniform(1, 4)]

    if method in ["ridge", "ip"]:
        config["reservoir"] = models["reservoir"]
        config["l2"] = models["l2"]
        config["weights"] = models["weights"]
        if method == "ip":
            config["reservoir"]["net_gain_and_bias"] = True
            config["ip"] = models["ip"]
    elif method == "gru":
        config["gru"] = models["gru"]
    elif method == "dt":
        config["dt"] = models["dt"]

    return config


def get_analysis(
    phase: Literal["model_selection", "retraining", "test"],
    exp_dir: str,
    mode: Literal["min", "max"],
):
    return tune.ExperimentAnalysis(
        f"{exp_dir}/{phase}", default_metric="eval_score", default_mode=mode
    )
