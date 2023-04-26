import os
from typing import Dict, Literal
from ray import air, tune

from ..trainables import get_trainable


def run(
    phase: Literal["model_selection", "retraining"],
    exp_dir: str,
    config: Dict,
    gpus_per_trial: float,
):
    os.makedirs(exp_dir, exist_ok=True)

    reporter = tune.CLIReporter(
        metric_columns={
            # "train_score": "TR-Score",
            # "eval_score": "VL-Score",
            # "test_score": "TS-Score",
            "train_accuracy": "TR-Acc",
            "eval_accuracy": "VL-Acc",
            "test_accuracy": "TS-Acc",
            "train_loss": "TR-Loss",
            "eval_loss": "VL-Loss",
            "test_loss": "TS-Loss",
        },
        infer_limit=3,
        metric="eval_loss",
        mode="min",
    )

    resources = {"cpu": 1, "gpu": gpus_per_trial}
    config["phase"] = phase
    num_samples = 700 if phase == "model_selection" else 100

    stopper = lambda trial_id, result: True

    tuner = tune.Tuner(
        tune.with_resources(get_trainable(config["method"]), resources),
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=num_samples, reuse_actors=True),
        run_config=air.RunConfig(
            name=phase,
            local_dir=exp_dir,
            stop=stopper,
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_order="min",
                checkpoint_score_attribute="eval_loss",
                checkpoint_frequency=1,
            ),
            verbose=1,
            progress_reporter=reporter,
        ),
    )
    return tuner.fit()
