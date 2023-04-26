from typing import Literal, Optional
import torch, os, json
from ray import tune

from ..util.functional import accuracy_fn, preprocess_fn, f1_fn
from ..dataset import AVLDataset, seq_collate_fn
from torch_esn.optimization.ridge_regression import validate_readout


def run(
    analysis: tune.ExperimentAnalysis,
    classification: bool,
    exp_dir: str,
    device: Optional[str] = None,
):
    mode = "max"
    cfg = analysis.get_best_config(metric="eval_score", mode=mode)
    test_data = AVLDataset("all", "test", True, cfg["norm"])
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, collate_fn=seq_collate_fn()
    )

    results = {"acc": [], "f1": []}
    trial = analysis.get_best_trial(metric="eval_score", mode=mode)
    for trial in analysis.trials:
        res_path = analysis.get_best_checkpoint(
            trial, metric="eval_score", mode=mode, return_path=True
        )
        model = torch.load(os.path.join(res_path, "model.pkl"))
        acc_value = validate_readout(
            model["readout"],
            test_loader,
            score_fn=accuracy_fn,
            preprocess_fn=model["reservoir"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        f1_value = validate_readout(
            model["readout"],
            test_loader,
            score_fn=f1_fn,
            preprocess_fn=model["reservoir"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        results["acc"].append(acc_value.item())
        results["f1"].append(f1_value.item())
    json.dump(results, open(os.path.join(exp_dir, "results.json"), "w"), indent=4)
    print(results, "saved.")
