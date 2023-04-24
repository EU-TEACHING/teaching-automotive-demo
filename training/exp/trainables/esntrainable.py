import os, torch
from ray import tune
import torch.nn.functional as F

from torch_esn.model.reservoir import Reservoir
from torch_esn.wrapper.base import ESNWrapper
from exp.util.functional import accuracy_fn, f1_fn
from torch_esn.optimization.ridge_regression import fit_and_validate_readout
from ..dataset import AVLDataset, seq_collate_fn
from typing import Dict, Optional, Union


class ESNTrainable(tune.Trainable):
    def setup(self, config: Dict):
        self.train_data, self.eval_data, self.test_data = (
            AVLDataset(scenarios="train", **config["data"]),
            AVLDataset(scenarios="eval", **config["data"]),
            AVLDataset(scenarios="test", **config["data"]),
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=1, shuffle=False, collate_fn=seq_collate_fn()
        )
        self.eval_loader = torch.utils.data.DataLoader(
            self.eval_data, batch_size=1, shuffle=False, collate_fn=seq_collate_fn()
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=1, shuffle=False, collate_fn=seq_collate_fn()
        )
        self.wrapper = ESNWrapper()
        self.model, self.states = None, None

    def step(self):
        cfg = self.get_config()
        model = {"reservoir": Reservoir(**cfg["reservoir"])}

        if "ip" in cfg["method"]:
            model["reservoir"] = self.wrapper.ip_step(
                self.train_loader, model["reservoir"], **cfg["ip"]
            ).eval()

        model["readout"], model["l2"], _ = fit_and_validate_readout(
            self.train_loader,
            self.eval_loader,
            l2_values=cfg["l2"],
            weights=cfg["weights"] if "weights" in cfg else None,
            score_fn=f1_fn,
            mode="max",
            preprocess_fn=model["reservoir"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        f1_train, acc_train = score_model(self.train_loader, model)
        f1_eval, acc_eval = score_model(self.eval_loader, model)
        f1_test, acc_test = score_model(self.test_loader, model)

        self.model = model
        return {
            "train_score": f1_train,
            "eval_score": f1_eval,
            "test_score": f1_test,
            "train_accuracy": acc_train,
            "eval_accuracy": acc_eval,
            "test_accuracy": acc_test,
        }

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        torch.save(self.model, os.path.join(checkpoint_dir, "model.pkl"))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: Union[Dict, str]):
        self.model = torch.load(os.path.join(checkpoint_dir, "model.pkl"))

    def reset_config(self, new_config):
        return True


@torch.no_grad()
def score_model(loader: torch.utils.data.DataLoader, model: Dict):
    f1, acc, n_samples = 0, 0, 0
    for x, y in loader:
        x = x.to("cuda" if torch.cuda.is_available() else "cpu")
        y = y.to("cuda" if torch.cuda.is_available() else "cpu")
        x = model["reservoir"](x)
        y_pred = F.linear(x.to(model["readout"]), model["readout"])
        y, y_pred = y.squeeze(), y_pred.squeeze()
        f1_b, acc_b = f1_fn(y, y_pred), accuracy_fn(y, y_pred)
        f1 += f1_b * y.shape[0]
        acc += acc_b * y.shape[0]
        n_samples += y.shape[0]
    return (f1 / n_samples).item(), (acc / n_samples).item()
