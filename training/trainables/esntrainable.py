import os, torch
from ray import tune
import torch.nn.functional as F

from torch_esn.model.reservoir import Reservoir
from torch_esn.wrapper.base import ESNWrapper
from torch_esn.optimization.ridge_regression import fit_and_validate_readout
from ..dataset import AVLDataset, seq_collate_fn

# from ..util.convert import convert_to_tf
from ..util.functional import accuracy_fn, f1_fn
from typing import Dict, Optional, Union


class ESNTrainable(tune.Trainable):
    def setup(self, config: Dict):
        self.model, self.states = None, None

    def step(self):
        config = self.get_config()
        train_data, eval_data, test_data = (
            AVLDataset(split="train", **config["data"]),
            AVLDataset(split="eval", **config["data"]),
            AVLDataset(split="test", **config["data"]),
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, collate_fn=seq_collate_fn()
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=1, shuffle=False, collate_fn=seq_collate_fn()
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False, collate_fn=seq_collate_fn()
        )
        wrapper = ESNWrapper()
        cfg = self.get_config()

        model = {"reservoir": Reservoir(**cfg["reservoir"])}
        if cfg["data"]["classification"]:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.MSELoss()

        if "ip" in cfg["method"]:
            model["reservoir"] = wrapper.ip_step(
                train_loader, model["reservoir"], **cfg["ip"]
            ).eval()

        model["readout"], model["l2"], _ = fit_and_validate_readout(
            train_loader,
            eval_loader,
            l2_values=cfg["l2"],
            weights=cfg["weights"] if "weights" in cfg else None,
            score_fn=loss,
            mode="min",
            preprocess_fn=model["reservoir"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        acc_train, loss_train = score_model(train_loader, model)
        acc_eval, loss_eval = score_model(eval_loader, model)
        acc_test, loss_test = score_model(test_loader, model)

        self.model = model
        return {
            "train_accuracy": acc_train,
            "eval_accuracy": acc_eval,
            "test_accuracy": acc_test,
            "train_loss": loss_train,
            "eval_loss": loss_eval,
            "test_loss": loss_test,
        }

    def save_checkpoint(self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        torch.save(self.model, os.path.join(checkpoint_dir, "model.pkl"))
        # convert_to_tf(self.model, os.path.join(checkpoint_dir, "tf_model"))
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir: Union[Dict, str]):
        self.model = torch.load(os.path.join(checkpoint_dir, "model.pkl"))

    def reset_config(self, new_config):
        return True


@torch.no_grad()
def score_model(loader: torch.utils.data.DataLoader, model: Dict):
    acc, mse, n_samples = 0, 0, 0
    mse_loss = torch.nn.MSELoss()
    for x, y in loader:
        x = x.to("cuda" if torch.cuda.is_available() else "cpu")
        y = y.to("cuda" if torch.cuda.is_available() else "cpu")
        x = model["reservoir"](x)
        y_pred = F.linear(x.to(model["readout"]), model["readout"])
        y, y_pred = y.squeeze(), y_pred.squeeze()
        acc_b, mse_b = (
            accuracy_fn(y, y_pred),
            mse_loss(y, y_pred),
        )
        # f1 += f1_b * y.shape[0]
        acc += acc_b * y.shape[0]
        mse += mse_b * y.shape[0]
        n_samples += y.shape[0]
    # return (f1 / n_samples).item(), (acc / n_samples).item(), (mse / n_samples).item()
    return (acc / n_samples).item(), (mse / n_samples).item()
