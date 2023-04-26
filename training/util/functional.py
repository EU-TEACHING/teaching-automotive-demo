import torch

from torch_esn.model.reservoir import Reservoir
from sklearn.metrics import f1_score


def preprocess_fn(reservoir: Reservoir, aggregation: str = "mean"):
    def _preprocess_fn(x):
        x = reservoir(x)
        if aggregation == "mean":
            x = x.mean(dim=0)
        elif aggregation == "sum":
            x = x.sum(dim=0)
        elif aggregation == "last":
            x = x[-1]
        return x

    return _preprocess_fn


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_pred = torch.clamp(torch.round(y_pred), 0, 1)
    y_true = torch.clamp(torch.round(y_true), 0, 1)
    return torch.sum(y_pred == y_true) / y_pred.shape[0]


def f1_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_pred = torch.clamp(torch.round(y_pred), 0, 1).numpy()
    y_true = torch.clamp(torch.round(y_true), 0, 1).numpy()
    return f1_score(y_true, y_pred)
