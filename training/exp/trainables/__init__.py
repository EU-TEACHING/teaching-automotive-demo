from .esntrainable import ESNTrainable
from .sklearntrainable import ScikitTrainable
from .rnntrainable import RNNTrainable


def get_trainable(method: str):
    if method in ["ridge", "ip"]:
        return ESNTrainable
    elif method in ["dt"]:
        return ScikitTrainable
    elif method in ["gru"]:
        return RNNTrainable
