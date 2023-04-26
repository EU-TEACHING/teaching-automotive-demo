from .esntrainable import ESNTrainable


def get_trainable(method: str):
    if method in ["ridge", "ip"]:
        return ESNTrainable
