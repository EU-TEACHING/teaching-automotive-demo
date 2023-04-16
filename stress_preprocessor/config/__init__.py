import json


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def __getattr__(self, attr):
        value = self.config[attr]
        if isinstance(value, dict):
            return ConfigDict(value)
        else:
            return value


class ConfigDict(dict):
    def __getattr__(self, attr):
        value = self[attr]
        if isinstance(value, dict):
            return ConfigDict(value)
        else:
            return value
