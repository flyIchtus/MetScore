import logging

import yaml


class Configurable:
    @classmethod
    def fromConfig(cls, config_data):
        if not isinstance(config_data, (str, dict)):
            raise TypeError("Invalid type for config_data. Expected str (file path) or dict.")

        if isinstance(config_data, str):
            try:
                with open(config_data, 'r') as file:
                    config_data = yaml.safe_load(file)
            except Exception as e:
                raise IOError(f"Error loading config file: {e}")

        if not isinstance(config_data, dict):
            raise TypeError("Invalid type for config_data. Expected dict after loading from YAML.")

        instance = cls(config_data)
        for key, value in config_data.items():
            logging.info(f"Setting {key} to {value}")
            setattr(instance, key, value)

        return instance
