import logging

import yaml


class Configurable:

    required_keys = []

    # aliases for the class name
    aliases = []

    @classmethod
    def fromConfig(cls, config_data, **kwargs):
        config_data = cls._safe_open(cls, config_data)
        _check_config(cls, config_data)
        instance = cls(config_data, **kwargs)
        for key, value in config_data.items():
            setattr(instance, key, value)

        return instance

    @classmethod
    def from_typed_config(cls, config_data, **kwargs):

        config_data = cls._safe_open(cls, config_data)

        type_name = config_data['type']
        a = cls.__subclasses__()
        print(cls)
        print(a)
        for subclass in cls.__subclasses__() + [cls]:
            if type_name in subclass.aliases + [subclass.__name__]:
                _check_config(subclass, config_data, typed=True)
                instance = subclass(config_data, **kwargs)
                for key, value in config_data.items():
                    setattr(instance, key, value)
                return instance

        raise Exception(f"Type {type_name} non trouvé, veuillez vérifier le fichier de configuration. "
                        f"Liste des types disponibles : {[el.__name__ for el in cls.__subclasses__()]}")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def _safe_open(self, config_data):
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

        return config_data

def _check_config(cls, config_data, typed=False):
    required_keys = []
    if typed:
        required_keys = cls.required_keys + ['type']
    current_class = cls
    while hasattr(current_class, 'required_keys'):
        required_keys += current_class.required_keys
        current_class = current_class.__base__
    for key in required_keys:
        if key not in config_data:
            raise ValueError(f"Missing required key: {key} for class {cls.__name__}")