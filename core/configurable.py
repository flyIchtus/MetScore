import yaml


class Configurable:
    """
    Base class for configurable objects.

    This class provides methods to load and validate configuration data from a YAML file or a dictionary. Subclasses
    should define `required_keys` and `aliases` class attributes. When a subclass is initialized with configuration
    data, the attributes defined in the configuration will be automatically set on the instance.
    """
    required_keys = []

    # aliases for the class name
    aliases = []

    @classmethod
    def fromConfig(cls, config_data, **kwargs):
        """
        Create an instance of the class from configuration data.

        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            instance: An instance of the class with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(cls, config_data)
        _check_config(cls, config_data)
        instance = cls(config_data, **kwargs)
        for key, value in config_data.items():
            setattr(instance, key, value)

        return instance

    @classmethod
    def from_typed_config(cls, config_data, **kwargs):
        """
        Create an instance of a subclass from typed configuration data.

        This method finds the correct subclass based on the 'type' key in the configuration data.

        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.
            **kwargs: Additional keyword arguments to pass to the class constructor.

        Returns:
            instance: An instance of the correct subclass with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(cls, config_data)
        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {cls.__name__}")

        def find_subclass_recursive(parent_cls):
            """
            Recursively search for the correct subclass based on the 'type' key.
            """
            for subclass in parent_cls.__subclasses__() + [parent_cls]:
                if type_name in subclass.aliases + [subclass.__name__]:
                    _check_config(subclass, config_data, typed=True)
                    instance = subclass(config_data, **config_data | kwargs)
                    for key, value in config_data.items():
                        setattr(instance, key, value)
                    return instance
                if subclass != parent_cls:
                    recursive_result = find_subclass_recursive(subclass)
                    if recursive_result:
                        return recursive_result
            return None

        result = find_subclass_recursive(cls)

        if result is not None:
            return result
        else:
            raise Exception(f"Type {type_name} non trouvé, veuillez vérifier le fichier de configuration. "
                            f"Liste des types disponibles : {[el.__name__ for el in cls.__subclasses__()]}")

    def __repr__(self):
        """
        Return a string representation of the instance.
        """
        return f"{self.__class__.__name__}({self.__dict__})"

    def _safe_open(self, config_data):
        """
        Open and load configuration data from a YAML file or return the provided dictionary.
        """
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
    """
    Check if the configuration data contains all required keys for the given class.

    Args:
        cls (type): The class to check.
        config_data (dict): The configuration data to be checked.
        typed (bool, optional): Indicates whether the check should include the 'type' key. Defaults to False.

    Raises:
        ValueError: If required keys are missing or invalid keys are present in the configuration.
    """
    required_keys = []
    if typed:
        required_keys = cls.required_keys + ['type']
    current_class = cls
    while hasattr(current_class, 'required_keys'):
        required_keys += current_class.required_keys
        current_class = current_class.__base__
    missing_keys = [key for key in required_keys if key not in config_data]
    if missing_keys:
        raise ValueError(f"Missing required keys for class {cls.__name__}: {', '.join(missing_keys)}")

    #invalid_keys = set(config_data.keys()) - set(required_keys) - set(cls.__dict__)
    #if invalid_keys:
    #    raise ValueError(f"Invalid keys in configuration for class {cls.__name__}: {', '.join(invalid_keys)}")
