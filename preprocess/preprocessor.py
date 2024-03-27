from core.configurable import Configurable


class Preprocessor(Configurable):
    """
    A base class for preprocessors.

    This class provides a method to preprocess a batch of data. Subclasses should define the `process_batch` method
    to perform the desired preprocessing steps.

    Attributes:
        config_data (dict): A dictionary containing the configuration data for the preprocessor.
    """

    def __init__(self, config_data, **kwargs):
        """
        Initialize the Preprocessor instance.

        Args:
            config_data (dict): A dictionary containing the configuration data for the preprocessor.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()

    def process_batch(self, batch):
        """
        Preprocess a batch of data.

        This method should be overridden by subclasses to perform the desired preprocessing steps.

        Args:
            batch: A batch of data to be preprocessed.

        Returns:
            The preprocessed batch of data.
        """
        return batch
