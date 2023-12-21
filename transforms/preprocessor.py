import numpy as np

from configurable import Configurable


class Preprocessor(Configurable):

    def __init__(self, config_data):
        """
        Sample for config yml file:
        name: Preprocessor
        config_file: where/your/config/file is
        """
        super().__init__()

    def process_batch(self, batch):
        return batch
