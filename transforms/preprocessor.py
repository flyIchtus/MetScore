import numpy as np

from configurable import Configurable


class Preprocessor(Configurable):

    def __init__(self, config_data):
        """
        Sample for config yml file:
        name: Preprocessor
        args:
            arg1: value1
            arg2: value2
        """
        super().__init__()

    def process_batch(self, batch):
        return batch
