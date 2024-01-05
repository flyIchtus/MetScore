from abc import ABC, abstractmethod
import logging

from configurable import Configurable


class Metric(ABC):
    isBatched: bool

    def __init__(self, isBatched=False, names=['metric'], var_channel=2, var_indices=[0,1,2], real_var_indices=[1,2,3]):
        self.isBatched = isBatched
        self.names = names
        # which channel of the data samples the variable indices are gonna be on. It should be either
        self.var_channel = var_channel 
        self.var_indices = var_indices # which indices to select (for different variables)
        self.real_var_indices = real_var_indices # which indices to select (for different variables, in case of real data)


    @classmethod
    def fromName(cls, metric):
        logging.debug(f"Creating metric {metric}")
        for subclass in Metric.__subclasses__():
            if subclass.__name__ == metric['type']:
                # metric["is_batched"], **metric['args']
                if 'args' not in metric:
                    metric['args'] = {}
                metric_cls = subclass(**metric['args'])
                print(metric_cls._preprocess, metric_cls._calculateCore)
                return metric_cls

        raise Exception(f"Metric {metric['type']} not found, check config file. "
                        f"List of available metrics: {Metric.__subclasses__()}")

    def calculate(self, *args, **kwargs):

        processed_data = self._preprocess(*args, **kwargs)
        result = self._calculateCore(processed_data)

        return result

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        # Common preprocessing logic for all metrics
        pass

    @abstractmethod
    def _calculateCore(self, processed_data):
        # Specific calculation logic for each metric
        pass

    def isBatched(self):
        return self.isBatched
