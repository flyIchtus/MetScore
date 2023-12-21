from abc import ABC, abstractmethod
import logging

from configurable import Configurable


class Metric(ABC):
    isBatched: bool

    def __init__(self, isBatched=False, name=None):
        self.isBatched = isBatched
        self.name = name

    @classmethod
    def fromName(cls, metric):
        logging.debug(f"Creating metric {metric}")
        for subclass in Metric.__subclasses__():
            if subclass.__name__ == metric['type']:
                # metric["is_batched"], **metric['args']
                if 'args' not in metric:
                    metric['args'] = {}
                metric_cls = subclass(metric, **metric['args'])
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
