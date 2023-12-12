from abc import ABC, abstractmethod


class Metric(ABC, object):
    def __init__(self, is_batched):
        self.isBatched = is_batched

    @classmethod
    def fromName(cls,metric):
        for subclass in Metric.__subclasses__():
            if subclass.__name__ == metric['name']:
                metric_cls = subclass(metric["is_batched"], **metric['args'])
                return metric_cls

        raise Exception(f"Metric {metric['name']} not found")


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
