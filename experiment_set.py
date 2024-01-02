import logging
from multiprocessing import Manager, Process

from dataloader import DataLoader
from dataset import Dataset
from configurable import Configurable
from metrics.metrics import Metric


class ExperimentSet(Configurable):
    def __init__(self, config_data):
        """
        Sample for config yml file:
        output_folder: path to output folder
        real_dataset_config:
            name: DatasetReal
            data_folder: path to data folder
            batch_size: 32
            preprocessor_config:
                name: Preprocessor
                args:
                    arg1: value1
                    arg2: value2
        fake_dataset_config:
            name: DatasetFake
            data_folder: path to data folder
            batch_size: 32
            preprocessor_config:
                name: Preprocessor
                args:
                    arg1: value1
                    arg2: value2
        obs_dataset_config:
            name: DatasetObs
            data_folder: path to data folder
            batch_size: 32
            preprocessor_config:
                name: Preprocessor
                args:
                    arg1: value1
                    arg2: value2
        metrics_list:
            - OrographyRMSE:
                is_batched: True
                usetorch: False

        """
        self.metrics = [Metric.fromName(metric) for metric in config_data['metrics_list']]
        self.batched_metrics = [metric for metric in self.metrics if metric.isBatched]
        self.not_batched_metrics = [metric for metric in self.metrics if not metric.isBatched]
        use_cache = self.not_batched_metrics is []
        logging.info(f"Using cache: {use_cache}")
        self.dataloader = DataLoader.fromConfig(config_data['dataloaders'], use_cache=use_cache)

    def run(self, index):
        logging.info(f"Running ExperimentSet {self.name}")

        batched_metric_results = {metric.name: [] for metric in self.batched_metrics}

        for (batch_fake, batch_real, batch_obs) in self.dataloader:
            for metric in self.batched_metrics:
                res = metric.calculate(batch_fake, batch_real, batch_obs)
                batched_metric_results[metric.name].append(res)

        for metric_name, results in batched_metric_results.items():
            # TODO moyenne par batch ?
            average_result = sum(results) / len(results)
            logging.info(f"{self.name} : Metric {metric_name} result: {average_result}")

        for metric in self.not_batched_metrics:
            res = metric.calculate(self.dataloader.get_all_data())
            logging.info(f"Metric {metric.name} result: {res}")
        logging.info(f"ExperimentSet {index} completed")
