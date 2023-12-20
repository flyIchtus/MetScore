import logging
from multiprocessing import Manager, Process

from dataset import Obs_dataset, Real_dataset, Fake_dataset
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
        
        self.datasetReal = Real_dataset.fromConfig(config_data['real_dataset_config'], use_cache=use_cache)
        self.datasetFake = Fake_dataset.fromConfig(config_data['fake_dataset_config'], use_cache=use_cache)
        self.datasetObs = Obs_dataset.fromConfig(config_data['obs_dataset_config'], use_cache=use_cache)
        
        
    def run(self, index):
        logging.info(f"Running ExperimentSet {self.name}")

        for (batch_real, batch_fake, batch_obs) in zip(self.datasetReal, self.datasetFake, self.datasetObs): 
            print(batch_real.shape, batch_fake.shape, batch_obs.shape)
            for metric in self.batched_metrics:
                res = metric.calculate(batch_real, batch_fake, batch_obs)
                logging.info(f"{self.name} : Metric {metric.name} result: {res}")

        for metric in self.not_batched_metrics:
            res = metric.calculate(self.datasetReal.get_all_data(), self.datasetFake.get_all_data(),
                                   self.datasetObs.get_all_data())
            logging.info(f"Metric {metric.name} result: {res}")
        logging.info(f"ExperimentSet {index} completed")
