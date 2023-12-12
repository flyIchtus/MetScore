import logging

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
        print(Metric.__subclasses__())
        self.metrics = [Metric.fromName(metric) for metric in config_data['metrics_list']]
        print(self.metrics)
        self.batched_metrics = [metric for metric in self.metrics if metric.isBatched]
        self.not_batched_metrics = [metric for metric in self.metrics if not metric.isBatched]

        self.datasetReal = Dataset.fromConfig( config_data['real_dataset_config'])
        self.datasetFake = Dataset.fromConfig( config_data['fake_dataset_config'])
        self.datasetObs = Dataset.fromConfig( config_data['obs_dataset_config'])

    def run(self, index):
        logging.info(f"Running ExperimentSet {index}")

        for (batch_real, batch_fake, batch_obs) in zip(self.datasetReal, self.datasetFake, self.datasetObs):
            for metric in self.batched_metrics:
                res = metric.calculate(batch_real, batch_fake, batch_obs)


        for metric in self.not_batched_metrics:
                res = metric.calculate(self.datasetReal.get_all_data(), self.datasetFake.get_all_data(), self.datasetObs.get_all_data())


        logging.info(f"ExperimentSet {index} completed")
