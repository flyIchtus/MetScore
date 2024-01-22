import logging
import numpy as np
from multiprocessing import Manager, Process

from dataloader import DateDataloader
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
        self.config_data = config_data
        use_cache = self.not_batched_metrics is []
        logging.info(f"Using cache: {use_cache}")

        self.dataloader = DateDataloader.fromConfig(config_data['dataloaders'], use_cache=use_cache)

    def run(self, index):
        logging.info(f"Running ExperimentSet {self.name}")

        batched_metric_results = {metric.names[0]: [] for metric in self.batched_metrics}
        # print(batched_metric_results)

        threshold = np.zeros((2,6))
        threshold[0] = self.config_data['threshold_ff']
        threshold[1] = self.config_data['threshold_t2m']
        for (batch_fake, batch_real, batch_obs) in self.dataloader:
            #print(batch_fake.shape, batch_real.shape)
            for metric in self.batched_metrics:
                res = metric.calculate(batch_fake, batch_real, batch_obs, self.config_data['debiasing'], self.config_data['debiasing_mode'], self.config_data['conditioning_members'], threshold)
                batched_metric_results[metric.names[0]].append(res)
                # print(batched_metric_results)

        for metric_name, results in batched_metric_results.items():
            # TODO moyenne par batch ?
            average_result = sum(results) / len(results)
            logging.info(f"{self.name} : Metric {metric_name} result: {average_result}")
        
        if self.not_batched_metrics is not []:
            r, f, o = self.dataloader.get_all_data()
            r,f = self.dataloader.randomize_and_cut(r,f)
        for metric in self.not_batched_metrics:
            res = metric.calculate(r, f, o)
            if res.size<25:
                logging.info(f"Metric {metric.names} result: {res}")
            else:
                logging.info(f"Metric {metric.names} : too long result to print")
        logging.info(f"ExperimentSet {index} completed")
