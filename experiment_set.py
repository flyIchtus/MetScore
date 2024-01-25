import logging
import numpy as np
from tqdm import tqdm

from dataloader import DateDataloader
from configurable import Configurable
from metrics.metrics import Metric

class ExperimentSet(Configurable):

    required_keys = ['name', 'output_path', 'dataloaders', 'metrics_list']

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
                ...
        fake_dataset_config:
            name: DatasetFake
            data_folder: path to data folder
            batch_size: 32
            preprocessor_config:
                name: Preprocessor
                ...
        obs_dataset_config:
            name: DatasetObs
            data_folder: path to data folder
            batch_size: 32
            preprocessor_config:
                name: Preprocessor
                ...
        metrics_list:
            - OrographyRMSE:
                is_batched: True
                usetorch: False

        """
        config_data_without_name = config_data.copy()
        del config_data_without_name['name']
        self.metrics = [Metric.from_typed_config(metric, **config_data_without_name) for metric in config_data['metrics_list']]
        self.batched_metrics = [metric for metric in self.metrics if metric.isBatched]
        self.not_batched_metrics = [metric for metric in self.metrics if not metric.isBatched]
        self.config_data = config_data
        use_cache = self.not_batched_metrics is not []
        logging.info(f"Using cache: {use_cache}")
        self.dataloader = DateDataloader.fromConfig(config_data['dataloaders'], use_cache=use_cache)

    def run(self, index):
        logging.info(f"Running ExperimentSet {self.name}")

        batched_metric_results = {metric.names[0]: [] for metric in self.batched_metrics}
        # logging.debug(batched_metric_results)

        threshold = np.zeros((2,6))
        threshold[0] = self.config_data['threshold_ff']
        threshold[1] = self.config_data['threshold_t2m']
        for (batch_fake, batch_real, batch_obs) in tqdm(self.dataloader, desc="Processing batches"):
            logging.debug(f"Shape : fake:{batch_fake.shape}, real:{batch_real.shape}, obs:{batch_obs.shape}")
            for metric in self.batched_metrics:
                res = metric.calculate(batch_fake, batch_real, batch_obs, self.config_data['debiasing'],
                                       self.config_data['debiasing_mode'], self.config_data['conditioning_members'],
                                       threshold)
                batched_metric_results[metric.names[0]].append(res)
                
                
                # logging.debug(batched_metric_results)

        for metric_name, results in batched_metric_results.items():
            # TODO moyenne par batch ?
            results_np = np.array(results, dtype=np.float32)
            np.save(self.config_data['output_path'] + metric_name, results_np)
            logging.info(f"{self.name} : Metric {metric_name} shape result: {results_np.shape}")


        if self.not_batched_metrics:
            real_data, fake_data, obs_data = self.dataloader.get_all_data()
            for metric in tqdm(self.not_batched_metrics, desc="Calculating non-batched metrics"):
                res = metric.calculate(real_data, fake_data, obs_data)
                if res.size < 25:
                    logging.info(f"Metric {metric.names} result: {res}")
                else:
                    logging.info(f"Metric {metric.names} : too long result to log")

        logging.info(f"ExperimentSet {index} completed")
