import logging
import numpy as np
from tqdm import tqdm

from dataloader import DateDataloader
from configurable import Configurable
from metrics.metrics import Metric


class ExperimentSet(Configurable):
    required_keys = ['name', 'output_path', 'dataloaders', 'metrics']

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
        data_metrics_config = {'var_indices': config_data['dataloaders']['fake_dataset_config']['var_indices'],
                                 'real_var_indices': config_data['dataloaders']['real_dataset_config']['var_indices'],
                                 'obs_var_indices': config_data['dataloaders']['obs_dataset_config']['var_indices'],
                                 }
        metrics_config = config_data['metrics']
        self.metrics = [Metric.from_typed_config(metric | metrics_config['args'] | data_metrics_config) for metric in
                        metrics_config['metrics_list']]
        self.batched_metrics = [metric for metric in self.metrics if metric.isBatched]
        self.not_batched_metrics = [metric for metric in self.metrics if not metric.isBatched]
        self.config_data = config_data
        use_cache = self.not_batched_metrics is not []
        logging.info(f"Using cache: {use_cache}")
        self.dataloader = DateDataloader.fromConfig(config_data['dataloaders'], use_cache=use_cache)

    def run(self, index):
        logging.info(f"Running ExperimentSet {self.name}")

        batched_metric_results = {metric.name: [] for metric in self.batched_metrics}
        # logging.debug(batched_metric_results)

        for (batch_fake, batch_real, batch_obs) in tqdm(self.dataloader, desc="Processing batches"):
            logging.debug(f"Shape : fake:{batch_fake.shape}, real:{batch_real.shape}, obs:{batch_obs.shape}")
            for metric in self.batched_metrics:
                res = metric.calculate(batch_fake, batch_real, batch_obs)
                batched_metric_results[metric.name].append(res)
                # logging.debug(batched_metric_results)

        for metric_name, results in batched_metric_results.items():
            # TODO moyenne par batch ?
            results_np = np.array(results)
            np.save(self.config_data['output_path'] + metric_name, results_np)
            # logging.info(f"{self.name} : Metric {metric_name} result: {average_result}")

        if self.not_batched_metrics:
            real_data, fake_data, obs_data = self.dataloader.get_all_data()
            for metric in tqdm(self.not_batched_metrics, desc="Calculating non-batched metrics"):
                res = metric.calculate(real_data, fake_data, obs_data)
                if res.size < 25:
                    logging.info(f"Metric {metric.name} result: {res}")
                else:
                    logging.info(f"Metric {metric.name} : too long result to log")

        logging.info(f"ExperimentSet {index} completed")
