import logging
import os

import pickle
import numpy as np
import yaml
from tqdm import tqdm

from core.configurable import Configurable
from core.dataloader import DataLoader
from metrics.metrics import Metric

class ExperimentSet(Configurable):
    required_keys = ['name', 'dataloaders', 'metrics']

    def __init__(self, config_data, output_folder='.'):
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
        self.config_data = config_data
        self.current_path = os.path.join(output_folder, config_data['name'])

    def _init_experiment(self):
        data_metrics_config = {'var_indices': self.config_data['dataloaders']['fake_dataset_config']['var_indices'],
                               'real_var_indices': self.config_data['dataloaders']['real_dataset_config']['var_indices'],
                               'obs_var_indices': self.config_data['dataloaders']['obs_dataset_config']['var_indices'],
                               }
        metrics_config = self.config_data['metrics']
        self.metrics = [Metric.from_typed_config(metric | metrics_config['args'] | data_metrics_config) for metric in
                        metrics_config['metrics_list']]
        self.batched_metrics = [metric for metric in self.metrics if metric.isBatched]
        self.not_batched_metrics = [metric for metric in self.metrics if not metric.isBatched]
        use_cache = self.not_batched_metrics is not []
        logging.info(f"Using cache: {use_cache}")
        self.config_data['dataloaders'].update(self.config_data)
        self.dataloader = DataLoader.from_typed_config(self.config_data['dataloaders'], use_cache=use_cache)

    def _prep_folder(self):
        if not os.path.exists(self.current_path):
            os.makedirs(self.current_path)
        else:
            logging.warning(f"Folder {self.current_path} already exists. Checking for existing config file.")
            try:
                with open(os.path.join(self.current_path, 'config.yml'), 'r') as f:
                    existing_config_data = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Folder {self.current_path} already exists, but no config file found."
                                        f"Please remove the folder or rename it or change {self.config_data['name']} in"
                                        f"config.")
            existing_metrics = existing_config_data['metrics']['metrics_list']
            new_metrics = self.config_data['metrics']['metrics_list']

            existing_config_data['metrics']['metrics_list'], self.config_data['metrics']['metrics_list'] = None, None

            if existing_config_data != self.config_data:
                raise ValueError(f"Folder {self.current_path} already exists, but config files do not match."
                                 f"Existing config: {existing_config_data}. New config: {self.config_data}")
            else:
                if existing_metrics == new_metrics:
                    raise ValueError(f"Folder {self.current_path} already exists, and metrics and config are the same.")
                else:
                    current_metrics = []
                    for metric in new_metrics:
                        if metric not in existing_metrics:
                            current_metrics.append(metric)
                            logging.debug(f"Added new metric: {metric}")
                self.config_data['metrics']['metrics_list'] = current_metrics
                logging.info(f"Folder {self.current_path} already exists. Running only new metrics: {current_metrics}")


    def run(self, index):
        logging.info(f"Running ExperimentSet {self.name}")

        self._prep_folder()

        self._init_experiment()

        batched_metric_results = {metric.name: [] for metric in self.batched_metrics}

        for (batch_fake, batch_real, batch_obs) in tqdm(self.dataloader, desc=f"{self.name}: Processing batches"):
            for metric in self.batched_metrics:
                logging.debug(f"Running Metric {type(metric)}")
                res = metric.calculate(batch_real, batch_fake, batch_obs)
                batched_metric_results[metric.name].append(res)

        for metric_name, results in tqdm(batched_metric_results.items(), desc=f"{self.name}: Saving batched results"):
            results_np = np.array(results, dtype=np.float32)
            np.save(os.path.join(self.current_path,  metric_name) + '.npy', results_np)
            logging.debug(f"{self.name} : Metric {metric_name} shape result: {results_np.shape}")

        if self.not_batched_metrics:
            real_data, fake_data, obs_data = self.dataloader.get_all_data()
            for metric in tqdm(self.not_batched_metrics, desc=f"{self.name}: Calculating non-batched metrics"):
                logging.debug(f"Running Metric {type(metric)}")
                results = metric.calculate(real_data, fake_data, obs_data)
                
                if type(results)==dict:
                    with open(os.path.join(self.current_path, metric.name) + '.p','wb') as f:
                        pickle.dump(results, f)
                else:
                    logging.debug(f"\nCalculated Metric. Result shape {results[0].shape}")
                    results_np = np.array(results, dtype=np.float32)
                    np.save(os.path.join(self.current_path, metric.name) + '.npy', results_np)
                    logging.debug(results_np.size)
                    if results_np.size < 25:
                        logging.info(f"Metric {metric.name} result: {results_np}")
                    else:
                        logging.info(f"Metric {metric.name} : too long result to log")

        logging.info(f"ExperimentSet {index} completed")
