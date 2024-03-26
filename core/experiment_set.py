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
    """
    A class to manage and run experiments using specified dataloaders and metrics.

    Attributes:
        config_data (dict): The configuration data for the experiment set.
        current_path (str): The path to the output folder for the experiment set.
        metrics (List[Metric]): The list of metrics to be used in the experiment set.
        batched_metrics (List[Metric]): The list of batched metrics.
        not_batched_metrics (List[Metric]): The list of non-batched metrics.
        dataloader (DataLoader): The dataloader instance to load data for the experiment set.
    """

    required_keys = ['name', 'dataloaders', 'metrics']

    def __init__(self, config_data, output_folder='.'):
        """
        Initialize the ExperimentSet instance.

        Args:
            config_data (dict): The configuration data for the experiment set.
            output_folder (str, optional): The path to the output folder. Defaults to '.'.
        """
        self.config_data = config_data
        self.current_path = os.path.join(output_folder, config_data['name'])

    def _init_experiment(self):
        """
        Initialize the metrics and dataloader for the experiment set.
        """
        # Prepare data_metrics_config by merging var_indices from fake, real, and obs datasets
        data_metrics_config = {'var_indices': self.config_data['dataloaders']['fake_dataset_config']['var_indices'],
                               'real_var_indices': self.config_data['dataloaders']['real_dataset_config'][
                                   'var_indices'],
                               'obs_var_indices': self.config_data['dataloaders']['obs_dataset_config']['var_indices'],
                               }
        metrics_config = self.config_data['metrics']

        # Initialize metrics using Metric.from_typed_config() and store them in self.metrics
        self.metrics = [Metric.from_typed_config(metric | metrics_config['args'] | data_metrics_config) for metric in
                        metrics_config['metrics_list']]

        # Separate batched and non-batched metrics
        self.batched_metrics = [metric for metric in self.metrics if metric.isBatched]
        self.not_batched_metrics = [metric for metric in self.metrics if not metric.isBatched]

        # Determine if cache should be used based on the presence of non-batched metrics
        use_cache = self.not_batched_metrics is not []
        logging.info(f"Using cache: {use_cache}")

        # Update the dataloaders configuration with the main configuration data
        self.config_data['dataloaders'].update(self.config_data)
        self.dataloader = DataLoader.from_typed_config(self.config_data['dataloaders'], use_cache=use_cache)


    def _prep_folder(self):
        """
        Prepare the output folder for the experiment set.
        """
        if not os.path.exists(self.current_path):
            os.makedirs(self.current_path)
            with open(os.path.join(self.current_path, 'config.yml'), 'w') as f:
                f.write(yaml.dump(self.config_data))
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

            existing_config_data['metrics']['metrics_list'], self.config_data['metrics'][
                'metrics_list'] = None, None

            if existing_config_data != self.config_data:
                raise ValueError(f"Folder {self.current_path} already exists, but config files do not match."
                                 f"Existing config: {existing_config_data}. New config: {self.config_data}")
            else:
                existing_names = {m['name'] for m in existing_metrics}
                new_names = {m['name'] for m in new_metrics}
                intersect = existing_names.intersection(new_names)
                if len(intersect)>0:
                    raise ValueError(
                        f"Folder {self.current_path} already exists, metrics coincide (common metric names : {intersect}) and config are the same.")
                else:
                    current_metrics = []
                    for metric in new_metrics:
                        current_metrics.append(metric)
                        logging.debug(f"Added new metric: {metric}")
                self.config_data['metrics']['metrics_list'] = current_metrics + existing_metrics
                with open(os.path.join(self.current_path, 'config.yml'), 'w') as f:
                    f.write(yaml.dump(self.config_data))
                self.config_data['metrics']['metrics_list'] = current_metrics
                logging.info(
                    f"Folder {self.current_path} already exists. Running only new metrics: {current_metrics}")

        

    def run(self, index):
        """
        Run the experiment set with the specified dataloaders and metrics.

        Args:
            index (int): The index of the experiment set.
        """
        logging.info(f"Running ExperimentSet {self.name}")

        self._prep_folder()

        self._init_experiment()

        batched_metric_results = {metric.name: [] for metric in self.batched_metrics}

        # Iterate through the dataloader and process batches
        for (batch_fake, batch_real, batch_obs) in tqdm(self.dataloader, desc=f"{self.name}: Processing batches"):
            for metric in self.batched_metrics:
                logging.debug(f"Running Metric {type(metric)}")
                # Calculate the metric for the current batch
                res = metric.calculate(batch_real, batch_fake, batch_obs)
                # Append the result to the corresponding list in batched_metric_results
                batched_metric_results[metric.name].append(res)

        # Save batched metric results as numpy arrays
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
