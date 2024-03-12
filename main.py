import argparse
import concurrent
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import yaml

from core.experiment_set import ExperimentSet


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


def setup_logger(debug=False):
    """
    Configure a logger with specified console and file handlers.
    Args:
        debug (bool): Whether to enable debug logging.
        log_file (str): The name of the log file.
    Returns:
        logging.Logger: The configured logger.
    """
    console_format = '%(asctime)s - %(levelname)s - %(message)s'

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    logger = setup_logger(debug=args.debug)

    logger.info("Starting program.")
    logger.info(f"Loading configuration from {args.config}")

    # Load the configuration
    config = load_yaml(args.config)
    assert 'output_folder' in config, f"output_path must be specified in {args.config}"
    os.makedirs(config['output_folder'], exist_ok=True)
    successful_experiments = []
    failed_experiments = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_experiment, experiment_config, config['output_folder'], index) for
                   index, experiment_config in
                   enumerate(config['experiments'])]

        # Get results
        for future in concurrent.futures.as_completed(futures):
            experiment_config, success, exception = future.result()
            if success:
                successful_experiments.append(experiment_config)
            else:
                failed_experiments.append((experiment_config, exception))
                logging.exception(f"Experiment {experiment_config['name']}\n failed with exception: {exception}")

    # Log results
    success_str = '\n'.join(exp_config['name'] for exp_config in successful_experiments)
    failure_str = '\n'.join(
        f"{exp_config['name']}\n Reason for failure: {exception}" for exp_config, exception in
        failed_experiments)

    logger.info("Experiment results:")
    if success_str:
        logger.info("Successful experiments:" + success_str)
    if failure_str:
        logger.exception("Failed experiments:" + failure_str)


def run_experiment(experiment_config, output_folder,index):
    try:
        experiment_set = ExperimentSet.fromConfig(experiment_config, output_folder=output_folder)
        experiment_set.run(index)
        return (experiment_config, True, None)  # Indique que l'expérience a réussi, sans exception
    except Exception as e:
        return (experiment_config, False, e)  # Indique que l'expérience a échoué



if __name__ == "__main__":
    main()
