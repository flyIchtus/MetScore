import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

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
    try:
        logger.info(f"Loading configuration from {args.config}")

        # Load the configuration
        config = load_yaml(args.config)
        assert 'output_folder' in config, f"output_path must be specified in {args.config}"
        os.makedirs(config['output_folder'], exist_ok=True)

        # Run each experiment in parallel
        with ThreadPoolExecutor() as executor:
            for experiment_config in config["experiments"]:
                experiment_set = ExperimentSet.fromConfig(experiment_config, output_folder=config['output_folder'])
                executor.map(experiment_set.run, range(len(config["experiments"]))[-1:])

        logger.info("Program completed.")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
