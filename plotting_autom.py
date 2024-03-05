
import pickle
import copy
import os

import argparse
import logging
import sys
import yaml

import plotting_functions as plf
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
    parser = argparse.ArgumentParser(description="Plot experiments.")
    parser.add_argument("--config", type=str, default="plotting_autom.yml", help="Path to the configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Configure logger
    logger = setup_logger(debug=args.debug)

    # Log the start of the program
    logger.info("Starting program.")

    try:
        logger.info(f"Loading configuration from {args.config}")

        # Load the configuration
        config = load_yaml(args.config)

        experiments = { k['name'] : k['folder'] for k in config['experiments'] }
        
        assert 'output_plots' in config, f"output_plots path must be specified in {args.config}"
        if not os.path.exists(config['output_plots']):
            os.mkdir(config['output_plots'])
        

        metrics = config['metrics']

        for metr_idx, metric in enumerate(metrics):
           logger.info(f"metric {metric['name']} being plotted")
           os.mkdirs(os.path.join(config['output_plots'],metric['folder']),exist_ok=True)

           plot_func = getattr(plf, f"plot_{metric['name']}")
           plot_func(experiments, metric, config)

        logger.info("Program completed.")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
