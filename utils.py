import logging
import os
import yaml
from pathlib import Path

def setup_logger(log_file="pipeline.log"):
    """
    Sets up a logger that outputs to both console and a file.
    Creates the directory for the log file if it doesn't exist.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    logger = logging.getLogger("BotnetDetection")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if instantiated multiple times
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        
    return logger

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Ensure required directories exist
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    
    return config
