"""
Configuration utilities for the code review model
"""
import os
import json

def save_config(config, config_path):
    """
    Save a configuration dictionary to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(config_path):
    """
    Load a configuration dictionary from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_default_config():
    """
    Get default configuration for the code review model.
    
    Returns:
        dict: Default configuration
    """
    return {
        "model_name": "Salesforce/codet5-base",
        "max_length": 256,
        "batch_size": 2,
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "output_dir": "./code_review_model",
        "final_model_dir": "./fine_tuned_code_review_model",
    }
