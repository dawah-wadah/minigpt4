import yaml


def load_config(path):
    """
    Load a YAML configuration file.
    
    Args:
        path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Loaded configuration as a dictionary.
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, path):
    """
    Save a dictionary as a YAML configuration file.
    
    Args:
        config (dict): Configuration to be saved.
        path (str): Path to save the YAML configuration file.
    """
    with open(path, 'w') as file:
        yaml.dump(config, file)
