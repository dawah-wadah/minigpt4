"""
Helper Functions
"""

import torch


def disabled_train(*args, **kwargs):
    """
    Disable training mode for a module.
    """
    pass


def autocast(*args, **kwargs):
    """
    Autocast context manager for mixed-precision training.
    """
    return torch.cuda.amp.autocast(*args, **kwargs)
