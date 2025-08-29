import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import sys
import logging
import os
from time import time
from copy import deepcopy
from abc import ABC, abstractmethod

from trainer import EarlyStopper, EpochStopper


def get_optimizer(model, name='adamw', **params):
    available_list = {
        'adam': optim.Adam, 
        'sgd': optim.SGD, 
        'adamw': optim.AdamW,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")

    selected = available_list[name]
    default_params = {'lr': 0.001}
    default_params.update(params)
    return selected(model.parameters(), **default_params)


def get_scheduler(optimizer, name='plateau', **params):
    available_list = {
        'step': optim.lr_scheduler.StepLR, 
        'multi_step': optim.lr_scheduler.MultiStepLR, 
        'exponential': optim.lr_scheduler.ExponentialLR, 
        'cosine': optim.lr_scheduler.CosineAnnealingLR, 
        'plateau': optim.lr_scheduler.ReduceLROnPlateau, 
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")
    
    selected = available_list[name]
    default_params = {}
    default_params.update(params)
    return selected(optimizer, **default_params)


def get_stopper(name='early_stop', **params):
    available_list = {
        'early_stop': EarlyStopper,
        'stop': EpochStopper,
    }
    name = name.lower()
    if name not in available_list:
        available_names = list(available_list.keys())
        raise ValueError(f"Unknown name: {name}. Available names: {available_names}")
    
    selected = available_list[name]
    default_params = {}
    default_params.update(params)
    return selected(**default_params)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'experiment.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


