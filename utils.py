import os
import random
import shutil

import yaml
import numpy as np
from dotmap import DotMap

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import models


def set_random_seed(np_seed, torch_seed):
    random.seed(np_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_train(config_path):
    # Open configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    exp_name = f'{config.model}_{config.dataset}'
    config.exp_name = exp_name
    os.makedirs(f'exps/logs/{exp_name}', exist_ok=True)
    os.makedirs(f'exps/ckpt', exist_ok=True)
    writer = SummaryWriter(f'exps/logs/{exp_name}')
    return config, writer


def build_model_optim(config, device, model_define_args):
    model = models.__dict__[config.model](config, device, model_define_args).to(device)
    if config.model == 'CKAN':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr), weight_decay=float(config.l2_weight))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    return model, optimizer
