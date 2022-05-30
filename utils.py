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
    writer = SummaryWriter(f'exps/logs/{exp_name}')
    os.makedirs(f'exps/configs', exist_ok=True)
    shutil.copy(config_path, f"exps/configs/{exp_name}.yaml")
    return config, writer


def build_model_optim_losses(config, n_entity, n_relation):
    model, optimizer, loss_func = None, None, None

    if config.model == 'CKAN':
        model = models.__dict__['Parallel'](models.__dict__['CKAN'](config, n_entity, n_relation)).cuda()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.lr),
            weight_decay=float(config.l2_weight),
        )
        loss_func = nn.BCELoss()
    elif config.model == 'KGIN'
    return model, optimizer, loss_func
