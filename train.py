import logging
import argparse

import numpy as np

import torch
from sklearn.metrics import roc_auc_score, f1_score

from dataloader import load_data
from utils import set_random_seed, prepare_train, build_model_optim


def ctr_eval(args, model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size, 2]
        scores = model.get_scores(data[start:start + args.batch_size])
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1


def train(config, datasets, writer, device):
    max_auc = 0.0
    train_data, valid_data, model_define_args = datasets
    ipe = np.ceil(train_data.shape[0] / config.batch_size)
    model, optim = build_model_optim(config, device, model_define_args)

    for epoch in range(config.n_epochs):
        np.random.shuffle(train_data)
        start = 0

        while start < train_data.shape[0]:
            data = train_data[start:start + config.batch_size, :]

            loss = model(data)

            optim.zero_grad()
            loss.backward()
            optim.step()

            start += config.batch_size
            writer.add_scalar('bce_loss', loss.item(), (epoch * ipe) + (start // config.batch_size))

        # Evaluate every epoch
        eval_auc, eval_f1 = ctr_eval(config, model, valid_data)

        # Log
        writer.add_scalar('valid/auc', eval_auc, epoch)
        writer.add_scalar('valid/f1', eval_f1, epoch)
        ctr_info = 'epoch %.2d    eval auc: %.4f f1: %.4f'
        logging.info(ctr_info, epoch, eval_auc, eval_f1)

        # Save best weight
        if max_auc < eval_auc:
            max_auc = eval_auc
            torch.save(model.state_dict(), f'exps/ckpt/{config.exp_name}_best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/CKAN_music.yaml', help='Configuration YAML path')
    parser.add_argument('--topK', default=True, action='store_true', help='Do top-k evaluation')
    args = parser.parse_args()

    set_random_seed(712933, 2021)
    config, writer = prepare_train(args.config)
    config.topK = args.topK

    # Load dataset
    datasets = load_data(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train(config, datasets, writer, device)



