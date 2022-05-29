import os
import shutil
import argparse

import yaml
from dotmap import DotMap

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from models.KGCN import KGCN
from data.movielens import KGDataLoader, KGCNDataset
from sklearn.model_selection import train_test_split


def prepare_train(args):
    # Open configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    exp_name = f'{config.model}_{config.dataset}'
    os.makedirs(f'exps/logs/{exp_name}', exist_ok=True)
    writer = SummaryWriter(f'exps/logs/{exp_name}')
    os.makedirs(f'exps/configs', exist_ok=True)
    shutil.copy(args.config, f"exps/configs/{exp_name}.yaml")

    data_loader = KGDataLoader(config.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=0.2, shuffle=False, random_state=999)
    train_dataset = KGCNDataset(x_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataset = KGCNDataset(x_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    return config, data_loader, kg, train_loader, test_loader, writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/KGCN.yaml', help='Configuration YAML path')
    args = parser.parse_args()

    config, data_loader, kg, train_loader, test_loader, logger = prepare_train(args)

    num_user, num_entity, num_relation = data_loader.get_num()
    user_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = KGCN(num_user, num_entity, num_relation, kg, config, device).to(device)
    criterion = torch.nn.BCELoss()      # BPR
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.l2_weight)

    # train
    loss_list = []
    test_loss_list = []
    auc_score_list = []
    iter_per_epoch = len(train_loader)
    for epoch in range(config.n_epochs):
        running_loss = 0.0
        for i, (user_ids, item_ids, labels) in enumerate(train_loader):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            logger.add_scalar('train/loss', loss.item(), iter_per_epoch*epoch+i)

        # print train loss per every epoch
        print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
        loss_list.append(running_loss / len(train_loader))

        # evaluate per every epoch
        with torch.no_grad():
            test_loss = 0
            total_roc = 0
            for user_ids, item_ids, labels in test_loader:
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids)
                test_loss += criterion(outputs, labels).item()
                total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            print('[Epoch {}]test_loss: '.format(epoch + 1), test_loss / len(test_loader))
            test_loss_list.append(test_loss / len(test_loader))
            auc_score_list.append(total_roc / len(test_loader))
