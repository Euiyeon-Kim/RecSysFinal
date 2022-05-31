import logging
import argparse

import numpy as np

import torch
from sklearn.metrics import roc_auc_score, f1_score

from dataloader import load_data
from utils import set_random_seed, prepare_train, build_model_optim_losses


def ctr_eval(args, model, data, user_triple_set, item_triple_set):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size, 2]
        scores = model(data, user_triple_set, item_triple_set, start, start + args.batch_size)
        scores = scores.detach().cpu().numpy()
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


def kgin_ctr_eval(args, model, data):
    auc_list = []
    f1_list = []
    model.eval()
    entity_gcn_emb, user_gcn_emb = model.generate()

    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size, 2]
        u_ids = torch.LongTensor(data[start:start + args.batch_size, 0]).cuda()
        u_g_embeddings = user_gcn_emb[u_ids]
        i_ids = torch.LongTensor(data[start:start + args.batch_size, 1]).cuda()
        i_g_embddings = entity_gcn_emb[i_ids]

        i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()
        print(i_rate_batch.shape)
        print(np.min(i_rate_batch), np.max(i_rate_batch))

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


def topk_eval(args, model, train_data, test_data, user_triple_set, item_triple_set, writer, epoch):
    def _show_recall_info(recall_prec_zip):
        res = ""
        for k, rec, prec in recall_prec_zip:
            res += "recall@%d:%.4f  " % (k, rec)
            res += "precision@%d:%.4f  " % (k, prec)
            writer.add_scalar(f'test/recall@{k}', rec, epoch)
            writer.add_scalar(f'test/precision@{k}', prec, epoch)
        logging.info(res)

    def _get_user_record(data, is_train):
        user_history_dict = dict()
        for rating in data:
            user = rating[0]
            item = rating[1]
            label = rating[2]
            if is_train or label == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = set()
                user_history_dict[user].add(item)
        return user_history_dict

    def _get_topk_feed_data(user, items):
        res = list()
        for item in items:
            res.append([user, item])
        return np.array(res)

    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    prec_list = {k: [] for k in k_list}

    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)

    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            scores = model(input_data, user_triple_set, item_triple_set, 0, args.batch_size)
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size
        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            scores = model(input_data, user_triple_set, item_triple_set, 0, args.batch_size)
            for item, score in zip(res_items, scores):
                item_score_map[item] = score
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
            prec_list[k].append(hit_num / k)

    model.train()
    recall = [np.mean(recall_list[k]) for k in k_list]
    precision = [np.mean(prec_list[k]) for k in k_list]
    _show_recall_info(zip(k_list, recall, precision))


def train_ckan(config, datasets, writer):
    train_data, valid_data, test_data, n_entity, n_relation, user_triple_set, item_triple_set = datasets
    print(f'{config.dataset} dataset has {n_entity} entities and {n_relation} relations')

    ipe = np.ceil(train_data.shape[0] / config.batch_size)
    model, optim, _ = build_model_optim_losses(config, n_entity=n_entity, n_relation=n_relation)
    for epoch in range(config.n_epochs):
        np.random.shuffle(train_data)
        start = 0

        while start < train_data.shape[0]:
            labels = train_data[start:start + config.batch_size, 2]
            scores = model(train_data, user_triple_set, item_triple_set, start, start + config.batch_size)

            loss = model.one_step(scores, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            start += config.batch_size
            writer.add_scalar('bce_loss', loss.item(), (epoch * ipe) + (start // config.batch_size))

        # Evaluate every epoch
        eval_auc, eval_f1 = ctr_eval(config, model, valid_data, user_triple_set, item_triple_set)
        test_auc, test_f1 = ctr_eval(config, model, test_data, user_triple_set, item_triple_set)
        writer.add_scalar('valid/auc', eval_auc, epoch)
        writer.add_scalar('valid/f1', eval_f1, epoch)
        writer.add_scalar('test/auc', test_auc, epoch)
        writer.add_scalar('test/f1', test_f1, epoch)

        ctr_info = 'epoch %.2d    eval auc: %.4f f1: %.4f    test auc: %.4f f1: %.4f'
        logging.info(ctr_info, epoch, eval_auc, eval_f1, test_auc, test_f1)

        # topk_eval(config, model, train_data, test_data, user_triple_set, item_triple_set, writer, epoch)


def train_kgin(config, datasets, writer):
    train_data, valid_data, test_data, n_params, graph, mat_list = datasets
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list
    print(f'{config.dataset} dataset has {n_params["n_entities"]} entities and {n_params["n_relations"]} relations')

    ipe = np.ceil(train_data.shape[0] / config.batch_size)
    model, optim, _ = build_model_optim_losses(config, n_params=n_params, graph=graph, mean_mat_list=mean_mat_list[0])

    for epoch in range(config.n_epochs):
        np.random.shuffle(train_data)
        start = 0

        while start < train_data.shape[0]:
            loss, _, _, cor = model(train_data[start:start + config.batch_size])

            optim.zero_grad()
            loss.backward()
            optim.step()

            start += config.batch_size
            writer.add_scalar('bce_loss', loss.item(), (epoch * ipe) + (start // config.batch_size))

        # Evaluate every epoch
        eval_auc, eval_f1 = kgin_ctr_eval(config, model, valid_data)
        test_auc, test_f1 = kgin_ctr_eval(config, model, test_data)
        writer.add_scalar('valid/auc', eval_auc, epoch)
        writer.add_scalar('valid/f1', eval_f1, epoch)
        writer.add_scalar('test/auc', test_auc, epoch)
        writer.add_scalar('test/f1', test_f1, epoch)

        ctr_info = 'epoch %.2d    eval auc: %.4f f1: %.4f    test auc: %.4f f1: %.4f'
        logging.info(ctr_info, epoch, eval_auc, eval_f1, test_auc, test_f1)

        # topk_eval(config, model, train_data, test_data, user_triple_set, item_triple_set, writer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/KGIN_music.yaml', help='Configuration YAML path')
    args = parser.parse_args()

    set_random_seed(712933, 2021)
    config, writer = prepare_train(args.config)

    # Load dataset
    datasets = load_data(config)

    # Train
    if config.model == 'CKAN':
        train_ckan(config, datasets, writer)
    elif config.model == 'KGIN':
        train_kgin(config, datasets, writer)



