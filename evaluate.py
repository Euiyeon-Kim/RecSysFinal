import logging
import argparse
from collections import defaultdict

import yaml
import numpy as np
from dotmap import DotMap

import torch

import models
from dataloader import load_data


def topk_eval(opt, model, train_data, test_data):
    def _show_rank_info(rank_zip, diversity):
        res = ""
        for k, rec, prec, f1, ndcg in rank_zip:
            res += "recall@%d:%.4f  " % (k, rec)
            res += "precision@%d:%.4f  " % (k, prec)
            res += "f1@%d:%.4f  " % (k, f1)
            res += "ndcg@%d:%.4f  " % (k, ndcg)
            logging.info(res)
            res = ""
        logging.info(f'diversity@10:{diversity:.4f}')

    def _get_user_record(data, is_train):
        user_history_dict = defaultdict(lambda: set())
        for u_id, i_id, r in data:
            if is_train or r == 1:
                user_history_dict[u_id].add(i_id)
        return user_history_dict

    def _get_topk_feed_data(user, items):
        res = list()
        for item in items:
            res.append([user, item])
        return np.array(res)

    def _get_ndcg_at_k(k, topk_items, test_items):
        dcg = 0
        for i in range(k):
            if topk_items[i] in test_items:
                dcg += (2 ** 1 - 1) / np.log2(i + 2)
        idcg = 0
        n = len(test_items) if len(test_items) < k else k
        for i in range(n):
            idcg += (2 ** 1 - 1) / np.log2(i + 2)
        if dcg == 0 or idcg == 0:
            return 0
        return dcg / idcg

    def _get_div(topkset_list):
        div_res = 0
        N = len(topkset_list)
        for i in range(N):
            for j in range(N):
                if j > i:
                    a = topkset_list[i].intersection(topkset_list[j])
                    b = topkset_list[i].union(topkset_list[j])
                    if len(b) != 0:
                        div_res = div_res + (1 - len(a) / len(b))
        return (2 / (N * (N - 1))) * div_res

    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    precision_list = {k: [] for k in k_list}
    f1_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}
    topkset_list = []

    item_set = set(train_data[:, 1].tolist() + test_data[:, 1].tolist())
    train_record = _get_user_record(train_data, True)
    test_record = _get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    test_user_num = 100
    if len(user_list) > test_user_num:
        user_list = np.random.choice(user_list, size=test_user_num, replace=False)

    model.eval()
    for user in user_list:
        test_item_list = list(item_set-set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start + opt.batch_size <= len(test_item_list):
            items = test_item_list[start:start + opt.batch_size]
            input_data = _get_topk_feed_data(user, items)
            scores = model.get_scores(input_data[0:opt.batch_size, :])
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += opt.batch_size

        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (opt.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            scores = model.get_scores(input_data[0:opt.batch_size, :])
            for item, score in zip(res_items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        topkset_list.append(set(item_sorted[:10]))
        for k in k_list:
            topk_items_list = item_sorted[:k]
            hit_num = len(set(topk_items_list) & set(test_record[user]))
            recall_list[k].append(hit_num / len(set(test_record[user])))
            precision_list[k].append(hit_num / k)
            f1_list[k].append((2 * hit_num) / (len(set(test_record[user])) + k))
            topk_items = list(set(topk_items_list))
            topk_items.sort(key=topk_items_list.index)
            ndcg_list[k].append(_get_ndcg_at_k(k, topk_items, list(set(test_record[user]))))

    recalls = [np.mean(recall_list[k]) for k in k_list]
    precisions = [np.mean(precision_list[k]) for k in k_list]
    f1s = [np.mean(f1_list[k]) for k in k_list]
    ndcgs = [np.mean(ndcg_list[k]) for k in k_list]
    diversity = _get_div(topkset_list)
    _show_rank_info(zip(k_list, recalls, precisions, f1s, ndcgs), diversity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/KGIN_music.yaml', help='Configuration YAML path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)

    _, _, model_define_args = load_data(config)
    train_data = np.load(f'data/{config.dataset}/train_data.npy')
    test_data = np.load(f'data/{config.dataset}/test_data.npy')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = models.__dict__[config.model](config, device, model_define_args)
    model.load_state_dict(torch.load(f'exps/ckpt/{config.model}_{config.dataset}_best.pth'))
    model = model.to(device)

    from train import ctr_eval
    auc, f1 = ctr_eval(config, model, test_data)
    logging.info(f"AUC: {auc:.4f}, F1: {f1:.4f}")
    topk_eval(config, model, train_data, test_data)
