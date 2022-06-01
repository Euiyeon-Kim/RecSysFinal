import logging
import argparse
from collections import defaultdict

import numpy as np


def topk_eval(args, model, train_data, test_data):
    def _show_rank_info(rank_zip):
        res = ""
        for k, rec, prec, f1, ndcg in rank_zip:
            res += "recall@%d:%.4f  " % (k, rec)
            res += "precision@%d:%.4f  " % (k, prec)
            res += "f1@%d:%.4f  " % (k, f1)
            res += "ndcg@%d:%.4f  " % (k, ndcg)
            logging.info(res)
            res = ""

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

    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}
    precision_list = {k: [] for k in k_list}
    f1_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

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
        while start + args.batch_size <= len(test_item_list):
            items = test_item_list[start:start + args.batch_size]
            input_data = _get_topk_feed_data(user, items)
            scores = model.get_scores(input_data[0:args.batch_size, :])
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size

        # padding the last incomplete mini-batch if exists
        if start < len(test_item_list):
            res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
            input_data = _get_topk_feed_data(user, res_items)
            scores = model.get_scores(input_data[0:args.batch_size, :])
            for item, score in zip(res_items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
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
    _show_rank_info(zip(k_list, recalls, precisions, f1s, ndcgs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/CKAN_music.yaml', help='Configuration YAML path')
    parser.add_argument('--topK', default=True, action='store_true', help='Do top-k evaluation')
    args = parser.parse_args()

