"""
Originated from https://github.com/weberrr/CKAN/blob/master/src/preprocess.py
"""

import argparse
import logging
import os

import pickle
import numpy as np


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

RATING_FILE_NAME = dict({'music': 'user_artists.dat', 'book': 'BX-Book-Ratings.csv', 'movie': 'ratings.csv'})
SEP = dict({'music': '\t', 'book': ';', 'movie': ','})
THRESHOLD = dict({'music': 0, 'book': 0, 'movie': 4})


def dataset_split(rating_np):
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    user_init_entity_set, item_init_entity_set = collaboration_propagation(rating_np, train_indices)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_init_entity_set.keys()]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_init_entity_set.keys()]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_init_entity_set.keys()]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set


def collaboration_propagation(rating_np, train_indices):
    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = dict()
    item_history_user_dict = dict()
    item_neighbor_item_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_item_dict:
                user_history_item_dict[user] = []
            user_history_item_dict[user].append(item)
            if item not in item_history_user_dict:
                item_history_user_dict[item] = []
            item_history_user_dict[item].append(user)

    logging.info("contructing items' initial entity set ...")
    for item in item_history_user_dict.keys():
        item_nerghbor_item = []
        for user in item_history_user_dict[item]:
            item_nerghbor_item = np.concatenate((item_nerghbor_item, user_history_item_dict[user]))
        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict


def read_item_index_to_entity_id_file(dataset):
    file = './data/' + dataset + '/item_index2entity_id.txt'
    logging.info("reading item index to entity id file: %s", file)
    item_index_old2new = dict()
    entity_id2index = dict()
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
    return item_index_old2new, entity_id2index


def convert_rating(dataset, item_index_old2new, entity_id2index):
    file = './data/' + dataset + '/' + RATING_FILE_NAME[dataset]
    logging.info("reading rating file: %s", file)

    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    enc = 'utf-8' if dataset != 'book' else 'CP1252'
    for line in open(file, encoding=enc).readlines()[1:]:
        array = line.strip().split(SEP[dataset])
        # remove prefix and suffix quotation marks for BX dataset
        if dataset == 'book':
            array = list(map(lambda x: x[1:-1], array))
        item_index_old = array[1]

        # if the item is not in the final item set
        if item_index_old not in item_index_old2new.keys():
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = array[0]
        rating = float(array[2])
        if rating >= THRESHOLD[dataset]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    write_file = './data/' + dataset + '/ratings_final'
    logging.info("converting rating file to: %s", write_file+'.txt')
    writer = open(write_file+'.txt', 'w', encoding='utf-8')
    writer_idx = 0
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]
        for item in pos_item_set:
            writer_idx += 1
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set

        a = unwatched_set.copy()
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]

        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer_idx += 1
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()

    logging.info("number of users: %d", user_cnt)
    logging.info("number of items: %d", len(item_set))
    logging.info("number of interactions: %d", writer_idx)

    rating_file = './data/' + dataset + '/ratings_final.txt'
    rating_np = np.loadtxt(rating_file, dtype=np.int32)
    train_data, valid_data, test_data, user_init_entity_set, item_init_entity_set = dataset_split(rating_np)
    with open(f"./data/{dataset}/user_init_entity_set.pkl", "wb") as tf:
        pickle.dump(user_init_entity_set, tf)
    with open(f"./data/{dataset}/item_init_entity_set.pkl", "wb") as tf:
        pickle.dump(item_init_entity_set, tf)
    np.save(f"./data/{dataset}/train_data.npy", train_data)
    np.save(f"./data/{dataset}/valid_data.npy", valid_data)
    np.save(f"./data/{dataset}/test_data.npy", test_data)
    os.remove(rating_file)


def convert_kg(dataset, entity_id2index):
    file = './data/' + dataset + '/' + 'kg.txt'
    logging.info("reading kg file: %s", file)
    write_file = './data/' + dataset + '/' + 'kg_final'
    logging.info("converting kg file to: %s", write_file + '.txt')
    entity_cnt = len(entity_id2index)
    relation_id2index = dict()
    relation_cnt = 0
    writer = open(write_file+'.txt', 'w', encoding='utf-8')
    writer_idx = 0
    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))
        writer_idx += 1
    writer.close()

    kg_np = np.loadtxt(write_file + '.txt', dtype=np.int32)
    np.save(write_file + '.npy', kg_np)
    os.remove(write_file + '.txt')

    logging.info("number of entities (containing items): %d", entity_cnt)
    logging.info("number of relations: %d", relation_cnt)
    logging.info("number of triples: %d", writer_idx)
    return entity_id2index, relation_id2index


if __name__ == '__main__':
    # we use the same random seed as RippleNet, KGCN, KGNN-LS for better comparison
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to preprocess')
    args = parser.parse_args()

    item_index_old2new, entity_id2index = read_item_index_to_entity_id_file(args.dataset)
    convert_rating(args.dataset, item_index_old2new, entity_id2index)
    entity_id2index, relation_id2index = convert_kg(args.dataset, entity_id2index)

    logging.info("data %s preprocess: done.", args.dataset)