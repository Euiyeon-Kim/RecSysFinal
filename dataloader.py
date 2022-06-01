import os
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import networkx as nx
import scipy.sparse as sp


DATAINFO = {
    'music': {
        'n_users': 1872,
        'n_items': 3846,
        'n_interactions': 42346,
        'n_entities': 9366,
        'n_relations': 60,
        'n_triplets': 15518,
    },
    'movie': {
        'n_users': 138159,
        'n_items': 16954,
        'n_interactions': 13501622,
        'n_entities': 102569,
        'n_relations': 32,
        'n_triplets': 499474,
    },
    'book': {
        'n_users': 17860,
        'n_items': 14967,
        'n_interactions': 139746,
        'n_entities': 77903,
        'n_relations': 25,
        'n_triplets': 151500,
    }
}
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    if args.model == 'CKAN':
        def _ckan_kg_propagation(args, kg, init_entity_set, set_size, is_user):
            # triple_sets: [n_obj][n_layer](h,r,t)x[set_size]
            triple_sets = defaultdict(list)
            for obj in init_entity_set.keys():
                if is_user and args.n_layer == 0:
                    n_layer = 1
                else:
                    n_layer = args.n_layer
                for l in range(n_layer):
                    h, r, t = [], [], []
                    if l == 0:
                        entities = init_entity_set[obj]
                    else:
                        entities = triple_sets[obj][-1][2]

                    for entity in entities:
                        for tail_and_relation in kg[entity]:
                            h.append(entity)
                            t.append(tail_and_relation[0])
                            r.append(tail_and_relation[1])

                    if len(h) == 0:
                        triple_sets[obj].append(triple_sets[obj][-1])
                    else:
                        indices = np.random.choice(len(h), size=set_size, replace=(len(h) < set_size))
                        h = [h[i] for i in indices]
                        r = [r[i] for i in indices]
                        t = [t[i] for i in indices]
                        triple_sets[obj].append((list(map(int, h)), r, t))
            return triple_sets

        def _ckan_construct_kg(kg_np):
            logging.info("constructing knowledge graph ...")
            kg = defaultdict(list)
            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
            return kg

        def _ckan_load_kg(args):
            kg_file = './data/' + args.dataset + '/kg_final'
            logging.info("loading kg file: %s.npy", kg_file)
            if os.path.exists(kg_file + '.npy'):
                kg_np = np.load(kg_file + '.npy')
            else:
                kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
                np.save(kg_file + '.npy', kg_np)
            n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
            n_relation = len(set(kg_np[:, 1]))
            kg = _ckan_construct_kg(kg_np)
            return n_entity, n_relation, kg

        logging.info("================== preparing data ===================")
        train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set = load_rating(args)
        n_entity, n_relation, kg = _ckan_load_kg(args)
        logging.info("contructing users' kg triple sets ...")
        user_triple_sets = _ckan_kg_propagation(args, kg, user_init_entity_set, args.user_triple_set_size, True)
        logging.info("contructing items' kg triple sets ...")
        item_triple_sets = _ckan_kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, False)
        model_define_args = {
            'n_entity': n_entity,
            'n_relation': n_relation,
            'user_triple_sets': user_triple_sets,
            'item_triple_sets': item_triple_sets
        }

        logging.info(f'{args.dataset} dataset has {n_entity} entities and {n_relation} relations')
        return train_data, eval_data, test_data, model_define_args

    else:
        n_users = DATAINFO[args.dataset]['n_users']
        n_items = DATAINFO[args.dataset]['n_items']
        n_entities = DATAINFO[args.dataset]['n_entities']
        n_nodes = n_entities + n_items

        def _kgin_read_triplets(file_name):
            can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
            can_triplets_np = np.unique(can_triplets_np, axis=0)

            if args.inverse_r:          # 원래 relation * 2 + 1
                # get triplets with inverse direction like <entity, is-aspect-of, item>
                inv_triplets_np = can_triplets_np.copy()
                inv_triplets_np[:, 0] = can_triplets_np[:, 2]
                inv_triplets_np[:, 2] = can_triplets_np[:, 0]
                inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
                # consider two additional relations --- 'interact' and 'be interacted'
                can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
                inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
                # get full version of knowledge graph
                triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
            else:                       # 원래 relation + 1
                # consider two additional relations --- 'interact'.
                can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
                triplets = can_triplets_np.copy()
            return triplets

        def _kgin_build_graph(train_data, triplets):
            ckg_graph = nx.MultiDiGraph()
            rd = defaultdict(list)

            logging.info("Begin to load interaction triples ...")
            for u_id, i_id, pos_neg in tqdm(train_data, ascii=True):
                if pos_neg == 1:
                    rd[0].append([u_id, i_id])

            logging.info("\nBegin to load knowledge graph triples ...")
            for h_id, r_id, t_id in tqdm(triplets, ascii=True):
                ckg_graph.add_edge(h_id, t_id, key=r_id)
                rd[r_id].append([h_id, t_id])

            return ckg_graph, rd

        def _kgin_build_sparse_relational_graph(relation_dict):
            def _bi_norm_lap(adj):
                # D^{-1/2}AD^{-1/2}
                rowsum = np.array(adj.sum(1))
                d_inv_sqrt = np.power(rowsum, -0.5).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

                # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
                bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
                return bi_lap.tocoo()

            def _si_norm_lap(adj):
                # D^{-1}A
                rowsum = np.array(adj.sum(1))

                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat_inv = sp.diags(d_inv)

                norm_adj = d_mat_inv.dot(adj)
                return norm_adj.tocoo()

            adj_mat_list = []
            print("Begin to build sparse relation matrix ...")
            for r_id in tqdm(relation_dict.keys()):
                np_mat = np.array(relation_dict[r_id])
                if r_id == 0:
                    cf = np_mat.copy()
                    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
                    vals = [1.] * len(cf)
                    adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
                else:
                    vals = [1.] * len(np_mat)
                    adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
                adj_mat_list.append(adj)
            norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
            mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]

            # interaction: user->item, [n_users, n_entities]
            norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
            mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

            return adj_mat_list, norm_mat_list, mean_mat_list

        logging.info("================== preparing data ===================")
        train_data, eval_data, test_data, _, _ = load_rating(args)
        logging.info('combinating train_cf and kg data ...')
        triplets = _kgin_read_triplets('./data/' + args.dataset + '/kg_final.txt')

        logging.info('building the graph ...')
        graph, relation_dict = _kgin_build_graph(train_data, triplets)
        n_relations = len(relation_dict)

        logging.info('building the adj mat ...')
        adj_mat_list, norm_mat_list, mean_mat_list = _kgin_build_sparse_relational_graph(relation_dict)

        n_params = {
            'n_users': int(n_users),
            'n_items': int(n_items),
            'n_entities': int(n_entities),
            'n_nodes': int(n_nodes),
            'n_relations': int(n_relations)
        }

        train_pos_data = np.array(list(filter(lambda x: x[2] == 1, train_data)))
        train_user_pos_dict = defaultdict(list)
        train_user_neg_dict = defaultdict(list)
        for u_id, i_id, r in train_data:
            if r == 1:
                train_user_pos_dict[u_id].append(i_id)
            else:
                train_user_neg_dict[u_id].append(i_id)

        model_define_args = {
            'n_params': n_params,
            'graph': graph,
            'mean_mat': mean_mat_list[0],
            'train_user_pos_dict': train_user_pos_dict,
            'train_user_neg_dict': train_user_neg_dict,
        }
        logging.info(f'{args.dataset} dataset has {int(n_entities)} entities and {int(n_relations)} relations')
        return train_pos_data, eval_data, test_data, model_define_args


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


def load_rating(args):
    rating_file = './data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    return dataset_split(rating_np)
