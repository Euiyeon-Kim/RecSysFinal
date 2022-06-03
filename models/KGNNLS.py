"""
    Modified from https://github.com/hwwang55/KGNN-LS/blob/master/src/model.py
"""
import torch
import torch.nn as nn


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act):
        super(Aggregator, self).__init__()
        self.dim = dim
        self.act = act
        self.dropout = dropout
        self.batch_size = batch_size

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))
            user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
            user_relation_scores_normalized = torch.softmax(user_relation_scores, dim=-1).unsqueeze(-1)
            neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_vectors, dim=2)
        else:
            neighbors_aggregated = torch.mean(neighbor_vectors, dim=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, config, dropout=0., act=nn.ReLU):
        super(SumAggregator, self).__init__(config.batch_size, config.dim, dropout=dropout, act=act)
        self.layer = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(p=1-self.dropout)

    def _init_weight(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        output = (self_vectors + neighbors_agg).view(-1, self.dim)
        output = self.dropout(output)
        output = self.layer(output).view(self.batch_size, -1, self.dim)

        return self.act(output)


class LabelAggregator(Aggregator):
    def __init__(self, config):
        super(LabelAggregator, self).__init__(config.batch_size, config.dim, 0., nn.Identity)

    def forward(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        user_embeddings = user_embeddings.view(self.batch_size, 1, 1, self.dim)
        user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
        user_relation_scores_normalized = torch.softmax(user_relation_scores, dim=-1)
        neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_labels, dim=-1)
        output = masks * self_labels + torch.logical_not(masks) * neighbors_aggregated
        return output


class KGNNLS(nn.Module):
    def __init__(self, config, device, model_define_args):
        super(KGNNLS, self).__init__()
        self.config = config
        self.device = device
        self.model_define_args = model_define_args

        self.n_users = model_define_args['n_users']
        self.n_entities = model_define_args['n_entities']
        self.n_relations = model_define_args['n_relations']
        self.adj_entity = model_define_args['adj_entity']
        self.adj_relation = model_define_args['adj_relation']

        self.emb_size = config.dim
        self.n_iter = config.n_iter
        self.batch_size = config.batch_size
        self.n_neighbor = config.neighbor_sample_size

        self._init_weight()
        self.user_emb = nn.Parameter(self.user_emb)
        self.entity_emb = nn.Parameter(self.entity_emb)
        self.relation_emb = nn.Parameter(self.relation_emb)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_emb = initializer(torch.empty(self.n_users, self.emb_size))
        self.entity_emb = initializer(torch.empty(self.n_entities, self.emb_size))
        self.relation_emb = initializer(torch.empty(self.n_relations, self.emb_size))

    def _get_feed_data(self, data):
        u_ids = torch.LongTensor(data[:, 0]).to(self.device)
        i_ids = torch.LongTensor(data[:, 1]).to(self.device)
        labels = torch.FloatTensor(data[:, 2]).to(self.device)
        return u_ids, i_ids, labels

    def get_neighbors(self, item_indices):
        seeds = item_indices.unsqueeze(1)
        entities = [seeds]
        relations = []
        print(seeds.shape)
        exit()
        for i in range(self.n_iter):
            neighbor_entities = torch.gather(self.adj_entity, entities[i]).view(self.batch_size, -1)
            neighbor_relations = torch.gather(self.adj_relation, entities[i]).view(self.batch_size, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations, user_embeddings):
        aggregators = []  # store all aggregators
        entity_vectors = self.entity_emb[entities, :]
        relation_vectors = self.relation_emb[relations, :]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregator(self.batch_size, self.emb_size, act=nn.Tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.emb_size)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.emb_size]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=entity_vectors[hop+1].view(shape),
                                    neighbor_relations=relation_vectors[hop+1].view(shape),
                                    user_embeddings=user_embeddings,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = entity_vectors[0].view(self.batch_size, self.emb_size)
        return res, aggregators

    def forward(self, data):
        u_ids, i_ids, labels = self._get_feed_data(data)
        user_embeddings = self.user_emb[u_ids, :]
        item_emb = self.user_emb[u_ids, :]

        entities, relations = self.get_neighbors(i_ids)
        item_embeddings, aggregators = self.aggregate(entities, relations, user_embeddings)

