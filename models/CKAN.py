"""
    Modified from https://github.com/weberrr/CKAN/blob/master/src/model.py

    How to distinguish the contribution of the different neighboring entities in KG
    How to fuse interaction and knowledge information into representations of users and items
    Collaboration Propagation
        user: 해당 유저가 interact한 Item
        item: 해당 아이템과 interact한 user가 interact한 item
    KG propagation: 그냥 layer diffusion
    Attentive Embedding Layer
        head와 relation mlp -> matmul with tail -> attentive weight
    Prediction layer
        Aggregation을 통해 user item embedding 구하고 matmul -> prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CKAN(nn.Module):
    def __init__(self, opt, device, n_entity, n_relation):
        super(CKAN, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation

        self.opt = opt
        self.device = device
        self.dim = opt.dim
        self.agg = opt.agg
        self.n_layer = opt.n_layer

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.attention = nn.Sequential(
           nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )

        self._init_weight()
        self.loss_fn = nn.BCELoss()

    def _get_triple_tensor(self, objs, triple_set):
        # [h,r,t]  h: [layers, batch_size, triple_set_size]
        h, r, t = [], [], []
        for i in range(self.n_layer):
            h.append(torch.LongTensor([triple_set[obj][i][0] for obj in objs]))
            r.append(torch.LongTensor([triple_set[obj][i][1] for obj in objs]))
            t.append(torch.LongTensor([triple_set[obj][i][2] for obj in objs]))

            h = list(map(lambda x: x.to(self.device), h))
            r = list(map(lambda x: x.to(self.device), r))
            t = list(map(lambda x: x.to(self.device), t))
        return [h, r, t]

    def _get_feed_data(self, data, user_triple_set, item_triple_set):
        # origin item
        items = torch.LongTensor(data[:, 1]).to(self.device)

        # kg propagation embeddings
        users_triple = self._get_triple_tensor(data[:, 0], user_triple_set)
        items_triple = self._get_triple_tensor(data[:, 1], item_triple_set)
        return items, users_triple, items_triple

    def forward(self, data, user_triple_set, item_triple_set):
        items, user_triple_set, item_triple_set = self._get_feed_data(data, user_triple_set, item_triple_set)
        user_embeddings = []
        user_emb_0 = self.entity_emb(user_triple_set[0][0])     # [batch_size, triple_set_size, dim]
        user_embeddings.append(user_emb_0.mean(dim=1))          # [batch_size, dim]

        for i in range(self.n_layer):
            h_emb = self.entity_emb(user_triple_set[0][i])      # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])    # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])      # [batch_size, triple_set_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)          # [batch_size, dim]
            user_embeddings.append(user_emb_i)

        item_embeddings = []
        item_emb_origin = self.entity_emb(items)                # [batch size, dim]
        item_embeddings.append(item_emb_origin)

        for i in range(self.n_layer):
            h_emb = self.entity_emb(item_triple_set[0][i])      # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])    # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])      # [batch_size, triple_set_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)          # [batch_size, dim]
            item_embeddings.append(item_emb_i)

        if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
            item_emb_0 = self.entity_emb(item_triple_set[0][0]) # [batch_size, triple_set_size, dim]
            item_embeddings.append(item_emb_0.mean(dim=1))                       # [batch_size, dim]

        scores = self.predict(user_embeddings, item_embeddings)
        return scores

    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]

        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u), dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v), dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores

    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        att_weights = self.attention(torch.cat((h_emb, r_emb), dim=-1)).squeeze(-1)     # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights, dim=-1)                               # [batch_size, triple_set_size]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)                   # [batch_size, triple_set_size, dim]
        emb_i = emb_i.sum(dim=1)                                                   # [batch_size, dim]
        return emb_i

    def one_step(self, scores, labels):
        labels = torch.FloatTensor(labels).to(self.device)
        return self.loss_fn(scores, labels)
