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

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        b = user_embeddings.shape[0]
        avg = False
        if not avg:
            user_embeddings = user_embeddings.view((b, 1, 1, self.dim))
            user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
            user_relation_scores_normalized = torch.softmax(user_relation_scores, dim=-1).unsqueeze(-1)
            neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_vectors, dim=2)
        else:
            neighbors_aggregated = torch.mean(neighbor_vectors, dim=2)
        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, config, dropout=0., act=nn.ReLU()):
        super(SumAggregator, self).__init__(config.batch_size, config.dim, dropout=dropout, act=act)
        self.layer = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        b = user_embeddings.shape[0]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        output = (self_vectors + neighbors_agg).view(-1, self.dim)
        output = self.dropout(output)
        output = self.layer(output).view(b, -1, self.dim)
        return self.act(output)


class KGNNLS(nn.Module):
    def __init__(self, config, device, model_define_args):
        super(KGNNLS, self).__init__()
        self.config = config
        self.device = device
        self.model_define_args = model_define_args

        self.n_users = model_define_args['n_users']
        self.n_entities = model_define_args['n_entities']
        self.n_relations = model_define_args['n_relations']
        self.adj_entity = torch.LongTensor(model_define_args['adj_entity']).to(device)
        self.adj_relation = torch.LongTensor(model_define_args['adj_relation']).to(device)
        self.interaction_dict = model_define_args['interaction_dict']
        self.offset = model_define_args['offset']

        self.emb_size = config.dim
        self.n_iter = config.n_iter
        self.n_neighbor = config.neighbor_sample_size
        self.ls_lambda = float(config.ls_weight)
        self.l2_lambda = float(config.l2_weight)

        self.aggregators = []
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregator(self.config, act=nn.Tanh()).to(self.device)
            else:
                aggregator = SumAggregator(self.config).to(self.device)
            self.aggregators.append(aggregator)

        self.user_emb = nn.Embedding(self.n_users, self.emb_size)
        self.entity_emb = nn.Embedding(self.n_entities, self.emb_size)
        self.relation_emb = nn.Embedding(self.n_relations + 1, self.emb_size)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def _get_feed_data(self, data):
        u_ids = torch.LongTensor(data[:, 0]).to(self.device)
        i_ids = torch.LongTensor(data[:, 1]).to(self.device)
        if self.training:
            labels = torch.FloatTensor(data[:, 2]).to(self.device)
        else:
            labels = None
        return u_ids, i_ids, labels

    def _get_neighbors(self, item_indices):
        b = item_indices.shape[0]
        seeds = item_indices.unsqueeze(1)
        entities = [seeds]
        relations = []

        for i in range(self.n_iter):
            index = torch.flatten(entities[i])
            neighbor_entities = torch.index_select(self.adj_entity, 0, index).reshape(b, -1)
            neighbor_relations = torch.index_select(self.adj_relation, 0, index).reshape(b, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations, user_embeddings):
        b = user_embeddings.shape[0]
        entity_vectors = [self.entity_emb(i) for i in entities]
        relation_vectors = [self.relation_emb(i) for i in relations]

        for i in range(self.n_iter):
            aggregator = self.aggregators[i]
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [b, -1, self.n_neighbor, self.emb_size]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=entity_vectors[hop+1].view(shape),
                                    neighbor_relations=relation_vectors[hop].view(shape),
                                    user_embeddings=user_embeddings,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = entity_vectors[0].view(b, self.emb_size)
        return res

    def label_smoothness_predict(self, user_embeddings, user, entities, relations):
        b = user_embeddings.shape[0]

        entity_labels = []

        # True means the label of this item is reset to initial value during label propagation
        reset_masks = []
        holdout_item_for_user = None

        for entities_per_iter in entities:
            users = torch.unsqueeze(user, dim=1)  # [batch_size, 1]
            user_entity_concat = users * self.offset + entities_per_iter  # [batch_size, n_neighbor^i]

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            def lookup_interaction_table(x, _):
                x = int(x)
                label = self.interaction_dict.setdefault(x, 0.5)
                return label

            initial_label = user_entity_concat.clone().cpu().double()
            initial_label.map_(initial_label, lookup_interaction_table)
            initial_label = initial_label.float().to(self.device)

            # False if the item is held out
            holdout_mask = (holdout_item_for_user - user_entity_concat).bool()
            # True if the entity is a labeled item
            reset_mask = (initial_label - 0.5).bool()
            reset_mask = torch.logical_and(reset_mask, holdout_mask)  # remove held-out items
            initial_label = holdout_mask.float() * initial_label + \
                            torch.logical_not(holdout_mask).float() * 0.5  # label initialization

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        # we do not need the reset_mask for the last iteration
        reset_masks = reset_masks[:-1]

        # label propagation
        relation_vectors = [self.relation_emb(i) for i in relations]
        for i in range(self.n_iter):
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                masks = reset_masks[hop]
                self_labels = entity_labels[hop]
                neighbor_labels = entity_labels[hop + 1].reshape(b, -1, self.n_neighbor)
                neighbor_relations = relation_vectors[hop].reshape(b, -1, self.n_neighbor, self.emb_size)

                # mix_neighbor_labels
                user_embeddings = user_embeddings.reshape(b, 1, 1, self.emb_size)
                user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
                user_relation_scores_normalized = torch.softmax(user_relation_scores, dim=-1)
                neighbors_aggregated_label = torch.mean(user_relation_scores_normalized * neighbor_labels, dim=2)
                output = masks.float() * self_labels + torch.logical_not(masks).float() * neighbors_aggregated_label

                entity_labels_next_iter.append(output)
            entity_labels = entity_labels_next_iter

        predicted_labels = entity_labels[0].squeeze(-1)
        return predicted_labels

    def forward(self, data):
        u_ids, i_ids, labels = self._get_feed_data(data)
        user_embeddings = self.user_emb(u_ids)

        entities, relations = self._get_neighbors(i_ids)
        item_embeddings = self.aggregate(entities, relations, user_embeddings)

        scores = (user_embeddings * item_embeddings).sum(dim=1)
        rec_loss = nn.BCEWithLogitsLoss()(scores, labels)

        predicted_labels = self.label_smoothness_predict(user_embeddings, u_ids, entities, relations)
        ls_loss = nn.BCEWithLogitsLoss()(predicted_labels, labels)

        l2_loss = (torch.norm(user_embeddings) ** 2 + torch.norm(item_embeddings) ** 2) / 2
        loss = rec_loss + self.ls_lambda * ls_loss + self.l2_lambda * l2_loss
        return loss

    def get_scores(self, data):
        u_ids, i_ids, _ = self._get_feed_data(data)
        user_embeddings = self.user_emb(u_ids)

        entities, relations = self._get_neighbors(i_ids)
        item_embeddings = self.aggregate(entities, relations, user_embeddings)

        scores = (user_embeddings * item_embeddings).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores.detach().cpu().numpy()