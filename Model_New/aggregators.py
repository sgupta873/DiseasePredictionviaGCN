import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import random


class MeanAggregator(nn.Module):
    """
    Computes mean-based or attention-based aggregation of neighboring node features.
    """

    def __init__(self, features_fn, features_dim=4096, use_cuda=False, gcn_style=False, mode="GCN"):
        """
        features_fn: function that maps node indices to feature tensors
        features_dim: dimensionality of the input/output features
        use_cuda: use GPU if True
        gcn_style: whether to include self-loops (like GCN) or not
        mode: "GCN", "GAT", or "GIN" to switch aggregation mechanism
        """
        super(MeanAggregator, self).__init__()

        self.features_fn = features_fn
        self.use_cuda = use_cuda
        self.gcn_style = gcn_style
        self.mode = mode
        self.attention_enabled = True if mode == "GAT" else False

        self.in_features = features_dim
        self.out_features = features_dim

        self.weight = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.att_weight = nn.Parameter(torch.FloatTensor(2 * self.out_features, 1))
        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim=1)

        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.att_weight)

    def forward(self, nodes, neighbors, num_sample=10, agg_type="max"):
        """
        nodes: list of central node indices
        neighbors: list of neighbor sets for each node
        num_sample: optional downsampling of neighbors
        agg_type: 'mean' or 'max' pooling for GCN aggregation
        """

        if num_sample is not None:
            sampled_neighbors = [set(random.sample(nbrs, num_sample)) if len(nbrs) >= num_sample else nbrs
                                 for nbrs in neighbors]
        else:
            sampled_neighbors = neighbors

        if self.gcn_style:
            sampled_neighbors = [nbrs | {nodes[i]} for i, nbrs in enumerate(sampled_neighbors)]

        all_unique_nodes = list(set.union(*sampled_neighbors))
        node_to_index = {n: i for i, n in enumerate(all_unique_nodes)}

        row_indices = []
        col_indices = []
        for i, nbrs in enumerate(sampled_neighbors):
            row_indices.extend([i] * len(nbrs))
            col_indices.extend([node_to_index[n] for n in nbrs])

        mask = Variable(torch.zeros(len(sampled_neighbors), len(all_unique_nodes)))
        mask[row_indices, col_indices] = 1

        zero_mask = -9e15 * torch.ones_like(mask)

        if self.use_cuda:
            mask = mask.cuda()

        neighbor_counts = mask.sum(1, keepdim=True)
        neighbor_counts[neighbor_counts == 0] = 1

        node_feats = self.features_fn(torch.LongTensor(nodes).cuda() if self.use_cuda else torch.LongTensor(nodes))
        neighbor_feats = self.features_fn(torch.LongTensor(all_unique_nodes).cuda() if self.use_cuda else torch.LongTensor(all_unique_nodes))

        if self.mode == "GAT":
            node_feats = torch.mm(node_feats, self.weight)
            neighbor_feats = torch.mm(neighbor_feats, self.weight)

            N = node_feats.size(0)
            M = neighbor_feats.size(0)

            a_input = torch.cat([
                node_feats.repeat(1, M).view(N * M, -1),
                neighbor_feats.repeat(N, 1)
            ], dim=1).view(N, M, -1)

            attn_scores = self.leakyrelu(torch.matmul(a_input, self.att_weight).squeeze(2))
            attention = torch.where(mask > 0, attn_scores, zero_mask)
            attention = self.softmax(attention)

            output = attention.mm(neighbor_feats)

        elif self.mode == "GCN":
            if agg_type == "mean":
                mask = mask.div(neighbor_counts)
                output = mask.mm(neighbor_feats)
            elif agg_type == "max":
                outputs = []
                for i, nbrs in enumerate(sampled_neighbors):
                    if nbrs:
                        embeddings = neighbor_feats[[node_to_index[n] for n in nbrs]]
                        max_embed, _ = torch.max(embeddings, dim=0)
                        outputs.append(max_embed)
                    else:
                        outputs.append(torch.zeros(neighbor_feats.size(1)).to(neighbor_feats.device))
                output = torch.stack(outputs)
        elif self.mode == "GIN":
            output = mask.mm(neighbor_feats)
        else:
            raise ValueError("Unknown aggregation mode")

        return output


class AttentionAggregator(nn.Module):
    """
    Aggregates neighbor features using simple dot-product attention.
    """

    def __init__(self, features_fn, in_features=4096, out_features=1024, use_cuda=False, gcn_style=False):
        """
        features_fn: function that returns feature vectors for given node indices
        in_features: input feature dimension
        out_features: output feature dimension
        use_cuda: if True, computations are performed on GPU
        gcn_style: whether to include self-loops in the neighbor set
        """
        super(AttentionAggregator, self).__init__()

        self.features_fn = features_fn
        self.use_cuda = use_cuda
        self.gcn_style = gcn_style

        self.in_features = in_features
        self.out_features = out_features
        self.softmax = nn.Softmax(dim=1)

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.att_weight = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.att_weight)

    def forward(self, nodes, neighbors, num_sample=10):
        """
        nodes: list of node indices
        neighbors: list of neighbor sets per node
        num_sample: max number of neighbors to use per node
        """
        if num_sample is not None:
            sampled_neighbors = [set(random.sample(nbrs, num_sample)) if len(nbrs) >= num_sample else nbrs
                                 for nbrs in neighbors]
        else:
            sampled_neighbors = neighbors

        if self.gcn_style:
            sampled_neighbors = [nbrs | {nodes[i]} for i, nbrs in enumerate(sampled_neighbors)]

        all_unique_nodes = list(set.union(*sampled_neighbors))
        node_to_index = {n: i for i, n in enumerate(all_unique_nodes)}

        row_indices = []
        col_indices = []
        for i, nbrs in enumerate(sampled_neighbors):
            row_indices.extend([i] * len(nbrs))
            col_indices.extend([node_to_index[n] for n in nbrs])

        mask = Variable(torch.zeros(len(sampled_neighbors), len(all_unique_nodes)))
        mask[row_indices, col_indices] = 1
        zero_mask = -9e15 * torch.ones_like(mask)

        if self.use_cuda:
            mask = mask.cuda()

        node_feats = self.features_fn(torch.LongTensor(nodes).cuda() if self.use_cuda else torch.LongTensor(nodes))
        neighbor_feats = self.features_fn(torch.LongTensor(all_unique_nodes).cuda() if self.use_cuda else torch.LongTensor(all_unique_nodes))

        attn_scores = node_feats.mm(neighbor_feats.t())
        attention = torch.where(mask > 0, attn_scores, zero_mask)
        attention = self.softmax(attention)

        aggregated = attention.mm(neighbor_feats)
        return aggregated

