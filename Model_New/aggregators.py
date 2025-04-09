import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings by computing the mean of its neighbors' embeddings.
    Supports different graph kernels: GCN, GAT, and GIN.
    """

    def __init__(self, features, feature_dim=4096, use_cuda=False, gcn_mode=False, kernel="GCN"):
        """
        Args:
            features (callable): Function mapping node IDs to feature tensors.
            feature_dim (int): Feature vector size.
            use_cuda (bool): If True, moves computations to GPU.
            gcn_mode (bool): If True, aggregates self-looped features like GCN.
            kernel (str): Graph aggregation type ("GCN", "GAT", "GIN").
        """
        super(MeanAggregator, self).__init__()

        self.features = features
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.gcn_mode = gcn_mode
        self.kernel = kernel

        # Only needed for GAT (Graph Attention Network)
        self.use_attention = kernel == "GAT"
        self.weight = nn.Parameter(torch.FloatTensor(feature_dim, feature_dim))
        self.attention_vector = nn.Parameter(torch.FloatTensor(2 * feature_dim, 1))

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)

        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention_vector)

    def forward(self, nodes, neighbor_sets, num_sample=10, aggregation="mean"):
        """
        Aggregates neighbor embeddings for a batch of nodes.

        Args:
            nodes (list): List of target nodes.
            neighbor_sets (list of sets): Each set contains the neighbors of a node.
            num_sample (int): Number of neighbors to sample (None for all).
            aggregation (str): Type of aggregation ("mean" for GCN, "sum" for GIN).

        Returns:
            torch.Tensor: Aggregated node embeddings.
        """
        # Sample neighbors if necessary
        if num_sample is not None:
            sampled_neighbors = [
                set(random.sample(neigh, num_sample)) if len(neigh) >= num_sample else neigh
                for neigh in neighbor_sets
            ]
        else:
            sampled_neighbors = neighbor_sets

        # If using GCN mode, include self-loops
        if self.gcn_mode:
            sampled_neighbors = [neigh | {nodes[i]} for i, neigh in enumerate(sampled_neighbors)]

        # Flatten and map unique nodes
        unique_nodes = list(set.union(*sampled_neighbors))
        node_index = {node: idx for idx, node in enumerate(unique_nodes)}

        # Construct adjacency mask
        row_indices = [i for i, neigh in enumerate(sampled_neighbors) for _ in neigh]
        col_indices = [node_index[n] for neigh in sampled_neighbors for n in neigh]

        mask = torch.zeros(len(sampled_neighbors), len(unique_nodes), device=self.device)
        mask[row_indices, col_indices] = 1

        # Load node and neighbor embeddings
        node_features = self.features(torch.LongTensor(nodes).to(self.device))
        neighbor_features = self.features(torch.LongTensor(unique_nodes).to(self.device))

        # Apply different aggregation methods
        if self.kernel == "GAT":
            # GAT attention mechanism
            node_transformed = node_features @ self.weight.T
            neighbor_transformed = neighbor_features @ self.weight.T
            num_nodes = node_transformed.shape[0]
            num_neighbors = neighbor_transformed.shape[0]

            # Compute pairwise attention scores
            combined = torch.cat(
                [
                    node_transformed.repeat(1, num_neighbors).view(num_nodes * num_neighbors, -1),
                    neighbor_transformed.repeat(num_nodes, 1),
                ],
                dim=1
            ).view(num_nodes, num_neighbors, -1)

            attention_scores = self.leaky_relu(combined @ self.attention_vector).squeeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
            attention_weights = self.softmax(attention_scores)

            aggregated_features = attention_weights @ neighbor_transformed

        elif self.kernel == "GCN":
            # Standard GCN-style mean aggregation
            degree = mask.sum(1, keepdim=True).clamp(min=1)
            mask = mask / degree  # Normalize by degree
            aggregated_features = mask @ neighbor_features

        elif self.kernel == "GIN":
            # Sum-based aggregation for GIN
            aggregated_features = mask @ neighbor_features

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

        return aggregated_features


class AttentionAggregator(nn.Module):
    """
    Implements an attention-based neighborhood aggregation mechanism 
    for graph-based learning.
    """

    def __init__(self, features, in_dim=4096, out_dim=1024, use_cuda=False, gcn_mode=False):
        """
        Args:
            features (callable): Function mapping node IDs to feature tensors.
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
            use_cuda (bool): Whether to use GPU.
            gcn_mode (bool): If True, includes self-loops like GCN.
        """
        super(AttentionAggregator, self).__init__()
        
        self.features = features
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.gcn_mode = gcn_mode

        # Attention weight parameters
        self.weight = nn.Parameter(torch.FloatTensor(out_dim, in_dim))
        self.attention_vector = nn.Parameter(torch.FloatTensor(2 * out_dim, 1))

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention_vector)

    def forward(self, nodes, neighbor_sets, num_sample=10):
        """
        Aggregates feature representations of neighbors using attention.

        Args:
            nodes (list): List of nodes in a batch.
            neighbor_sets (list of sets): Each set contains the neighbors of a node.
            num_sample (int): Number of neighbors to sample (no sampling if None).

        Returns:
            torch.Tensor: Aggregated node embeddings.
        """
        # Sample neighbors if required
        if num_sample is not None:
            sampled_neighbors = [
                set(random.sample(neigh, num_sample)) if len(neigh) >= num_sample else neigh
                for neigh in neighbor_sets
            ]
        else:
            sampled_neighbors = neighbor_sets

        # Include self-loops in GCN mode
        if self.gcn_mode:
            sampled_neighbors = [neigh | {nodes[i]} for i, neigh in enumerate(sampled_neighbors)]

        # Flatten neighbors and create index mapping
        unique_nodes = list(set.union(*sampled_neighbors))
        node_index = {node: idx for idx, node in enumerate(unique_nodes)}

        # Construct adjacency mask
        row_indices = [i for i, neigh in enumerate(sampled_neighbors) for _ in neigh]
        col_indices = [node_index[n] for neigh in sampled_neighbors for n in neigh]

        mask = torch.zeros(len(sampled_neighbors), len(unique_nodes), device=self.device)
        mask[row_indices, col_indices] = 1

        # Retrieve feature matrices for nodes and neighbors
        node_features = self.features(torch.LongTensor(nodes).to(self.device))
        neighbor_features = self.features(torch.LongTensor(unique_nodes).to(self.device))

        # Transform features using learned weights
        node_transformed = node_features @ self.weight.T
        neighbor_transformed = neighbor_features @ self.weight.T

        # Compute attention scores
        num_nodes = node_transformed.shape[0]
        num_neighbors = neighbor_transformed.shape[0]

        combined = torch.cat(
            [
                node_transformed.repeat(1, num_neighbors).view(num_nodes * num_neighbors, -1),
                neighbor_transformed.repeat(num_nodes, 1),
            ],
            dim=1
        ).view(num_nodes, num_neighbors, -1)

        attention_scores = self.leaky_relu(combined @ self.attention_vector).squeeze(2)

        # Apply mask to attention scores
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = self.softmax(attention_scores)

        # Compute weighted sum of neighbor embeddings
        aggregated_features = attention_weights @ neighbor_transformed

        return aggregated_features