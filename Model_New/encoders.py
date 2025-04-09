import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.mlp import MLP

class Encoder(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator,
                 num_layers=3, num_sample=10, base_model=None, gcn=False, cuda=False, kernel="GCN"):
        super(Encoder, self).__init__()
        
        self.features = features  # Node feature matrix
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists  # Adjacency list
        self.aggregator = aggregator  # Aggregation function
        self.num_sample = num_sample  # Number of neighbors to sample
        self.gcn = gcn  # Whether to use GCN-style propagation
        self.embed_dim = embed_dim  # Embedding dimension
        self.cuda = cuda  # CUDA flag
        self.kernel = kernel  # Graph kernel type (GCN, GAT, GIN, etc.)
        self.device = torch.device("cuda" if cuda else "cpu")  # Device handling
        
        if base_model:
            self.base_model = base_model  # Optional base model for hierarchical learning
        
        # Xavier initialization for weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(embed_dim, feature_dim if gcn else 2 * feature_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Dynamic layer selection for MLP-based encoding
        self.num_layers = num_layers
        self.mlps = nn.ModuleList([MLP(2, feature_dim if i == 0 else embed_dim, embed_dim, embed_dim) for i in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(embed_dim) for _ in range(num_layers)])
        self.linears_prediction = nn.ModuleList([nn.Linear(feature_dim if i == 0 else embed_dim, embed_dim) for i in range(num_layers)])

    def forward(self, nodes):
        nodes = torch.LongTensor(nodes).to(self.device)  # Ensure nodes are on the correct device
        
        # GIN Kernel: Uses MLPs and batch norms
        if self.kernel == "GIN":
            neigh_feats = self.aggregator(nodes, self.adj_lists[nodes], self.num_sample, average="sum")  # Vectorized adjacency lookup
            self_feats = self.features(nodes)
            h = self_feats + neigh_feats  # Element-wise sum
            for layer in range(self.num_layers):
                h = F.relu(self.batch_norms[layer](self.mlps[layer](h)))  # MLP with batch norm
            return h.t()
        else:
            neigh_feats = self.aggregator(nodes, self.adj_lists[nodes], self.num_sample)  # Vectorized adjacency lookup
            
            if not self.gcn:
                self_feats = self.features(nodes)
                combined = torch.cat([self_feats, neigh_feats], dim=1)  # Concatenation for non-GCN models
            else:
                combined = neigh_feats  # Use neighbor features directly for GCN
            
            # GAT Kernel: Uses ELU activation, others use ReLU
            if self.kernel == "GAT":
                return F.elu(self.weight @ combined.t())
            return F.relu(self.weight @ combined.t())
