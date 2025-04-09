import functools
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, recall_score, 
    precision_score, roc_curve, auc, precision_recall_fscore_support
)
from collections import defaultdict
from Model.encoders import Encoder
from Model.aggregators import MeanAggregator
from Utils.RARE_INFO import RareInfo
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def log_execution_time(func):
    """Decorator to log execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f'{func.__name__} executed in {duration:.4f}s')
        return result
    return wrapper

class UnsupervisedGraphSage(nn.Module):
    """GraphSAGE model for unsupervised learning."""
    def __init__(self, encoder):
        super(UnsupervisedGraphSage, self).__init__()
        self.encoder = encoder
        self.weight = nn.Parameter(torch.FloatTensor(1, encoder.embed_dim))
        self.log_sigmoid = nn.LogSigmoid()
        self.mse_loss_fn = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_u, node_v):
        embed_u = self.encoder(node_u)
        embed_v = self.encoder(node_v)
        return nn.functional.cosine_similarity(embed_u.t(), embed_v.t())

    def mse_loss(self, nodes, labels):
        embeddings = self.encoder(nodes)
        scores = self.weight.mm(embeddings)
        return self.mse_loss_fn(scores.squeeze(), labels.squeeze())

    def positive_loss(self, node_u, node_v):
        return self.log_sigmoid(self.forward(node_u, node_v))

    def negative_loss(self, nodes_u, negative_samples):
        loss = 0
        for u in nodes_u:
            embed_u = self.encoder([u])
            embed_negatives = self.encoder(list(negative_samples[u]))
            scores = nn.functional.cosine_similarity(embed_u.t(), embed_negatives.t())
            loss += self.log_sigmoid(torch.mean(-scores))
        return loss

    def compute_loss(self, nodes_u, nodes_v, negative_samples):
        return -sum(self.positive_loss(nodes_u, nodes_v)) - self.negative_loss(nodes_u, negative_samples)

def evaluate_performance(data_name, predictions, true_labels, indices, roc=True):
    """Evaluate model performance using various metrics."""
    print("----" * 25)
    print(f"\nEvaluation Results for {data_name}:\n")

    pred_np = predictions.data.numpy().argmax(axis=1)
    true_np = true_labels[indices]

    if roc:
        fpr, tpr, _ = roc_curve(true_np, predictions.data.numpy().max(axis=1))
        print("ROC AUC Score:", roc_auc_score(true_np, predictions.data.numpy().max(axis=1)))

    print("Precision - Recall - F1 Score:")
    print(precision_recall_fscore_support(true_np, pred_np))
    print("\nMacro F1:", f1_score(true_np, pred_np, average="macro"))
    print("Macro Recall:", recall_score(true_np, pred_np, average="macro"))
    print("Macro Precision:", precision_score(true_np, pred_np, average="macro"))
    print()
    print("Weighted F1:", f1_score(true_np, pred_np, average="weighted"))
    print("Weighted Recall:", recall_score(true_np, pred_np, average="weighted"))
    print("Weighted Precision:", precision_score(true_np, pred_np, average="weighted"))
    print()

class RarePredictor:
    """Predictor class for rare event detection using GraphSAGE."""
    def __init__(self, feature_matrix, binary_labels, multi_labels, adjacency_list,
                 feature_dim, enc_layer_sizes, num_samples,
                 train_set, test_set, cuda=False, walk_length=5, num_walks=6):
        
        self.cuda = cuda
        self.train_set = train_set
        self.test_set = [i for i in np.where((multi_labels < RareInfo().OTHERS))[0] if i in test_set]
        self.multi_train_set = [i for i in np.where((multi_labels > RareInfo().NON_RARE) & 
                                                    (multi_labels < RareInfo().OTHERS))[0] if i in train_set]
        self.multi_test_set = [i for i in np.where((multi_labels > RareInfo().NON_RARE) & 
                                                   (multi_labels < RareInfo().OTHERS))[0] if i in test_set]
        
        self.binary_labels = binary_labels
        self.multi_labels = multi_labels
        self.feature_dim = feature_dim
        self.adjacency_list = adjacency_list
        self.enc_layer_sizes = enc_layer_sizes
        self.num_samples = num_samples
        self.walk_length = walk_length
        self.num_walks = num_walks
        
        self.node_features = nn.Embedding(len(feature_matrix), feature_dim)
        self.node_features.weight = nn.Parameter(torch.FloatTensor(feature_matrix), requires_grad=False)
        
        self.agg1 = MeanAggregator(self.node_features, cuda=cuda)
        self.enc1 = Encoder(self.node_features, feature_dim, enc_layer_sizes[0], adjacency_list, self.agg1, gcn=True, cuda=cuda)
        self.agg2 = MeanAggregator(lambda nodes: self.enc1(nodes).t(), cuda=cuda)
        self.enc2 = Encoder(lambda nodes: self.enc1(nodes).t(), self.enc1.embed_dim, enc_layer_sizes[1], adjacency_list, self.agg2, gcn=True, cuda=cuda)
        
        self.neg_samples = self.generate_negative_samples()

    def generate_negative_samples(self):
        """Generate negative samples for training."""
        neg_samples = {}
        for node in range(len(self.adjacency_list)):
            neighbors = {node}
            frontier = {node}
            for _ in range(self.walk_length):
                frontier = {neighbor for outer in frontier for neighbor in self.adjacency_list[int(outer)]} - neighbors
                neighbors |= frontier
            far_nodes = set(self.train_set) - neighbors
            neg_samples[node] = random.sample(far_nodes, 1) if far_nodes else []
        return neg_samples

    def train_unsupervised(self, epochs=100, batch_size=512, learning_rate=0.01):
        """Train the GraphSAGE model using unsupervised learning."""
        model = UnsupervisedGraphSage(self.enc2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            batch_nodes = random.sample(self.train_set, batch_size)
            optimizer.zero_grad()
            pos_pairs = [(node, neighbor) for node in batch_nodes for neighbor in self.adjacency_list[node]]
            nodes_u, nodes_v = zip(*pos_pairs)
            loss = model.compute_loss(nodes_u, nodes_v, self.neg_samples)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
