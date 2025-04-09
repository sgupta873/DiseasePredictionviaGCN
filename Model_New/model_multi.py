import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

import time
import random
import numpy as np
import functools
import warnings
import sklearn.exceptions

from Model.encoders import Encoder
from Model.aggregators import MeanAggregator

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def timer(func):
    """Function decorator for timing."""
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} completed in {time.time() - start:.2f}s")
        return result
    return wrapped


class DiseaseClassifier(nn.Module):
    def __init__(self, output_size, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss_fn = nn.BCEWithLogitsLoss()
        embed_dim = encoder.embed_dim  # get the embedding dimension from the encoder
        self.weight = nn.Parameter(torch.randn(output_size, embed_dim))
        self.classifier_weights = nn.Parameter(torch.FloatTensor(output_size, encoder.embed_dim))
        init.xavier_uniform_(self.classifier_weights)
        self.linear = nn.Linear(encoder.embed_dim, 1)

    @staticmethod
    def compute_binary_loss(preds, targets):
        preds = torch.from_numpy(preds)
        log_loss = (targets * preds.clamp(1e-12).log() +
                    (1 - targets) * (1 - preds).clamp(1e-12).log()).mean()
        return -log_loss

    def forward(self, node_ids):
        embeddings = self.encoder(node_ids)  # shape: [embed_dim, batch_size]
        scores = self.weight @ embeddings     # shape: [num_classes, batch_size]
        return scores.T

    def compute_loss(self, node_ids, targets):
        outputs = self.forward(node_ids)
        return self.loss_fn(outputs, targets)

    def forward_hinge(self, node_ids):
        return self.linear(self.encoder(node_ids).T)

    def hinge_loss(self, node_ids, targets):
        outputs = self.forward_hinge(node_ids)
        return torch.mean(torch.clamp(1 - outputs * targets, min=0))


def evaluate_metrics(label, predictions, true_labels, node_indices, top_k=(1, 2, 3, 4, 5)):
    print(f"\n{'-' * 100}\n{label}\n")

    true_tensor = torch.LongTensor(true_labels[node_indices])
    top_k_max = max(top_k)
    batch_size = true_tensor.size(0)

    _, ranked_preds = predictions.topk(top_k_max, 1)
    hits = torch.zeros_like(ranked_preds)

    for i in range(batch_size):
        for k in range(top_k_max):
            hits[i, k] = 1 if true_tensor[i][ranked_preds[i, k]] == 1 else 0
    hits = hits.T

    correct_per_sample = true_tensor.sum(1).float()

    for k in top_k:
        correct_top_k = hits[:k].sum(0).float()
        prec, rec = 0.0, 0.0
        for i in range(batch_size):
            prec += correct_top_k[i] / k
        prec /= batch_size

        recall = (correct_top_k / correct_per_sample).sum() / batch_size
        f1 = 2 * prec * recall / (prec + recall + 1e-8)

        print(f"Precision@{k}: {prec:.5f}, Recall@{k}: {recall:.5f}, F1@{k}: {f1:.5f}")


class DiseasePredictionPipeline:
    def __init__(self, node_features, binary_labels, num_classes, multilabels, graph, feat_dim,
                 enc_depth, enc_dims, sample_sizes, train_idx, test_idx,
                 kernel_type='gcn', top_k=(1, 2, 3, 4, 5), class_weights_enabled=False,
                 class_weights=[0.5, 0.5], use_gcn=False, agg_with_gcn=True, use_cuda=False):

        self.train_ids = train_idx.copy()
        self.test_ids = test_idx.copy()

        self.rare_test_ids = [i for i in np.where((binary_labels > 0))[0] if i in self.test_ids]
        self.rare_test_indices = [self.test_ids.index(i) for i in self.rare_test_ids]

        self.features = nn.Embedding(len(node_features), feat_dim)
        self.features.weight = nn.Parameter(torch.FloatTensor(node_features), requires_grad=False)

        self.label_data = multilabels
        self.binary_labels = binary_labels

        self.graph = graph
        self.enc_dims = enc_dims
        self.sample_sizes = sample_sizes
        self.kernel = kernel_type
        self.top_k = top_k
        self.feature_dim = feat_dim
        self.encoder_depth = enc_depth
        self.class_weights_flag = class_weights_enabled
        self.class_weights = torch.FloatTensor(class_weights)

        self.use_gcn = use_gcn
        self.agg_gcn = agg_with_gcn
        self.cuda = use_cuda

        self.build_encoder_stack()

    def build_encoder_stack(self):
        self.agg_layers = []
        self.enc_layers = []

        current_input = self.features
        current_dim = self.feature_dim

        for i in range(len(self.enc_dims)):
            agg = MeanAggregator(current_input, features_dim=current_dim,
                                 cuda=self.cuda, kernel=self.kernel, gcn=self.agg_gcn)
            enc = Encoder(current_input, current_dim, self.enc_dims[i], self.graph,
                          agg, gcn=self.use_gcn, cuda=self.cuda, kernel=self.kernel)
            current_input = lambda nodes: enc(nodes).T
            current_dim = self.enc_dims[i]
            self.agg_layers.append(agg)
            self.enc_layers.append(enc)

        for i, enc in enumerate(self.enc_layers):
            enc.num_samples = self.sample_sizes[i]

    def get_classifier(self, output_size, enc_index):
        return DiseaseClassifier(output_size, self.enc_layers[enc_index - 1])

    def train_model(self, model, epochs=100, batch_size=512, lr=0.01):
        np.random.seed(1)
        random.seed(1)

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        epoch_times = []

        for epoch in range(epochs):
            batch_nodes = self.train_ids[:batch_size]
            random.shuffle(self.train_ids)

            optimizer.zero_grad()
            label_batch = Variable(torch.FloatTensor(self.label_data[np.array(batch_nodes)]))
            loss = model.compute_loss(batch_nodes, label_batch)
            loss.backward()
            optimizer.step()
            epoch_times.append(time.time())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        print(f"\nAverage Training Time per Epoch: {np.mean(np.diff(epoch_times)):.4f}s\n")

    def execute(self, epochs, batch_size, lr):
        classifier = self.get_classifier(self.label_data.shape[1], self.encoder_depth)
        self.train_model(classifier, epochs, batch_size, lr)

        predictions = classifier.forward(self.test_ids)

        evaluate_metrics("Overall Evaluation", predictions,
                         self.label_data, self.test_ids, top_k=self.top_k)

        print(f"Rare Samples Count: {len(self.rare_test_indices)}")
        evaluate_metrics("Rare Class Evaluation", predictions[self.rare_test_indices],
                         self.label_data, self.rare_test_ids, top_k=self.top_k)
