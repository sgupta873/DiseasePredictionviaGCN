import functools
import time
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.autograd import Variable
from collections import defaultdict
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning

from Model.encoders import Encoder
from Model.aggregators import MeanAggregator
from Utils.RARE_INFO import RareInfo

# Suppress warnings for undefined metrics in sklearn
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Function decorator for logging execution time
def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.5f}s")
        return result
    return wrapper

class DiseaseClassifier(nn.Module):
    """A neural network model for disease classification using embeddings."""

    def __init__(self, num_classes, encoder):
        super(DiseaseClassifier, self).__init__()
        self.encoder = encoder
        self.loss_fn = nn.CrossEntropyLoss()  # Cross-entropy for multi-class classification

        # Learnable weight matrix
        self.weights = nn.Parameter(torch.FloatTensor(num_classes, encoder.embed_dim))
        init.xavier_uniform_(self.weights)

        # Linear layer for hinge loss computation
        self.linear_layer = nn.Linear(encoder.embed_dim, 1)

    @staticmethod
    def compute_binary_loss(predictions, targets):
        """Computes binary cross-entropy loss with numerical stability."""
        predictions = torch.from_numpy(predictions)
        log_probs = (targets * predictions.clamp(1e-12).log()) + ((1 - targets) * (1 - predictions).clamp(1e-12).log())
        return -log_probs.mean()

    def forward(self, nodes):
        embeddings = self.encoder(nodes)
        scores = self.weights.mm(embeddings)
        return scores.t()

    def compute_loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.loss_fn(scores, labels.squeeze())

    def compute_hinge_loss(self, nodes, labels):
        hinge_output = self.linear_layer(self.encoder(nodes).t())
        return torch.mean(torch.clamp(1 - hinge_output * labels, min=0))

def evaluate_performance(name, predictions, true_labels, indices, calculate_roc=True):
    """Evaluates model performance using classification metrics."""
    print("-" * 100)
    print(f"\n{name} Classification Results:")

    predicted_classes = predictions.data.numpy().argmax(axis=1)
    
    if calculate_roc:
        fpr, tpr, _ = metrics.roc_curve(true_labels[indices], predicted_classes)
        auc_score = metrics.auc(fpr, tpr)
        print("ROC AUC Score:", metrics.roc_auc_score(true_labels[indices], predictions.data.numpy().max(axis=1)))

    print("Precision - Recall - F1 Score:")
    print(metrics.precision_recall_fscore_support(true_labels[indices], predicted_classes))

    print(f"\nMacro F1 Score: {metrics.f1_score(true_labels[indices], predicted_classes, average='macro')}")
    print(f"Macro Recall: {metrics.recall_score(true_labels[indices], predicted_classes, average='macro')}")
    print(f"Macro Precision: {metrics.precision_score(true_labels[indices], predicted_classes, average='macro')}")
    
    print(f"\nWeighted F1 Score: {metrics.f1_score(true_labels[indices], predicted_classes, average='weighted')}")
    print(f"Weighted Recall: {metrics.recall_score(true_labels[indices], predicted_classes, average='weighted')}")
    print(f"Weighted Precision: {metrics.precision_score(true_labels[indices], predicted_classes, average='weighted')}\n")

class DiseasePredictor:
    """Graph-based disease prediction model leveraging neural networks for feature aggregation."""

    def __init__(self, features, binary_labels, multi_labels, adjacency_list, feature_size, 
                 encoder_count, encoder_dims, sample_sizes, train_indices, test_indices,
                 model_type='gcn', use_weights=False, class_weights=[0.5, 0.5], 
                 use_gcn=False, use_agg_gcn=True, use_cuda=False):
        
        # Device and model settings
        self.use_cuda = use_cuda
        self.use_gcn = use_gcn
        self.use_agg_gcn = use_agg_gcn
        self.train_indices = train_indices
        self.test_indices = test_indices

        # Identify rare disease samples in test set
        self.rare_test_samples = [i for i in np.where(binary_labels > 0)[0].squeeze() if i in test_indices]
        self.rare_test_indices = [test_indices.index(i) for i in self.rare_test_samples]

        # Labels
        self.binary_labels = binary_labels
        self.multi_labels = multi_labels
        self.num_binary_classes = 2
        self.num_multi_classes = multi_labels.max() + 1

        # Feature embedding setup
        self.features = nn.Embedding(len(features), feature_size)
        self.features.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
        self.adjacency_list = adjacency_list

        # Model parameters
        self.encoder_dims = encoder_dims
        self.encoder_count = encoder_count
        self.model_type = model_type
        self.use_attention = model_type == "GAT"
        self.feature_size = feature_size
        self.sample_sizes = sample_sizes

        # Weighted loss settings
        self.use_weights = use_weights
        self.class_weights = torch.FloatTensor(class_weights)

        # Aggregator and encoders
        self.agg1 = MeanAggregator(self.features, feature_size, use_cuda, model_type, use_agg_gcn)
        self.enc1 = Encoder(self.features, feature_size, encoder_dims[0], adjacency_list, 
                            self.agg1, gcn=use_gcn, cuda=use_cuda, kernel=model_type)

        self.agg2 = MeanAggregator(lambda nodes: self.enc1(nodes).t(), self.enc1.embed_dim, use_cuda, model_type, use_agg_gcn)
        self.enc2 = Encoder(lambda nodes: self.enc1(nodes).t(), self.enc1.embed_dim, encoder_dims[1], adjacency_list, 
                            self.agg2, base_model=self.enc1, gcn=use_gcn, cuda=use_cuda, kernel=model_type)

        self.agg3 = MeanAggregator(lambda nodes: self.enc2(nodes).t(), self.enc2.embed_dim, use_cuda, model_type, use_agg_gcn)
        self.enc3 = Encoder(lambda nodes: self.enc2(nodes).t(), self.enc2.embed_dim, encoder_dims[2], adjacency_list, 
                            self.agg3, base_model=self.enc2, gcn=use_gcn, cuda=use_cuda, kernel=model_type)

        self.agg4 = MeanAggregator(lambda nodes: self.enc3(nodes).t(), self.enc3.embed_dim, use_cuda, model_type, use_agg_gcn)
        self.enc4 = Encoder(lambda nodes: self.enc3(nodes).t(), self.enc3.embed_dim, encoder_dims[3], adjacency_list, 
                            self.agg4, base_model=self.enc3, gcn=use_gcn, cuda=use_cuda, kernel=model_type)

        # Define sample counts per encoder
        self.enc1.num_samples = sample_sizes[0]
        self.enc2.num_samples = sample_sizes[1]
        self.enc3.num_samples = sample_sizes[2]
        self.enc4.num_samples = sample_sizes[3]

    def initialize_classifier(self, num_classes, encoder_level):
        """Selects the appropriate encoder level for classification."""
        classifiers = [self.enc1, self.enc2, self.enc3, self.enc4]
        return DiseaseClassifier(num_classes, classifiers[encoder_level - 1])

    def train_and_evaluate(self, epochs, batch_size, learning_rate):
        """Runs the training and evaluation pipeline for disease classification."""
        classifier = self.initialize_classifier(self.num_multi_classes, self.encoder_count)
        self._train_model(classifier, self.train_indices, self.multi_labels, epochs, batch_size, learning_rate)

        predictions = classifier.forward(self.test_indices)

        evaluate_performance("Overall Multi-Class Classification", predictions, self.multi_labels, self.test_indices, False)
        evaluate_performance("Rare Disease Classification", predictions[self.rare_test_indices], self.multi_labels, self.rare_test_samples, False)

    def _train_model(self, model, train_data, labels, epochs, batch_size, learning_rate):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        for epoch in range(epochs):
            batch_nodes = random.sample(train_data, batch_size)
            optimizer.zero_grad()
            loss = model.compute_loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes, dtype=np.int64)])))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
