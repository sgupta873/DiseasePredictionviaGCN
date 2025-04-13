import functools

import sklearn
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random

from Model.encoders import Encoder
from Model.aggregators import MeanAggregator

import warnings
import sklearn.exceptions


warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        res = func(*args, **kw)
        end_time = time.time()
        print('%s executed in %ss' % (func.__name__, end_time - start_time))
        return res

    return wrapper


class DiseasesClassifier(nn.Module):

    def __init__(self, num_classes, enc):
        super(DiseasesClassifier, self).__init__()
        self.enc = enc

        # self.xent = nn.CrossEntropyLoss()
        self.xent = nn.BCEWithLogitsLoss()

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

        self.a = nn.Linear(enc.embed_dim, 1)

    @staticmethod
    def binary_loss(y_pred, y):
        y_pred = torch.from_numpy(y_pred)
        logits = (y * y_pred.clamp(1e-12).log() + (1 - y)
                  * (1 - y_pred).clamp(1e-12).log()).mean()
        return -logits

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels)

    def forward_hinge(self, nodes):
        embeds = self.enc(nodes)
        return self.a(embeds.t())  # a = nn.Linear(enc.embed_dim, 1)

    def hinge_loss(self, nodes, labels):
        h_loss = self.forward_hinge(nodes)
        return torch.mean(torch.clamp(1 - h_loss * labels, min=0))


def evaluate(data_name, val_output, test_labels, val, topk=(1, 2, 3, 4, 5,)):
    print("----" * 25)
    print()
    print("%s: " % data_name)

    # shape: batchnum * classnum
    target = torch.LongTensor(test_labels[val])
    output = val_output  # shape: batchnum * classnum

    # print(target.shape, output.shape)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print(pred)
    correct = torch.zeros_like(pred)
    for i in range(batch_size):
        for k in range(maxk):
            correct[i, k] = 1 if target[i][pred[i, k]] == 1 else 0
    correct = correct.t()

    correct_target = target.sum(1, keepdim=True).squeeze().float()

    for k in topk:
        correct_k = correct[:k].sum(0, keepdim=True).squeeze().float()

        precision_k = 0.0
        # recall_k = 0.0
        for i in range(0, batch_size):
            # _k = k if k < correct_target[i].data else correct_target[i]
            _k = k
            precision_k += correct_k[i] / _k
            # recall_k += correct_k[i] / correct_target[i]
        precision_k = precision_k / batch_size
        # recall_k = recall_k / batch_size

        # print("precision @", k, precision_k.data)
        # print("recall @", k, recall_k.data)

        # precision_k = correct_k / k
        # precision_k = precision_k.sum() / batch_size

        recall_k = correct_k / correct_target
        recall_k = recall_k.sum() / batch_size

        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)

        print("precision @ %d : %.5f, recall @ %d : %.5f, f1 @ %d : %.5f" % (
            k, precision_k.data, k, recall_k.data, k, f1_k.data))
        # print("precision @", k, precision_k.data)
        # print("recall @", k, recall_k.data)
        # print("f1 @", k, f1_k.data)

        # print("precision@%d: %f" & (k, precision_k))
        # print("recall@%d: %f" & (k, recall_k))
    print()


class DiseasesPredictor:
    def __init__(self, feat_data, b_labels, multi_class_num, labels, adj_lists, feature_dim,
                 train_enc_num, train_enc_dim, train_sample_num,
                 train, test,
                 kernel='gcn',  topk=(1, 2, 3, 4, 5,),
                 weights_flag=False, weights=[0.5, 0.5],
                 gcn=False, agg_gcn=True, cuda=False, agg_type = "Mean"):

        self.cuda = cuda
        self.gcn = gcn
        self.agg_gcn = agg_gcn
        self.agg_type = agg_type
        self.train_original = train.copy()
        self.test_original = test.copy()

        self.train = train
        self.test = test
        self.test_rare = [i for i in np.where(
            (b_labels > 0))[0].squeeze() if i in self.test]
        self.test_rare_index = [self.test.index(i) for i in self.test_rare]

        self.b_labels = b_labels
        self.labels = labels

        self.bi_class_num = 2
        self.multi_class_num = multi_class_num

        # nodes' features (random setting)
        self.features = nn.Embedding(len(feat_data), feature_dim)
        self.features.weight = nn.Parameter(
            torch.FloatTensor(feat_data), requires_grad=False)
        self.adj_lists = adj_lists  # edges' information

        # model parameters
        self.train_enc_dim = train_enc_dim
        self.train_enc_num = train_enc_num
        self.kernel = kernel
        self.attention = True if kernel == "GAT" else False
        self.feature_dim = feature_dim
        self.train_sample_num = train_sample_num

        self.topk = topk

        # weighted cross-entropy
        self.weights_flag = weights_flag
        self.class_weights = torch.FloatTensor(weights)

        # labels for test
        self.test_b_labels = b_labels
        self.test_adj = adj_lists

        # inductive settings
        self.is_inductive = False
        self.test_sample_num = train_sample_num
        self.test_features = self.features

        # build aggregator and encoders
        # default: transductive setting

        self.agg1 = MeanAggregator(self.features, features_dim=feature_dim,
                                   use_cuda=self.cuda, mode=self.kernel)
        self.enc1 = Encoder(self.features, feature_dim, train_enc_dim[0], adj_lists, self.agg1,
                            gcn=self.gcn, cuda=self.cuda, kernel=self.kernel,agg_type = self.agg_type)

        self.agg2 = MeanAggregator(lambda nodes: self.enc1(nodes).t(), features_dim=self.enc1.embed_dim,
                                   use_cuda=self.cuda, mode=self.kernel)
        self.enc2 = Encoder(lambda nodes: self.enc1(nodes).t(), self.enc1.embed_dim, train_enc_dim[1], adj_lists,
                            self.agg2, base_model=self.enc1, gcn=self.gcn, cuda=self.cuda, kernel=self.kernel,agg_type = self.agg_type)

        self.agg3 = MeanAggregator(lambda nodes: self.enc2(nodes).t(), features_dim=self.enc2.embed_dim,
                                   use_cuda=self.cuda, mode=self.kernel)
        self.enc3 = Encoder(lambda nodes: self.enc2(nodes).t(), self.enc2.embed_dim, train_enc_dim[2], adj_lists,
                            self.agg3, base_model=self.enc2, gcn=self.gcn, cuda=self.cuda, kernel=self.kernel,agg_type = self.agg_type)
        self.agg4 = MeanAggregator(lambda nodes: self.enc3(nodes).t(), features_dim=self.enc3.embed_dim,
                                   use_cuda=self.cuda, mode=self.kernel)
        self.enc4 = Encoder(lambda nodes: self.enc3(nodes).t(), self.enc3.embed_dim, train_enc_dim[3], adj_lists,
                            self.agg4, base_model=self.enc3, gcn=self.gcn, cuda=self.cuda, kernel=self.kernel,agg_type = self.agg_type)
        self.enc1.num_samples = self.train_sample_num[0]
        self.enc2.num_samples = self.train_sample_num[1]
        self.enc3.num_samples = self.train_sample_num[2]
        self.enc4.num_samples = self.train_sample_num[3]

    def set_classifier(self, class_num, train_enc_num):
        classifier = DiseasesClassifier(class_num, self.enc2)
        if train_enc_num == 1:
            classifier = DiseasesClassifier(class_num, self.enc1)
        elif train_enc_num == 2:
            classifier = DiseasesClassifier(class_num, self.enc2)
        elif train_enc_num == 3:
            classifier = DiseasesClassifier(class_num, self.enc3)
        elif train_enc_num == 4:
            classifier = DiseasesClassifier(class_num, self.enc4)
        return classifier

    def run(self, loop_num, batch_num, lr):
        print("Running predictor...") 
        if loop_num is None:
            loop_num = [100, 500]
            

        multi_classifier = self.set_classifier(
            class_num=self.multi_class_num, train_enc_num=self.train_enc_num)

        self.__train__(multi_classifier, train=self.train, labels=self.labels,
                       loop_num=loop_num, batch_num=batch_num, lr=lr)

        multi_result_direct = multi_classifier.forward(self.test)

        evaluate("multi classification (overall)",
                 multi_result_direct,
                 self.labels,
                 self.test,
                 topk=self.topk)
        print("len of rare:", len(self.test_rare_index))
        evaluate("multi classification (rare)",
                 multi_result_direct[self.test_rare_index],
                 self.labels,
                 self.test_rare,
                 topk=self.topk)

    def __train__(self, selected_model, train, labels, loop_num=100, batch_num=512, lr=0.01):
        np.random.seed(1)
        random.seed(1)

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, selected_model.parameters()), lr=lr)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, selected_model.parameters()), lr=lr, betas=(0.9, 0.99))
        times = []

        for batch in range(loop_num):
            batch_nodes = train[:batch_num]
            random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = selected_model.loss(batch_nodes,
                                       Variable(torch.FloatTensor(labels[np.array(batch_nodes, dtype=np.int64)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            print(batch, loss.data)

        print()
        print("Average batch time:", np.mean(times))
        print()
