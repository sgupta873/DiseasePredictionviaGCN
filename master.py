#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import time
import pickle
from Model_New.model_multi import DiseasePredictionPipeline  # Updated class name

# Load data from pickled files
file_path = "./data/sample_data/sample_garph"
node_list = pickle.load(open(file_path + ".nodes.pkl", "rb"))
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
rare_patient = pickle.load(open(file_path + ".rare.label.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
node_map = pickle.load(open(file_path + ".map.pkl", "rb"))
train_nodes = pickle.load(open(file_path + ".train.pkl", "rb"))
test_nodes = pickle.load(open(file_path + ".test.pkl", "rb"))

# Configuration settings
multi_class_num = 108
feature_dim = 10000
epochs = 1000
batch_size = 200
learning_rate = 0.3
train_enc_dims = [1000, 1000, 1000, 1000]
num_samples = [5, 5, 5, 5]

# Simulated feature data
feat_data = np.random.random((50000, feature_dim))

# Start timer
start_time = time.time()

# Instantiate the model pipeline
model = DiseasePredictionPipeline(
    node_features=feat_data,
    binary_labels=rare_patient,
    num_classes=multi_class_num,
    multilabels=labels,
    graph=adj_lists,
    feat_dim=feature_dim,
    enc_depth=1,  # Depth of encoder to use during classification
    enc_dims=train_enc_dims,
    sample_sizes=num_samples,
    train_idx=train_nodes,
    test_idx=test_nodes,
    kernel_type='GCN',
    top_k=(1, 2, 3, 4, 5)
)

# Run the training and evaluation process
model.execute(epochs, batch_size, learning_rate)

# Output summary
print("Feature dimension:", feature_dim)
print("Encoder dimensions:", train_enc_dims)
print("Total run time: {:.2f}s".format(time.time() - start_time))


# In[ ]:




