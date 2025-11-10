import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# grid-specific parameters
DATA = ['./v1', './v2', './v3'] # totaling 809 days
N_NODES = 187
ATTACKED_NODES = [32, 33, 162]  # zero indexed
ATTACK_WINDOW = (16, 61)


class GraphConv(layers.Layer):
    # graph convolution: h' = σ(D^(-1/2) A D^(-1/2) h W + b)
    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs, adjacency):
        # adjacency @ inputs @ w: aggregate neighbor features, transform, activate
        return self.activation(tf.matmul(adjacency, tf.matmul(inputs, self.w)) + self.b)

def gcn(n_nodes, n_features, adj_matrix):
    inp = layers.Input(shape=(n_nodes, n_features))

    # symmetric normalization: D^(-1/2) A D^(-1/2) for stable gradients
    adj = adj_matrix + np.eye(adj_matrix.shape[0])  # add self-loops
    d_inv_sqrt = np.power(np.sum(adj, axis=1), -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    adj_norm = np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
    adj_tf = tf.constant(adj_norm, dtype=tf.float32)

    # encoder: extract spatial patterns through graph convolutions
    x = GraphConv(128)(inp, adj_tf)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = GraphConv(64)(x, adj_tf)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # residual connection: helps gradient flow
    x = layers.Add()([x, layers.Dense(64)(inp)])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # per-node binary classification
    out = GraphConv(1, activation='sigmoid')(x, adj_tf)
    return keras.Model(inputs=inp, outputs=out)

def load_adj_matrix(path):
    """grid topology from csv (undirected edges)"""
    adj = np.zeros((N_NODES, N_NODES))
    for _, row in pd.read_csv(path).iterrows():
        i, j = int(row['source']), int(row['target'])
        if i < N_NODES and j < N_NODES:
            adj[i, j] = adj[j, i] = 1
    return adj

def load_all_days(dirs, parse_day):
    if isinstance(dirs, str):  # single & list of dirs
        dirs = [dirs]

    X_list, y_list, d = [], [], []
    for dir in dirs:
        files = sorted(os.listdir(dir))
        for fname in tqdm(files, desc=f'Loading {os.path.basename(dir)}'):
            X, y = parse_day(os.path.join(dir, fname))
            X_list.append(X)
            y_list.append(y)
            d.append(f"{os.path.basename(dir)}/day{fname.split('day')[1].split('_')[0].split('.')[0]}")
    return np.stack(X_list), np.stack(y_list), d

class WeightedBCE(keras.losses.Loss):
    # weighted binary cross-entropy: higher penalty for missing attacks
    def __init__(self, pos_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.clip_by_value(tf.reshape(y_pred, [-1]), 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-y_true * tf.math.log(y_pred) * self.pos_weight - (1 - y_true) * tf.math.log(1 - y_pred))
