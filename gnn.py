import os, re
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from multiprocessing import Pool, cpu_count
from functools import partial

DATA_TRAIN = ['./storage/v1', './storage/v2', './storage/v3']
DATA_TEST = ['./storage/v4', './storage/v5']
CUT = 1.0
N_NODES = 187
ALARM_THRESHOLD = 0.6
PERSISTENCE = 1
START, END = PERSISTENCE, 80
EPS_ATTACK = 1e-4
AUG_PROB = 0.6
AUG_MAX_FLIP = 3
STORAGE_NODES = [86, 103, 106, 111, 112, 113, 114, 124, 126, 127, 129, 130]


# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORK; h' = σ(D^(-1/2) A D^(-1/2) h W + b)
class GraphConv(layers.Layer):
    def __init__(self, units, act='relu', reg=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.act = keras.activations.get(act)
        self.reg = reg

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='glorot_uniform',
                                 trainable=True, regularizer=keras.regularizers.l2(self.reg))
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inp, adj):
        return self.act(tf.matmul(adj, tf.matmul(inp, self.w)) + self.b)


def gcn(n_nodes, n_feat, adj, reduced=False):
    adj = adj + np.eye(adj.shape[0])
    d = np.power(np.sum(adj, axis=1), -0.5)
    d[np.isinf(d)] = 0.0
    a_norm = tf.constant(np.diag(d) @ adj @ np.diag(d), dtype=tf.float32)

    inp = layers.Input(shape=(n_nodes, n_feat))
    if reduced:
        x1 = GraphConv(32)(inp, a_norm)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.5)(x1)
        x1 = GraphConv(16)(x1, a_norm)
        x2 = layers.Dense(16, activation='relu')(inp)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.4)(x2)
    else:
        x1 = GraphConv(128)(inp, a_norm)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.3)(x1)
        x1 = GraphConv(64)(x1, a_norm)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.2)(x1)
        x2 = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.3)(x2)

    out = layers.Dense(1, activation='sigmoid')(layers.Concatenate()([x1, x2]))
    return keras.Model(inputs=inp, outputs=out)


class WeightedBCE(keras.losses.Loss):
    def __init__(self, pos_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def call(self, true, pred):
        true = tf.reshape(true, [-1])
        pred = tf.clip_by_value(tf.reshape(pred, [-1]), 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-true * tf.math.log(pred) * self.pos_weight - (1 - true) * tf.math.log(1 - pred))


def load_adj(path):
    adj = np.zeros((N_NODES, N_NODES))
    df = pd.read_csv(path)
    adj[df['source'], df['target']] = adj[df['target'], df['source']] = 1
    return adj


def accumulate(dv, dt, vs, ts, adj, esb=None):
    T = dv.shape[1]
    t_range = np.arange(1, T + 1)

    dv_mean = np.cumsum(dv, axis=1) / t_range
    dt_mean = np.cumsum(dt, axis=1) / t_range

    dv_std = np.sqrt(np.maximum((np.cumsum(dv ** 2, axis=1) / t_range) - (dv_mean ** 2), 0))
    dt_std = np.sqrt(np.maximum((np.cumsum(dt ** 2, axis=1) / t_range) - (dt_mean ** 2), 0))

    # vector corr
    deg = adj.sum(axis=1, keepdims=True) + 1e-8
    neighbors_dv_avg = (adj @ dv) / deg
    neighbors_dt_avg = (adj @ dt) / deg

    if T > 10:
        mu_y = neighbors_dv_avg.cumsum(axis=1) / t_range
        cov = np.cumsum(dv * neighbors_dv_avg, axis=1) / t_range - (dv_mean * mu_y)
        denom = dv_std * np.sqrt(np.maximum((np.cumsum(neighbors_dv_avg ** 2, axis=1) / t_range) - (mu_y ** 2), 0))
        dv_corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-8)

        mu_y_t = neighbors_dt_avg.cumsum(axis=1) / t_range
        cov_t = np.cumsum(dt * neighbors_dt_avg, axis=1) / t_range - (dt_mean * mu_y_t)
        denom_t = dt_std * np.sqrt(np.maximum((np.cumsum(neighbors_dt_avg ** 2, axis=1) / t_range) - (mu_y_t ** 2), 0))
        dt_corr = np.divide(cov_t, denom_t, out=np.zeros_like(cov_t), where=denom_t > 1e-8)
    else:
        dv_corr = np.zeros((dv.shape[0], 1))
        dt_corr = np.zeros((dt.shape[0], 1))

    return np.concatenate([
        dv[:, -1:], dt[:, -1:],
        dv_mean[:, -1:], dv_std[:, -1:], np.maximum.accumulate(np.abs(dv), axis=1)[:, -1:],
        dt_mean[:, -1:], dt_std[:, -1:], np.maximum.accumulate(np.abs(dt), axis=1)[:, -1:],
        np.mean(vs, axis=1, keepdims=True), np.std(vs, axis=1, keepdims=True),
        np.mean(ts, axis=1, keepdims=True), np.std(ts, axis=1, keepdims=True),
        np.percentile(dv, 90, axis=1, keepdims=True), np.percentile(dv, 10, axis=1, keepdims=True),
        np.percentile(dt, 90, axis=1, keepdims=True), np.percentile(dt, 10, axis=1, keepdims=True),
        neighbors_dv_avg[:, -1:], neighbors_dt_avg[:, -1:],
        (dv - neighbors_dv_avg)[:, -1:], (dt - neighbors_dt_avg)[:, -1:],
        dv_corr[:, -1:], dt_corr[:, -1:],
        esb
    ], axis=1)


def process_pkl(path, adj, start_t, end_t, augment=False):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    att_steps = [f"x{i}" for i in range(1, END + 1)]
    if len(att_steps) < end_t:
        return None

    vs = np.array([data['sch_v'][s] for s in att_steps[:END]]).T
    ts = np.array([data['sch_θ'][s] for s in att_steps[:END]]).T
    va = np.array([data['attack_v'].get(s, data['sch_v'][s]) for s in att_steps[:END]]).T
    ta = np.array([data['attack_θ'].get(s, data['sch_θ'][s]) for s in att_steps[:END]]).T
    dv, dt = va - vs, ta - ts

    esb = np.zeros((N_NODES, 1), dtype=np.float32)
    esb[STORAGE_NODES] = 1.0

    y_day = np.zeros(N_NODES, dtype=np.float32)
    if 'esset_btm_a' in data:   # v6-10
        y_day[np.array(data['esset_btm_a']['A']) - 1] = 1.0
    else:                       # v1-5
        y_day[np.array(data['esset_btm']['A']) - 1] = 1.0

    att_idx = np.where(y_day == 1)[0]
    attack_mask = np.zeros((N_NODES, END), dtype=np.float32)
    if len(att_idx) > 0:
        attack_mask[att_idx, :] = ((np.abs(dv[att_idx, :]) > EPS_ATTACK) |
                                   (np.abs(dt[att_idx, :]) > EPS_ATTACK)).astype(np.float32)
    attack_cum = np.maximum.accumulate(attack_mask, axis=1)

    # fast seq build
    x_arr = np.array([accumulate(dv[:, :t + 1], dt[:, :t + 1], vs[:, :t + 1], ts[:, :t + 1], adj, esb)
                      for t in range(start_t, end_t)])
    y_arr = attack_cum[:, start_t : end_t].T

    # aug via flip for training
    if augment and len(att_idx) > 0 and np.random.rand() < AUG_PROB:
        n_flip = min(len(att_idx), np.random.randint(1, AUG_MAX_FLIP + 1))
        flip_idx = np.random.choice(att_idx, n_flip, replace=False)

        # zero-out physics deviations (creates physically consistent normal samples)
        dv2, dt2, ac2 = dv.copy(), dt.copy(), attack_cum.copy()
        dv2[flip_idx, :] = 0.0
        dt2[flip_idx, :] = 0.0
        ac2[flip_idx, :] = 0.0

        x2 = np.array([accumulate(dv2[:, :t + 1], dt2[:, :t + 1], vs[:, :t + 1], ts[:, :t + 1], adj, esb)
                       for t in range(start_t, end_t)])
        y2 = ac2[:, start_t : end_t].T
        return np.concatenate([x_arr, x2], axis=0), np.concatenate([y_arr, y2], axis=0), f"{os.path.basename(path)}_aug"

    return x_arr, y_arr, os.path.basename(path)


def holdoff(pred, thresh, persist=1):
    res = np.zeros_like(pred, dtype=int)
    for t in range(persist - 1, len(pred)):
        res[t] = (pred[t - persist + 1 : t + 1] > thresh).all(axis=0).astype(int)
    return res


def smooth_operator(pred, window=1):
    res = np.zeros_like(pred)
    for t in range(len(pred)):
        res[t] = pred[max(0, t - window + 1) : t + 1].mean(axis=0)
    return res


def temporal_loader(dirs, adj, start, end, frac=1.0, files=None, augment=False):
    if isinstance(dirs, str): dirs = [dirs]
    if files is None:
        files = []
        for d in dirs:
            fs = sorted([os.path.join(d, x) for x in os.listdir(d)])
            if frac < 1.0:
                np.random.shuffle(fs)
                fs = fs[:int(frac * len(fs))]
            files.extend(fs)

    x_all, y_all, meta = [], [], []
    with Pool(min(cpu_count(), 8)) as p:
        res = list(tqdm(p.imap(partial(process_pkl, adj=adj, start_t=start, end_t=end, augment=augment), files), total=len(files)))

    for X, y, m in res:
        if X is not None:
            x_all.append(X)
            y_all.append(y)
            meta.extend([m] * len(X))
    return np.concatenate(x_all), np.concatenate(y_all), meta


if __name__ == "__main__":
    adj = load_adj('topology.csv')
    print(f"nodes: {N_NODES}, edges: {int(adj.sum()) // 2}")
    print("load training sets...")
    X_train, y_train, _ = temporal_loader(DATA_TRAIN, adj, START, END, CUT, augment=True)
    print("load testing sets...")
    X_test, y_test, meta = temporal_loader(DATA_TEST, adj, START, END, CUT, augment=False)
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    # split val
    np.random.seed(7)
    perm = np.random.permutation(len(X_train))
    n_val = int(0.1 * len(X_train))
    X_val, y_val = X_train[perm[:n_val]], y_train[perm[:n_val]]
    X_train, y_train = X_train[perm[n_val:]], y_train[perm[n_val:]]

    # std
    xm, xs = X_train.mean((0, 1), keepdims=True), X_train.std((0, 1), keepdims=True) + 1e-8
    X_train = (X_train - xm) / xs
    X_val = (X_val - xm) / xs
    X_test = (X_test - xm) / xs

    w = (y_train.size - y_train.sum()) / (y_train.sum() + 1e-8) * 0.01   # adjust factor to balance prec & rec
    print(f"pos weight: {w:.2f}")   # accounting augmented data

    model = gcn(N_NODES, X_train.shape[2], adj, reduced=CUT < 0.3)
    ep, bs = 30, 32
    lr = keras.optimizers.schedules.CosineDecay(2e-3, ep * (len(X_train) // bs), 1e-3)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=WeightedBCE(w),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.5),
                 keras.metrics.AUC(curve='PR'),
                 keras.metrics.Precision(thresholds=0.5),
                 keras.metrics.Recall(thresholds=0.5)]
    )

    h = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=ep, batch_size=bs, verbose=1,
                  callbacks=[keras.callbacks.EarlyStopping('val_recall', patience=5, restore_best_weights=True, mode='max')])

    # history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, m in enumerate(['loss', 'auc', 'precision', 'recall']):
        ax = axes[i // 2, i % 2]
        ax.plot(h.history[m], label='train')
        ax.plot(h.history[f'val_{m}'], label='val')
        ax.set_title(m.upper())
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # predict
    n_steps = END - START
    n_days = len(X_test) // n_steps
    raw = model.predict(X_test, verbose=0)[:, :, 0]

    ## calibrate
    #priors = np.percentile(raw, 25, axis=0)
    #print(f"bias (N131): {priors[130]:.3f}")
    #print(f"bias (N130): {priors[129]:.3f}")
    #calib = np.maximum(0, raw - 1.0 * priors)

    smooth = smooth_operator(raw)
    day_pred = smooth.reshape(n_days, n_steps, N_NODES)
    preds = holdoff(smooth, ALARM_THRESHOLD, PERSISTENCE)

    mode = ['normal', 'attacked']
    print(classification_report(y_test.flatten(), preds.flatten(), target_names=mode, digits=3))

    cm = confusion_matrix(y_test.flatten(), preds.flatten())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=mode, yticklabels=mode)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.show()

    # DEBUGGING
    node_p = day_pred.max(1).mean(0)
    node_l = y_test.max(0)
    print("\nattacked conf:")
    for i in np.where(node_l == 1)[0]: print(f" N{i + 1}: {node_p[i]:.4f}")
    print("\ntop false:")
    for i in np.where(node_l == 0)[0][np.argsort(node_p[np.where(node_l == 0)[0]])[-10:][::-1]]:
        print(f" N{i + 1}: {node_p[i]:.4f}")

    d = 1
    data = day_pred[d]
    tgt = y_test.reshape(n_days, n_steps, N_NODES)[d]

    base = dt.datetime(2000, 1, 1)  # arbitrary ref
    times = [(base + dt.timedelta(minutes=t * 15)).strftime("%H:%M") for t in range(START, END)]

    plt.figure(figsize=(14, 7))
    att_idx = np.where(tgt.max(0) == 1)[0]
    if len(att_idx) > 0:
        clrs = plt.get_cmap('tab10')(np.linspace(0, 1, len(att_idx)))
        for i, idx in enumerate(att_idx):
            plt.plot(data[:, idx], color=clrs[i])
            plt.text(len(data)-1, data[-1, idx], f' N{idx+1}', color=clrs[i], va='center')

    t_att = np.flatnonzero(tgt.max(1))
    if len(t_att) > 0:
        plt.axvspan(t_att[0], t_att[-1], color='r', alpha=0.05, label='Attack')
    plt.axhline(ALARM_THRESHOLD, color='k', ls='--', label='Thresh')

    plt.xticks(np.arange(0, len(times), 4), times[::4], rotation=45)
    plt.ylabel('confidence')
    plt.title(f'Attack Probability Evolution ({meta[d * n_steps]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()