import os
import re
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from multiprocessing import Pool, cpu_count
from functools import partial

DATA_TRAIN = ['./v1', './v2', './v3']
DATA_TEST = ['./v4', './v5']
DEMON = 0   # portion of testset used for training
CUT = 1.0
N_NODES = 187
ALARM_THRESHOLD = 0.9
PERSISTENCE = 1
START, END = PERSISTENCE, 79


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
    adj = adj + 5 * np.eye(adj.shape[0])    # trust own sensors more than neighbours'
    d = np.power(np.sum(adj, axis=1), -0.5)
    d[np.isinf(d)] = 0.0
    a_norm = tf.constant(np.diag(d) @ adj @ np.diag(d), dtype=tf.float32)

    inp = layers.Input(shape=(n_nodes, n_feat))
    if reduced: # simpler architecture for limited dataset
        x1 = GraphConv(32)(inp, a_norm)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.5)(x1)
        x1 = GraphConv(16)(x1, a_norm)

        x2 = layers.Dense(16, activation='relu')(inp)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.4)(x2)
    else:
        # branch 1: neighbors
        x1 = GraphConv(128)(inp, a_norm)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.3)(x1)
        x1 = GraphConv(64)(x1, a_norm)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.2)(x1)

        # branch 2: self
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
    """grid topology from csv (undirected edges)"""
    adj = np.zeros((N_NODES, N_NODES))
    df = pd.read_csv(path)
    adj[df['source'], df['target']] = adj[df['target'], df['source']] = 1
    return adj


def accumulate(dv, dt, vs, ts, dfp_hist, dfq_hist, cent, adj):
    T = dv.shape[1]
    t_range = np.arange(1, T + 1)   # for vectorized feature map

    dv_mean = np.cumsum(dv, axis=1) / t_range
    dt_mean = np.cumsum(dt, axis=1) / t_range

    dv_std = np.sqrt(np.maximum((np.cumsum(dv ** 2, axis=1) / t_range) - (dv_mean ** 2), 0))
    dt_std = np.sqrt(np.maximum((np.cumsum(dt ** 2, axis=1) / t_range) - (dt_mean ** 2), 0))

    dv_max = np.maximum.accumulate(np.abs(dv), axis=1)
    dt_max = np.maximum.accumulate(np.abs(dt), axis=1)

    deg = adj.sum(axis=1, keepdims=True) + 1e-8
    neighbors_dv_avg = (adj @ dv) / deg
    neighbors_dt_avg = (adj @ dt) / deg

    dv_diff_neighbors = dv - neighbors_dv_avg
    dt_diff_neighbors = dt - neighbors_dt_avg

    #neighbors_dfp_avg = (adj @ dfp_hist) / deg
    #neighbors_dfq_avg = (adj @ dfq_hist) / deg

    if T > 10:
        # current means/stds (last column of the cumulative arrays)
        mu_y = neighbors_dv_avg.cumsum(axis=1) / t_range
        cov = np.cumsum(dv * neighbors_dv_avg, axis=1) / t_range - (dv_mean * mu_y)

        # recalculate neighbor std dev cumulatively
        denom = dv_std * np.sqrt(np.maximum((np.cumsum(neighbors_dv_avg ** 2, axis=1) / t_range) - (mu_y ** 2), 0))
        dv_corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-8)

        # repeat for voltage angle
        mu_y_t = neighbors_dt_avg.cumsum(axis=1) / t_range
        cov_t = np.cumsum(dt * neighbors_dt_avg, axis=1) / t_range - (dt_mean * mu_y_t)
        denom_t = dt_std * np.sqrt(np.maximum((np.cumsum(neighbors_dt_avg ** 2, axis=1) / t_range) - (mu_y_t ** 2), 0))
        dt_corr = np.divide(cov_t, denom_t, out=np.zeros_like(cov_t), where=denom_t > 1e-8)

        # take the last column to match feature shape
        dv_corr = dv_corr[:, -1:]
        dt_corr = dt_corr[:, -1:]
    else:
        dv_corr = np.zeros((dv.shape[0], 1))
        dt_corr = np.zeros((dt.shape[0], 1))

    feats = [
        dv[:, -1:], dt[:, -1:],
        dv_mean[:, -1:], dv_std[:, -1:], dv_max[:, -1:],
        dt_mean[:, -1:], dt_std[:, -1:], dt_max[:, -1:],
        #(dv**2).sum(axis=1, keepdims=True) / T,
        #(dt**2).sum(axis=1, keepdims=True) / T,
        np.mean(vs, axis=1, keepdims=True),
        np.std(vs, axis=1, keepdims=True),
        np.mean(ts, axis=1, keepdims=True),
        np.std(ts, axis=1, keepdims=True),
        np.percentile(dv, 90, axis=1, keepdims=True),
        np.percentile(dv, 10, axis=1, keepdims=True),
        np.percentile(dt, 90, axis=1, keepdims=True),
        np.percentile(dt, 10, axis=1, keepdims=True),
        #np.gradient(dv, axis=1)[:, -1:] if T > 1 else np.zeros((dv.shape[0], 1)),
        #np.gradient(dt, axis=1)[:, -1:] if T > 1 else np.zeros((dt.shape[0], 1)),
        neighbors_dv_avg[:, -1:], neighbors_dt_avg[:, -1:],
        dv_diff_neighbors[:, -1:], dt_diff_neighbors[:, -1:],
        #np.abs(dv_diff_neighbors).mean(axis=1, keepdims=True),
        #np.abs(dt_diff_neighbors).mean(axis=1, keepdims=True),
        dv_corr, dt_corr,
        #dfp_hist[:, -1:], dfq_hist[:, -1:],
        #neighbors_dfp_avg[:, -1:], neighbors_dfq_avg[:, -1:],
        cent
    ]
    return np.concatenate(feats, axis=1)


def process_pkl(path, cent, adj, start_t, end_t):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    att_steps = [f"x{i}" for i in range(1, END + 1)]
    if len(att_steps) < end_t:
        return None

    vs = np.array([data['sch_v'][s] for s in att_steps[:END]]).T
    ts = np.array([data['sch_θ'][s] for s in att_steps[:END]]).T
    va = np.array([data['attack_v'].get(s, data['sch_v'][s]) for s in att_steps[:END]]).T
    ta = np.array([data['attack_θ'].get(s, data['sch_θ'][s]) for s in att_steps[:END]]).T

    dv = va - vs
    dt = ta - ts

    def safe_load(d, s, n): # load power data
        if s not in d:
            return np.zeros(n)
        arr = np.asarray(d[s])
        return arr[:n] if len(arr) >= n else np.pad(arr, (0, n - len(arr)))

    fps = np.array([safe_load(data['sch_fp'], s, N_NODES) for s in att_steps[:END]]).T
    fqs = np.array([safe_load(data['sch_fq'], s, N_NODES) for s in att_steps[:END]]).T
    fpa = np.array([safe_load(data['attack_fp'], s, N_NODES) for s in att_steps[:END]]).T
    fqa = np.array([safe_load(data['attack_fq'], s, N_NODES) for s in att_steps[:END]]).T

    dfp = fpa - fps
    dfq = fqa - fqs

    y_day = np.zeros(N_NODES, dtype=np.float32)
    y_day[np.array(data['esset_btm']['A']) - 1] = 1.0

    x_list = []
    for t in range(start_t, end_t):
        sl = slice(None, t + 1)
        x_list.append(accumulate(dv[:, sl], dt[:, sl], vs[:, sl], ts[:, sl], dfp[:, sl], dfq[:, sl], cent, adj))

    dir = os.path.basename(os.path.dirname(path))
    pkl = re.sub(r'data_|_v\d+|\.pkl', '', os.path.basename(path))
    return np.array(x_list), np.tile(y_day, (len(x_list), 1)), f"{dir}/{pkl}"


def holdoff(pred, thresh, window):
    """mark timestep as attacked if prob > thresh for last 'window' consecutive steps"""
    result = np.zeros_like(pred, dtype=int)
    for t in range(window - 1, pred.shape[0]):  # start from 3rd timestep
        result[t, :] = (pred[t - window + 1 : t + 1, :] > thresh).all(axis=0).astype(int)
    return result


def smooth_operator(pred, window=1):
    smoothed = np.zeros_like(pred)
    for t in range(pred.shape[0]):
        start = max(0, t - window + 1)
        smoothed[t] = pred[start : t + 1].mean(axis=0)  # avg over time for each node
    return smoothed


def temporal_loader(dirs, adj, start, end, frac=1.0, files=None):
    if isinstance(dirs, str):
        dirs = [dirs]

    if files is None:
        all_files = []
        for d in dirs:
            f = sorted([os.path.join(d, x) for x in os.listdir(d)])
            if frac < 1.0:
                np.random.shuffle(f)
                f = f[:int(frac * len(f))]
            all_files.extend(f)
    else:
        all_files = files

    cent = np.sum(adj, axis=1, keepdims=True)
    cent = cent / (cent.max() + 1e-8)
    x_all, y_all, meta_all = [], [], []
    with Pool(min(cpu_count(), 8)) as pool:
        results = list(tqdm(pool.imap(partial(process_pkl, cent=cent, adj=adj, start_t=start, end_t=end), all_files),
                            total=len(all_files)))

    for result in results:
        if result is not None:
            X, y, meta = result
            x_all.append(X)
            y_all.append(y)
            meta_all.extend([meta] * len(X))    # repeat meta for each timestep

    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0), meta_all


if __name__ == "__main__":
    adj = load_adj('topology.csv')
    print(f"grid bus: {N_NODES}, edges: {int(adj.sum()) // 2}")
    if DEMON > 0:
        tous = [f for d in DATA_TEST for f in sorted([os.path.join(d, x) for x in os.listdir(d)])]
        np.random.seed(7)
        np.random.shuffle(tous)
        n_split = int(len(tous) * DEMON)

        print("loading training sets...")
        X_tr1, y_tr1, _ = temporal_loader(DATA_TRAIN, adj, START, END, CUT)
        X_tr2, y_tr2, _ = temporal_loader(None, adj, START, END, files=tous[:n_split])
        X_train = np.concatenate([X_tr1, X_tr2], axis=0)
        y_train = np.concatenate([y_tr1, y_tr2], axis=0)

        print("loading testing sets...")
        X_test, y_test, meta = temporal_loader(None, adj, START, END, files=tous[n_split:])

    else:
        print("loading training sets...")
        X_train, y_train, _ = temporal_loader(DATA_TRAIN, adj, START, END, frac=CUT)
        print("loading testing sets...")
        X_test, y_test, meta = temporal_loader(DATA_TEST, adj, START, END, frac=CUT)
    print(f"train {X_train.shape}, test {X_test.shape}")

    X_mean = X_train.mean((0, 1), keepdims=True)
    X_std = X_train.std((0, 1), keepdims=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    weight = (y_train.size - y_train.sum()) / (y_train.sum() + 1e-8) * 0.4  # balance prec & rec
    print(f"positive class weight: {weight:.2f}")
    print(f"attack ratio: train {y_train.mean():.4f}, test {y_test.mean():.4f}\n")

    model = gcn(N_NODES, X_train.shape[2], adj, reduced=CUT < 0.3)
    EP, BS = 50, 32
    lr_sch = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-3,
        decay_steps=EP * (len(X_train) // BS),
        alpha=1e-3
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_sch),
        loss=WeightedBCE(pos_weight=weight),
        metrics=[
            keras.metrics.BinaryAccuracy(name='acc', threshold=0.5),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='prec', thresholds=0.5),
            keras.metrics.Recall(name='rec', thresholds=0.5)
        ]
    )

    h = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EP, batch_size=BS, verbose=1,
                  callbacks=[keras.callbacks.EarlyStopping(monitor='val_rec', patience=10, restore_best_weights=True, mode='max')])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['loss', 'auc', 'prec', 'rec']
    for i, m in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(h.history[m], label='train')
        ax.plot(h.history[f'val_{m}'], label='val')
        ax.set_title(m.upper())
        ax.set_xlabel('ep')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    n_steps = END - START
    n_days = len(X_test) // n_steps
    pred_raw = model.predict(X_test, verbose=0)[:, :, 0]
    pred_smoothed = smooth_operator(pred_raw, 3)
    day_pred = pred_smoothed.reshape(n_days, n_steps, N_NODES)
    timestep_labels = y_test
    timestep_preds = holdoff(pred_smoothed, ALARM_THRESHOLD, PERSISTENCE)

    mode = ['normal', 'attacked']
    print(classification_report(timestep_labels.flatten(), timestep_preds.flatten(), target_names=mode))

    cm = confusion_matrix(timestep_labels.flatten(), timestep_preds.flatten())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=mode, yticklabels=mode)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.show()

    node_preds = day_pred.max(axis=1).mean(axis=0)
    node_labels = y_test.max(axis=0)

    # DEBUGGING
    print(f"\nattacked nodes (avg conf across timesteps):")
    for idx in np.where(node_labels == 1)[0]:
        print(f"  N{idx + 1}: {node_preds[idx]:.4f}")
    print(f"\ntop 10 false alarms:")
    normal = np.where(node_labels == 0)[0]
    for idx in normal[np.argsort(node_preds[normal])[-10:][::-1]]:
        print(f"  N{idx + 1}: {node_preds[idx]:.4f}")

    d = 1
    day_data = day_pred[d]
    tgt = y_test.reshape(n_days, n_steps, N_NODES)[d]

    base = dt.datetime(2000, 1, 1, 0, 0)    # arbitrary ref
    times = [(base + dt.timedelta(minutes=t * 15)).strftime("%H:%M") for t in range(START, END)]

    plt.figure(figsize=(14, 7))
    att_indices = np.where(tgt.max(axis=0) == 1)[0]
    if len(att_indices) > 0:
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(att_indices)))
        for i, idx in enumerate(att_indices):
            c = colors[i]
            plt.plot(day_data[:, idx], color=c, linewidth=2.5)
            plt.text(len(day_data) - 1, day_data[-1, idx], f' N{idx + 1}', color=c, va='center')

    plt.axvspan(*(np.flatnonzero(tgt.max(1))[[0, -1]]), color='r', alpha=0.05, label='Attack Window')
    plt.axhline(y=ALARM_THRESHOLD, color='k', linestyle='--', label=f'Thresh ({ALARM_THRESHOLD})')

    ticks = np.arange(0, len(times), 4)
    plt.xticks(ticks, [times[i] for i in ticks], rotation=45)
    plt.ylabel('confidence')
    plt.title(f'Attack Probability Evolution ({meta[d * n_steps]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()