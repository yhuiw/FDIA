import os
import datetime
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

DATA_TRAIN = ['./v1', './v2', './v3']
DATA_TEST = ['./v4', './v5']

N_NODES = 187
START, END = 30, 79
DATA_FRAC = 1.0
IS_REDUCED = DATA_FRAC < 0.3
ALARM_THRESHOLD = 0.5
USE_ALL_FEATURE = False
PERSISTENCE = 4 # alert iff consecutive steps to reduce false alarm


class GraphConv(layers.Layer):
    # h' = σ(D^(-1/2) A D^(-1/2) h W + b); SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORK
    def __init__(self, units, act='relu', reg=1e-4, **kwargs):
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
    # sym norm adj
    adj = adj + np.eye(adj.shape[0])
    d = np.power(np.sum(adj, axis=1), -0.5)
    d[np.isinf(d)] = 0.0
    a_norm = tf.constant(np.diag(d) @ adj @ np.diag(d), dtype=tf.float32)

    inp = layers.Input(shape=(n_nodes, n_feat))
    if reduced:
        x = GraphConv(32)(inp, a_norm)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = GraphConv(16)(x, a_norm)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Add()([x, layers.Dense(16)(inp)])
        x = layers.Activation('relu')(x)
    else:
        # increased capacity for weak nodes
        x = GraphConv(256)(inp, a_norm)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = GraphConv(128)(x, a_norm)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Add()([x, layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))(inp)])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)

    out = GraphConv(1, act='sigmoid')(x, a_norm)
    return keras.Model(inputs=inp, outputs=out)


class WeightedBCE(keras.losses.Loss): # handling imbalance
    def __init__(self, pos_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.clip_by_value(tf.reshape(y_pred, [-1]), 1e-7, 1 - 1e-7)
        return tf.reduce_mean(-y_true * tf.math.log(y_pred) * self.pos_weight - (1 - y_true) * tf.math.log(1 - y_pred))


class BinaryFocalLoss(keras.losses.Loss):   # address node 104 / 128 weak confidence
    def __init__(self, gamma=3.0, alpha=0.90, **kwargs): # adjust alpha to trade off FN and FP
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_factor * tf.math.pow(1. - pt, self.gamma)
        return tf.reduce_mean(focal_weight * bce)


def load_adj(path):
    """grid topology from csv (undirected edges)"""
    adj = np.zeros((N_NODES, N_NODES))
    for _, r in pd.read_csv(path).iterrows():
        i, j = int(r['source']), int(r['target'])
        if i < N_NODES and j < N_NODES:
            adj[i, j] = adj[j, i] = 1
    return adj


def cumul_feat(dv_hist, dt_hist, vs_hist, ts_hist, cent, dfp_hist=None, dfq_hist=None, fps_hist=None, fqs_hist=None):
    T = dv_hist.shape[1]

    # cumulative sum for mean calculation
    dv_cumsum = np.cumsum(dv_hist, axis=1)
    dt_cumsum = np.cumsum(dt_hist, axis=1)

    # cumulative mean: cumsum / t
    t_range = np.arange(1, T + 1)
    dv_mean = dv_cumsum / t_range
    dt_mean = dt_cumsum / t_range

    # cumulative std: use running variance formula
    dv_sq_cumsum = np.cumsum(dv_hist ** 2, axis=1)
    dt_sq_cumsum = np.cumsum(dt_hist ** 2, axis=1)
    dv_var = (dv_sq_cumsum / t_range) - (dv_mean ** 2)
    dt_var = (dt_sq_cumsum / t_range) - (dt_mean ** 2)
    dv_std = np.sqrt(np.maximum(dv_var, 0))
    dt_std = np.sqrt(np.maximum(dt_var, 0))

    # cumulative max/min
    dv_abs = np.abs(dv_hist)
    dt_abs = np.abs(dt_hist)
    dv_max = np.maximum.accumulate(dv_abs, axis=1)
    dt_max = np.maximum.accumulate(dt_abs, axis=1)
    dv_min = np.minimum.accumulate(dv_abs, axis=1)
    dt_min = np.minimum.accumulate(dt_abs, axis=1)

    feats = [   # get last timestep values (current cumulative)
        # instantaneous deviation
        dv_hist[:, -1:], dt_hist[:, -1:],
        # cumulative stats so far
        dv_mean[:, -1:], dv_std[:, -1:], dv_max[:, -1:], dv_min[:, -1:],
        dt_mean[:, -1:], dt_std[:, -1:], dt_max[:, -1:], dt_min[:, -1:],

        # energy-based features (more robust than percentiles), suggested by GenAI
        (dv_hist**2).sum(axis=1, keepdims=True) / T,  # normalized energy
        (dt_hist**2).sum(axis=1, keepdims=True) / T,

        # baseline operational stats
        np.mean(vs_hist, axis=1, keepdims=True),
        np.std(vs_hist, axis=1, keepdims=True),
        np.mean(ts_hist, axis=1, keepdims=True),
        np.std(ts_hist, axis=1, keepdims=True),

        # distribution shape
        np.percentile(dv_hist, 90, axis=1, keepdims=True),
        np.percentile(dv_hist, 10, axis=1, keepdims=True),
        np.percentile(dt_hist, 90, axis=1, keepdims=True),
        np.percentile(dt_hist, 10, axis=1, keepdims=True),

        # temporal patterns: rate of change
        np.gradient(dv_hist, axis=1)[:, -1:] if T > 1 else np.zeros((dv_hist.shape[0], 1)),
        np.gradient(dt_hist, axis=1)[:, -1:] if T > 1 else np.zeros((dt_hist.shape[0], 1)),

        # outlier persistence
        (dv_abs > np.std(dv_hist, axis=1, keepdims=True) * 2).sum(axis=1, keepdims=True).astype(float) / T,
        (dt_abs > np.std(dt_hist, axis=1, keepdims=True) * 2).sum(axis=1, keepdims=True).astype(float) / T,

        cent
    ]

    if USE_ALL_FEATURE and dfp_hist is not None:
        dfp_cumsum = np.cumsum(dfp_hist, axis=1)
        dfq_cumsum = np.cumsum(dfq_hist, axis=1)
        dfp_mean = dfp_cumsum / t_range
        dfq_mean = dfq_cumsum / t_range

        dfp_sq_cumsum = np.cumsum(dfp_hist ** 2, axis=1)
        dfq_sq_cumsum = np.cumsum(dfq_hist ** 2, axis=1)
        dfp_var = (dfp_sq_cumsum / t_range) - (dfp_mean ** 2)
        dfq_var = (dfq_sq_cumsum / t_range) - (dfq_mean ** 2)
        dfp_std = np.sqrt(np.maximum(dfp_var, 0))
        dfq_std = np.sqrt(np.maximum(dfq_var, 0))

        dfp_abs = np.abs(dfp_hist)
        dfq_abs = np.abs(dfq_hist)
        dfp_max = np.maximum.accumulate(dfp_abs, axis=1)
        dfq_max = np.maximum.accumulate(dfq_abs, axis=1)

        feats.extend([
            dfp_hist[:, -1:], dfq_hist[:, -1:],
            dfp_mean[:, -1:], dfp_max[:, -1:], dfp_std[:, -1:],
            dfq_mean[:, -1:], dfq_max[:, -1:], dfq_std[:, -1:],
            np.mean(fps_hist, axis=1, keepdims=True),
            np.std(fps_hist, axis=1, keepdims=True),
            np.mean(fqs_hist, axis=1, keepdims=True),
            np.std(fqs_hist, axis=1, keepdims=True),
        ])

    return np.concatenate(feats, axis=1)


def process_pkl(path, cent, start_t, end_t):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    att_nodes = [n-1 for n in data['esset_btm']['A']]
    # att_window = data['timeset_a']['A']
    # att_steps = [f"x{i}" for i in range(min(att_window), max(att_window) + 1)]
    att_steps = [f"x{i}" for i in range(1, END + 1)]
    if len(att_steps) < end_t:
        return None

    vs = np.array([data['sch_v'][s] for s in att_steps[:END]]).T
    ts = np.array([data['sch_θ'][s] for s in att_steps[:END]]).T
    va = np.array([data['attack_v'].get(s, data['sch_v'][s]) for s in att_steps[:END]]).T
    ta = np.array([data['attack_θ'].get(s, data['sch_θ'][s]) for s in att_steps[:END]]).T

    dv = va - vs
    dt = ta - ts

    if USE_ALL_FEATURE:
        def safe_load(d, s, n):
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
    else:
        dfp = dfq = fps = fqs = None

    y_day = np.zeros(N_NODES, dtype=np.float32)
    for n in att_nodes:
        y_day[n] = 1.0

    x_list = []
    for t in range(start_t, end_t):
        sl = slice(None, t + 1)
        dv_t  = dv[:, sl]
        dt_t  = dt[:, sl]
        vs_t  = vs[:, sl]
        ts_t  = ts[:, sl]

        if USE_ALL_FEATURE:
            feats = cumul_feat(
                dv_t, dt_t, vs_t, ts_t, cent,
                dfp_hist=dfp[:, sl], dfq_hist=dfq[:, sl],
                fps_hist=fps[:, sl], fqs_hist=fqs[:, sl]
            )
        else:
            feats = cumul_feat(dv_t, dt_t, vs_t, ts_t, cent)

        x_list.append(feats)

    return np.array(x_list), np.tile(y_day, (len(x_list), 1))


def temporal_loader(dirs, adj, start, end, frac=1.0):
    if isinstance(dirs, str):
        dirs = [dirs]

    cent = np.sum(adj, axis=1, keepdims=True)
    cent = cent / (cent.max() + 1e-8)

    all_files = []
    for d in dirs:
        files = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.pkl')])
        if frac < 1.0:
            np.random.shuffle(files)
            files = files[:int(frac * len(files))]
        all_files.extend(files)

    x_all, y_all = [], []
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(partial(process_pkl, cent=cent, start_t=start, end_t=end), all_files),
            total=len(all_files)))

    for result in results:
        if result is not None:
            X, y = result
            x_all.append(X)
            y_all.append(y)

    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)

if __name__ == "__main__":
    # adj = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    adj = load_adj('topology.csv')
    print(f"grid bus: {N_NODES}, edges: {int(adj.sum()) // 2}")
    print(f"using full features: {USE_ALL_FEATURE}\n")

    print("loading training sets...")
    X_train, y_train = temporal_loader(DATA_TRAIN, adj, START, END, frac=DATA_FRAC)
    print("loading testing sets...")
    X_test, y_test = temporal_loader(DATA_TEST, adj, START, END, frac=DATA_FRAC)
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"features per node: {X_train.shape[2]}")

    # normalize
    X_mean = X_train.mean((0, 1), keepdims=True)
    X_std = X_train.std((0, 1), keepdims=True) + 1e-8
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    weight = (y_train.size - y_train.sum()) / (y_train.sum() + 1e-8) * 0.5
    print(f"positive class weight: {weight:.2f}")
    print(f"attack ratio: train {y_train.mean():.4f}, test {y_test.mean():.4f}\n")

    model = gcn(N_NODES, X_train.shape[2], adj, reduced=IS_REDUCED)
    lr_sch = keras.optimizers.schedules.CosineDecay(    # cosine annealing for better convergence
        initial_learning_rate=2e-3,
        decay_steps=50 * (len(X_train) // 32),
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

    h = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1,
                  callbacks=[   # early stopping upon perfect AUC
                  keras.callbacks.EarlyStopping(monitor='val_rec', patience=15, restore_best_weights=True, mode='max'),
                  #keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=8, min_lr=1e-6, mode='max')
                  ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['loss', 'auc', 'prec', 'rec']
    for i, m in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        ax.plot(h.history[m], label='train')
        ax.plot(h.history[f'val_{m}'], label='val')
        ax.set_title(m.upper())
        ax.set_xlabel('ep')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    pred_raw = model.predict(X_test, verbose=0)[:, :, 0]

    n_steps = END - START
    n_days = len(X_test) // n_steps
    day_pred = pred_raw.reshape(n_days, n_steps, N_NODES)
    labels_by_day = y_test.reshape(n_days, n_steps, N_NODES)
    day_node_labels = labels_by_day.max(axis=1)
    day_node_preds = ((day_pred > ALARM_THRESHOLD).astype(int).sum(axis=1) >= PERSISTENCE).astype(int)
    mode = ['normal', 'attacked']
    print(classification_report(day_node_labels.flatten(), day_node_preds.flatten(), target_names=mode))

    cm = confusion_matrix(day_node_labels.flatten(), day_node_preds.flatten())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=mode, yticklabels=mode)
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.show()

    ## DEBUGGING, per-node analysis ↓
    node_preds = day_pred.max(axis=1).mean(axis=0)
    node_labels = day_node_labels.max(axis=0)

    print(f"attacked nodes (avg confidence):")
    for idx in np.where(node_labels == 1)[0]:
        print(f"  N{idx + 1}: {node_preds[idx]:.4f}")

    print(f"\ntop 10 false alarms:")
    normal = np.where(node_labels == 0)[0]
    for idx in normal[np.argsort(node_preds[normal])[-10:][::-1]]:
        print(f"N{idx + 1}: {node_preds[idx]:.4f}")
    ## DEBUGGING ↑

    d = 0   # probability traces for arbitrary test day
    day_data = day_pred[d]
    tgt = labels_by_day[d]

    times = []
    for t in range(START, END):
        times.append((datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(minutes=t * 15)).strftime("%H:%M"))

    plt.figure(figsize=(14, 7))
    att_indices = np.where(tgt.max(axis=0) == 1)[0]
    if len(att_indices) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(att_indices)))
        for i, idx in enumerate(att_indices):
            c = colors[i]
            plt.plot(day_data[:, idx], color=c)
            plt.text(len(day_data) - 1, day_data[-1, idx], f' N{idx + 1}', color=c, va='center')
    for i, idx in enumerate(np.where(tgt.max(axis=0) == 0)[0][:5]):
        plt.plot(day_data[:, idx], label='normal' if i == 0 else "")

    attack_active_steps = np.where(tgt.max(axis=1) == 1)[0]
    if len(attack_active_steps) > 0:
        plt.axvspan(attack_active_steps[0], attack_active_steps[-1], color='r', alpha=0.05)
    plt.axhline(y=ALARM_THRESHOLD, color='k', linestyle='--')

    ticks = np.arange(0, len(times), 4)
    plt.xticks(ticks, [times[i] for i in ticks], rotation=45)
    plt.ylabel('confidence')
    plt.title(f'Attack Probability Evolution (Day {d + 1})')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()