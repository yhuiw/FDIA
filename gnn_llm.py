"""
GNN + Local LLM Hybrid FDIA Detection
Uses vLLM for fast local inference on 3x A6000
"""

import os, re, warnings
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
from functools import partial, lru_cache
from pathlib import Path
from typing import List, Dict, Tuple
#import yaml
from vllm import LLM, SamplingParams

# configs
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

LLM_ENABLED = True
LLM_TOP_K = 10
LLM_CONFIDENCE_THRESHOLD = 0.5
LOCAL_MODEL_PATH = "./models/llama-70b"

warnings.filterwarnings('ignore')


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
    if 'esset_btm_a' in data:
        y_day[np.array(data['esset_btm_a']['A']) - 1] = 1.0
    else:
        y_day[np.array(data['esset_btm']['A']) - 1] = 1.0

    att_idx = np.where(y_day == 1)[0]
    attack_mask = np.zeros((N_NODES, END), dtype=np.float32)
    if len(att_idx) > 0:
        attack_mask[att_idx, :] = ((np.abs(dv[att_idx, :]) > EPS_ATTACK) |
                                   (np.abs(dt[att_idx, :]) > EPS_ATTACK)).astype(np.float32)
    attack_cum = np.maximum.accumulate(attack_mask, axis=1)

    x_arr = np.array([accumulate(dv[:, :t + 1], dt[:, :t + 1], vs[:, :t + 1], ts[:, :t + 1], adj, esb)
                      for t in range(start_t, end_t)])
    y_arr = attack_cum[:, start_t:end_t].T

    if augment and len(att_idx) > 0 and np.random.rand() < AUG_PROB:
        n_flip = min(len(att_idx), np.random.randint(1, AUG_MAX_FLIP + 1))
        flip_idx = np.random.choice(att_idx, n_flip, replace=False)

        dv2, dt2, ac2 = dv.copy(), dt.copy(), attack_cum.copy()
        dv2[flip_idx, :] = 0.0
        dt2[flip_idx, :] = 0.0
        ac2[flip_idx, :] = 0.0

        x2 = np.array([accumulate(dv2[:, :t + 1], dt2[:, :t + 1], vs[:, :t + 1], ts[:, :t + 1], adj, esb)
                       for t in range(start_t, end_t)])
        y2 = ac2[:, start_t:end_t].T
        return np.concatenate([x_arr, x2], axis=0), np.concatenate([y_arr, y2], axis=0), f"{os.path.basename(path)}_aug"

    return x_arr, y_arr, os.path.basename(path)


def holdoff(pred, thresh, persist=1):
    res = np.zeros_like(pred, dtype=int)
    for t in range(persist - 1, len(pred)):
        res[t] = (pred[t - persist + 1:t + 1] > thresh).all(axis=0).astype(int)
    return res


def smooth_operator(pred, window=1):
    res = np.zeros_like(pred)
    for t in range(len(pred)):
        res[t] = pred[max(0, t - window + 1):t + 1].mean(axis=0)
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
        res = list(tqdm(p.imap(partial(process_pkl, adj=adj, start_t=start, end_t=end, augment=augment), files),
                        total=len(files), desc="loading data"))

    for X, y, m in res:
        if X is not None:
            x_all.append(X)
            y_all.append(y)
            meta.extend([m] * len(X))
    return np.concatenate(x_all), np.concatenate(y_all), meta


# LOCAL LLM COMPONENTS (vLLM-based, multi-gpu)
class LocalAttackValidator:
    """local llm validator using vllm for multi-gpu inference"""

    SYSTEM_PROMPT = """You are an expert in power grid cybersecurity, specializing in False Data Injection Attacks (FDIA) on distributed energy resources.

Given measurements for a power grid node, determine if it is under cyberattack.

Attack indicators:
- Large voltage/angle deviations from scheduled values
- Unusual patterns compared to neighboring nodes
- High correlation with neighbors but own deviations exceed thresholds

Response format (exactly 3 lines):
Line 1: "Yes" or "No"
Line 2: Confidence score (0.0 to 1.0)
Line 3: One-sentence technical reason

Be decisive and concise."""

    def __init__(self, model_path: str, tensor_parallel_size: int = 2):
        """
        model_path: path to local model
        tensor_parallel_size: number of gpus (2 for 70b, 1 for 8b)
        """
        print(f"loading local llm from {model_path}...")
        print(f"using {tensor_parallel_size} gpus with tensor parallelism...")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            max_model_len=2048,  # short context for speed
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=100,
            stop=["\n\n"]  # stop at blank line
        )

        print("llm loaded successfully!")
        self.cache = {}  # simple cache for repeated queries

    def extract_signature(self, X_sample: np.ndarray, node_id: int, adj: np.ndarray) -> str:
        """extract concise node signature"""
        feats = X_sample[node_id]

        dv, dt = feats[0], feats[1]
        dv_max, dt_max = feats[4], feats[7]
        neighbor_dv, neighbor_dt = feats[16], feats[17]
        diff_v, diff_t = feats[18], feats[19]
        corr_v, corr_t = feats[20], feats[21]
        is_storage = int(feats[22])
        n_neighbors = int(adj[node_id].sum())

        return (
            f"node: N{node_id + 1} ({'storage' if is_storage else 'regular'}), neighbors: {n_neighbors}\n"
            f"voltage_dev: current={dv:.4f}, max={dv_max:.4f}\n"
            f"angle_dev: current={dt:.4f}, max={dt_max:.4f}\n"
            f"neighbor_avg: v={neighbor_dv:.4f}, θ={neighbor_dt:.4f}\n"
            f"diff_from_neighbors: v={diff_v:.4f}, θ={diff_t:.4f}\n"
            f"correlation: v={corr_v:.3f}, θ={corr_t:.3f}"
        )

    def validate_batch(self, signatures: List[str]) -> List[Tuple[int, float, str]]:
        """batch validation for efficiency"""
        prompts = [
            f"{self.SYSTEM_PROMPT}\n\n{sig}\n\nIs this node under cyberattack?"
            for sig in signatures
        ]

        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for output in outputs:
            try:
                text = output.outputs[0].text.strip()
                lines = text.split('\n')

                prediction = 1 if lines[0].strip().lower() == 'yes' else 0
                confidence = float(lines[1].strip()) if len(lines) > 1 else (0.9 if prediction else 0.1)
                explanation = lines[2].strip() if len(lines) > 2 else "no explanation"

                results.append((prediction, confidence, explanation))
            except Exception as e:
                results.append((0, 0.5, f"parse error: {e}"))

        return results


class HybridDetector:
    """gnn + local llm hybrid"""

    def __init__(self, gnn_model, llm_validator: LocalAttackValidator, adj: np.ndarray):
        self.gnn = gnn_model
        self.llm = llm_validator
        self.adj = adj
        self.llm_calls = 0
        self.llm_corrections = 0

    def predict_with_explanation(
            self,
            X_test: np.ndarray,
            batch_size: int = 32  # batch for llm efficiency
    ) -> Tuple[np.ndarray, Dict]:
        """hybrid prediction with batched llm inference"""
        n_samples = len(X_test)

        print("gnn inference...")
        gnn_probs = self.gnn.predict(X_test, verbose=0)[:, :, 0]
        gnn_smooth = smooth_operator(gnn_probs)

        print(f"llm validation (batch_size={batch_size})...")
        llm_adjustments = {}
        explanations = {}

        # collect all queries first
        queries = []  # (sample_idx, node_idx, signature)
        for i in tqdm(range(n_samples), desc="collecting queries"):
            probs = gnn_smooth[i]
            uncertain_mask = (probs > LLM_CONFIDENCE_THRESHOLD) & (probs < ALARM_THRESHOLD + 0.2)

            if not uncertain_mask.any():
                continue

            uncertain_nodes = np.where(uncertain_mask)[0]
            if len(uncertain_nodes) > LLM_TOP_K:
                top_k_idx = np.argsort(probs[uncertain_nodes])[-LLM_TOP_K:]
                uncertain_nodes = uncertain_nodes[top_k_idx]

            for node in uncertain_nodes:
                sig = self.llm.extract_signature(X_test[i], node, self.adj)
                queries.append((i, node, sig, probs[node]))

        print(f"total llm queries: {len(queries)}")

        # batch inference
        for batch_start in tqdm(range(0, len(queries), batch_size), desc="llm inference"):
            batch = queries[batch_start:batch_start + batch_size]
            signatures = [q[2] for q in batch]

            results = self.llm.validate_batch(signatures)

            for (i, node, _, gnn_prob), (pred, conf, expl) in zip(batch, results):
                self.llm_calls += 1

                if (pred == 1 and conf > 0.7 and gnn_prob < ALARM_THRESHOLD):
                    llm_adjustments[(i, node)] = 1
                    self.llm_corrections += 1
                    explanations[(i, node)] = f"LLM: {expl}"
                elif (pred == 0 and conf > 0.7 and gnn_prob > ALARM_THRESHOLD):
                    llm_adjustments[(i, node)] = 0
                    self.llm_corrections += 1
                    explanations[(i, node)] = f"LLM: {expl}"

        # apply corrections
        final_preds = holdoff(gnn_smooth, ALARM_THRESHOLD, PERSISTENCE)
        for (i, node), new_val in llm_adjustments.items():
            final_preds[i, node] = new_val

        metadata = {
            'gnn_probs': gnn_smooth,
            'llm_calls': self.llm_calls,
            'llm_corrections': self.llm_corrections,
            'explanations': explanations
        }

        return final_preds, metadata


if __name__ == "__main__":
    adj = load_adj('topology.csv')
    print(f"nodes: {N_NODES}, edges: {int(adj.sum()) // 2}")
    X_train, y_train, _ = temporal_loader(DATA_TRAIN, adj, START, END, CUT, augment=True)
    X_test, y_test, meta = temporal_loader(DATA_TEST, adj, START, END, CUT, augment=False)
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    np.random.seed(7)
    perm = np.random.permutation(len(X_train))
    n_val = int(0.1 * len(X_train))
    X_val, y_val = X_train[perm[:n_val]], y_train[perm[:n_val]]
    X_train, y_train = X_train[perm[n_val:]], y_train[perm[n_val:]]

    xm, xs = X_train.mean((0, 1), keepdims=True), X_train.std((0, 1), keepdims=True) + 1e-8
    X_train = (X_train - xm) / xs
    X_val = (X_val - xm) / xs
    X_test = (X_test - xm) / xs

    print("TRAINING GNN")

    w = (y_train.size - y_train.sum()) / (y_train.sum() + 1e-8) * 0.01
    print(f"pos weight: {w:.2f}")

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

    print("GNN-ONLY RESULTS")

    n_steps = END - START
    n_days = len(X_test) // n_steps
    gnn_raw = model.predict(X_test, verbose=0)[:, :, 0]
    gnn_smooth = smooth_operator(gnn_raw)
    gnn_preds = holdoff(gnn_smooth, ALARM_THRESHOLD, PERSISTENCE)

    print(classification_report(y_test.flatten(), gnn_preds.flatten(), target_names=['normal', 'attacked'], digits=3))

    if LLM_ENABLED:
        use_llm = input("\nenable local llm validation? (y/n, FREE with your gpus): ").lower() == 'y'

        if use_llm:
            print("LOADING LOCAL LLM")

            # check which model exists
            if Path(LOCAL_MODEL_PATH).exists():
                tensor_parallel = 2 if "70b" in LOCAL_MODEL_PATH.lower() else 1
                llm_validator = LocalAttackValidator(LOCAL_MODEL_PATH, tensor_parallel_size=tensor_parallel)
            else:
                print(f"error: model not found at {LOCAL_MODEL_PATH}")
                print("download model first with:")
                print("  huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir ./models/llama-70b")
                exit(1)

            print("HYBRID GNN+LLM DETECTION")

            test_size = int(input(f"test on how many samples? (max {len(X_test)}, 0 for all): ") or len(X_test))
            test_size = min(test_size, len(X_test)) if test_size > 0 else len(X_test)

            X_test_subset = X_test[:test_size]
            y_test_subset = y_test[:test_size]

            hybrid = HybridDetector(model, llm_validator, adj)
            hybrid_preds, metadata = hybrid.predict_with_explanation(X_test_subset, batch_size=32)

            print("HYBRID RESULTS")
            print(f"llm calls: {metadata['llm_calls']}")
            print(f"llm corrections: {metadata['llm_corrections']}")
            print(f"correction rate: {100 * metadata['llm_corrections'] / max(1, metadata['llm_calls']):.2f}%")

            print("\nperformance:")
            print(classification_report(y_test_subset.flatten(), hybrid_preds.flatten(), target_names=['normal', 'attacked'], digits=3))

            if metadata['explanations']:
                print("\nsample llm explanations:")
                for (i, node), expl in list(metadata['explanations'].items())[:10]:
                    print(f"  sample {i}, N{node + 1}: {expl}")

    cm = confusion_matrix(y_test.flatten(), gnn_preds.flatten())
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False,
                xticklabels=['normal', 'attacked'],
                yticklabels=['normal', 'attacked'])
    plt.ylabel('actual')
    plt.xlabel('predicted')
    plt.title('GNN Detection Results')
    plt.tight_layout()