import os, pickle, yaml, time
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from functools import partial

DATA_TRAIN = ['./str/v6', './str/v7', './str/v8']
DATA_TEST = ['./str/v9', './str/v10']
N_NODES = 187
ESB = [86, 103, 106, 111, 112, 113, 114, 124, 126, 127, 129, 130]   # 0-indexed
START, END = 1, 80
EPS_ATTACK = 1e-4
SCALE = 1e4
EVAL_TIMESTEPS = [20, 40, 60, 79]  # evaluate at these timesteps


def load_adj(path):
    adj = np.zeros((N_NODES, N_NODES))
    df = pd.read_csv(path)
    adj[df['source'], df['target']] = adj[df['target'], df['source']] = 1
    return adj


def safe_corr(x, y):
    if len(x) < 2 or len(y) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    try:
        c = np.corrcoef(x, y)[0, 1]
        return c if not np.isnan(c) else 0.0
    except:
        return 0.0


def feat_extract(path, adj, timestep):
    """extract temporal features up to given timestep"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    steps = [f"x{i}" for i in range(1, timestep + 1)]
    vs = np.array([data['sch_v'][s] for s in steps]).T
    ts = np.array([data['sch_θ'][s] for s in steps]).T
    va = np.array([data['attack_v'].get(s, data['sch_v'][s]) for s in steps]).T
    ta = np.array([data['attack_θ'].get(s, data['sch_θ'][s]) for s in steps]).T

    dv, dt = va - vs, ta - ts

    # labels: match gnn logic
    y = np.zeros(N_NODES, dtype=int)
    if 'esset_btm_a' in data:  # v6-10
        for node in data['esset_btm_a']['A']:
            if node - 1 < N_NODES:
                y[node - 1] = 1
    elif 'esset_btm' in data:  # v1-5
        for node in data['esset_btm']['A']:
            if node - 1 < N_NODES:
                y[node - 1] = 1

    # attack cumulative mask: once attack detected, stays 1
    att_idx = np.where(y == 1)[0]
    attack_cum = np.zeros(N_NODES, dtype=int)
    if len(att_idx) > 0:
        for n in att_idx:
            if (np.abs(dv[n, :]) > EPS_ATTACK).any() or (np.abs(dt[n, :]) > EPS_ATTACK).any():
                attack_cum[n] = 1

    T = dv.shape[1]
    feats = []

    for n in range(N_NODES):
        nbrs = np.where(adj[n] > 0)[0]
        dv_n, dt_n = dv[n], dt[n]

        # cumulative statistics (like gnn)
        dv_mean = np.mean(dv_n)
        dv_std = np.std(dv_n)
        dv_max = np.max(np.abs(dv_n))
        dt_mean = np.mean(dt_n)
        dt_std = np.std(dt_n)
        dt_max = np.max(np.abs(dt_n))

        # neighbor aggregation
        if len(nbrs) > 0:
            dv_nbr = dv[nbrs]
            dv_nbr_mean = dv_nbr.mean(axis=0)
            dv_nbr_avg = dv_nbr_mean.mean()
            dv_nbr_std = dv_nbr.std()
            corr_v = safe_corr(dv_n, dv_nbr_mean)
        else:
            dv_nbr_avg = 0.0
            dv_nbr_std = 0.0
            corr_v = 0.0

        # temporal patterns
        dv_grad = np.gradient(dv_n)
        dv_grad_max = np.max(np.abs(dv_grad))

        # afternoon indicator (x60-x80)
        is_afternoon = 1 if timestep >= 60 else 0
        dv_recent = np.abs(dv_n[-min(10, T):]).max() if T > 0 else 0

        feat = {
            'node_id': n + 1,
            'timestep': timestep,
            'is_storage': 1 if n in ESB else 0,
            'n_neighbors': len(nbrs),
            # cumulative deviation stats
            'dv_mean': float(dv_mean) * SCALE,
            'dv_std': float(dv_std) * SCALE,
            'dv_max': float(dv_max) * SCALE,
            'dv_p90': float(np.percentile(np.abs(dv_n), 90)) * SCALE,
            'dt_mean': float(dt_mean) * SCALE,
            'dt_std': float(dt_std) * SCALE,
            'dt_max': float(dt_max) * SCALE,
            # temporal dynamics
            'dv_grad_max': float(dv_grad_max) * SCALE,
            'dv_recent': float(dv_recent) * SCALE,
            # neighbor comparison
            'dv_vs_nbr': float(np.abs(dv_mean - dv_nbr_avg)) * SCALE,
            'dv_ratio': float(np.abs(dv_n).mean() / (np.abs(dv_nbr_avg) + 1e-6)),
            'corr_v': float(corr_v),
            'nbr_std': float(dv_nbr_std) * SCALE,

            'is_afternoon': is_afternoon,   # time context

            'label': int(attack_cum[n])
        }
        feats.append(feat)

    return pd.DataFrame(feats)


def process_file(path, adj, timesteps):
    """process one file at multiple timesteps"""
    dfs = []
    for t in timesteps:
        try:
            df = feat_extract(path, adj, t)
            dfs.append(df)
        except:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_dataset(dirs, adj, timesteps, max_files=60):
    if isinstance(dirs, str):
        dirs = [dirs]
    files = []
    for d in dirs:
        if os.path.exists(d):
            files.extend([os.path.join(d, x) for x in sorted(os.listdir(d)) if x.endswith('.pkl')])

    files = files[:max_files]
    with Pool(min(cpu_count(), 8)) as p:
        dfs = list(tqdm(p.imap(partial(process_file, adj=adj, timesteps=timesteps), files), total=len(files)))

    dfs = [df for df in dfs if df is not None]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


class TemporalLLMDetector:
    def __init__(self, train_df, config_path='configs.yaml'):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.key = cfg['api_key']
        self.model = cfg['model']
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        # compute global statistics per timestep range
        early = train_df[train_df.timestep <= 40]
        late = train_df[train_df.timestep > 40]

        attacks = train_df[train_df.label == 1]
        normals = train_df[train_df.label == 0]

        self.stats = {
            'attack_dv_mean': attacks['dv_max'].mean(),
            'attack_dv_std': attacks['dv_max'].std(),
            'attack_dv_min': attacks['dv_max'].min(),
            'attack_dv_q25': attacks['dv_max'].quantile(0.25),
            'attack_dv_q75': attacks['dv_max'].quantile(0.75),
            'normal_dv_mean': normals['dv_max'].mean(),
            'normal_dv_std': normals['dv_max'].std(),
            'normal_dv_q95': normals['dv_max'].quantile(0.95),
            'early_attack_rate': early[early.label == 1]['dv_max'].mean() if len(early[early.label == 1]) > 0 else 0,
            'late_attack_rate': late[late.label == 1]['dv_max'].mean() if len(late[late.label == 1]) > 0 else 0,
        }

        self.examples = self._select_examples(train_df)

    def _select_examples(self, df):
        """select diverse examples across severity and time"""
        attacks = df[df.label == 1].copy()
        normals = df[df.label == 0].copy()

        examples = []

        if len(attacks) >= 4:
            att_sorted = attacks.sort_values(['dv_max', 'timestep'])
            examples.append(att_sorted.iloc[0])  # weak early
            examples.append(att_sorted.iloc[len(attacks) // 3])  # medium
            examples.append(att_sorted.iloc[2 * len(attacks) // 3])  # strong
            examples.append(att_sorted.iloc[-1])  # strongest late

        if len(normals) >= 4:
            nor_sorted = normals.sort_values(['dv_max', 'timestep'])
            examples.append(nor_sorted.iloc[0])  # clean
            examples.append(nor_sorted.iloc[len(normals) // 3])
            examples.append(nor_sorted.iloc[2 * len(normals) // 3])
            examples.append(nor_sorted.iloc[-1])  # boundary

        return pd.DataFrame(examples).sample(frac=1, random_state=7) if examples else pd.DataFrame()

    def _format_example(self, row, show_label=True):
        label_str = f" → {'ATK' if row['label'] else 'NOR'}" if show_label else " →"
        return (
            f"N{row['node_id']}@t{row['timestep']} (stor={row['is_storage']},nbr={row['n_neighbors']},aft={row['is_afternoon']}): "
            f"dVmax={row['dv_max']:.2f} dVstd={row['dv_std']:.2f} dVp90={row['dv_p90']:.2f} "
            f"grad={row['dv_grad_max']:.2f} rcnt={row['dv_recent']:.2f} "
            f"vs_nbr={row['dv_vs_nbr']:.2f} ratio={row['dv_ratio']:.2f} corr={row['corr_v']:.2f}"
            f"{label_str}"
        )

    def _build_prompt(self, target_row):
        stats_str = (
            f"ATTACK SIGNATURES (×10⁴ p.u.):\n"
            f"Attack: dVmax μ={self.stats['attack_dv_mean']:.2f} σ={self.stats['attack_dv_std']:.2f} "
            f"Q1={self.stats['attack_dv_q25']:.2f} Q3={self.stats['attack_dv_q75']:.2f}\n"
            f"Normal: dVmax μ={self.stats['normal_dv_mean']:.2f} σ={self.stats['normal_dv_std']:.2f} "
            f"95th={self.stats['normal_dv_q95']:.2f}\n"
            f"Temporal: early_atk={self.stats['early_attack_rate']:.2f} late_atk={self.stats['late_attack_rate']:.2f}\n\n"
        )

        rules_str = (
            f"DETECTION RULES (cumulative features, once attacked stays attacked):\n"
            f"STRONG ATTACK:\n"
            f"  - dVmax > {self.stats['normal_dv_q95']:.2f} (exceeds 95% of normals)\n"
            f"  - dVmax > 1.5 AND (rcnt > 1.0 OR grad > 0.8) [sudden onset]\n"
            f"  - ratio > 2.5 AND corr < 0.5 [independent from neighbors]\n"
            f"MODERATE ATTACK:\n"
            f"  - dVmax in [{self.stats['attack_dv_q25']:.2f},{self.stats['attack_dv_q75']:.2f}] AND stor=1 [storage node]\n"
            f"  - is_afternoon=1 AND dVmax > 0.8 AND vs_nbr > 0.5 [afternoon pattern]\n"
            f"NORMAL:\n"
            f"  - dVmax < {self.stats['normal_dv_mean'] + self.stats['normal_dv_std']:.2f}\n"
            f"  - ratio < 1.5 AND corr > 0.7 [coupled with neighbors]\n"
            f"KEY: Higher timestep + high deviation = likely attack already occurred\n\n"
        )

        examples_str = "EXAMPLES:\n" + "\n".join([self._format_example(ex) for _, ex in self.examples.iterrows()]) + "\n"
        test_str = f"\nTEST:\n{self._format_example(target_row, show_label=False)}\n"

        return (
            f"Detect FDIA on power grid. Features are CUMULATIVE (aggregated over time).\n\n"
            f"{stats_str}{rules_str}{examples_str}{test_str}\n"
            f"Respond: 'ATK' or 'NOR'"
        )

    def predict(self, test_df):
        preds = []
        errors = []

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="llm inference"):
            pred, raw = self._query(row)
            preds.append(pred)

            if pred != row['label']:
                errors.append({
                    'true': row['label'],
                    'pred': pred,
                    'node': row['node_id'],
                    't': row['timestep'],
                    'dv': row['dv_max'],
                    'raw': raw[:80]
                })

        if errors:
            print(f"\nerrors ({len(errors)} total, showing first 5):")
            for e in errors[:5]:
                print(f"  N{e['node']}@t{e['t']}: true={e['true']} pred={e['pred']} dv={e['dv']:.2f} → {e['raw']}")

        return np.array(preds)

    def _query(self, row, max_retries=3):
        prompt = self._build_prompt(row)

        for attempt in range(max_retries):
            try:
                headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 5
                }

                time.sleep(1.2)  # rate limit
                resp = requests.post(self.url, headers=headers, json=data, timeout=15)

                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content'].strip().upper()
                    if 'ATK' in content:
                        return 1, content
                    elif 'NOR' in content:
                        return 0, content
                    return self._fallback(row), content

                elif resp.status_code == 429:
                    time.sleep((attempt + 1) * 3)
                    continue

            except Exception as e:
                if attempt == max_retries - 1:
                    return self._fallback(row), f"err:{e}"
                time.sleep(1)

        return self._fallback(row), "timeout"

    def _fallback(self, row):
        """heuristic based on gnn threshold patterns"""
        if row['dv_max'] > 1.2 and row['corr_v'] < 0.5:
            return 1
        if row['dv_max'] > 2.0:
            return 1
        if row['is_afternoon'] and row['dv_max'] > 0.8 and row['dv_ratio'] > 2.0:
            return 1
        return 0


if __name__ == "__main__":
    adj = load_adj('topology.csv')
    print(f"nodes: {N_NODES}, edges: {int(adj.sum()) // 2}\n")
    print("loading train...")
    df_train = load_dataset(DATA_TRAIN, adj, EVAL_TIMESTEPS, max_files=60)
    print("loading test...")
    df_test = load_dataset(DATA_TEST, adj, EVAL_TIMESTEPS, max_files=40)
    if len(df_train) == 0 or len(df_test) == 0:
        exit("no data")

    print(f"\n{len(df_train)} train samples, attack ratio {df_train.label.mean():.3f}")
    print(f"{len(df_test)} test samples, attack ratio {df_test.label.mean():.3f}")

    clf = TemporalLLMDetector(df_train)

    # balanced test
    attacks = df_test[df_test.label == 1]
    normals = df_test[df_test.label == 0]
    n_samples = min(len(attacks), len(normals), 80)

    test_subset = pd.concat([
        attacks.sample(n=n_samples, random_state=7),
        normals.sample(n=n_samples, random_state=7)
    ]).sample(frac=1, random_state=7)

    print(f"\nRunning on {len(test_subset)} balanced samples...")
    y_pred = clf.predict(test_subset)
    y_true = test_subset['label'].values

    mode = ['normal', 'attack']
    print(classification_report(y_true, y_pred, target_names=mode, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=mode, yticklabels=mode)
    axes[0].set_ylabel('actual')
    axes[0].set_xlabel('predicted')
    axes[0].set_title('confusion matrix')

    timesteps = sorted(test_subset['timestep'].unique())
    prec_by_t, rec_by_t = [], []
    for t in timesteps:
        mask = test_subset['timestep'] == t
        y_t = y_true[mask]
        p_t = y_pred[mask]
        tp = ((p_t == 1) & (y_t == 1)).sum()
        fp = ((p_t == 1) & (y_t == 0)).sum()
        fn = ((p_t == 0) & (y_t == 1)).sum()
        prec_by_t.append(tp / (tp + fp + 1e-8))
        rec_by_t.append(tp / (tp + fn + 1e-8))

    axes[1].plot(timesteps, prec_by_t, 'o-', label='precision', linewidth=2)
    axes[1].plot(timesteps, rec_by_t, 's-', label='recall', linewidth=2)
    axes[1].set_xlabel('timestep')
    axes[1].set_ylabel('score')
    axes[1].set_title('temporal performance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()