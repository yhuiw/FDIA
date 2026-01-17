import os, pickle, yaml, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

DATA_TRAIN = ['./storage/v6', './storage/v7', './storage/v8']
DATA_TEST = ['./storage/v9', './storage/v10']
N_NODES = 187
STORAGE_NODES = [87, 104, 107, 112, 113, 114, 115, 125, 127, 128, 131]  # one-indexed
N_EXAMPLES = 8
SCALE = 1e4

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

def extract_features(path, adj):
    """extract rich statistical features for each node"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    steps = [f"x{i}" for i in range(1, 81)]
    vs = np.array([data['sch_v'][s] for s in steps]).T
    ts = np.array([data['sch_θ'][s] for s in steps]).T
    va = np.array([data['attack_v'].get(s, data['sch_v'][s]) for s in steps]).T
    ta = np.array([data['attack_θ'].get(s, data['sch_θ'][s]) for s in steps]).T

    dv, dt = va - vs, ta - ts

    # ground truth labels
    y = np.zeros(N_NODES, dtype=int)
    if 'esset_btm' in data and 'A' in data['esset_btm']:
        for node in data['esset_btm']['A']:
            if node - 1 < N_NODES:
                y[node - 1] = 1

    feats = []
    for n in range(N_NODES):
        nbrs = np.where(adj[n] > 0)[0]
        dv_n, dt_n = dv[n], dt[n]

        # neighbor statistics
        if len(nbrs) > 0:
            dv_nbr = dv[nbrs]
            dt_nbr = dt[nbrs]
            dv_nbr_mean = dv_nbr.mean(axis=0)
            dt_nbr_mean = dt_nbr.mean(axis=0)
            dv_nbr_std = dv_nbr.std(axis=0).mean()
            #dt_nbr_std = dt_nbr.std(axis=0).mean()
        else:
            dv_nbr_mean = np.zeros_like(dv_n)
            dt_nbr_mean = np.zeros_like(dt_n)
            dv_nbr_std = 0.0
            #dt_nbr_std = 0.0

        # temporal patterns
        dv_grad = np.gradient(dv_n)
        #dt_grad = np.gradient(dt_n)

        afternoon_slice = slice(60, 80) # when attacks concentrate
        dv_afternoon_max = np.abs(dv_n[afternoon_slice]).max() if len(dv_n) > 60 else 0

        feat = {
            'node_id': n + 1,
            'is_storage': 1 if (n + 1) in STORAGE_NODES else 0,
            'n_neighbors': len(nbrs),
            # voltage magnitude deviation stats
            'dv_mean': float(np.mean(dv_n)) * SCALE,
            'dv_std': float(np.std(dv_n)) * SCALE,
            'dv_max': float(np.max(np.abs(dv_n))) * SCALE,
            'dv_p90': float(np.percentile(np.abs(dv_n), 90)) * SCALE,
            # angle deviation stats
            'dt_mean': float(np.mean(dt_n)) * SCALE,
            'dt_std': float(np.std(dt_n)) * SCALE,
            'dt_max': float(np.max(np.abs(dt_n))) * SCALE,
            # temporal dynamics
            'dv_grad_max': float(np.max(np.abs(dv_grad))) * SCALE,
            'dv_afternoon_max': float(dv_afternoon_max) * SCALE,
            # neighbor comparison
            'dv_vs_nbr_mean': float(np.mean(np.abs(dv_n - dv_nbr_mean))) * SCALE,
            'dv_vs_nbr_ratio': float(np.abs(dv_n).mean() / (np.abs(dv_nbr_mean).mean() + 1e-6)),
            'dt_vs_nbr_mean': float(np.mean(np.abs(dt_n - dt_nbr_mean))) * SCALE,
            'corr_v_nbr': float(safe_corr(dv_n, dv_nbr_mean)),
            'corr_t_nbr': float(safe_corr(dt_n, dt_nbr_mean)),
            'nbr_std_v': float(dv_nbr_std) * SCALE,

            'label': int(y[n])
        }
        feats.append(feat)

    return pd.DataFrame(feats)

def load_dataset(dirs, max_files=60):
    if isinstance(dirs, str):
        dirs = [dirs]
    files = []
    for d in dirs:
        if os.path.exists(d):
            files.extend([os.path.join(d, x) for x in sorted(os.listdir(d)) if x.endswith('.pkl')])

    dfs = []
    for f in tqdm(files[:max_files], desc=f"from {','.join([os.path.basename(d) for d in dirs])}"):
        dfs.append(extract_features(f, load_adj('topology.csv')))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

class LLMDetector:
    def __init__(self, train_df, config_path='configs.yaml'):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.key = cfg['api_key']
        self.model = cfg['model']
        self.url = "https://api.groq.com/openai/v1/chat/completions"

        # compute global statistics for context
        attacks = train_df[train_df.label == 1]
        normals = train_df[train_df.label == 0]

        self.stats = {
            'attack_dv_mean': attacks['dv_max'].mean(),
            'attack_dv_std': attacks['dv_max'].std(),
            'attack_dv_min': attacks['dv_max'].min(),
            'attack_dv_25': attacks['dv_max'].quantile(0.25),
            'attack_dv_75': attacks['dv_max'].quantile(0.75),
            'normal_dv_mean': normals['dv_max'].mean(),
            'normal_dv_std': normals['dv_max'].std(),
            'normal_dv_95': normals['dv_max'].quantile(0.95),
        }

        # select diverse representative examples
        self.examples = self._select_examples(train_df)

    def _select_examples(self, df):
        attacks = df[df.label == 1].copy()
        normals = df[df.label == 0].copy()

        examples = []

        # attack examples, cover severity spectrum
        if len(attacks) >= 4:
            attacks_sorted = attacks.sort_values('dv_max')
            examples.append(attacks_sorted.iloc[0])                 # weakest attack
            examples.append(attacks_sorted.iloc[len(attacks) // 3]) # low-medium
            examples.append(attacks_sorted.iloc[2 * len(attacks) // 3]) # medium-high
            examples.append(attacks_sorted.iloc[-1])    # strongest attack

        # normal examples, clean vs boundary cases
        if len(normals) >= 4:
            normals_sorted = normals.sort_values('dv_max')
            examples.append(normals_sorted.iloc[0])  # cleanest
            examples.append(normals_sorted.iloc[len(normals) // 4])  # low
            examples.append(normals_sorted.iloc[len(normals) // 2])  # median
            examples.append(normals_sorted.iloc[-1])  # highest (boundary)

        return pd.DataFrame(examples).sample(frac=1, random_state=7)

    def _format_example(self, row, show_label=True):
        label_str = f" → {'ATTACK' if row['label'] else 'NORMAL'}" if show_label else " →"
        return (
            f"Node{row['node_id']} (Storage={row['is_storage']}, Nbrs={row['n_neighbors']}): "
            f"dV_max={row['dv_max']:.2f}, dV_std={row['dv_std']:.2f}, "
            f"dV_p90={row['dv_p90']:.2f}, dT_max={row['dt_max']:.2f}, "
            f"grad={row['dv_grad_max']:.2f}, afternoon={row['dv_afternoon_max']:.2f}, "
            f"vs_nbr={row['dv_vs_nbr_mean']:.2f}, ratio={row['dv_vs_nbr_ratio']:.2f}, "
            f"corr={row['corr_v_nbr']:.2f}{label_str}"
        )

    def _build_prompt(self, target_row):
        # statistical context
        stats_str = (
            f"DATASET STATISTICS (×10⁴ p.u.):\n"
            f"Attack class: dV_max μ={self.stats['attack_dv_mean']:.2f}, σ={self.stats['attack_dv_std']:.2f}, "
            f"range=[{self.stats['attack_dv_min']:.2f}, Q1={self.stats['attack_dv_25']:.2f}, Q3={self.stats['attack_dv_75']:.2f}]\n"
            f"Normal class: dV_max μ={self.stats['normal_dv_mean']:.2f}, σ={self.stats['normal_dv_std']:.2f}, "
            f"95th={self.stats['normal_dv_95']:.2f}\n\n"
        )

        # decision rules
        rules_str = (
            f"CLASSIFICATION RULES:\n"
            f"1. STRONG ATTACK INDICATORS (high confidence):\n"
            f"   - dV_max > {self.stats['normal_dv_95']:.2f} (exceeds 95th percentile of normals)\n"
            f"   - dV_max > 1.5 AND (afternoon > 1.0 OR grad > 0.8)\n"
            f"   - vs_nbr > 0.8 AND ratio > 2.0 (self deviation >> neighbors)\n"
            f"2. WEAK ATTACK INDICATORS (moderate confidence):\n"
            f"   - dV_max in [{self.stats['attack_dv_25']:.2f}, {self.stats['attack_dv_75']:.2f}] AND is_storage=1\n"
            f"   - afternoon > 0.6 AND vs_nbr > 0.5\n"
            f"3. NORMAL INDICATORS:\n"
            f"   - dV_max < {self.stats['normal_dv_mean'] + self.stats['normal_dv_std']:.2f}\n"
            f"   - ratio < 1.5 AND corr > 0.7 (coupled with neighbors, not independent attack)\n\n"
        )

        examples_str = "TRAINING EXAMPLES:\n"
        for _, ex in self.examples.iterrows():
            examples_str += self._format_example(ex) + "\n"

        # test case
        test_str = f"\nTEST CASE:\n{self._format_example(target_row, show_label=False)}"

        prompt = (
            f"You are detecting False Data Injection Attacks on power grid nodes.\n\n"
            f"{stats_str}{rules_str}{examples_str}{test_str}\n\n"
            f"Analyze the test case using the statistical context and rules. "
            f"Respond with exactly: 'ATTACK' or 'NORMAL'"
        )

        return prompt

    def predict(self, test_df):
        preds, raw_responses = [], []
        errors = []

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="inferencing"):
            pred, raw = self._query_with_retry(row)
            preds.append(pred)
            raw_responses.append(raw)

            if pred != row['label']:
                errors.append({
                    'idx': idx,
                    'true': row['label'],
                    'pred': pred,
                    'dv_max': row['dv_max'],
                    'response': raw
                })

        if errors:
            print(f"\nerror analysis (first 5/{len(errors)}):")
            for e in errors[:5]:
                print(f"  true={e['true']}, pred={e['pred']}, dv_max={e['dv_max']:.2f}")
                print(f"  LLM: '{e['response'][:100]}'")

        return np.array(preds)

    ## CODE COPIED FROM GENAI
    def _query_with_retry(self, row, max_retries=3):
        prompt = self._build_prompt(row)

        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 10
                }

                time.sleep(1.5)

                resp = requests.post(self.url, headers=headers, json=data, timeout=15)

                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content'].strip().upper()
                    if 'ATTACK' in content and 'NORMAL' not in content:
                        return 1, content
                    elif 'NORMAL' in content:
                        return 0, content
                    return self._heuristic_fallback(row), content

                elif resp.status_code == 429:   # rate limit hit, wait longer
                    wait_time = (attempt + 1) * 5
                    print(f"\n[429] rate limited. sleeping {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    print(f"\n[API Error] {resp.status_code}: {resp.text[:100]}")
                    time.sleep(1)

            except Exception as e:
                print(f"\n[Connection Error] {e}")
                time.sleep(1)

        return self._heuristic_fallback(row), "FAILED_ALL_RETRIES"

    def _heuristic_fallback(self, row):
        if row['dv_max'] > 0.8 and row['corr_v_nbr'] < 0.5:
            return 1
        if row['dv_max'] > 2.0:
            return 1
        return 0

if __name__ == "__main__":
    print("loading training data")
    df_train = load_dataset(DATA_TRAIN, max_files=60)

    print("loading test data")
    df_test = load_dataset(DATA_TEST, max_files=40)

    if len(df_train) == 0 or len(df_test) == 0:
        exit("insufficient data")

    print(f"\ntrain: {len(df_train)} samples, attack ratio: {df_train.label.mean():.3f}")
    print(f"test: {len(df_test)} samples, attack ratio: {df_test.label.mean():.3f}")

    clf = LLMDetector(df_train) # build classifier

    # balanced test subset
    attacks = df_test[df_test.label == 1]
    normals = df_test[df_test.label == 0]

    n_samples = min(len(attacks), len(normals), 60)
    test_subset = pd.concat([
        attacks.sample(n=n_samples, random_state=7),
        normals.sample(n=n_samples, random_state=7)
    ]).sample(frac=1, random_state=7)

    print(f"\nRunning on {len(test_subset)} balanced samples")
    y_pred = clf.predict(test_subset)
    y_true = test_subset['label'].values

    ## PLOTTING CODES BELOW BY GENAI, reference only
    print(classification_report(y_true, y_pred, target_names=['normal', 'attack'], digits=3))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['normal', 'attack'], yticklabels=['normal', 'attack'])
    axes[0].set_ylabel('actual')
    axes[0].set_xlabel('predicted')
    axes[0].set_title('confusion matrix')

    # feature importance visualization
    test_attacks = test_subset[test_subset.label == 1]
    test_normals = test_subset[test_subset.label == 0]

    x_pos = np.arange(2)
    features = ['dv_max', 'dv_std', 'dv_afternoon_max', 'dv_vs_nbr_ratio']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for i, feat in enumerate(features):
        axes[1].bar(x_pos + i * 0.15,
                    [test_normals[feat].mean(), test_attacks[feat].mean()],
                    width=0.15, label=feat, color=colors[i])

    axes[1].set_xticks(x_pos + 0.225)
    axes[1].set_xticklabels(['normal', 'attack'])
    axes[1].set_ylabel('mean value')
    axes[1].set_title('feature distributions')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()