import os
import pickle
import time
import yaml
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

with open('configs.yaml') as f:
    CFG = yaml.safe_load(f)

if CFG.get('provider') == 'local':
    from vllm import LLM, SamplingParams


def corr(A, B):
    """fast vectorized correlation"""
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    cov = (A * B).sum(axis=1)
    stds = np.sqrt((A**2).sum(axis=1) * (B**2).sum(axis=1))
    return np.divide(cov, stds, out=np.zeros_like(cov), where=stds != 0)


def process_file(path, adj):
    """load file once & extract feat for all timesteps vectorially"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # parse labels
    y = np.zeros(CFG['nodes'], dtype=int)
    btm = data.get('esset_btm_a', data.get('esset_btm', {}))
    if 'A' in btm:
        idx = np.array(btm['A']) - 1
        y[idx[idx < CFG['nodes']]] = 1

    # pre compute neighbor degree for averaging
    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1

    dfs = []
    for t in CFG['eval_timesteps']:
        steps = [f"x{i}" for i in range(1, t + 1)]

        # extract matrices (N_nodes x T_steps)
        vs = np.array([data['sch_v'][s] for s in steps]).T
        va = np.array([data['attack_v'].get(s, data['sch_v'][s]) for s in steps]).T
        dv = va - vs

        # neighbor avg matrix
        dv_nbr = (adj @ dv) / deg

        feats = pd.DataFrame({
            'node_id': np.arange(1, CFG['nodes'] + 1),
            'timestep': t,
            'is_storage': np.isin(np.arange(CFG['nodes']), np.array(CFG['esb']) - 1).astype(int),
            'n_neighbors': adj.sum(axis=1).astype(int),
            'is_afternoon': 1 if t >= 60 else 0,

            # scaled stats
            'dv_max': np.abs(dv).max(axis=1) * CFG['scale'],
            'dv_std': dv.std(axis=1) * CFG['scale'],
            'dv_p90': np.percentile(np.abs(dv), 90, axis=1) * CFG['scale'],

            # temporal
            'dv_grad_max': np.abs(np.gradient(dv, axis=1)).max(axis=1) * CFG['scale'],
            'dv_recent': np.abs(dv[:, -min(10, t):]).max(axis=1) * CFG['scale'],

            # spatial
            'dv_vs_nbr': np.abs(dv.mean(1) - dv_nbr.mean(1)) * CFG['scale'],
            'dv_ratio': np.abs(dv).mean(1) / (np.abs(dv_nbr).mean(1) + 1e-6),
            'corr_v': corr(dv, dv_nbr),

            'label': (y & ((np.abs(dv) > CFG['eps_attack']).any(axis=1))).astype(int)
        })
        dfs.append(feats)

    return pd.concat(dfs, ignore_index=True)


def load_dataset(dirs, adj, max_files=None):
    files = []
    for d in dirs:
        if os.path.exists(d):
            files.extend([os.path.join(d, x) for x in sorted(os.listdir(d))])

    if max_files:
        files = files[:max_files]

    with Pool(min(cpu_count(), 8)) as p:
        dfs = list(tqdm(p.imap(partial(process_file, adj=adj), files), total=len(files)))

    return pd.concat([d for d in dfs if d is not None], ignore_index=True)


class LLMDetector:
    def __init__(self, train_df):
        self.provider = CFG.get('provider', 'groq')

        # global stats
        atk = train_df[train_df.label == 1]
        nor = train_df[train_df.label == 0]
        self.stats = {
            'atk_mu': atk.dv_max.mean(), 'atk_std': atk.dv_max.std(),
            'nor_mu': nor.dv_max.mean(), 'nor_std': nor.dv_max.std(),
            'nor_95': nor.dv_max.quantile(0.95)
        }
        self.examples = self._select_examples(train_df)

        # provider setup
        if self.provider == 'local':
            print(f"init local model: {CFG['path']}")
            self.llm = LLM(
                model=CFG['path'],
                tensor_parallel_size=2,
                trust_remote_code=True,
                gpu_memory_utilization=0.7,
                max_model_len=4096, # adjust based on prompt length
                dtype='bfloat16'
            )
            self.params = SamplingParams(
                temperature=0,
                max_tokens=10,
                top_p=1.0,
                #stop=['ATK', 'NOR', '\n'] # DEBUGGING: uncomment to test fallback performance
            )
        else:
            self.key = CFG['api_key']
            self.model = CFG['model']
            self.url = "https://api.groq.com/openai/v1/chat/completions"

    def _select_examples(self, df):
        exs = []
        for label, count in [(1, 4), (0, 4)]:
            sub = df[df.label == label].sort_values(['dv_max', 'timestep'])
            if len(sub) >= count:
                exs.append(sub.iloc[np.linspace(0, len(sub) - 1, count, dtype=int)])
        return pd.concat(exs).sample(frac=1, random_state=7)

    def _format(self, r, label=True):
        lbl = f" → {'ATK' if r.label else 'NOR'}" if label else " →"
        return (f"N{int(r.node_id)}@t{int(r.timestep)} (stor={int(r.is_storage)},nbr={int(r.n_neighbors)},aft={int(r.is_afternoon)}): "
                f"dVmax={r.dv_max:.2f} dVstd={r.dv_std:.2f} dVp90={r.dv_p90:.2f} "
                f"grad={r.dv_grad_max:.2f} rcnt={r.dv_recent:.2f} "
                f"vs_nbr={r.dv_vs_nbr:.2f} ratio={r.dv_ratio:.2f} corr={r.corr_v:.2f}{lbl}")

    def _prompt(self, tgt, ex_str=None):
        if ex_str is None:
            ex_str = "\n".join([self._format(r) for _, r in self.examples.iterrows()])
        return (f"Detect FDIA. Cumulative features.\n"
                f"Stats: Normal 95th={self.stats['nor_95']:.2f}, Atk Mean={self.stats['atk_mu']:.2f}\n"
                f"Rules: dVmax > {self.stats['nor_95']:.2f} (Suspicious). dVmax > 1.5 & grad > 0.8 (Strong). ratio > 2.5 (Isolated).\n"
                f"EXAMPLES:\n{ex_str}\nTEST:\n{self._format(tgt, False)}\n"
                f"Respond ONLY with either 'ATK' or 'NOR'. Do not explain.")

    def predict(self, test_df):
        if self.provider == 'local':
            return self._predict_local(test_df)

        # api fallback
        preds = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="API inference"):
            preds.append(self._query(row))
        return np.array(preds)

    def _predict_local(self, test_df):
        """batch inference with vllm"""
        ex_str = "\n".join([self._format(r) for _, r in self.examples.iterrows()])
        prompts = [self._prompt(row, ex_str) for _, row in test_df.iterrows()]

        print(f"running batch inference on {len(prompts)} samples locally...")
        outputs = self.llm.generate(prompts, self.params, use_tqdm=True)
        preds = []
        for i, o in enumerate(outputs):
            txt = o.outputs[0].text.strip().upper()
            if 'ATK' in txt and 'NOR' not in txt:
                preds.append(1)
            elif 'NOR' in txt:
                preds.append(0)
            else:   # ambiguous response, use fallback
                preds.append(self._fallback(test_df.iloc[i]))
                print(f"  [fallback] sample {i}: {txt[:50]}")
        return np.array(preds)

    def _query(self, row):
        for _ in range(3):
            try:
                time.sleep(1.0)
                res = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role":"user","content": self._prompt(row)}],
                        "temperature":0,
                        "max_tokens":5
                    },
                    timeout=10)
                if res.status_code == 200:
                    txt = res.json()['choices'][0]['message']['content'].upper()
                    return 1 if 'ATK' in txt else (0 if 'NOR' in txt else self._fallback(row))
            except:
                pass
        return self._fallback(row)

    def _fallback(self, r):
        return 1 if (r.dv_max > 2.0 or (r.dv_max > 1.2 and r.corr_v < 0.5)) else 0


if __name__ == "__main__":
    adj = np.zeros((CFG['nodes'], CFG['nodes']))
    t = pd.read_csv('topology.csv')
    adj[t.source, t.target] = adj[t.target, t.source] = 1

    print("loading data...")
    df_train = load_dataset(CFG['train_dirs'], adj, 60)
    df_test = load_dataset(CFG['test_dirs'], adj, 40)

    # balanced evaluation
    pos, neg = df_test[df_test.label==1], df_test[df_test.label==0]
    n = min(len(pos), len(neg), 80)
    tst_sub = pd.concat([pos.sample(n, random_state=7), neg.sample(n, random_state=7)]).sample(frac=1)

    print(f"testing on {len(tst_sub)} samples...")
    y_pred = LLMDetector(df_train).predict(tst_sub)
    mode = ['normal', 'attacked']

    print(classification_report(tst_sub.label, y_pred, target_names=mode, digits=3))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(tst_sub.label, y_pred), annot=True, fmt='d', xticklabels=mode, yticklabels=mode)
    plt.show()