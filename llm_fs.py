import os
import time
import pickle
import random
import requests
import yaml
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report as cr, confusion_matrix as cm
from tqdm import tqdm

#from transformers import AutoTokenizer

with open('configs.yaml') as f:
    CFG = yaml.safe_load(f)
    NODES = CFG['n_node']

if CFG.get('provider') == 'local':
    from vllm import LLM, SamplingParams

SCALE = 1e4     # LLM struggle with small decimals since all power data in p.u.; save token
SHOTS = 5       # normal or attack example count, 0 for ZSL
PROMPTS = 10    # API prompt batch size; quicker but degrading perf as this increases & heavier cost for fallback
BALANCED = True # DB: test mode, set to False for "real-life" scenario (<1% attack)


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

    y = np.zeros(NODES, dtype=int)  # parse labels
    btm = data.get('esset_btm_a', data.get('esset_btm', {}))
    if 'A' in btm:
        idx = np.array(btm['A']) - 1
        y[idx[idx < NODES]] = 1

    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1

    ## CHANGED: Robust check for timeset format (int vs string)
    raw_steps = data['timeset_a']['A']
    if raw_steps and isinstance(raw_steps[0], int):
        valid_steps = sorted(raw_steps)
    else:
        valid_steps = sorted([int(t[1:]) for t in raw_steps])

    dfs = []
    # loop through dynamic attack window only
    for t in valid_steps:
        if t < 2: continue # skip first step if needed for gradient/history

        steps = [f"x{i}" for i in range(1, t + 1)]

        # extract voltage matrices (N_nodes x T_steps)
        vs = np.array([data['sch_v'][s] for s in steps]).T
        va = np.array([data['attack_v'].get(s, data['sch_v'][s]) for s in steps]).T
        dv = va - vs
        dv_nbr = (adj @ dv) / deg   # neighbour avg matrix

        # Power Injection (PI) features
        pis = np.array([data['sch_pi'][s] for s in steps]).T
        pia = np.array([data['attack_pi'].get(s, data['sch_pi'][s]) for s in steps]).T
        dpi = pia - pis

        feats = pd.DataFrame({
            'node_id': np.arange(1, NODES + 1),
            'timestep': t,
            'is_storage': np.isin(np.arange(NODES), np.array(CFG['esb']) - 1).astype(int),
            'n_neighbors': adj.sum(axis=1).astype(int),

            # scaled stats
            'dv_max': np.abs(dv).max(axis=1) * SCALE,
            'dv_std': dv.std(axis=1) * SCALE,
            'dv_p90': np.percentile(np.abs(dv), 90, axis=1) * SCALE,
            'dpi_max': np.abs(dpi).max(axis=1) * SCALE,

            # temporal
            'dv_grad_max': np.abs(np.gradient(dv, axis=1)).max(axis=1) * SCALE,
            'dv_recent': np.abs(dv[:, -min(10, t):]).max(axis=1) * SCALE,

            # spatial
            'dv_vs_nbr': np.abs(dv.mean(1) - dv_nbr.mean(1)) * SCALE,
            'dv_ratio': np.abs(dv).mean(1) / (np.abs(dv_nbr).mean(1) + 1e-6),
            'corr_v': corr(dv, dv_nbr),

            'label': (y & ((np.abs(dv) > CFG['eps_attack']).any(axis=1))).astype(int)   ##
        })
        dfs.append(feats)

    return pd.concat(dfs, ignore_index=True)


def ldr(dirs, adj, max_file=None):
    files = []
    for d in dirs:
        files.extend([os.path.join(d, x) for x in sorted(os.listdir(d))])

    if max_file:
        random.seed(7)
        random.shuffle(files)
        files = files[:max_file]

    with Pool(min(cpu_count(), 8)) as p:
        dfs = list(tqdm(p.imap(partial(process_file, adj=adj), files), total=len(files)))

    return pd.concat([d for d in dfs if d is not None], ignore_index=True)


class LLMDetector:
    def __init__(self, data):
        self.provider = CFG.get('provider', 'groq')

        # global stats
        atk = data[data.label == 1]
        nor = data[data.label == 0]
        self.stats = {
            'atk_mu': atk.dv_max.mean(),
            'atk_std': atk.dv_max.std(),
            'nor_mu': nor.dv_max.mean(),
            'nor_std': nor.dv_max.std(),
            'nor_95': nor.dv_max.quantile(0.95),
            'nor_99': nor.dv_max.quantile(0.99) ## feat suggested by GenAI
        }
        self.ctr_fallback = 0
        self.ex_str = "\n".join([self._format(r) for _, r in self._select(data).iterrows()])

        if self.provider == 'local':
            self.llm = LLM(
                model=CFG['path'],
                tensor_parallel_size=2,
                gpu_memory_utilization=0.7, # adjust based on VRAM & current usage
                #trust_remote_code=True,
                max_model_len=4096, # adjust based on prompt length
                dtype='bfloat16'
            )
            self.param = SamplingParams(
                temperature=0,
                max_tokens=10,
                top_p=1.0,
                #stop=['ATK', 'NOR', '\n'] # DB: uncomment to test fallback performance
            )
        else:
            self.key = os.environ['GROQ_K1']
            self.model = CFG['model']
            self.url = "https://api.groq.com/openai/v1/chat/completions"
            #self.url = "https://openrouter.ai/api/v1/chat/completions"

    def _select(self, df):
        exs = []
        for label, ctr in [(1, SHOTS), (0, SHOTS)]:
            if label == 1:
                sub = df[df.label == 1].sort_values(['dv_max', 'timestep'])
                if len(sub) >= ctr:
                    # choose attack samples evenly spaced by severity
                    exs.append(sub.iloc[np.linspace(0, len(sub) - 1, ctr, dtype=int)])
                    # or randomly choose in each tier
                    #chunks = np.array_split(sub.index.to_numpy(), ctr)
                    #exs.append(pd.concat([sub.loc[idx].sample(1, random_state=7)for idx in chunks]))
            else:   # hard negatives that look like attack + random
                high_noise = df[df.label == 0].nlargest(ctr // 2, 'dv_max')
                rand_normal = df[df.label == 0].sample(ctr - len(high_noise), random_state=7)
                exs.append(pd.concat([high_noise, rand_normal]))

        return pd.concat(exs).sample(frac=1, random_state=7)

    def _format(self, r, label=True):
        lbl = f" → {'ATK' if r.label else 'NOR'}" if label else " →"
        return (f"N{int(r.node_id)}@t{int(r.timestep)} "
                f"(stor={int(r.is_storage)},nbr={int(r.n_neighbors)}): "
                f"dVmax={r.dv_max:.2f} dVstd={r.dv_std:.2f} dVp90={r.dv_p90:.2f} dpiMax={r.dpi_max:.2f} "
                f"grad={r.dv_grad_max:.2f} rcnt={r.dv_recent:.2f} "
                f"vs_nbr={r.dv_vs_nbr:.2f} ratio={r.dv_ratio:.2f} corr={r.corr_v:.2f}{lbl}")

    def _format_batch_prompt(self, batch_rows):
        targets_str = ""
        for i, (_, row) in enumerate(batch_rows):
            targets_str += f"Sample {i}: {self._format(row, False)}\n"

        example = f"EXAMPLES:\n{self.ex_str}\n" if self.ex_str else "ZERO-SHOT TASK."
        return (f"Detect FDIA. Cumulative features. Base rate varies by context.\n"
                f"CONTEXT: FDIA typically targets afternoon/evening periods (duck curve peak). Attackers build up silently to drain reserve margin, maximizing damage as evening load approaches.\n"
                f"STATS: Normal 95th={self.stats['nor_95']:.2f}, Normal 99th={self.stats['nor_99']:.2f}, Atk Mean={self.stats['atk_mu']:.2f}\n"
                f"RULES: dVmax > {self.stats['nor_95']:.2f} (Suspicious). dVmax > 1.5 & grad > 0.8 (Strong). ratio > 2.5 (Isolated).\n"
                f"Evaluate each case on its merits.\n"
                f"{example}\n"
                f"TEST BATCH:\n{targets_str}\n"
                f"Output ONLY a list of lines like '0: NOR' or '1: ATK'. Do not explain.")

    def _prompt(self, tgt):
        # Fallback for single/local inference (reusing batch logic for consistency)
        return self._format_batch_prompt([(0, tgt)]).replace("TEST BATCH", "TARGET").replace("Sample 0:", "Target:")

    def predict(self, tester):
        self.ctr_fallback = 0 # rst
        if self.provider == 'local':
            return self._predict_local(tester)

        # batch API requests for efficiency
        preds = []
        rows = list(tester.iterrows())

        for i in tqdm(range(0, len(rows), PROMPTS), desc="API inference"):
            batch_rows = rows[i:i + PROMPTS]
            batch_preds = self._query_batch(batch_rows)
            preds.extend(batch_preds)

        print(f"{self.ctr_fallback} fallback(s) ({self.ctr_fallback / len(tester):.1%})")
        return np.array(preds)

    def _predict_local(self, tester):
        """batch inference via vllm"""
        prompts = [self._prompt(row) for _, row in tester.iterrows()]
        outputs = self.llm.generate(prompts, self.param, use_tqdm=True)
        preds = []
        for i, o in enumerate(outputs):
            txt = o.outputs[0].text.strip().upper()
            if 'ATK' in txt:
                preds.append(1)
            elif 'NOR' in txt:
                preds.append(0)
            else:   # ambiguous response, resort to fallback
                self.ctr_fallback += 1
                preds.append(self._fallback(tester.iloc[i]))
                print(f"  [fallback] sample {i}: {txt}")    # DB

        print(f"{self.ctr_fallback} fallback(s) ({self.ctr_fallback / len(tester):.1%})")
        return np.array(preds)

    def _query_batch(self, batch_rows):
        prompt = self._format_batch_prompt(batch_rows)
        batch_results = [-1] * len(batch_rows)

        for attempt in range(5):    # retry logic to handle rate limits (429)
            try:
                res = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.key}",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a strict classifier. Respond ONLY with format 'Index: Label'. Example:\n0: NOR\n1: ATK"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": 100, # Increased for batch output
                        #"stop": ["ATK", "NOR", "\n"]
                    },
                    timeout=20
                )

                if res.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    # tqdm.write(f"rate limit hit. retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                res.raise_for_status()  # DB: expose any other err

                content = res.json()['choices'][0]['message']['content']

                # Parse lines like "0: NOR"
                for line in content.strip().split('\n'):
                    line = line.upper().strip()
                    if ':' in line: parts = line.split(':', 1)
                    elif '.' in line: parts = line.split('.', 1)
                    else: continue

                    try:
                        idx = int(parts[0].strip())
                        lbl = parts[1].strip()
                        if idx < len(batch_results):
                            if 'ATK' in lbl: batch_results[idx] = 1
                            elif 'NOR' in lbl: batch_results[idx] = 0
                    except: pass

                # If parsed most items successfully, stop retrying
                if batch_results.count(-1) <= 1:
                    break
            except Exception as e:
                print(f"API Error: {e}")
                pass

        # Fill any missing/unparsed items with fallback logic
        final_preds = []
        for i, (idx, row) in enumerate(batch_rows):
            if batch_results[i] != -1:
                final_preds.append(batch_results[i])
            else:
                self.ctr_fallback += 1
                final_preds.append(self._fallback(row))

        return final_preds

    def _fallback(self, r): ## may tune to integrate gnn
        return 1 if (r.dv_max > 2.0 or (r.dv_max > 1.2 and r.corr_v < 0.5)) else 0


if __name__ == "__main__":
    adj = np.zeros((NODES, NODES))
    t = pd.read_csv('topology.csv')
    adj[t.source, t.target] = adj[t.target, t.source] = 1   # adjacency matrix

    print("loading datasets...")
    trn = ldr(CFG['train_dirs'], adj)
    tst = ldr(CFG['test_dirs'], adj)

    if BALANCED:
        pos, neg = tst[tst.label == 1], tst[tst.label == 0]
        n = min(len(pos), len(neg), 50)
        tst_sub = pd.concat([pos.sample(n, random_state=7), neg.sample(n, random_state=7)]).sample(frac=1)
    else:   # adjust sample count per budget
        tst_sub = tst.sample(300, random_state=9)

    print(f"attack rate: {tst_sub.label.mean():.1%}")
    y_pred = LLMDetector(trn).predict(tst_sub)
    cond = ['normal', 'attacked']
    print(cr(tst_sub.label, y_pred, target_names=cond, digits=3))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm(tst_sub.label, y_pred), annot=True, fmt='d', xticklabels=cond, yticklabels=cond)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.show()