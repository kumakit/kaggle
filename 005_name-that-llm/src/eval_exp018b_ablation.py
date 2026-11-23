"""
005_name-that-llm: Exp #018b Post-Processing & Twin-Weight Ablation on 8-Backbone OOF
Tests sensitivity of Exp #018 (8-Backbone + Qwen3-Emb-8B Twin) to Cluster Sinkhorn, Global Sinkhorn,
and higher Twin/Exp018 weights via Strict 5-Fold Nested-CV.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "005_name-that-llm/src")
from eval_exp012_blend import cluster_sinkhorn, global_sinkhorn

SEED = 42
DATA_DIR = "005_name-that-llm/data"


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    classes = sorted(df["model"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = df["model"].map(c2i).values
    groups = df["prompt_id"].values
    G_tfidf = np.load(os.path.join(DATA_DIR, "tmp_exp012/capped_agglo_G.npy"))
    gates = np.load(os.path.join(DATA_DIR, "tmp_exp012/oof_exp013_gates.npz"))
    families = {f: np.array([i for i, c in enumerate(classes) if c.startswith(f)]) for f in ("claude", "gpt", "gemini", "glm")}
    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]

    eps = 1e-7
    files = [
        "oof_exp005_openpool_modernbert.csv",
        "oof_exp006_qwen3_lora.csv",
        "oof_exp007_qwen3_4b_lora.csv",
        "oof_exp008_modernbert_large.csv",
        "oof_exp012_deberta_large.csv",
        "oof_exp013_qwen3_8b_lora.csv",
        "oof_exp015_llama3_8b_lora.csv",
        "oof_exp018_qwen3_emb8b_plain.csv",
        "oof_exp018_qwen3_emb8b_centered.csv",
        "oof_exp011_stylegram.csv",
        "oof_exp013_stylegram_v2.csv",
        "oof_exp010_glm_blend.csv",
    ]
    probs = [np.clip(pd.read_csv(os.path.join(DATA_DIR, fn))[classes].values, eps, 1.0) for fn in files]
    l5, l6, l7, l8, l12, l13q8, l15ll, l18p, l18c, l11sg, l13sg, ll = [np.log(p) for p in probs]
    neu15 = 0.225 * l13q8 + 0.2025 * l7 + 0.2025 * l8 + 0.135 * l12 + 0.10125 * l6 + 0.03375 * l5 + 0.10 * l15ll

    def run_pipe(idx, w18, ac, c_sk, g_sk=0.35, w_sg1=0.35, w_sg2=0.35, lam=0.15, cov=0.30):
        l18 = log_softmax((1.0 - ac) * l18p[idx] + ac * l18c[idx], axis=1)
        neu = (1.0 - w18) * neu15[idx] + w18 * l18
        bl = log_softmax(neu + w_sg1 * l11sg[idx] + w_sg2 * l13sg[idx], axis=1)
        if lam > 0:
            out = bl.copy()
            for fam, idxs in families.items():
                base = out[:, idxs]
                base_lp = log_softmax(base, axis=1)
                sp = np.sort(np.exp(base_lp), axis=1)
                conf_gate = np.clip((0.45 - (sp[:, -1] - sp[:, -2])) / 0.45, 0.0, 1.0)
                strength = min(0.45, lam / (1.0 + lam)) * conf_gate
                mixed_lp = log_softmax((1.0 - strength[:, None]) * base_lp + strength[:, None] * gates[fam][idx], axis=1)
                out[:, idxs] = logsumexp(base, axis=1)[:, None] + mixed_lp
            bl = out
        if c_sk > 0:
            bl = cluster_sinkhorn(bl, G_tfidf[idx], 3, c_sk, 6, 18)
        if g_sk > 0:
            bl = global_sinkhorn(bl, g_sk)
        glm_mass = np.exp(bl)[:, glm_idxs].sum(1)
        sel = np.sort(np.argsort(-glm_mass, kind="stable")[: int(round(cov * len(bl)))])
        before = bl.argmax(1)
        after = before.copy()
        after[sel] = (bl[sel] + 1.15 * ll[idx][sel]).argmax(1)
        glm_rel = np.isin(before, glm_idxs) | np.isin(after, glm_idxs)
        after[(before != after) & ~glm_rel] = before[(before != after) & ~glm_rel]
        return after

    all_i = np.arange(len(y))
    p18_base = run_pipe(all_i, 0.30, 0.35, c_sk=0.85, g_sk=0.35)
    sc18_base = balanced_accuracy_score(y, p18_base)
    print(f"Exp #018 Baseline (w18=0.30, ac=0.35, c_sk=0.85, g_sk=0.35): {sc18_base:.5f}")

    print("\n=== Cluster Sinkhorn vs Global Sinkhorn Sweep (w18=0.30, ac=0.35) ===")
    for c_sk in [0.00, 0.35, 0.65, 0.85]:
        for g_sk in [0.30, 0.35]:
            p = run_pipe(all_i, 0.30, 0.35, c_sk=c_sk, g_sk=g_sk)
            sc = balanced_accuracy_score(y, p)
            print(f"  c_sk={c_sk:.2f}, g_sk={g_sk:.2f} -> OOF={sc:.5f} ({sc - sc18_base:+.5f}) | agree18={float((p == p18_base).mean()):.3f}")

    print("\n=== Higher w18 / ac Sweep at c_sk=0.85 vs c_sk=0.00 ===")
    for c_sk in [0.85, 0.00]:
        for w18 in [0.30, 0.35, 0.40]:
            for ac in [0.35, 0.45, 0.55]:
                p = run_pipe(all_i, w18, ac, c_sk=c_sk, g_sk=0.35)
                sc = balanced_accuracy_score(y, p)
                print(f"  c_sk={c_sk:.2f}, w18={w18:.2f}, ac={ac:.2f} -> OOF={sc:.5f}")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cands = [(w18, ac, c_sk) for w18 in [0.30, 0.35, 0.40] for ac in [0.35, 0.45, 0.55] for c_sk in [0.00, 0.65, 0.85]]
    nested = np.zeros_like(y)
    chosen = []
    for tr, va in splitter.split(df, y, groups=groups):
        bsc, bcfg = -1.0, None
        for cfg in cands:
            sc = balanced_accuracy_score(y[tr], run_pipe(tr, *cfg))
            if sc > bsc:
                bsc, bcfg = sc, cfg
        chosen.append(bcfg)
        nested[va] = run_pipe(va, *bcfg)
    print(f"\n=== Strict 5-Fold Nested-CV (8-Backbone Extended Grid): {balanced_accuracy_score(y, nested):.5f} ===")
    print(f"  chosen={chosen}")


if __name__ == "__main__":
    main()
