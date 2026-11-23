"""
005_name-that-llm: Exp #019 9-Backbone (+ Qwen3-14B-27L Twin LoRA) + SGv1/v2 + Gates + Sinkhorn + GLM Evaluator
Evaluates correlation with Exp #018 & existing 7 backbones, Neural-Only OOF, Full Pipeline OOF,
per-class recall comparison, and strict 5-Fold Nested-CV vs Exp #018 (LB 0.668) baseline.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.metrics import balanced_accuracy_score, recall_score
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
        "oof_exp019_qwen3_14b_plain.csv",
        "oof_exp019_qwen3_14b_centered.csv",
        "oof_exp011_stylegram.csv",
        "oof_exp013_stylegram_v2.csv",
        "oof_exp010_glm_blend.csv",
    ]
    probs = [np.clip(pd.read_csv(os.path.join(DATA_DIR, fn))[classes].values, eps, 1.0) for fn in files]
    l5, l6, l7, l8, l12, l13q8, l15ll, l18p, l18c, l19p, l19c, l11sg, l13sg, ll = [np.log(p) for p in probs]

    print("=== Exp #019 Single-Model & Twin Head OOFs ===")
    print(f"  Plain-Only OOF        : {balanced_accuracy_score(y, l19p.argmax(1)):.5f}")
    print(f"  Centered-Only OOF     : {balanced_accuracy_score(y, l19c.argmax(1)):.5f}")
    for alpha_c in [0.25, 0.35, 0.45, 0.50]:
        ltw = log_softmax((1.0 - alpha_c) * l19p + alpha_c * l19c, axis=1)
        print(f"  Twin (alpha_c={alpha_c:.2f})   : {balanced_accuracy_score(y, ltw.argmax(1)):.5f}")

    names = ["MB-B", "Q3-1.7B", "Q3-4B", "MB-L", "Deb-L", "Q3-8B", "Llama3-8B", "Exp018-P", "Exp018-C"]
    print("\n=== Probability Correlations vs Exp #019 (Qwen3-14B-27L Twin) ===")
    for nm, p in zip(names, probs[:9]):
        c_p = np.corrcoef(p.ravel(), probs[9].ravel())[0, 1]
        c_c = np.corrcoef(p.ravel(), probs[10].ravel())[0, 1]
        print(f"  {nm:10s}: plain={c_p:.4f} | centered={c_c:.4f}")

    neu15 = 0.225 * l13q8 + 0.2025 * l7 + 0.2025 * l8 + 0.135 * l12 + 0.10125 * l6 + 0.03375 * l5 + 0.10 * l15ll
    l18_base = log_softmax(0.65 * l18p + 0.35 * l18c, axis=1)
    neu18 = 0.70 * neu15 + 0.30 * l18_base

    print(f"\n=== Neural-Only Sweep ((1 - w19) * neu18 + w19 * Exp019_Twin) ===")
    print(f"  Exp #018 Neural Baseline: {balanced_accuracy_score(y, neu18.argmax(1)):.5f}")
    for w19 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for ac19 in [0.00, 0.25, 0.35, 0.45]:
            l19 = log_softmax((1.0 - ac19) * l19p + ac19 * l19c, axis=1)
            nb = (1.0 - w19) * neu18 + w19 * l19
            print(f"  w19={w19:.2f}, ac19={ac19:.2f} -> Neural OOF={balanced_accuracy_score(y, nb.argmax(1)):.5f}")

    def run_pipe(idx, w19, ac19, w_sg1=0.35, w_sg2=0.35, lam=0.15, c_sk=0.85, g_sk=0.35, cov=0.30):
        l19 = log_softmax((1.0 - ac19) * l19p[idx] + ac19 * l19c[idx], axis=1)
        neu = (1.0 - w19) * neu18[idx] + w19 * l19
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
    p18 = run_pipe(all_i, 0.00, 0.00)
    sc18 = balanced_accuracy_score(y, p18)
    print(f"\n=== Full Pipeline OOF Sweep (Baseline Exp #018 = {sc18:.5f}) ===")
    for w19 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for ac19 in [0.00, 0.25, 0.35, 0.45]:
            p = run_pipe(all_i, w19, ac19)
            sc = balanced_accuracy_score(y, p)
            rec = recall_score(y, p, average=None)
            agree = float((p == p18).mean())
            print(f"  w19={w19:.2f}, ac19={ac19:.2f} -> OOF={sc:.5f} ({sc - sc18:+.5f}) | min_rec={rec.min():.3f} | agree18={agree:.3f}")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    base_val = np.zeros_like(y)
    for _, va in splitter.split(df, y, groups=groups):
        base_val[va] = run_pipe(va, 0.00, 0.00)
    sc18_val = balanced_accuracy_score(y, base_val)

    cands = [(0.00, 0.00)] + [(w19, ac19) for w19 in [0.10, 0.15, 0.20, 0.25, 0.30] for ac19 in [0.25, 0.35, 0.45]]
    nested = np.zeros_like(y)
    chosen = []
    for tr, va in splitter.split(df, y, groups=groups):
        bsc, bcfg = -1.0, None
        for w19, ac19 in cands:
            sc = balanced_accuracy_score(y[tr], run_pipe(tr, w19, ac19))
            if sc > bsc:
                bsc, bcfg = sc, (w19, ac19)
        chosen.append(bcfg)
        nested[va] = run_pipe(va, bcfg[0], bcfg[1])
    nsc = balanced_accuracy_score(y, nested)
    print(f"\n=== Strict 5-Fold Nested-CV ===")
    print(f"  Exp #018 Val-Split: {sc18_val:.5f} -> Exp #019 Nested-CV: {nsc:.5f} ({nsc - sc18_val:+.5f}) | chosen={chosen}")


if __name__ == "__main__":
    main()
