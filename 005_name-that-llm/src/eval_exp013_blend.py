"""
005_name-that-llm: Exp #013 Full 6-Backbone + StyleGram v2 + Family Gates + Sinkhorn + GLM-5.3/Flash Evaluator
Evaluates exact OOF & Nested-CV Balanced Accuracy with real oof_exp013_qwen3_8b_lora.csv + oof_exp013_stylegram_v2.csv.
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
    G = np.load(os.path.join(DATA_DIR, "tmp_exp012/capped_agglo_G.npy"))
    gates = np.load(os.path.join(DATA_DIR, "tmp_exp012/oof_exp013_gates.npz"))
    families = {f: np.array([i for i, c in enumerate(classes) if c.startswith(f)]) for f in ("claude", "gpt", "gemini", "glm")}

    eps = 1e-7
    l5, l6, l7, l8, l12, l13q8, l13sg, ll = [
        np.log(np.clip(pd.read_csv(os.path.join(DATA_DIR, fn))[classes].values, eps, 1.0))
        for fn in [
            "oof_exp005_openpool_modernbert.csv",
            "oof_exp006_qwen3_lora.csv",
            "oof_exp007_qwen3_4b_lora.csv",
            "oof_exp008_modernbert_large.csv",
            "oof_exp012_deberta_large.csv",
            "oof_exp013_qwen3_8b_lora.csv",
            "oof_exp013_stylegram_v2.csv",
            "oof_exp010_glm_blend.csv",
        ]
    ]
    print(f"Exp013 Qwen3-8B single OOF : {balanced_accuracy_score(y, l13q8.argmax(1)):.5f}")
    print(f"Exp013 StyleGram v2 OOF    : {balanced_accuracy_score(y, l13sg.argmax(1)):.5f}")
    print(f"Corr(Qwen3-8B_v2, Qwen3-4B): {np.corrcoef(np.exp(l13q8).ravel(), np.exp(l7).ravel())[0,1]:.4f}")

    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]

    def run_pipeline(idx, w_q8, w_sg, lam, c_sk, cov=0.30):
        rem = 1.0 - w_q8
        neu = (
            w_q8 * l13q8[idx]
            + rem * (0.30 * l7[idx] + 0.30 * l8[idx] + 0.20 * l12[idx] + 0.15 * l6[idx] + 0.05 * l5[idx])
        )
        bl = log_softmax(neu + w_sg * l13sg[idx], axis=1)
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
            bl = cluster_sinkhorn(bl, G[idx], 3, c_sk, 6, 18)
        bl = global_sinkhorn(bl, 0.35)
        glm_mass = np.exp(bl)[:, glm_idxs].sum(1)
        sel = np.sort(np.argsort(-glm_mass, kind="stable")[: int(round(cov * len(bl)))])
        before = bl.argmax(1)
        after = before.copy()
        after[sel] = (bl[sel] + 1.15 * ll[idx][sel]).argmax(1)
        glm_rel = np.isin(before, glm_idxs) | np.isin(after, glm_idxs)
        after[(before != after) & ~glm_rel] = before[(before != after) & ~glm_rel]
        return after

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cands = [
        (w_q8, w_sg, lam, c_sk)
        for w_q8 in [0.15, 0.20, 0.25]
        for w_sg in [0.50, 0.65]
        for lam in [0.25, 0.35, 0.50]
        for c_sk in [0.65, 0.85]
    ]
    nested_preds = np.zeros_like(y)
    chosen = []
    for tr_idx, va_idx in splitter.split(df, y, groups=groups):
        best_sc, best_cfg = -1.0, None
        for cfg in cands:
            sc = balanced_accuracy_score(y[tr_idx], run_pipeline(tr_idx, *cfg))
            if sc > best_sc:
                best_sc, best_cfg = sc, cfg
        chosen.append(best_cfg)
        nested_preds[va_idx] = run_pipeline(va_idx, *best_cfg)

    print(f"\nNested-CV Full Pipeline BalAcc: {balanced_accuracy_score(y, nested_preds):.5f}")
    for i, c in enumerate(chosen):
        print(f"  Fold {i + 1} selected: w_q8={c[0]}, w_sg={c[1]}, lam={c[2]}, c_sk={c[3]}")

    all_i = np.arange(len(y))
    print("\n=== Global Grid Summary (c_sk=0.65 vs 0.85, cov=0.30) ===")
    for w_q8 in [0.00, 0.15, 0.20, 0.25]:
        for w_sg in [0.50, 0.65]:
            for lam in [0.35]:
                for c_sk in [0.65, 0.85]:
                    p = run_pipeline(all_i, w_q8, w_sg, lam, c_sk)
                    sc = balanced_accuracy_score(y, p)
                    rec = recall_score(y, p, average=None)
                    print(f"  w_q8={w_q8:.2f} w_sg={w_sg:.2f} lam={lam:.2f} c_sk={c_sk:.2f} -> BalAcc={sc:.5f} | min_rec={rec.min():.3f} ({classes[rec.argmin()]})")


if __name__ == "__main__":
    main()
