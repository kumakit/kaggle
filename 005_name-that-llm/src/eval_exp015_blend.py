"""
005_name-that-llm: Exp #015 7-Backbone (+ Llama-3.1-8B 24L) + SGv1/v2 + Gates + Sinkhorn + GLM-5.3/Flash Evaluator
Evaluates exact OOF, 5-Fold Nested-CV, per-class recall, and optimal 7-backbone weight allocation.
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
    l5, l6, l7, l8, l12, l13q8, l15ll, l11sg, l13sg, ll = [
        np.log(np.clip(pd.read_csv(os.path.join(DATA_DIR, fn))[classes].values, eps, 1.0))
        for fn in [
            "oof_exp005_openpool_modernbert.csv",
            "oof_exp006_qwen3_lora.csv",
            "oof_exp007_qwen3_4b_lora.csv",
            "oof_exp008_modernbert_large.csv",
            "oof_exp012_deberta_large.csv",
            "oof_exp013_qwen3_8b_lora.csv",
            "oof_exp015_llama3_8b_lora.csv",
            "oof_exp011_stylegram.csv",
            "oof_exp013_stylegram_v2.csv",
            "oof_exp010_glm_blend.csv",
        ]
    ]

    # Exp014 base 6-backbone weights (sum=1.0): 0.25 Q8, 0.225 Q4, 0.225 MB-L, 0.15 Deb-L, 0.1125 Q1.7, 0.0375 MB-B
    base6_all = (
        0.25 * l13q8
        + 0.225 * l7
        + 0.225 * l8
        + 0.15 * l12
        + 0.1125 * l6
        + 0.0375 * l5
    )

    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]

    def run_pipeline(idx, w_ll, w_q8_extra=0.0, w_sg1=0.35, w_sg2=0.35, lam=0.15, c_sk=0.85, g_sk=0.35, cov=0.30):
        # Mix Exp014 6-backbone blend with Llama3-8B (and optional extra Qwen3-8B shift)
        w_b6 = max(0.0, 1.0 - w_ll - w_q8_extra)
        neu = w_b6 * base6_all[idx] + w_ll * l15ll[idx] + w_q8_extra * l13q8[idx]
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
            bl = cluster_sinkhorn(bl, G[idx], 3, c_sk, 6, 18)
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
    print("=== Neural-Only Sweep (before SG/Gates/Sinkhorn/GLM) ===")
    for w_ll in [0.00, 0.08, 0.12, 0.15, 0.18, 0.22]:
        neu = (1.0 - w_ll) * base6_all + w_ll * l15ll
        print(f"  w_llama={w_ll:.2f} -> Neural BalAcc={balanced_accuracy_score(y, neu.argmax(1)):.5f}")

    print("\n=== Full Pipeline Sweep vs Exp #014 (w_ll=0.00, sg1=0.35, sg2=0.35, lam=0.15, c_sk=0.85) ===")
    for w_ll in [0.00, 0.08, 0.12, 0.15, 0.18, 0.22]:
        p = run_pipeline(all_i, w_ll=w_ll)
        sc = balanced_accuracy_score(y, p)
        rec = recall_score(y, p, average=None)
        print(f"  w_ll={w_ll:.2f} (lam=0.15, c_sk=0.85) -> OOF BalAcc={sc:.5f} | min_rec={rec.min():.3f} ({classes[rec.argmin()]})")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cands = [
        (w_ll, w_sg1, w_sg2, lam, c_sk)
        for w_ll in [0.00, 0.10, 0.14, 0.18, 0.22]
        for (w_sg1, w_sg2) in [(0.35, 0.35), (0.30, 0.45)]
        for lam in [0.15, 0.25]
        for c_sk in [0.85]
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

    nested_sc = balanced_accuracy_score(y, nested_preds)
    print(f"\n=== 5-Fold Nested-CV Full Pipeline BalAcc: {nested_sc:.5f} ===")
    for i, c in enumerate(chosen):
        print(f"  Fold {i + 1} selected: w_ll={c[0]}, sg1={c[1]}, sg2={c[2]}, lam={c[3]}, c_sk={c[4]}")

    # Check Exp014 baseline vs best Exp015 per-class recall
    p_base = run_pipeline(all_i, w_ll=0.00)
    p_best = run_pipeline(all_i, w_ll=0.15)
    r_base = recall_score(y, p_base, average=None)
    r_best = recall_score(y, p_best, average=None)
    print("\n=== Class Recall Comparison (Exp014 w_ll=0.00 -> Exp015 w_ll=0.15) ===")
    for i, cls in enumerate(classes):
        print(f"  {cls:24s}: {r_base[i]:.3f} -> {r_best[i]:.3f} ({r_best[i] - r_base[i]:+.3f})")


if __name__ == "__main__":
    main()
