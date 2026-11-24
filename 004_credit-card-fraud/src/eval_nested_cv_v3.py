"""
Fast Nested-CV v3 evaluation incorporating Exp #042 (10-Fold GLM-Boost LGBM) + all clean OOF banks.
Uses newton-cholesky and greedy forward selection, verified on 2 independent outer CV splits.
"""
import gc
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append("004_credit-card-fraud/src")
from eval_nested_cv_base import load_all_clean_models, deduplicate_models, to_rank_logit
from eval_nested_cv_v2 import load_busyaprime_models, eval_nested_logreg

DATA_DIR = "004_credit-card-fraud/data"


def main():
    y = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy().astype(int)
    models = load_all_clean_models(y)
    models = load_busyaprime_models(models, y)

    # Add Exp #042 (10-fold GLM-boost LightGBM, both foldavg and refit test)
    oof42 = pd.read_csv(f"{DATA_DIR}/exp042_out/oof_exp042_gb10.csv")["oof_exp042_gb10"].to_numpy()
    te42_refit = pd.read_csv(f"{DATA_DIR}/exp042_out/submission_exp042_refit.csv")["is_fraud"].to_numpy()
    models["ours_exp042_gb10_refit"] = (oof42.astype(np.float32), te42_refit.astype(np.float32), "ours_gbdt", roc_auc_score(y, oof42))

    print(f"Total clean models: {len(models)}", flush=True)
    single_models = {k: v for k, v in models.items() if k != "pub_wide_lm_stack"}
    kept_singles = deduplicate_models(single_models, y, corr_thresh=0.9994)
    print(f"Deduplicated single models: {len(kept_singles)}", flush=True)

    base_name = "pub_wide_lm_stack"
    base_z = to_rank_logit(models[base_name][0])
    print(f"Baseline {base_name}: AUC = {roc_auc_score(y, base_z):.6f}", flush=True)

    cand_names = []
    for k in kept_singles:
        if (
            k.startswith(("ours_", "busya_tabm", "busya_rmlp", "busya_ebm", "busya_lgb_g2_plaininit", "next2060_", "bizzi_", "weni_", "megayak_", "queyrusi_", "aman_sameer", "aman_tabpfn", "aman_gpt2", "aman_yekenot"))
            or models[k][3] >= 0.9508
        ):
            cand_names.append(k)

    Z_cand = {k: to_rank_logit(models[k][0]) for k in cand_names}
    selected = [base_name]
    cur_z = base_z.copy()
    cur_auc = roc_auc_score(y, cur_z)
    pool = list(cand_names)

    for step in range(15):
        best_k = None
        best_score = cur_auc
        best_next_z = None
        for k in pool:
            zk = Z_cand[k]
            for w in [0.02, 0.04, 0.07, 0.10, 0.14]:
                cand_z = (1.0 - w) * cur_z + w * zk
                auc = roc_auc_score(y, cand_z)
                if auc > best_score + 1e-6:
                    best_score = auc
                    best_k = (k, w)
                    best_next_z = cand_z
        if best_k is None:
            break
        k, w = best_k
        selected.append(k)
        pool.remove(k)
        cur_z = (best_next_z - best_next_z.mean()) / best_next_z.std()
        cur_auc = best_score
        print(f"  Step {step+1:2d}: +{k:32s} (w={w:.2f}) -> AUC={cur_auc:.6f}", flush=True)

    Z_sel = np.column_stack([base_z] + [Z_cand[k] for k in selected[1:]])
    print(f"\nStrict 2-Seed Outer Nested-CV on {len(selected)} selected anchors:", flush=True)
    for C in [0.005, 0.01, 0.02, 0.05, 0.1]:
        a2026, _ = eval_nested_logreg(Z_sel, y, C, seed=2026)
        a777, _ = eval_nested_logreg(Z_sel, y, C, seed=777)
        print(f"  [Selected LogReg C={C:<5}] Nested2026={a2026:.6f} | Nested777={a777:.6f}", flush=True)

    # Also test Non-negative Ridge on selected anchors
    skf2026 = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    skf777 = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    for alpha in [1000.0, 5000.0, 15000.0]:
        p26 = np.zeros(len(y))
        for tr, va in skf2026.split(Z_sel, y):
            p26[va] = Ridge(alpha=alpha, positive=True, max_iter=1000).fit(Z_sel[tr], y[tr]).predict(Z_sel[va])
        p77 = np.zeros(len(y))
        for tr, va in skf777.split(Z_sel, y):
            p77[va] = Ridge(alpha=alpha, positive=True, max_iter=1000).fit(Z_sel[tr], y[tr]).predict(Z_sel[va])
        print(f"  [Selected NonNeg Ridge alpha={alpha:<7.0f}] Nested2026={roc_auc_score(y, p26):.6f} | Nested777={roc_auc_score(y, p77):.6f}", flush=True)


if __name__ == "__main__":
    main()
