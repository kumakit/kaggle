"""
Direct ROC-AUC Nested Coordinate Ascent (Hill Climbing with replacement / fine step)
evaluated rigorously across TWO unseen outer CV splits (seed=2026 and seed=777).
"""
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"
CACHE_DIR = os.path.join(DATA_DIR, "cache_v2")


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast exact Mann-Whitney ROC-AUC using argsort (2-3x faster than sklearn check_array)."""
    order = np.argsort(y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    # Sum of 1-based ranks of positives (assuming no exact float ties after blending)
    pos_ranks = np.where(y_sorted == 1)[0] + 1
    return float((np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_auc_hill_climb(Xtr: np.ndarray, ytr: np.ndarray, steps: int = 40, base_init_steps: int = 12):
    """Start with base_init_steps of col 0 (WideLM), then greedily add 1 unit of best col per step."""
    d = Xtr.shape[1]
    counts = np.zeros(d, dtype=np.int32)
    counts[0] = base_init_steps
    cur_sum = Xtr[:, 0].astype(np.float64) * base_init_steps
    cur_auc = fast_auc(ytr, cur_sum)
    total_w = base_init_steps

    for _ in range(steps):
        best_j = -1
        best_auc = cur_auc
        for j in range(d):
            cand = cur_sum + Xtr[:, j]
            auc = fast_auc(ytr, cand)
            if auc > best_auc + 1e-7:
                best_auc = auc
                best_j = j
        if best_j < 0:
            break
        counts[best_j] += 1
        cur_sum += Xtr[:, best_j]
        total_w += 1
        cur_auc = best_auc

    return counts / counts.sum()


def main():
    t0 = time.time()
    y = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy().astype(int)
    with open(os.path.join(CACHE_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["names"]
    Z_oof = np.load(os.path.join(CACHE_DIR, "Z_oof.npy"))
    base_oof = np.load(os.path.join(CACHE_DIR, "base_oof.npy"))
    name_to_idx = {n: i for i, n in enumerate(names)}

    candidates = [
        "aman_sameer_glm10f",
        "megayak_realmlp_v2",
        "weni_realmlp_bpe_v2",
        "ours_exp023_glm_xgb",
        "next2060_glmboost_pair",
        "next2060_glmboost",
        "ours_exp021_glm_lgb",
        "ours_exp019_xgb_d2",
        "queyrusi_digit_cnn",
        "aman_tabpfn_slimypunk_tp1",
        "gbank_gbn_bag5_w512",
        "busya_tabm_dl",
        "weni_bpe_lgb_l4",
        "next2060_realmlp",
    ]
    valid = [k for k in candidates if k in name_to_idx]
    Z_sub = np.column_stack([base_oof] + [Z_oof[:, name_to_idx[k]] for k in valid]).astype(np.float64)
    labels = ["WideLM"] + valid

    print(f"Baseline WideLM AUC = {roc_auc_score(y, base_oof):.6f}", flush=True)

    for base_init in [8, 12, 20]:
        for seed in [2026, 777]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            oof_hc = np.zeros(len(y), dtype=np.float64)
            ws = []
            for tr, va in skf.split(Z_sub, y):
                w = fit_auc_hill_climb(Z_sub[tr], y[tr], steps=35, base_init_steps=base_init)
                ws.append(w)
                oof_hc[va] = Z_sub[va] @ w
            w_mean = np.mean(ws, axis=0)
            print(f"[AUC HillClimb init={base_init:2d} seed={seed}] Nested AUC = {roc_auc_score(y, oof_hc):.6f}", flush=True)
            if seed == 2026:
                w_str = ", ".join(f"{l}:{wi:.3f}" for l, wi in sorted(zip(labels, w_mean), key=lambda x: -x[1]) if wi >= 0.005)
                print(f"  Weights: {w_str}", flush=True)

    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
