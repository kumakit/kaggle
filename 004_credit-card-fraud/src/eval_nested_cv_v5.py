"""
Nested-CV v5: Multi-resolution Coordinate Ascent directly optimizing ROC-AUC
with non-negative simplex weights, plus Soft-Rank-Logit vs Standard-Rank-Logit comparison,
and Hybrid blend with L2-LogReg(C=0.05) across 2 unseen outer splits (2026 & 777).
"""
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"
CACHE_DIR = os.path.join(DATA_DIR, "cache_v2")


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    pos_ranks = np.where(y_sorted == 1)[0] + 1
    return float((np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def to_soft_rank_logit(arr: np.ndarray, k: int) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    eps = 0.5 / k
    z = np.log((r + eps) / (1.0 - r + eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float64)


def fit_coord_auc(Xtr: np.ndarray, ytr: np.ndarray, reg_l2: float = 0.0):
    """Multi-scale coordinate ascent directly optimizing ROC-AUC on simplex w >= 0, sum(w) = 1."""
    d = Xtr.shape[1]
    w = np.zeros(d, dtype=np.float64)
    w[0] = 1.0  # start from WideLM

    def score(wv):
        s = Xtr @ wv
        return fast_auc(ytr, s) - reg_l2 * np.sum(wv[1:] ** 2)

    cur_score = score(w)
    for step in [0.06, 0.03, 0.015, 0.005]:
        improved = True
        while improved:
            improved = False
            for j in range(1, d):
                for sign in [1.0, -1.0]:
                    cand = w.copy()
                    delta = step * sign
                    if cand[j] + delta < 0.0 or cand[0] - delta < 0.0:
                        continue
                    cand[j] += delta
                    cand[0] -= delta
                    sc = score(cand)
                    if sc > cur_score + 1e-7:
                        cur_score = sc
                        w = cand
                        improved = True
    return w / w.sum()


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
        "gbank_gbn_bag5_w512",
        "queyrusi_digit_cnn",
        "aman_tabpfn_slimypunk_tp1",
        "ours_exp021_glm_lgb",
        "busya_tabm_dl",
    ]
    valid = [k for k in candidates if k in name_to_idx]
    Z_std = np.column_stack([base_oof] + [Z_oof[:, name_to_idx[k]] for k in valid]).astype(np.float64)
    labels = ["WideLM"] + valid

    # Also build Soft Rank-Logit space (k = len(labels))
    Z_soft = np.column_stack([to_soft_rank_logit(Z_std[:, j], k=len(labels)) for j in range(Z_std.shape[1])])

    print(f"Baseline WideLM AUC = {roc_auc_score(y, base_oof):.6f}", flush=True)

    for space_name, Z_mat in [("StdRankLogit", Z_std), ("SoftRankLogit", Z_soft)]:
        print(f"\n=== {space_name} ({len(labels)} models) ===", flush=True)
        for reg_l2 in [0.0, 0.0005]:
            for seed in [2026, 777]:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                oof_ca = np.zeros(len(y), dtype=np.float64)
                oof_lr = np.zeros(len(y), dtype=np.float64)
                ws = []
                for tr, va in skf.split(Z_mat, y):
                    w = fit_coord_auc(Z_mat[tr], y[tr], reg_l2=reg_l2)
                    ws.append(w)
                    oof_ca[va] = Z_mat[va] @ w
                    if reg_l2 == 0.0:
                        clf = LogisticRegression(C=0.05, solver="newton-cholesky", max_iter=80).fit(Z_mat[tr], y[tr])
                        oof_lr[va] = clf.decision_function(Z_mat[va])

                auc_ca = roc_auc_score(y, oof_ca)
                msg = f"  [CoordAUC reg={reg_l2:<6} seed={seed}] Nested={auc_ca:.6f}"
                if reg_l2 == 0.0:
                    zca = (oof_ca - oof_ca.mean()) / oof_ca.std()
                    zlr = (oof_lr - oof_lr.mean()) / oof_lr.std()
                    auc_lr = roc_auc_score(y, zlr)
                    auc_mix = roc_auc_score(y, 0.65 * zca + 0.35 * zlr)
                    msg += f" | LogReg(0.05)={auc_lr:.6f} | Mix(65/35)={auc_mix:.6f}"
                print(msg, flush=True)
                if seed == 2026:
                    w_mean = np.mean(ws, axis=0)
                    w_str = ", ".join(f"{l}:{wi:.3f}" for l, wi in sorted(zip(labels, w_mean), key=lambda x: -x[1]) if wi >= 0.003)
                    print(f"    Weights: {w_str}", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
