"""
Exp #044: Pure Nested-CV Grand Robust Ensemble
Integrates:
  - WideLM Stack (Nested-CV 201 models)
  - Exp #042 (Our own 10-Fold GLM-Boosted LightGBM + 7 Joint Pairs + Full-Data Refit 2-Seed)
  - Sameer 10-Fold GLM-Boost (external 10-Fold anchor)
  - Megayak RealMLP v2 (8-Seed 2-epoch)
  - Megayak RealMLP 4-epoch (mlp4, ultra-orthogonal deep net)
  - Exp #023 (Our own GLM-Boosted XGBoost Depth 2)
Verified rigorously across TWO unseen outer CV splits (seed=2026 and seed=777).
"""
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"
CACHE_DIR = os.path.join(DATA_DIR, "cache_v2")


def to_std_rank_logit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    z = logit(np.clip(r, eps, 1.0 - eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float64)


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score, kind="mergesort")
    y_sorted = y_true[order]
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    pos_ranks = np.where(y_sorted == 1)[0] + 1
    return float((np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_coord_auc(Xtr: np.ndarray, ytr: np.ndarray):
    d = Xtr.shape[1]
    w = np.zeros(d, dtype=np.float64)
    w[0] = 1.0
    cur_score = fast_auc(ytr, Xtr @ w)
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
                    sc = fast_auc(ytr, Xtr @ cand)
                    if sc > cur_score + 1e-7:
                        cur_score = sc
                        w = cand
                        improved = True
    return w / w.sum()


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=["id"])
    y = train["is_fraud"].to_numpy().astype(int)

    # Load Exp #042
    df_42_oof = pd.read_csv(f"{DATA_DIR}/exp042/oof_exp042_gb10.csv")
    df_42_ref = pd.read_csv(f"{DATA_DIR}/exp042/submission_exp042_refit.csv")
    assert (df_42_oof["id"].values == train["id"].values).all()
    assert (df_42_ref["id"].values == test["id"].values).all()

    auc_42 = roc_auc_score(y, df_42_oof["oof_exp042_gb10"])
    print(f"[Exp #042 Our 10-Fold GLM-Boost] OOF AUC = {auc_42:.6f}", flush=True)

    # Load cache components
    with open(os.path.join(CACHE_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    name_to_idx = {n: i for i, n in enumerate(meta["names"])}
    Z_oof = np.load(os.path.join(CACHE_DIR, "Z_oof.npy"))
    Z_te = np.load(os.path.join(CACHE_DIR, "Z_te.npy"))
    base_oof = np.load(os.path.join(CACHE_DIR, "base_oof.npy")).astype(np.float64)
    base_te = np.load(os.path.join(CACHE_DIR, "base_te.npy")).astype(np.float64)

    z42_oof = to_std_rank_logit(df_42_oof["oof_exp042_gb10"].to_numpy())
    z42_te = to_std_rank_logit(df_42_ref["is_fraud"].to_numpy())

    z_mlp4_oof = to_std_rank_logit(np.load(f"{DATA_DIR}/oof_banks/megayak_realmlp/mlp4_oof.npy"))
    z_mlp4_te = to_std_rank_logit(np.load(f"{DATA_DIR}/oof_banks/megayak_realmlp/mlp4_test.npy"))

    def get_pair(k):
        idx = name_to_idx[k]
        return Z_oof[:, idx].astype(np.float64), Z_te[:, idx].astype(np.float64)

    z_sam_oof, z_sam_te = get_pair("aman_sameer_glm10f")
    z_rmlp_oof, z_rmlp_te = get_pair("megayak_realmlp_v2")
    z_weni_oof, z_weni_te = get_pair("weni_realmlp_bpe_v2")
    z_23_oof, z_23_te = get_pair("ours_exp023_glm_xgb")

    print(f"Corr(Exp042, WideLM)      = {np.dot(z42_oof, base_oof)/len(y):.6f}", flush=True)
    print(f"Corr(Exp042, Sameer10F)   = {np.dot(z42_oof, z_sam_oof)/len(y):.6f}", flush=True)
    print(f"Corr(Exp042, Exp023_XGB)  = {np.dot(z42_oof, z_23_oof)/len(y):.6f}", flush=True)

    # Assemble candidate matrix
    labels = ["WideLM", "Exp042_Our10F", "Sameer10F", "Megayak_RealMLP2", "Megayak_RealMLP4", "Weni_RealMLP_BPE", "Exp023_OurXGB"]
    X_oof = np.column_stack([base_oof, z42_oof, z_sam_oof, z_rmlp_oof, z_mlp4_oof, z_weni_oof, z_23_oof])
    X_te = np.column_stack([base_te, z42_te, z_sam_te, z_rmlp_te, z_mlp4_te, z_weni_te, z_23_te])

    # Optimize inside TWO unseen outer 5-fold CV splits
    w_folds = []
    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        nested = np.zeros(len(y), dtype=np.float64)
        ws = []
        for tr, va in skf.split(X_oof, y):
            w = fit_coord_auc(X_oof[tr], y[tr])
            ws.append(w)
            nested[va] = X_oof[va] @ w
        w_mean = np.mean(ws, axis=0)
        w_folds.append(w_mean)
        print(f"\n[Nested CoordAUC seed={seed}] Nested-CV AUC = {roc_auc_score(y, nested):.6f}", flush=True)
        for lbl, wi in zip(labels, w_mean):
            print(f"  {lbl:20s}: {wi:.4f}", flush=True)

    # Final averaged non-negative weights across all 10 nested folds (robust, zero leakage)
    w_final = np.mean(w_folds, axis=0)
    w_final = w_final / w_final.sum()
    print("\n=== Final Averaged Nested Weights ===", flush=True)
    for lbl, wi in zip(labels, w_final):
        print(f"  {lbl:20s}: {wi:.4f}", flush=True)

    oof_final = X_oof @ w_final
    te_final = X_te @ w_final
    final_auc = roc_auc_score(y, oof_final)
    print(f"\n[Exp #044 Final] Full OOF AUC = {final_auc:.6f}", flush=True)

    # Compare vs Exp043
    oof43 = np.load(f"{DATA_DIR}/oof_exp043.npy").astype(np.float64)
    print(f"Gain vs Exp043 (0.952603): {final_auc - roc_auc_score(y, oof43):+.6f}", flush=True)

    # Check fold wins vs WideLM & vs Exp043
    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        wins_w = sum(roc_auc_score(y[va], oof_final[va]) > roc_auc_score(y[va], base_oof[va]) for _, va in skf.split(X_oof, y))
        wins_43 = sum(roc_auc_score(y[va], oof_final[va]) > roc_auc_score(y[va], oof43[va]) for _, va in skf.split(X_oof, y))
        print(f"Seed {seed}: wins {wins_w}/5 vs WideLM, {wins_43}/5 vs Exp043", flush=True)

    p_te = expit(te_final)
    sub = pd.DataFrame({"id": test["id"], "is_fraud": p_te.astype(np.float64)})
    out_csv = os.path.join(DATA_DIR, "submission_exp044.csv")
    out_oof = os.path.join(DATA_DIR, "oof_exp044.npy")
    sub.to_csv(out_csv, index=False)
    np.save(out_oof, oof_final.astype(np.float32))
    print(f"Saved {out_csv} (rows={len(sub)}, nulls={sub['is_fraud'].isna().sum()}) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
