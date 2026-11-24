"""
Fast Nested-CV v4: True Non-Negative Simplex Optimization (SLSQP without softmax bias)
and Nested Coordinate Ascent (0.005 step) over WideLM + diverse post-Nov11 & our own models.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"
CACHE_DIR = os.path.join(DATA_DIR, "cache_v2")


def fit_simplex_logreg(Xtr: np.ndarray, ytr: np.ndarray):
    """Exact non-negative weights w >= 0, sum(w) = 1 minimizing binary logloss of a*(Xtr@w)+b."""
    d = Xtr.shape[1]

    def obj(params):
        w = params[:d]
        a = params[d]
        b = params[d + 1]
        z = a * (Xtr @ w) + b
        p = expit(z)
        loss = -np.mean(ytr * np.log(p + 1e-12) + (1.0 - ytr) * np.log(1.0 - p + 1e-12))
        dz = (p - ytr) / len(ytr)
        gw = a * (Xtr.T @ dz)
        ga = np.dot(dz, Xtr @ w)
        gb = np.sum(dz)
        return loss, np.concatenate([gw, [ga, gb]])

    p0 = np.zeros(d + 2)
    p0[0] = 0.85
    if d > 1:
        p0[1:] = 0.15 / (d - 1)
    p0[d] = 1.2
    p0[d + 1] = -4.5

    bounds = [(0.0, 1.0)] * d + [(0.1, 10.0), (-15.0, 0.0)]
    cons = {"type": "eq", "fun": lambda p: np.sum(p[:d]) - 1.0, "jac": lambda p: np.array([1.0] * d + [0.0, 0.0])}
    res = minimize(obj, p0, jac=True, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 100, "ftol": 1e-8})
    w = np.clip(res.x[:d], 0.0, 1.0)
    return w / w.sum()


def main():
    t0 = time.time()
    y = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy().astype(int)
    with open(os.path.join(CACHE_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["names"]
    aucs = meta["aucs"]
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
        "weni_bpe_cat_d5",
        "next2060_realmlp",
        "bizzi_lgb3_s1",
    ]
    valid = [k for k in candidates if k in name_to_idx]
    Z_sub = np.column_stack([base_oof] + [Z_oof[:, name_to_idx[k]] for k in valid]).astype(np.float64)
    labels = ["WideLM"] + valid

    print(f"Baseline WideLM AUC = {roc_auc_score(y, base_oof):.6f}", flush=True)

    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_slsqp = np.zeros(len(y), dtype=np.float64)
        ws = []
        for tr, va in skf.split(Z_sub, y):
            w = fit_simplex_logreg(Z_sub[tr], y[tr].astype(np.float64))
            ws.append(w)
            oof_slsqp[va] = Z_sub[va] @ w
        w_mean = np.mean(ws, axis=0)
        print(f"\n[Nested SLSQP Simplex seed={seed}] AUC = {roc_auc_score(y, oof_slsqp):.6f}", flush=True)
        for lbl, wi in sorted(zip(labels, w_mean), key=lambda x: -x[1]):
            if wi >= 0.002:
                print(f"  {lbl:32s}: {wi:.4f}", flush=True)

    # Also test Non-negative L2 LogReg vs pure LogReg C=0.05 + Simplex blend (hybrid ensemble)
    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p_sim = np.zeros(len(y), dtype=np.float64)
        p_lr = np.zeros(len(y), dtype=np.float64)
        for tr, va in skf.split(Z_sub, y):
            w = fit_simplex_logreg(Z_sub[tr], y[tr].astype(np.float64))
            p_sim[va] = Z_sub[va] @ w
            clf = LogisticRegression(C=0.05, solver="newton-cholesky", max_iter=80).fit(Z_sub[tr], y[tr])
            p_lr[va] = clf.decision_function(Z_sub[va])
        # Standardized mix
        zs = (p_sim - p_sim.mean()) / p_sim.std()
        zl = (p_lr - p_lr.mean()) / p_lr.std()
        print(
            f"[Seed {seed}] Simplex={roc_auc_score(y, zs):.6f} | LogReg(0.05)={roc_auc_score(y, zl):.6f} "
            f"| 50/50={roc_auc_score(y, 0.5*zs+0.5*zl):.6f} | 70/30={roc_auc_score(y, 0.7*zs+0.3*zl):.6f}",
            flush=True,
        )

    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
