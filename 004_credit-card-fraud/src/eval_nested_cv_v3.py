"""
Fast Nested-CV v3 using cached Z_oof/Z_te in data/cache_v2/:
1. Evaluates curated diverse anchor sets + WideLM stack under 2-Seed Outer Nested-CV (LogReg & NonNeg SLSQP).
2. Compares Standard Rank-Logit vs Soft Rank-Logit.
3. Also tests non-redundant Top-40 single models (without pub_wide_lm_stack) with LogReg.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"
CACHE_DIR = os.path.join(DATA_DIR, "cache_v2")


def to_soft_rank_logit(arr: np.ndarray, k: int) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    eps = 0.5 / k
    z = np.log((r + eps) / (1.0 - r + eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float32)


def eval_nested_logreg(Z: np.ndarray, y: np.ndarray, C: float, seed: int):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=np.float64)
    coefs = []
    for tr, va in skf.split(Z, y):
        clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=80, tol=1e-5)
        clf.fit(Z[tr], y[tr])
        pred[va] = clf.decision_function(Z[va])
        coefs.append(clf.coef_.ravel())
    return roc_auc_score(y, pred), pred, np.mean(coefs, axis=0)


def eval_nested_nonneg_blend(Z: np.ndarray, y: np.ndarray, seed: int):
    """Non-negative convex weights optimizing LogLoss on Rank-Logits (zero extrapolation)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=np.float64)
    d = Z.shape[1]
    ws = []
    # Fit on fast subsample (or full fold with analytic gradient) via logistic loss on standardized Z
    from scipy.special import expit
    for tr, va in skf.split(Z, y):
        Xtr, ytr = Z[tr].astype(np.float64), y[tr].astype(np.float64)
        # Fit slope a and intercept b via simple 1D logit scaling + convex combo
        # Optimize convex weights w >= 0, sum(w) = 1 minimizing binary cross-entropy of (a * (Xtr @ w) + b)
        def loss_and_grad(params):
            w_raw = params[:d]
            a = params[d]
            b = params[d + 1]
            # softmax parameterization avoids equality constraint headaches
            e = np.exp(w_raw - np.max(w_raw))
            w = e / e.sum()
            z = a * (Xtr @ w) + b
            p = expit(z)
            # binary logloss + tiny L2 on w_raw
            loss = -np.mean(ytr * np.log(p + 1e-12) + (1.0 - ytr) * np.log(1.0 - p + 1e-12)) + 1e-5 * np.sum(w_raw**2)
            dz = (p - ytr) / len(ytr)
            grad_w = a * (Xtr.T @ dz)
            # Jacobian of softmax: w * (g - sum(w*g))
            grad_w_raw = w * (grad_w - np.dot(w, grad_w)) + 2e-5 * w_raw
            grad_a = np.dot(dz, Xtr @ w)
            grad_b = np.sum(dz)
            return loss, np.concatenate([grad_w_raw, [grad_a, grad_b]])

        p0 = np.zeros(d + 2)
        p0[0] = 3.0  # start heavy on base WideLM
        p0[d] = 1.2
        p0[d + 1] = -4.5
        res = minimize(loss_and_grad, p0, jac=True, method="L-BFGS-B", options={"maxiter": 120})
        e = np.exp(res.x[:d] - np.max(res.x[:d]))
        w = e / e.sum()
        ws.append(w)
        pred[va] = Z[va] @ w
    return roc_auc_score(y, pred), pred, np.mean(ws, axis=0)


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

    print(f"Loaded cache in {time.time() - t0:.2f}s | Base WideLM AUC={roc_auc_score(y, base_oof):.6f}", flush=True)

    # Curated diverse post-Nov11 + ours anchors
    anchor_sets = {
        "Set2_Top2": ["aman_sameer_glm10f", "megayak_realmlp_v2"],
        "Set5_Core": [
            "aman_sameer_glm10f",
            "megayak_realmlp_v2",
            "weni_realmlp_bpe_v2",
            "ours_exp023_glm_xgb",
            "next2060_glmboost_pair",
        ],
        "Set9_Diverse": [
            "aman_sameer_glm10f",
            "megayak_realmlp_v2",
            "weni_realmlp_bpe_v2",
            "ours_exp023_glm_xgb",
            "next2060_glmboost_pair",
            "next2060_glmboost",
            "ours_exp021_glm_lgb",
            "queyrusi_digit_cnn",
            "aman_tabpfn_slimypunk_tp1",
        ],
        "Set13_FullFam": [
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
        ],
    }

    for sname, keys in anchor_sets.items():
        valid_keys = [k for k in keys if k in name_to_idx]
        Z_sub = np.column_stack([base_oof] + [Z_oof[:, name_to_idx[k]] for k in valid_keys])
        print(f"\n=== {sname} (WideLM + {len(valid_keys)} anchors) ===", flush=True)
        for C in [0.02, 0.05, 0.1, 0.3]:
            a2026, _, coef = eval_nested_logreg(Z_sub, y, C, seed=2026)
            a777, _, _ = eval_nested_logreg(Z_sub, y, C, seed=777)
            print(f"  LogReg C={C:<4}: Nested2026={a2026:.6f} | Nested777={a777:.6f}", flush=True)
        ann2026, _, w_nn = eval_nested_nonneg_blend(Z_sub, y, seed=2026)
        ann777, _, _ = eval_nested_nonneg_blend(Z_sub, y, seed=777)
        print(f"  NonNeg Convex: Nested2026={ann2026:.6f} | Nested777={ann777:.6f}", flush=True)
        w_str = ", ".join(f"{n}:{w:.3f}" for n, w in zip(["WideLM"] + valid_keys, w_nn) if w >= 0.005)
        print(f"    NonNeg weights: {w_str}", flush=True)

    # Also test Scratch Top-45 diverse single models (without WideLM) with LogReg
    top_idx = sorted(range(len(names)), key=lambda i: aucs[i], reverse=True)[:45]
    # Add diverse NN/CNN/TabPFN/Ours if not in top 45
    extra = [name_to_idx[k] for k in ["megayak_realmlp_v2", "weni_realmlp_bpe_v2", "queyrusi_digit_cnn", "aman_tabpfn_slimypunk_tp1", "busya_tabm_dl", "ours_exp019_xgb_d2", "ours_exp036_cat_d3"] if k in name_to_idx]
    comb_idx = sorted(set(top_idx + extra))
    Z_top = Z_oof[:, comb_idx]
    print(f"\n=== Scratch Top-Diverse ({len(comb_idx)} models, NO WideLM) ===", flush=True)
    for C in [0.02, 0.05, 0.1, 0.3]:
        a2026, _, _ = eval_nested_logreg(Z_top, y, C, seed=2026)
        a777, _, _ = eval_nested_logreg(Z_top, y, C, seed=777)
        print(f"  Scratch LogReg C={C:<4}: Nested2026={a2026:.6f} | Nested777={a777:.6f}", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
