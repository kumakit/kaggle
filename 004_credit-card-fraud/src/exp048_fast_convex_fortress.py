"""
Exp #048: Fast Pure Nested-CV Convex Fortress Ensemble (Zero Extrapolation & Zero LB Probing)
Directly loads the top orthogonal models (WideLM + Sameer10F + Exp046 XGB10 + Exp042 LGB10
+ Exp041/047 3-Seed RealMLP + Megayak RealMLP + Weni RealMLP BPE + DigitCNN + TabPFN)
and optimizes with analytic gradient L-BFGS-B + LogReg on 2-Seed Outer Nested-CV.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"
BANKS_DIR = os.path.join(DATA_DIR, "oof_banks")


def to_rank_logit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    z = logit(np.clip(r, eps, 1.0 - eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float64)


def fit_nonneg_logreg_analytic(Z: np.ndarray, y: np.ndarray, l2: float = 1e-4):
    """Fit non-negative linear combination + intercept via analytic gradient L-BFGS-B."""
    k = Z.shape[1]
    n = len(y)
    theta0 = np.zeros(k + 1, dtype=np.float64)
    theta0[0] = logit(y.mean())
    theta0[1] = 1.5
    theta0[2:] = 0.05

    def obj_grad(theta):
        b = theta[0]
        w = theta[1:]
        logits = b + Z @ w
        p = expit(logits)
        eps = 1e-7
        p_clip = np.clip(p, eps, 1.0 - eps)
        loss = -np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)) + 0.5 * l2 * np.sum(w ** 2)
        diff = (p - y) / n
        grad_b = np.sum(diff)
        grad_w = Z.T @ diff + l2 * w
        return loss, np.concatenate([[grad_b], grad_w])

    bounds = [(None, None)] + [(0.0, 10.0) for _ in range(k)]
    res = minimize(obj_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    return res.x[0], res.x[1:]


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=["id"])
    y = train["is_fraud"].to_numpy().astype(int)

    # Load curated elite orthogonal models directly
    z_am = np.load(f"{BANKS_DIR}/amanatar/rank1_q17_streams.npz", allow_pickle=False)
    q_cnn = np.load(f"{BANKS_DIR}/queyrusi_cnn/predictions.npz", allow_pickle=False)

    oof41 = pd.read_csv(f"{DATA_DIR}/exp041_out/oof_exp041_realmlp.csv")["oof_realmlp_exp041"].to_numpy()
    te41 = pd.read_csv(f"{DATA_DIR}/exp041_out/submission_exp041_refit.csv")["is_fraud"].to_numpy()
    oof47 = pd.read_csv(f"{DATA_DIR}/exp047_out/oof_exp047_realmlp.csv")["oof_realmlp_exp041"].to_numpy()
    te47 = pd.read_csv(f"{DATA_DIR}/exp047_out/submission_exp047_refit.csv")["is_fraud"].to_numpy()
    oof_rmlp3 = (rankdata(oof41) / len(y) + 2.0 * rankdata(oof47) / len(y)) / 3.0
    te_rmlp3 = (rankdata(te41) / len(test) + 2.0 * rankdata(te47) / len(test)) / 3.0

    oof42 = pd.read_csv(f"{DATA_DIR}/exp042_out/oof_exp042_gb10.csv")["oof_exp042_gb10"].to_numpy()
    te42 = pd.read_csv(f"{DATA_DIR}/exp042_out/submission_exp042_refit.csv")["is_fraud"].to_numpy()
    oof46 = pd.read_csv(f"{DATA_DIR}/exp046_out/oof_exp046_xgb10.csv")["oof_exp046_xgb10"].to_numpy()
    te46 = pd.read_csv(f"{DATA_DIR}/exp046_out/submission_exp046_refit.csv")["is_fraud"].to_numpy()

    items = [
        ("pub_wide_lm_stack", np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_nested.npy"),
         np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_test.npy")),
        ("aman_sameer_glm10f", z_am["oof_sameer_glm10f"], z_am["test_sameer_glm10f"]),
        ("ours_exp042_lgb10_refit", oof42, te42),
        ("ours_exp046_xgb10_refit", oof46, te46),
        ("megayak_realmlp_v2", np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_oof.npy"),
         np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_test.npy")),
        ("ours_realmlp_3seed_refit", oof_rmlp3, te_rmlp3),
        ("weni_realmlp_bpe_v2", np.load(f"{BANKS_DIR}/wenimohamed/oof_weni_realmlp_bpe_v2.npy"),
         np.load(f"{BANKS_DIR}/wenimohamed/test_weni_realmlp_bpe_v2.npy")),
        ("queyrusi_digit_cnn", q_cnn["oof"], q_cnn["test"]),
        ("aman_tabpfn_tp1", z_am["oof_tabpfn_slimypunk_tp1"], z_am["test_tabpfn_slimypunk_tp1"]),
    ]

    names = [x[0] for x in items]
    Z_oof = np.column_stack([to_rank_logit(x[1]) for x in items])
    Z_te = np.column_stack([to_rank_logit(x[2]) for x in items])

    print(f"Baseline pub_wide_lm_stack = {roc_auc_score(y, Z_oof[:, 0]):.6f}", flush=True)

    # 1. Sweep Non-negative Convex Blend (analytic gradient L-BFGS-B) across 2 seeds
    best_l2, best_nonneg_auc = None, -1.0
    print("\n--- Non-Negative Convex Blend (Zero Negative Weights, 2-Seed Outer Nested-CV) ---", flush=True)
    for l2 in [1e-5, 5e-5, 2e-4, 1e-3, 5e-3]:
        aucs = []
        for seed in [2026, 777]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            p = np.zeros(len(y))
            for tr, va in skf.split(Z_oof, y):
                b, w = fit_nonneg_logreg_analytic(Z_oof[tr], y[tr], l2=l2)
                p[va] = b + Z_oof[va] @ w
            aucs.append(roc_auc_score(y, p))
        m_auc = float(np.mean(aucs))
        print(f"  NonNeg l2={l2:<7.5f}: Nested2026={aucs[0]:.6f} | Nested777={aucs[1]:.6f} | Mean={m_auc:.6f}", flush=True)
        if m_auc > best_nonneg_auc:
            best_nonneg_auc = m_auc
            best_l2 = l2

    # 2. Compare with Unconstrained LogReg C=0.05 on same 9 positive anchors
    print("\n--- Unconstrained LogReg on 9 Positive Anchors (2-Seed Outer Nested-CV) ---", flush=True)
    best_c, best_lr_auc = None, -1.0
    for C in [0.02, 0.05, 0.1, 0.2]:
        aucs = []
        for seed in [2026, 777]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            p = np.zeros(len(y))
            for tr, va in skf.split(Z_oof, y):
                clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=100, tol=1e-6).fit(Z_oof[tr], y[tr])
                p[va] = clf.decision_function(Z_oof[va])
            aucs.append(roc_auc_score(y, p))
        m_auc = float(np.mean(aucs))
        print(f"  LogReg C={C:<5}: Nested2026={aucs[0]:.6f} | Nested777={aucs[1]:.6f} | Mean={m_auc:.6f}", flush=True)
        if m_auc > best_lr_auc:
            best_lr_auc = m_auc
            best_c = C

    # Build final 2-seed bagged predictions using the best method
    use_nonneg = best_nonneg_auc >= best_lr_auc
    oof_final = np.zeros(len(y))
    te_final = np.zeros(len(test))
    ws = []

    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_s = np.zeros(len(y))
        te_s = np.zeros(len(test))
        for tr, va in skf.split(Z_oof, y):
            if use_nonneg:
                b, w = fit_nonneg_logreg_analytic(Z_oof[tr], y[tr], l2=best_l2)
                oof_s[va] = b + Z_oof[va] @ w
                te_s += (b + Z_te @ w) / 5.0
                ws.append(w / w.sum())
            else:
                clf = LogisticRegression(C=best_c, solver="newton-cholesky", max_iter=100, tol=1e-6).fit(Z_oof[tr], y[tr])
                oof_s[va] = clf.decision_function(Z_oof[va])
                te_s += clf.decision_function(Z_te) / 5.0
                ws.append(clf.coef_[0])
        oof_final += to_rank_logit(oof_s) / 2.0
        te_final += to_rank_logit(te_s) / 2.0

    final_auc = roc_auc_score(y, oof_final)
    method_tag = f"NonNeg-Convex(l2={best_l2})" if use_nonneg else f"LogReg(C={best_c})"
    print(f"\n=== Final Exp #048 Fortress ({method_tag}) ===", flush=True)
    print(f"  2-Seed Outer Nested-CV AUC = {final_auc:.6f}", flush=True)
    mean_w = np.mean(ws, axis=0)
    for n, w in sorted(zip(names, mean_w), key=lambda x: -abs(x[1])):
        print(f"  {n:28s}: {w:.4f}", flush=True)

    p_final = expit(te_final)
    sub = pd.DataFrame({"id": test["id"], "is_fraud": p_final})
    out_sub = os.path.join(DATA_DIR, "submission_exp048.csv")
    out_oof = os.path.join(DATA_DIR, "oof_exp048.npy")
    sub.to_csv(out_sub, index=False)
    np.save(out_oof, expit(oof_final).astype(np.float32))
    assert sub["is_fraud"].isnull().sum() == 0
    assert len(sub) == 299984
    print(f"Saved: {out_sub} & {out_oof}", flush=True)


if __name__ == "__main__":
    main()
