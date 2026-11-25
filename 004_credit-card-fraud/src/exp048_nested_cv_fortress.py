"""
Exp #048: Pure Nested-CV Fortress Ensemble (Zero LB probing)
Integrates:
  - Exp #046: 10-Fold GLM-Boosted XGBoost Depth 2 + Full-Data Refit (2-Seeds)
  - Exp #042: 10-Fold GLM-Boosted LightGBM 4-leaf + Full-Data Refit (2-Seeds)
  - Exp #041 + #047: 3-Seed Digit-TE RealMLP + Full-Data Refit (Seeds 42, 2026, 777)
  - Exp #023: 5-Fold GLM-Boosted XGBoost Depth 2
  - Clean diverse public anchors (WideLM Stack, Sameer10F, Megayak RealMLP, Weni BPE RealMLP, TabM, EBM, DigitCNN, TabPFN)

Compares Standard Rank-Logit LogReg vs Soft Rank-Logit LogReg vs Non-Negative Convex Blend
on TWO independent Outer Nested-CV splits (seed=2026 & seed=777).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append("004_credit-card-fraud/src")
from eval_nested_cv_base import load_all_clean_models, to_rank_logit
from eval_nested_cv_v2 import load_busyaprime_models

DATA_DIR = "004_credit-card-fraud/data"


def to_soft_rank_logit(arr: np.ndarray, k: int = 12) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    pad = 0.5 / k
    z = np.log((r + pad) / (1.0 - r + pad))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float32)


def fit_nonneg_convex_logloss(Z: np.ndarray, y: np.ndarray, l2: float = 1e-4):
    # Initialize heavily on strongest column (col 0)
    k = Z.shape[1]
    w0 = np.zeros(k)
    w0[0] = 0.8
    w0[1:] = 0.2 / max(1, k - 1)

    def obj(w):
        z = Z @ w
        p = expit(z)
        # Binary cross entropy + tiny L2
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)
        bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return bce + l2 * np.sum(w ** 2)

    bounds = [(0.0, 1.0) for _ in range(k)]
    res = minimize(obj, w0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 150})
    return res.x


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=["id"])
    y = train["is_fraud"].to_numpy().astype(int)

    models = load_all_clean_models(y)
    models = load_busyaprime_models(models, y)

    # Load Exp #046 (10-Fold GLM-XGBoost)
    oof46 = pd.read_csv(f"{DATA_DIR}/exp046_out/oof_exp046_xgb10.csv")["oof_exp046_xgb10"].to_numpy()
    te46_refit = pd.read_csv(f"{DATA_DIR}/exp046_out/submission_exp046_refit.csv")["is_fraud"].to_numpy()
    models["ours_exp046_xgb10_refit"] = (oof46.astype(np.float32), te46_refit.astype(np.float32), "ours_gbdt", roc_auc_score(y, oof46))

    # Load Exp #042 (10-Fold GLM-LightGBM)
    oof42 = pd.read_csv(f"{DATA_DIR}/exp042_out/oof_exp042_gb10.csv")["oof_exp042_gb10"].to_numpy()
    te42_refit = pd.read_csv(f"{DATA_DIR}/exp042_out/submission_exp042_refit.csv")["is_fraud"].to_numpy()
    models["ours_exp042_gb10_refit"] = (oof42.astype(np.float32), te42_refit.astype(np.float32), "ours_gbdt", roc_auc_score(y, oof42))

    # Load Exp #041 (seed 42) + Exp #047 (seeds 2026, 777) -> 3-Seed RealMLP Ensemble!
    oof41 = pd.read_csv(f"{DATA_DIR}/exp041_out/oof_exp041_realmlp.csv")["oof_realmlp_exp041"].to_numpy()
    te41 = pd.read_csv(f"{DATA_DIR}/exp041_out/submission_exp041_refit.csv")["is_fraud"].to_numpy()
    oof47 = pd.read_csv(f"{DATA_DIR}/exp047_out/oof_exp047_realmlp.csv")["oof_realmlp_exp041"].to_numpy()
    te47 = pd.read_csv(f"{DATA_DIR}/exp047_out/submission_exp047_refit.csv")["is_fraud"].to_numpy()

    oof_rmlp3 = (rankdata(oof41) / len(y) + 2.0 * rankdata(oof47) / len(y)) / 3.0
    te_rmlp3 = (rankdata(te41) / len(test) + 2.0 * rankdata(te47) / len(test)) / 3.0
    models["ours_realmlp_3seed_refit"] = (oof_rmlp3.astype(np.float32), te_rmlp3.astype(np.float32), "ours_nn", roc_auc_score(y, oof_rmlp3))

    print("=== New Self-Trained Models OOF AUC ===")
    print(f"  Exp #041 RealMLP (1-seed):   {roc_auc_score(y, oof41):.6f}")
    print(f"  Exp #047 RealMLP (2-seeds):  {roc_auc_score(y, oof47):.6f}")
    print(f"  Exp #041+047 RealMLP (3-sd): {models['ours_realmlp_3seed_refit'][3]:.6f}")
    print(f"  Exp #042 GB-10F LGBM (2-sd): {models['ours_exp042_gb10_refit'][3]:.6f}")
    print(f"  Exp #046 GB-10F XGB  (2-sd): {models['ours_exp046_xgb10_refit'][3]:.6f}")

    anchors = [
        "pub_wide_lm_stack",         # 0.952565
        "aman_sameer_glm10f",        # 0.951595
        "ours_exp046_xgb10_refit",   # NEW 10-Fold GLM-XGB
        "ours_exp042_gb10_refit",    # 10-Fold GLM-LGBM
        "ours_realmlp_3seed_refit",  # NEW 3-Seed RealMLP
        "megayak_realmlp_v2",        # 0.950168
        "weni_realmlp_bpe_v2",       # 0.950005
        "ours_exp023_glm_xgb",       # 0.951165
        "gbank_gbn_bag5_w512",       # 0.951223
        "busya_ebm_g2_glminit",      # 0.950591
        "busya_tabm_dl",             # 0.949703
        "queyrusi_digit_cnn",        # 0.949738
        "aman_tabpfn_slimypunk_tp1", # 0.949294
    ]
    anchors = [k for k in anchors if k in models]

    Z_oof = np.column_stack([to_rank_logit(models[k][0]) for k in anchors])
    Z_te = np.column_stack([to_rank_logit(models[k][1]) for k in anchors])

    Z_soft_oof = np.column_stack([to_soft_rank_logit(models[k][0], k=len(anchors)) for k in anchors])
    Z_soft_te = np.column_stack([to_soft_rank_logit(models[k][1], k=len(anchors)) for k in anchors])

    base_auc = roc_auc_score(y, Z_oof[:, 0])
    print(f"\nBaseline pub_wide_lm_stack = {base_auc:.6f}")
    for idx in [1, 2, 3, 4]:
        corr = float(np.dot(Z_oof[:, 0], Z_oof[:, idx]) / len(y))
        g = max(roc_auc_score(y, (1 - w) * Z_oof[:, 0] + w * Z_oof[:, idx]) for w in [0.02, 0.04, 0.08, 0.12, 0.16]) - base_auc
        print(f"  1D vs {anchors[idx]:26s}: corr={corr:.4f}, solo={models[anchors[idx]][3]:.6f}, gain={g*1e4:+.3f} pts")

    # Evaluate Standard Rank-Logit LogReg across 2 seeds
    print("\n--- Method 1: Standard Rank-Logit LogReg (2-Seed Nested-CV) ---")
    best_std = (None, -1.0)
    for C in [0.03, 0.06, 0.08, 0.12, 0.2]:
        aucs = []
        for seed in [2026, 777]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            p = np.zeros(len(y))
            for tr, va in skf.split(Z_oof, y):
                clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=100, tol=1e-6).fit(Z_oof[tr], y[tr])
                p[va] = clf.decision_function(Z_oof[va])
            aucs.append(roc_auc_score(y, p))
        m_auc = float(np.mean(aucs))
        print(f"  C={C:<4}: Nested2026={aucs[0]:.6f} | Nested777={aucs[1]:.6f} | Mean={m_auc:.6f}")
        if m_auc > best_std[1]:
            best_std = (C, m_auc)

    # Evaluate Soft Rank-Logit LogReg across 2 seeds
    print("\n--- Method 2: Soft Rank-Logit LogReg (2-Seed Nested-CV) ---")
    best_soft = (None, -1.0)
    for C in [0.03, 0.06, 0.08, 0.12, 0.2]:
        aucs = []
        for seed in [2026, 777]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            p = np.zeros(len(y))
            for tr, va in skf.split(Z_soft_oof, y):
                clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=100, tol=1e-6).fit(Z_soft_oof[tr], y[tr])
                p[va] = clf.decision_function(Z_soft_oof[va])
            aucs.append(roc_auc_score(y, p))
        m_auc = float(np.mean(aucs))
        print(f"  C={C:<4}: Nested2026={aucs[0]:.6f} | Nested777={aucs[1]:.6f} | Mean={m_auc:.6f}")
        if m_auc > best_soft[1]:
            best_soft = (C, m_auc)

    # Method 3: Non-negative Convex Blend on top 5 positive anchors (zero negative weights!)
    pos_idx = [0, 1, 2, 3, 4, 5, 6, 11, 12]
    Z_pos_oof = Z_oof[:, pos_idx]
    Z_pos_te = Z_te[:, pos_idx]
    pos_names = [anchors[i] for i in pos_idx]
    print("\n--- Method 3: Non-Negative Convex Blend (Zero Extrapolation, 2-Seed Nested-CV) ---")
    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p = np.zeros(len(y))
        for tr, va in skf.split(Z_pos_oof, y):
            w = fit_nonneg_convex_logloss(Z_pos_oof[tr], y[tr], l2=1e-4)
            p[va] = Z_pos_oof[va] @ w
        print(f"  NonNeg Convex (seed={seed}): AUC={roc_auc_score(y, p):.6f}")

    # Build final OOF & Test predictions using best Standard LogReg + best Soft LogReg + NonNeg Convex Blend
    skf26 = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_m1, te_m1 = np.zeros(len(y)), np.zeros(len(test))
    oof_m2, te_m2 = np.zeros(len(y)), np.zeros(len(test))
    oof_m3, te_m3 = np.zeros(len(y)), np.zeros(len(test))
    coefs_m1 = []
    weights_m3 = []

    for tr, va in skf26.split(Z_oof, y):
        c1 = LogisticRegression(C=best_std[0], solver="newton-cholesky", max_iter=100, tol=1e-6).fit(Z_oof[tr], y[tr])
        oof_m1[va] = c1.decision_function(Z_oof[va])
        te_m1 += c1.decision_function(Z_te) / 5.0
        coefs_m1.append(c1.coef_[0])

        c2 = LogisticRegression(C=best_soft[0], solver="newton-cholesky", max_iter=100, tol=1e-6).fit(Z_soft_oof[tr], y[tr])
        oof_m2[va] = c2.decision_function(Z_soft_oof[va])
        te_m2 += c2.decision_function(Z_soft_te) / 5.0

        w3 = fit_nonneg_convex_logloss(Z_pos_oof[tr], y[tr], l2=1e-4)
        oof_m3[va] = Z_pos_oof[va] @ w3
        te_m3 += (Z_pos_te @ w3) / 5.0
        weights_m3.append(w3)

    print("\n--- Final Meta-Ensemble Comparison (Nested2026) ---")
    print(f"  M1 Std-Logit LogReg (C={best_std[0]}):   {roc_auc_score(y, oof_m1):.6f}")
    print(f"  M2 Soft-Logit LogReg (C={best_soft[0]}): {roc_auc_score(y, oof_m2):.6f}")
    print(f"  M3 NonNeg Convex:                        {roc_auc_score(y, oof_m3):.6f}")

    # Blend M1 + M2 + M3 (meta-stabilization)
    for w3_share in [0.0, 0.2, 0.35, 0.5]:
        w12 = (1.0 - w3_share) / 2.0
        oof_blend = w12 * to_rank_logit(oof_m1) + w12 * to_rank_logit(oof_m2) + w3_share * to_rank_logit(oof_m3)
        print(f"  MetaBlend (M1+M2 {(1-w3_share):.2f} + M3 {w3_share:.2f}): {roc_auc_score(y, oof_blend):.6f}")

    # Choose optimal stable blend: 40% M1 + 40% M2 + 20% M3 (or best of M1/M2/Meta)
    # Use 0.40 * M1 + 0.40 * M2 + 0.20 * M3 if >= both, else 0.5 * M1 + 0.5 * M2
    oof_50_50 = 0.5 * to_rank_logit(oof_m1) + 0.5 * to_rank_logit(oof_m2)
    oof_40_40_20 = 0.4 * to_rank_logit(oof_m1) + 0.4 * to_rank_logit(oof_m2) + 0.2 * to_rank_logit(oof_m3)
    if roc_auc_score(y, oof_40_40_20) >= roc_auc_score(y, oof_50_50):
        oof_final = oof_40_40_20
        te_final = 0.4 * to_rank_logit(te_m1) + 0.4 * to_rank_logit(te_m2) + 0.2 * to_rank_logit(te_m3)
        tag = "40% StdLogit + 40% SoftLogit + 20% NonNegConvex"
    else:
        oof_final = oof_50_50
        te_final = 0.5 * to_rank_logit(te_m1) + 0.5 * to_rank_logit(te_m2)
        tag = "50% StdLogit + 50% SoftLogit"

    final_auc = roc_auc_score(y, oof_final)
    print(f"\nSelected Fortress ({tag}): Nested-CV AUC = {final_auc:.6f}")

    print("\nMean M1 Coefficients:")
    for n, w in sorted(zip(anchors, np.mean(coefs_m1, axis=0)), key=lambda x: -abs(x[1])):
        print(f"  {n:28s}: {w:+.4f}")

    print("\nMean M3 NonNeg Weights:")
    mw3 = np.mean(weights_m3, axis=0)
    mw3 = mw3 / mw3.sum()
    for n, w in sorted(zip(pos_names, mw3), key=lambda x: -x[1]):
        print(f"  {n:28s}: {w:.4f}")

    p_final = expit(te_final)
    sub = pd.DataFrame({"id": test["id"], "is_fraud": p_final})
    out_sub = os.path.join(DATA_DIR, "submission_exp048.csv")
    out_oof = os.path.join(DATA_DIR, "oof_exp048.npy")
    sub.to_csv(out_sub, index=False)
    np.save(out_oof, expit(oof_final).astype(np.float32))
    assert sub["is_fraud"].isnull().sum() == 0
    assert len(sub) == 299984
    print(f"\nSaved {out_sub} & {out_oof}")


if __name__ == "__main__":
    main()
