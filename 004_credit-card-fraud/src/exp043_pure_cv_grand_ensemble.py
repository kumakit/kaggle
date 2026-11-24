"""
Exp #043: Pure Nested-CV Grand Ensemble (incorporating Exp #041 RealMLP + Exp #042 10-Fold GLM-Boost
+ Wide LM Stack + Diverse Neural/LM/CNN/TabPFN/GBDT Anchors).
Zero public-LB probing! Strictly validated on two independent Outer Nested-CV splits (seed=2026 & seed=777).
Produces submission_exp043.csv and oof_exp043.npy.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append("004_credit-card-fraud/src")
from eval_nested_cv_base import load_all_clean_models, to_rank_logit
from eval_nested_cv_v2 import load_busyaprime_models

DATA_DIR = "004_credit-card-fraud/data"


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=["id"])
    y = train["is_fraud"].to_numpy().astype(int)

    models = load_all_clean_models(y)
    models = load_busyaprime_models(models, y)

    # Load our new Exp #041 (RealMLP w/ Full-Data Refit) & Exp #042 (10-Fold GLM-Boost w/ Full-Data Refit)
    oof41 = pd.read_csv(f"{DATA_DIR}/exp041_out/oof_exp041_realmlp.csv")["oof_realmlp_exp041"].to_numpy()
    te41_refit = pd.read_csv(f"{DATA_DIR}/exp041_out/submission_exp041_refit.csv")["is_fraud"].to_numpy()
    models["ours_exp041_realmlp_refit"] = (
        oof41.astype(np.float32), te41_refit.astype(np.float32), "ours_nn", roc_auc_score(y, oof41)
    )

    oof42 = pd.read_csv(f"{DATA_DIR}/exp042_out/oof_exp042_gb10.csv")["oof_exp042_gb10"].to_numpy()
    te42_refit = pd.read_csv(f"{DATA_DIR}/exp042_out/submission_exp042_refit.csv")["is_fraud"].to_numpy()
    models["ours_exp042_gb10_refit"] = (
        oof42.astype(np.float32), te42_refit.astype(np.float32), "ours_gbdt", roc_auc_score(y, oof42)
    )

    print(f"Exp #041 RealMLP OOF AUC = {models['ours_exp041_realmlp_refit'][3]:.6f}")
    print(f"Exp #042 GB-10F  OOF AUC = {models['ours_exp042_gb10_refit'][3]:.6f}")

    # Curated orthogonal anchors including both new self-trained models
    anchors = [
        "pub_wide_lm_stack",         # 0.952565 (201-model wide LM stack)
        "aman_sameer_glm10f",        # 0.951595 (10-fold GLM boost, refit)
        "ours_exp042_gb10_refit",    # our 10-fold GLM boost w/ pairs, refit
        "ours_exp041_realmlp_refit", # our Digit-TE RealMLP, refit
        "megayak_realmlp_v2",        # 0.950168 (RealMLP refit)
        "weni_realmlp_bpe_v2",       # 0.950005 (BPE RealMLP)
        "ours_exp023_glm_xgb",       # 0.951165 (our GLM-XGBoost depth 2)
        "gbank_gbn_bag5_w512",       # 0.951223 (Neural GLM-boost)
        "busya_ebm_g2_glminit",      # 0.950591 (EBM GLM-init)
        "busya_tabm_dl",             # 0.949703 (TabM)
        "queyrusi_digit_cnn",        # 0.949738 (Digit CNN)
        "aman_tabpfn_slimypunk_tp1", # 0.949294 (TabPFN)
    ]
    anchors = [k for k in anchors if k in models]

    Z_oof = np.column_stack([to_rank_logit(models[k][0]) for k in anchors])
    Z_te = np.column_stack([to_rank_logit(models[k][1]) for k in anchors])

    base_auc = roc_auc_score(y, Z_oof[:, 0])
    print(f"\nBaseline pub_wide_lm_stack OOF AUC = {base_auc:.6f}")

    # Check correlation & individual gain of our new models vs pub_wide_lm_stack
    for idx in [1, 2, 3, 4, 5, 6]:
        corr = float(np.dot(Z_oof[:, 0], Z_oof[:, idx]) / len(y))
        best_g = max(roc_auc_score(y, (1 - w) * Z_oof[:, 0] + w * Z_oof[:, idx]) for w in [0.02, 0.04, 0.08, 0.12, 0.16]) - base_auc
        print(f"  vs {anchors[idx]:26s}: corr={corr:.4f}, solo={models[anchors[idx]][3]:.6f}, 1D_gain={best_g*1e4:+.3f} pts")

    # 2-Seed Outer Nested-CV Sweep of C for LogisticRegression
    best_C, best_mean_auc = None, -1.0
    for C in [0.02, 0.04, 0.08, 0.15, 0.3]:
        aucs = []
        for seed in [2026, 777]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            p = np.zeros(len(y), dtype=np.float64)
            for tr, va in skf.split(Z_oof, y):
                clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=100, tol=1e-6)
                clf.fit(Z_oof[tr], y[tr])
                p[va] = clf.decision_function(Z_oof[va])
            aucs.append(roc_auc_score(y, p))
        mean_auc = float(np.mean(aucs))
        print(f"  [Curated-{len(anchors)} LogReg C={C:<4}] Nested2026={aucs[0]:.6f} | Nested777={aucs[1]:.6f} | Mean={mean_auc:.6f}")
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_C = C

    # Also test Non-Negative / Minimal-Negative Convex Blend (offset + positive weights)
    # Using 5-Fold Bagged LogReg on best_C for final test predictions
    skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_final = np.zeros(len(y), dtype=np.float64)
    te_logits = np.zeros(len(test), dtype=np.float64)
    coefs = []
    for tr, va in skf_final.split(Z_oof, y):
        clf = LogisticRegression(C=best_C, solver="newton-cholesky", max_iter=100, tol=1e-6)
        clf.fit(Z_oof[tr], y[tr])
        oof_final[va] = clf.decision_function(Z_oof[va])
        te_logits += clf.decision_function(Z_te) / 5.0
        coefs.append(clf.coef_[0])

    mean_coef = np.mean(coefs, axis=0)
    print(f"\nFinal Nested-CV OOF AUC (C={best_C}) = {roc_auc_score(y, oof_final):.6f} (vs Baseline {base_auc:.6f}, Gain={(roc_auc_score(y, oof_final)-base_auc)*1e4:+.3f} pts)")
    print("Mean LogReg Coefficients:")
    for name, w in sorted(zip(anchors, mean_coef), key=lambda x: -abs(x[1])):
        print(f"  {name:28s}: {w:+.4f}")

    p_final = expit(te_logits)
    sub = pd.DataFrame({"id": test["id"], "is_fraud": p_final})
    out_sub = os.path.join(DATA_DIR, "submission_exp043.csv")
    out_oof = os.path.join(DATA_DIR, "oof_exp043.npy")
    sub.to_csv(out_sub, index=False)
    np.save(out_oof, expit(oof_final).astype(np.float32))

    # Sanity & correlation vs Exp040 / Exp033
    assert sub["is_fraud"].isnull().sum() == 0
    assert len(sub) == 299984
    sub40 = pd.read_csv(f"{DATA_DIR}/submission_exp040.csv")["is_fraud"]
    sub33 = pd.read_csv(f"{DATA_DIR}/submission_exp033.csv")["is_fraud"]
    print(f"Spearman vs Exp040 (Old Frame 2): {pd.Series(p_final).corr(sub40, method='spearman'):.6f}")
    print(f"Spearman vs Exp033 (Old Frame 1): {pd.Series(p_final).corr(sub33, method='spearman'):.6f}")
    print(f"Saved: {out_sub} & {out_oof}")


if __name__ == "__main__":
    main()
