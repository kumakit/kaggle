"""
Fine-grained Nested-CV optimization v4 on selected diverse orthogonal anchors.
Tests LogReg C sweep + Nelder-Mead/Powell rank-logit blend inside 2-seed Outer Nested-CV.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append("004_credit-card-fraud/src")
from eval_nested_cv_base import load_all_clean_models, to_rank_logit
from eval_nested_cv_v2 import load_busyaprime_models, eval_nested_logreg

DATA_DIR = "004_credit-card-fraud/data"


def main():
    y = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy().astype(int)
    models = load_all_clean_models(y)
    models = load_busyaprime_models(models, y)

    oof42 = pd.read_csv(f"{DATA_DIR}/exp042_out/oof_exp042_gb10.csv")["oof_exp042_gb10"].to_numpy()
    te42_refit = pd.read_csv(f"{DATA_DIR}/exp042_out/submission_exp042_refit.csv")["is_fraud"].to_numpy()
    models["ours_exp042_gb10_refit"] = (oof42.astype(np.float32), te42_refit.astype(np.float32), "ours_gbdt", roc_auc_score(y, oof42))

    # Curated orthogonal anchors (1-2 strong representatives per distinct family/signal)
    anchors = [
        "pub_wide_lm_stack",        # 0.952565 (201-model wide LM stack)
        "aman_sameer_glm10f",       # 0.951595 (10-fold GLM boost, refit)
        "ours_exp042_gb10_refit",   # 0.951449 (our 10-fold GLM boost w/ pairs, refit)
        "ours_exp023_glm_xgb",      # 0.951165 (our GLM-XGBoost depth 2)
        "ours_exp021_glm_lgb",      # 0.950852 (our GLM-LightGBM 4-leaf)
        "megayak_realmlp_v2",       # 0.950168 (RealMLP refit)
        "weni_realmlp_bpe_v2",      # 0.950005 (BPE RealMLP)
        "next2060_realmlp",         # 0.950033 (RealMLP)
        "next2060_glmboost_pair",   # 0.950864 (GLMBoost pair refit3)
        "weni_bpe_lgb_l4",          # 0.950931 (BPE LightGBM)
        "gbank_gbn_bag5_w512",      # 0.951223 (Neural GLM-boost)
        "busya_ebm_g2_glminit",     # 0.950591 (EBM GLM-init)
        "busya_tabm_dl",            # 0.949703 (TabM)
        "queyrusi_digit_cnn",       # 0.949738 (Digit CNN)
        "aman_tabpfn_slimypunk_tp1",# 0.949294 (TabPFN)
    ]
    anchors = [k for k in anchors if k in models]
    print(f"Curated anchors ({len(anchors)}):", anchors, flush=True)

    Z = np.column_stack([to_rank_logit(models[k][0]) for k in anchors])
    base_z = Z[:, 0]
    print(f"Baseline pub_wide_lm_stack = {roc_auc_score(y, base_z):.6f}", flush=True)

    # Fine-grained greedy forward selection
    selected = [0]
    cur_z = base_z.copy()
    cur_auc = roc_auc_score(y, cur_z)
    for step in range(12):
        best = None
        best_auc = cur_auc
        best_next = None
        for idx in range(1, len(anchors)):
            if idx in selected:
                continue
            zk = Z[:, idx]
            for w in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16]:
                cand = (1.0 - w) * cur_z + w * zk
                auc = roc_auc_score(y, cand)
                if auc > best_auc + 1e-7:
                    best_auc = auc
                    best = (idx, w)
                    best_next = cand
        if best is None:
            break
        idx, w = best
        selected.append(idx)
        cur_z = (best_next - best_next.mean()) / best_next.std()
        cur_auc = best_auc
        print(f"  Greedy Step {step+1:2d}: +{anchors[idx]:28s} (w={w:.3f}) -> AUC={cur_auc:.6f}", flush=True)

    # Test LogReg C sweep on ALL curated anchors vs GREEDY-selected anchors in 2-seed Outer Nested-CV
    for tag, cols in [("Curated-15", list(range(len(anchors)))), ("Greedy-Sel", selected)]:
        Z_sub = Z[:, cols]
        print(f"\n--- {tag} ({len(cols)} models) 2-Seed Outer Nested-CV ---", flush=True)
        for C in [0.01, 0.03, 0.08, 0.2, 0.5, 1.0]:
            a26, _ = eval_nested_logreg(Z_sub, y, C, seed=2026)
            a77, _ = eval_nested_logreg(Z_sub, y, C, seed=777)
            print(f"  LogReg C={C:<4}: Nested2026={a26:.6f} | Nested777={a77:.6f}", flush=True)


if __name__ == "__main__":
    main()
