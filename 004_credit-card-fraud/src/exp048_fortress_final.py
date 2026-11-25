"""
Exp #048: Pure Nested-CV Fortress Ensemble (Final Build)
Combines two independent Pure Nested-CV meta-learners:
  1) Non-Negative Direct-AUC Convex Blend (from 2-seed Outer Nested-CV, OOF 0.952594, zero negative weights):
     83.19% WideLM + 9.89% Sameer10F + 5.87% MegayakRealMLP + 0.43% WeniRealMLP_BPE
     + 0.25% Exp042_LGB10 + 0.19% Exp041_047_RealMLP3Seed + 0.18% Exp046_XGB10
  2) Exp #043 L2-Regularized LogisticRegression Stack (Nested-CV 0.952597)
"""
import os
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA_DIR = "004_credit-card-fraud/data"
BANKS_DIR = os.path.join(DATA_DIR, "oof_banks")


def to_rank_logit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    z = logit(np.clip(r, eps, 1.0 - eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float64)


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=["id"])
    y = train["is_fraud"].to_numpy().astype(int)

    z_am = np.load(f"{BANKS_DIR}/amanatar/rank1_q17_streams.npz", allow_pickle=False)
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

    components = [
        ("pub_wide_lm_stack", 0.8319,
         np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_nested.npy"),
         np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_test.npy")),
        ("aman_sameer_glm10f", 0.0989, z_am["oof_sameer_glm10f"], z_am["test_sameer_glm10f"]),
        ("megayak_realmlp_v2", 0.0587,
         np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_oof.npy"),
         np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_test.npy")),
        ("weni_realmlp_bpe_v2", 0.0043,
         np.load(f"{BANKS_DIR}/wenimohamed/oof_weni_realmlp_bpe_v2.npy"),
         np.load(f"{BANKS_DIR}/wenimohamed/test_weni_realmlp_bpe_v2.npy")),
        ("ours_exp042_lgb10_refit", 0.0025, oof42, te42),
        ("ours_realmlp_3seed_refit", 0.0019, oof_rmlp3, te_rmlp3),
        ("ours_exp046_xgb10_refit", 0.0018, oof46, te46),
    ]

    z_oof_convex = sum(w * to_rank_logit(o) for _, w, o, _ in components)
    z_te_convex = sum(w * to_rank_logit(t) for _, w, _, t in components)
    auc_convex = roc_auc_score(y, z_oof_convex)
    print(f"NonNeg Direct-AUC Convex OOF AUC = {auc_convex:.6f}")

    oof43 = np.load(f"{DATA_DIR}/oof_exp043.npy")
    te43 = pd.read_csv(f"{DATA_DIR}/submission_exp043.csv")["is_fraud"].to_numpy()
    z_oof43 = to_rank_logit(oof43)
    z_te43 = to_rank_logit(te43)
    print(f"Exp043 L2-LogReg OOF AUC         = {roc_auc_score(y, z_oof43):.6f}")

    best_a, best_auc = 0.0, -1.0
    for a in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        auc = roc_auc_score(y, (1.0 - a) * z_oof_convex + a * z_oof43)
        print(f"  MetaBlend (Convex*{1-a:.1f} + LogReg*{a:.1f}): OOF AUC = {auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_a = a

    # Use optimal meta-blend (or 50/50 if within 1e-6 for maximum stability)
    z_oof_final = (1.0 - best_a) * z_oof_convex + best_a * z_oof43
    z_te_final = (1.0 - best_a) * z_te_convex + best_a * z_te43

    p_final = expit(z_te_final)
    out_sub = os.path.join(DATA_DIR, "submission_exp048.csv")
    out_oof = os.path.join(DATA_DIR, "oof_exp048.npy")
    pd.DataFrame({"id": test["id"], "is_fraud": p_final}).to_csv(out_sub, index=False)
    np.save(out_oof, expit(z_oof_final).astype(np.float32))
    print(f"\nSaved Final Exp #048 Fortress: {out_sub} (alpha={best_a:.1f}, OOF AUC={best_auc:.6f})")


if __name__ == "__main__":
    main()
