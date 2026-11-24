"""
Baseline Nested-CV evaluation across all clean local + collected OOF banks (before Exp #041 RealMLP).
N100 protection: n_jobs <= 2, float32 arrays, explicit gc.
"""
import gc
import glob
import os
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

DATA_DIR = "004_credit-card-fraud/data"
BANKS_DIR = os.path.join(DATA_DIR, "oof_banks")


def to_rank_logit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    z = logit(np.clip(r, eps, 1.0 - eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float32)


def load_all_clean_models(y: np.ndarray):
    n_tr = len(y)
    n_te = 299984
    models = {}  # name -> (oof_raw, test_raw, family)

    def add(name, oof, te, family):
        oof = np.asarray(oof, dtype=np.float32).ravel()
        te = np.asarray(te, dtype=np.float32).ravel()
        if len(oof) != n_tr or len(te) != n_te:
            return
        if np.isnan(oof).any() or np.isnan(te).any():
            return
        auc = roc_auc_score(y, oof)
        if auc < 0.945:
            return
        models[name] = (oof, te, family, auc)

    # 1. Our own models
    add("ours_exp019_xgb_d2", pd.read_csv(f"{DATA_DIR}/oof_exp019_colab.csv")["oof_xgb_d2_te"],
        pd.read_csv(f"{DATA_DIR}/submission_exp019_colab.csv")["is_fraud"], "ours_gbdt")
    add("ours_exp021_glm_lgb", np.load(f"{DATA_DIR}/oof_exp021.npy"),
        pd.read_csv(f"{DATA_DIR}/submission_exp021.csv")["is_fraud"], "ours_gbdt")
    add("ours_exp023_glm_xgb", pd.read_csv(f"{DATA_DIR}/oof_exp023_colab.csv")["oof_glm_xgb_d2"],
        pd.read_csv(f"{DATA_DIR}/submission_exp023_colab.csv")["is_fraud"], "ours_gbdt")
    add("ours_exp036_cat_d3", pd.read_csv(f"{DATA_DIR}/oof_exp036_catboost.csv")["oof_pred"],
        pd.read_csv(f"{DATA_DIR}/submission_exp036_catboost.csv")["is_fraud"], "ours_gbdt")

    # 2. goodpjw_artifacts wide_lm_stack_nested
    add("pub_wide_lm_stack", np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_nested.npy"),
        np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_test.npy"), "stack")

    # 3. goodpjw_bank (70+ single/bagged members, prefer _full test if available)
    bank_dir = f"{BANKS_DIR}/goodpjw_bank"
    for f in sorted(glob.glob(f"{bank_dir}/oof_*.npy")):
        base = os.path.basename(f)[4:-4]  # strip oof_ and .npy
        te_full = os.path.join(bank_dir, f"test_{base}_full.npy")
        te_norm = os.path.join(bank_dir, f"test_{base}.npy")
        te_p = te_full if os.path.exists(te_full) else te_norm
        if os.path.exists(te_p):
            fam = "ebm" if "ebm" in base else ("nn" if "realmlp" in base or "gbn" in base else ("glm" if base.startswith("glm") else "gbdt"))
            add(f"gbank_{base}", np.load(f), np.load(te_p), fam)

    # 4. amanatar diverse streams (exclude anchor_095370 which is board-tuned)
    z_am = np.load(f"{BANKS_DIR}/amanatar/rank1_q17_streams.npz", allow_pickle=False)
    for k in z_am.files:
        if k.startswith("oof_") and k != "oof_anchor_095370":
            tk = "test_" + k[4:]
            if tk in z_am.files:
                fam = "lm" if ("lm" in k or "gpt2" in k) else ("tabpfn" if "tabpfn" in k else ("nn" if "realmlp" in k else "gbdt"))
                add(f"aman_{k[4:]}", z_am[k], z_am[tk], fam)

    # 5. next2060 (use *_refit3 for glmboost / glmboost_pair)
    nx_oof = pd.read_csv(f"{BANKS_DIR}/next2060/oof.csv")
    nx_te = pd.read_csv(f"{BANKS_DIR}/next2060/test.csv")
    add("next2060_glmboost", nx_oof["glmboost"], nx_te["glmboost_refit3"], "gbdt")
    add("next2060_glmboost_pair", nx_oof["glmboost_pair"], nx_te["glmboost_pair_refit3"], "gbdt")
    add("next2060_realmlp", nx_oof["realmlp"], nx_te["realmlp"], "nn")

    # 6. bizziberri (refit)
    bz_oof = pd.read_csv(f"{BANKS_DIR}/bizziberri/oof.csv")
    bz_te = pd.read_csv(f"{BANKS_DIR}/bizziberri/test.csv")
    for col in ["glm", "lgb3_s1", "lgb4", "xgb2"]:
        add(f"bizzi_{col}", bz_oof[col], bz_te[col], "glm" if col == "glm" else "gbdt")

    # 7. wenimohamed (BPE features + CatBoost / LGB / GLM / RealMLP / TabM)
    w_dir = f"{BANKS_DIR}/wenimohamed"
    for col in ["weni_bpe_lgb_l4", "weni_bpe_cat_d5", "weni_bpe_cat_d3", "weni_bpe_glm", "weni_realmlp_bpe_v2", "weni_tabm_bpe"]:
        add(col, np.load(f"{w_dir}/oof_{col}.npy"), np.load(f"{w_dir}/test_{col}.npy"),
            "nn" if ("realmlp" in col or "tabm" in col) else ("glm" if "glm" in col else "gbdt"))

    # 8. megayak_realmlp & megayak_lm
    add("megayak_realmlp_v2", np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_oof.npy"),
        np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_test.npy"), "nn")
    add("megayak_lm_5f", np.load(f"{BANKS_DIR}/megayak_lm/lm_oof.npy"),
        np.load(f"{BANKS_DIR}/megayak_lm/lm_test.npy"), "lm")

    # 9. queyrusi_cnn (Digit CNN + gen_predictions)
    q_cnn = np.load(f"{BANKS_DIR}/queyrusi_cnn/predictions.npz", allow_pickle=False)
    add("queyrusi_digit_cnn", q_cnn["oof"], q_cnn["test"], "cnn")
    q_gen = np.load(f"{BANKS_DIR}/queyrusi_cnn/gen_predictions.npz", allow_pickle=False)
    add("queyrusi_gen_stack", q_gen["oof"], q_gen["test"], "gbdt")

    return models


def deduplicate_models(models: dict, y: np.ndarray, corr_thresh: float = 0.9998):
    # Sort by AUC descending so stronger models are kept first, prioritizing ours_ first on ties
    items = sorted(models.items(), key=lambda kv: (kv[1][3] + (0.002 if kv[0].startswith("ours_") else 0.0)), reverse=True)
    kept_names = []
    kept_z = []
    for name, (oof, te, fam, auc) in items:
        z = to_rank_logit(oof)
        is_dup = False
        for kz in kept_z:
            c = np.dot(z, kz) / len(z)
            if c > corr_thresh:
                is_dup = True
                break
        if not is_dup:
            kept_names.append(name)
            kept_z.append(z)
    return kept_names


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    y = train["is_fraud"].to_numpy().astype(int)

    models = load_all_clean_models(y)
    print(f"Loaded {len(models)} valid clean models.")

    # Separate single models vs pre-stacked pub_wide_lm_stack
    single_models = {k: v for k, v in models.items() if k != "pub_wide_lm_stack"}
    kept_singles = deduplicate_models(single_models, y, corr_thresh=0.9995)
    print(f"Kept {len(kept_singles)} deduplicated single models (corr <= 0.9995).")

    # Build rank-logit matrices for single models
    Z_oof = np.column_stack([to_rank_logit(models[k][0]) for k in kept_singles])
    Z_te = np.column_stack([to_rank_logit(models[k][1]) for k in kept_singles])
    aucs = [models[k][3] for k in kept_singles]
    print("Top 15 single models:")
    for k, a in sorted(zip(kept_singles, aucs), key=lambda x: -x[1])[:15]:
        print(f"  {k:32s}: {a:.6f} ({models[k][2]})")

    # Outer Nested-CV using a DIFFERENT split seed (2026) to avoid overfitting to fold-42 boundaries
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

    # Method A: L2-Regularized LogisticRegression on single models (C sweep)
    for C in [0.001, 0.005, 0.02, 0.1]:
        nested_oof = np.zeros(len(y), dtype=np.float64)
        for tr_i, va_i in skf_outer.split(Z_oof, y):
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=500, n_jobs=1)
            clf.fit(Z_oof[tr_i], y[tr_i])
            nested_oof[va_i] = clf.decision_function(Z_oof[va_i])
        print(f"[Nested-CV Singles LogReg C={C:<5}] AUC = {roc_auc_score(y, nested_oof):.6f}")

    # Method B: Non-negative SLSQP optimization on single models (zero extrapolation / max stability)
    nested_nn = np.zeros(len(y), dtype=np.float64)
    for tr_i, va_i in skf_outer.split(Z_oof, y):
        # Fit Ridge with positive=True (fast & exact non-negative least squares)
        reg = Ridge(alpha=5000.0, positive=True, max_iter=2000)
        reg.fit(Z_oof[tr_i], y[tr_i])
        nested_nn[va_i] = reg.predict(Z_oof[va_i])
    print(f"[Nested-CV Singles NonNeg Ridge(alpha=5000)] AUC = {roc_auc_score(y, nested_nn):.6f}")

    # Method C: pub_wide_lm_stack + New Post-Nov11 models (ours + next2060 + bizziberri + weni + megayak_realmlp + queyrusi_cnn)
    post_keys = [k for k in kept_singles if any(k.startswith(p) for p in ("ours_", "next2060_", "bizzi_", "weni_", "megayak_realmlp", "queyrusi_digit", "aman_tabpfn"))]
    Z_aug_oof = np.column_stack([to_rank_logit(models["pub_wide_lm_stack"][0])] + [to_rank_logit(models[k][0]) for k in post_keys])
    print(f"\nAugmented Stack: pub_wide_lm_stack + {len(post_keys)} post-Nov11/ours models: {post_keys}")
    print(f"Baseline pub_wide_lm_stack OOF AUC = {models['pub_wide_lm_stack'][3]:.6f}")

    for C in [0.002, 0.01, 0.05]:
        nested_aug = np.zeros(len(y), dtype=np.float64)
        for tr_i, va_i in skf_outer.split(Z_aug_oof, y):
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=500, n_jobs=1)
            clf.fit(Z_aug_oof[tr_i], y[tr_i])
            nested_aug[va_i] = clf.decision_function(Z_aug_oof[va_i])
        print(f"[Nested-CV Augmented LogReg C={C:<5}] AUC = {roc_auc_score(y, nested_aug):.6f}")

    nested_aug_nn = np.zeros(len(y), dtype=np.float64)
    for tr_i, va_i in skf_outer.split(Z_aug_oof, y):
        reg = Ridge(alpha=5000.0, positive=True, max_iter=2000)
        reg.fit(Z_aug_oof[tr_i], y[tr_i])
        nested_aug_nn[va_i] = reg.predict(Z_aug_oof[va_i])
    print(f"[Nested-CV Augmented NonNeg Ridge(5000)] AUC = {roc_auc_score(y, nested_aug_nn):.6f}")


if __name__ == "__main__":
    main()
