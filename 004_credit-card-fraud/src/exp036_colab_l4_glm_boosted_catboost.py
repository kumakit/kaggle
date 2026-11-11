"""
004_credit-card-fraud: Exp #036 - Colab L4 GPU GLM-Boosted CatBoost GPU (Oblivious Trees, 2 Seeds)
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Key Innovations:
  1. The 3rd and Final Pillar of GBDT: CatBoost GPU with Oblivious (Symmetric) Trees!
     - LightGBM uses Leaf-wise splits; XGBoost uses Depth-wise splits.
     - CatBoost uses Symmetric Trees (all leaves at a depth share identical split feature & threshold).
     - This unique structure provides the strongest structural orthogonality against the existing ladder & GBDT models!
  2. GLM Additive Skeleton Injection:
     - GLM (LogisticRegression solver='newton-cholesky', C=0.3) captures the underlying additive formula.
     - Raw decision_function logit is passed directly into CatBoost `Pool(..., baseline=z)`.
  3. Shallow Oblivious Trees (depth=3, 8 leaves) fit only the residual non-linear interactions without overfitting.
  4. 2-Seed Averaging (Seeds: 42, 2026) x 5-Fold Stratified CV on Colab L4 GPU.
  5. Dual recovery: Google Drive + Base64 stdout for test submission CSV & 699,960-row OOF predictions CSV.
"""

import os
import sys
import subprocess
import glob
import json
import base64
import gc
import warnings
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FLOAT_COLS = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']

PAIRS = [
    ("merchant_category", "merchant_risk_score"),
    ("is_foreign_transaction", "distance_from_home_km"),
    ("channel", "distance_from_home_km"),
    ("channel", "hours_since_last_txn"),
    ("card_type", "amount_usd"),
    ("card_type", "account_balance_usd"),
    ("device_type", "velocity_score"),
    ("device_type", "distance_from_home_km"),
    ("auth_method", "amount_usd"),
    ("auth_method", "hours_since_last_txn"),
    ("merchant_category", "amount_usd"),
    ("merchant_category", "velocity_score"),
    ("channel", "velocity_score"),
]

def setup_environment():
    print("=== 1. Setting up Colab L4 Environment ===")
    try:
        import catboost
        print("  - CatBoost is already installed.")
    except ImportError:
        print("  - Installing catboost...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost", "scikit-learn", "kaggle"], check=True)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    username = os.environ.get("KAGGLE_USERNAME", "nobuyukikumagai")
    key = os.environ.get("KAGGLE_KEY", "757b1a8d9ce71f3b129fbae7cc34b1b2")
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)
        print(f"  - Configured kaggle.json for {username}")

    data_dir = "/content/data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("  - Downloading dataset from Kaggle...")
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "playground-series-s6e11", "-p", data_dir], check=True)
        for zip_file in glob.glob(os.path.join(data_dir, "*.zip")):
            subprocess.run(["unzip", "-q", "-o", zip_file, "-d", data_dir], check=True)

    print(f"  - Dataset ready at: {data_dir}")
    return data_dir


def extract_digit_keys(df):
    keys = {}
    for c in FLOAT_COLS:
        units = np.rint(df[c].to_numpy() * 100).astype(np.int64)
        integer = units // 100
        s = pd.Series(integer).astype(str)
        keys[f"{c}__int"] = integer
        keys[f"{c}__int10"] = units // 1000
        keys[f"{c}__tenths"] = units // 10
        keys[f"{c}__cents"] = units % 100
        keys[f"{c}__lead1"] = (s.str.len() * 10 + s.str[0].astype(int)).to_numpy()
        keys[f"{c}__lead2"] = (s.str.len() * 100 + s.str.slice(0, 2).astype(int)).to_numpy()
    return pd.DataFrame(keys, index=df.index)


def extract_pair_keys(df):
    pairs = {}
    for a, b in PAIRS:
        sa = df[a].astype(str)
        sb = df[b].astype(str)
        pairs[f"{a}_X_{b}"] = (sa + "__" + sb).astype("category").cat.codes.to_numpy()
    return pd.DataFrame(pairs, index=df.index)


def prepare_data(data_dir):
    print("=== 2. Preparing Data & High-Cardinality Keys ===")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"  - Train: {train.shape}, Test: {test.shape}")

    CAT = [c for c in train.columns if train[c].dtype == "object" and c != "is_fraud"]
    NUM = [c for c in train.columns if train[c].dtype in ["float64", "int64"] and c not in ["id", "is_fraud"]]
    BOOL = [c for c in train.columns if train[c].dtype == "bool"]
    if "is_foreign_transaction" in BOOL:
        pass
    else:
        BOOL = [c for c in train.columns if c.startswith("is_")]

    train_dk = extract_digit_keys(train)
    test_dk = extract_digit_keys(test)

    train_pk = extract_pair_keys(train)
    test_pk = extract_pair_keys(test)

    # Encode categories to contiguous integers
    GLM_KEYS = list(train_dk.columns) + list(train_pk.columns) + CAT
    print(f"  - Total High-Cardinality Keys for GLM Target Encoding: {len(GLM_KEYS)}")

    full_keys = pd.concat([
        pd.concat([train_dk, train_pk, train[CAT]], axis=1),
        pd.concat([test_dk, test_pk, test[CAT]], axis=1)
    ], axis=0, ignore_index=True)

    CODES = {}
    N_LEVELS = {}
    for c in GLM_KEYS:
        codes, uniques = pd.factorize(full_keys[c], sort=True)
        CODES[c] = (codes[:len(train)], codes[len(train):])
        N_LEVELS[c] = len(uniques)

    return train, test, CAT, NUM, BOOL, GLM_KEYS, CODES, N_LEVELS


def target_encode_fold(keys, codes, n_levels, tr_idx, va_idx, y_tr, seed=42, m=50.0):
    global_mean = y_tr.mean()
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    E_tr = {}
    E_va = {}
    E_te = {}

    for c in keys:
        tr_code, te_code = codes[c]
        c_tr = tr_code[tr_idx]
        c_va = tr_code[va_idx]
        K = n_levels[c]

        oof_te = np.full(len(c_tr), global_mean, dtype=np.float64)
        for in_tr, in_va in kf.split(c_tr):
            cnt = np.bincount(c_tr[in_tr], minlength=K)
            pos = np.bincount(c_tr[in_tr], weights=y_tr[in_tr], minlength=K)
            smoothed = (pos + m * global_mean) / (cnt + m)
            oof_te[in_va] = smoothed[c_tr[in_va]]
        E_tr[f"te_{c}"] = oof_te

        cnt_full = np.bincount(c_tr, minlength=K)
        pos_full = np.bincount(c_tr, weights=y_tr, minlength=K)
        smoothed_full = (pos_full + m * global_mean) / (cnt_full + m)

        E_va[f"te_{c}"] = smoothed_full[c_va]
        E_te[f"te_{c}"] = smoothed_full[te_code]

    return pd.DataFrame(E_tr), pd.DataFrame(E_va), pd.DataFrame(E_te)


def to_logit(df, eps=1e-5):
    return logit(np.clip(df.to_numpy(), eps, 1.0 - eps))


def log_num(df):
    res = []
    for c in FLOAT_COLS:
        x = df[c].to_numpy()
        res.append(np.log1p(np.maximum(0, x)))
    return np.column_stack(res)


def main():
    data_dir = setup_environment()
    from catboost import CatBoostClassifier, Pool

    train, test, CAT, NUM, BOOL, GLM_KEYS, CODES, N_LEVELS = prepare_data(data_dir)
    y = train["is_fraud"].to_numpy().astype(int)

    ALL_KEYS = [f"{c}__{s}" for c in FLOAT_COLS for s in ["int", "tenths", "cents", "lead1"]]

    # Features for CatBoost
    cat_features_base = NUM + BOOL

    SEEDS = [42, 2026]
    N_SPLITS = 5

    oof_preds_all_seeds = np.zeros(len(train))
    test_preds_all_seeds = np.zeros(len(test))

    print(f"\n=== 3. Training GLM-Boosted CatBoost GPU ({len(SEEDS)} Seeds x {N_SPLITS} Folds) ===")
    print("  - Structure: Oblivious Trees, depth=3 (8 leaves), lr=0.03, l2_reg=3.0")

    for seed_idx, seed in enumerate(SEEDS, 1):
        print(f"\n--- [Seed {seed_idx}/{len(SEEDS)}: {seed}] ---")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        oof_seed = np.zeros(len(train))
        test_seed = np.zeros(len(test))

        for fold, (tr, va) in enumerate(skf.split(train, y), 1):
            # 1. Target Encoding for GLM
            E_tr, E_va, E_te = target_encode_fold(GLM_KEYS, CODES, N_LEVELS, tr, va, y[tr], seed=seed + fold, m=50.0)

            # 2. GLM fitting
            sc_e = StandardScaler().fit(to_logit(E_tr))
            sc_n = StandardScaler().fit(log_num(train.iloc[tr]))
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[CAT].iloc[tr])

            def build_glm_design(E, df):
                return np.hstack([
                    sc_e.transform(to_logit(E)),
                    oh.transform(df[CAT]),
                    sc_n.transform(log_num(df)),
                    df[BOOL].to_numpy(float)
                ])

            X_glm_tr = build_glm_design(E_tr, train.iloc[tr])
            X_glm_va = build_glm_design(E_va, train.iloc[va])
            X_glm_te = build_glm_design(E_te, test)

            glm = LogisticRegression(C=0.3, solver="newton-cholesky", tol=1e-8, max_iter=200)
            glm.fit(X_glm_tr, y[tr])

            z_tr = glm.decision_function(X_glm_tr)
            z_va = glm.decision_function(X_glm_va)
            z_te = glm.decision_function(X_glm_te)

            glm_fold_auc = roc_auc_score(y[va], expit(z_va))

            # 3. CatBoost on top of GLM baseline
            cb_E_tr = E_tr[[f"te_{c}" for c in ALL_KEYS]].to_numpy()
            cb_E_va = E_va[[f"te_{c}" for c in ALL_KEYS]].to_numpy()
            cb_E_te = E_te[[f"te_{c}" for c in ALL_KEYS]].to_numpy()

            X_cb_tr = np.hstack([train[cat_features_base].iloc[tr].to_numpy(float), cb_E_tr])
            X_cb_va = np.hstack([train[cat_features_base].iloc[va].to_numpy(float), cb_E_va])
            X_cb_te = np.hstack([test[cat_features_base].to_numpy(float), cb_E_te])

            # Use Pool with baseline logit margin
            train_pool = Pool(X_cb_tr, label=y[tr], baseline=z_tr)
            val_pool = Pool(X_cb_va, label=y[va], baseline=z_va)
            test_pool = Pool(X_cb_te, baseline=z_te)

            cb_model = CatBoostClassifier(
                iterations=3500,
                learning_rate=0.03,
                depth=3,                    # Oblivious Trees: 8 leaves
                l2_leaf_reg=3.0,
                loss_function='Logloss',
                eval_metric='AUC',
                task_type='GPU',
                random_seed=seed + fold,
                early_stopping_rounds=150,
                verbose=False
            )

            cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)

            # Predictions (CatBoost predict_proba automatically accounts for baseline!)
            p_va = cb_model.predict_proba(val_pool)[:, 1]
            p_te = cb_model.predict_proba(test_pool)[:, 1]

            fold_auc = roc_auc_score(y[va], p_va)
            print(f"  Seed {seed} Fold {fold}: GLM AUC = {glm_fold_auc:.5f} | CatBoost AUC = {fold_auc:.5f}")

            oof_seed[va] = p_va
            test_seed += p_te / N_SPLITS

            gc.collect()

        seed_auc = roc_auc_score(y, oof_seed)
        print(f"  >>> Seed {seed} Overall OOF AUC: {seed_auc:.5f} <<<")

        oof_preds_all_seeds += oof_seed / len(SEEDS)
        test_preds_all_seeds += test_seed / len(SEEDS)

    total_oof_auc = roc_auc_score(y, oof_preds_all_seeds)
    print("\n" + "=" * 60)
    print(f"  ★ Exp #036 2-Seed CatBoost GPU Final OOF AUC: {total_oof_auc:.5f} ★")
    print("=" * 60)

    # 4. Save results
    sub = pd.DataFrame({
        "id": test["id"],
        "is_fraud": test_preds_all_seeds
    })
    sub_path = "/content/submission_exp036_catboost.csv"
    sub.to_csv(sub_path, index=False)

    oof_df = pd.DataFrame({
        "id": train["id"],
        "oof_pred": oof_preds_all_seeds,
        "is_fraud": y
    })
    oof_path = "/content/oof_exp036_catboost.csv"
    oof_df.to_csv(oof_path, index=False)

    # Copy to Google Drive if mounted
    drive_dir = "/content/drive/MyDrive/kaggle_004_s6e11"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(drive_dir, exist_ok=True)
        import shutil
        shutil.copy(sub_path, os.path.join(drive_dir, "submission_exp036_catboost.csv"))
        shutil.copy(oof_path, os.path.join(drive_dir, "oof_exp036_catboost.csv"))
        print(f"  - Successfully synced to Google Drive: {drive_dir}")

    # Fallback Base64 Output
    with open(sub_path, "rb") as f:
        sub_b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{sub_b64}\n===BASE64_SUBMISSION_END===")

    with open(oof_path, "rb") as f:
        oof_b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_OOF_START===\n{oof_b64}\n===BASE64_OOF_END===")

    print("\n=== Exp #036 Colab L4 Job Finished Successfully! ===")


if __name__ == "__main__":
    main()
