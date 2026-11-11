"""
004_credit-card-fraud: Exp #023 - Colab L4 GPU GLM-Boosted XGBoost Depth 2 (2 Seeds)
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Key Innovations:
  1. Transfer of the Breakthrough from LightGBM (Exp021: 0.95113) to XGBoost GPU!
  2. Additive GLM (LogisticRegression solver='newton-cholesky', C=0.3) captures the underlying logistic formula using 73 target-encoded keys (36 digit keys + 13 interaction pairs + 24 base features).
  3. GLM Logit injected directly into XGBoost Depth 2 via `base_margin` in DMatrix.
  4. XGBoost Depth 2 (max_depth=2) fits strictly the non-linear interactions & residual errors.
  5. 2-Seed Averaging (Seeds: 42, 2026) x 5-Fold Stratified CV on Colab L4 GPU.
  6. Base64 export for test submission CSV AND 699,960-row OOF predictions CSV.
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
import xgboost as xgb

warnings.filterwarnings("ignore")

FLOAT_COLS = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']

PAIRS = [
    ("merchant_category", "merchant_risk_score"),
    ("is_foreign_transaction", "ip_country_mismatch"),
    ("merchant_category", "q:amount_usd"),
    ("txn_count_last_24h", "q:velocity_score"),
    ("merchant_category", "customer_age"),
    ("q:amount_usd", "q:merchant_risk_score"),
    ("q:hours_since_last_txn", "q:velocity_score"),
    ("customer_age", "q:merchant_risk_score"),
    ("is_new_merchant", "day_of_week"),
    ("is_ai_generated_scam_attempt", "q:velocity_score"),
    ("customer_age", "card_age_months"),
    ("time_of_day_hour", "day_of_week"),
    ("txn_count_last_24h", "customer_age")
]


def ensure_environment_and_data():
    """Verify packages, Kaggle credentials, and dataset on Colab VM."""
    print("[INIT] Verifying packages and datasets...")

    try:
        import xgboost
        import sklearn
    except ImportError:
        print("  - Installing missing ML packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost", "scikit-learn", "kaggle"], check=True)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    username = os.environ.get("KAGGLE_USERNAME", "nobuyukikumagai")
    key = os.environ.get("KAGGLE_KEY", "757b1a8d9ce71f3b129fbae7cc34b1b2")
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)
        print(f"  - Configured kaggle.json for user: {username}")

    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        candidates = ["/content/data", "004_credit-card-fraud/data", "/content/kaggle/004_credit-card-fraud/data"]
        for c in candidates:
            if os.path.exists(os.path.join(c, "train.csv")):
                data_dir = c
                break

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
        keys[f"{c}__lead2"] = (s.str.len() * 100 + s.str[:2].astype(int)).to_numpy()
    return pd.DataFrame(keys, index=df.index)


def factorize_shared(frame_tr, frame_te, cols):
    codes, n_levels = {}, {}
    for c in cols:
        combined = pd.concat([frame_tr[c], frame_te[c]], ignore_index=True)
        k, uniques = pd.factorize(combined)
        k = k.astype(np.int32)
        codes[c] = (k[:len(frame_tr)], k[len(frame_tr):])
        n_levels[c] = len(uniques)
    return codes, n_levels


def te_table(codes, target, n_levels, m):
    frauds = np.bincount(codes, weights=target, minlength=n_levels)
    rows = np.bincount(codes, minlength=n_levels)
    return (frauds + m * target.mean()) / (rows + m)


def target_encode_fold(cols, codes_dict, levels_dict, tr, va, y_all, seed, m=50, inner_cv=5):
    y_tr = y_all[tr].astype(float)
    inner = list(KFold(inner_cv, shuffle=True, random_state=seed).split(tr))

    E_tr = np.empty((len(tr), len(cols)), dtype=np.float32)
    E_va = np.empty((len(va), len(cols)), dtype=np.float32)
    n_test = len(codes_dict[cols[0]][1])
    E_te = np.empty((n_test, len(cols)), dtype=np.float32)

    for j, c in enumerate(cols):
        c_train, c_test = codes_dict[c]
        k_tr = c_train[tr]
        n_lev = levels_dict[c]

        for fit_idx, enc_idx in inner:
            tab = te_table(k_tr[fit_idx], y_tr[fit_idx], n_lev, m)
            E_tr[enc_idx, j] = tab[k_tr[enc_idx]]

        full_tab = te_table(k_tr, y_tr, n_lev, m)
        E_va[:, j] = full_tab[c_train[va]]
        E_te[:, j] = full_tab[c_test]

    names = [f"te_{c}" for c in cols]
    return (pd.DataFrame(E_tr, columns=names),
            pd.DataFrame(E_va, columns=names),
            pd.DataFrame(E_te, columns=names))


def main():
    print("=" * 80)
    print("  004 Credit Card Fraud: Exp #023 - Colab L4 GPU GLM-Boosted XGBoost Depth 2")
    print("=" * 80)

    data_dir = ensure_environment_and_data()

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    print("\n[1/5] Loading data and detecting feature types...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train: {train.shape}, Test: {test.shape}")
    y = train["is_fraud"].values

    FEATURES = [c for c in test.columns if c != "id"]
    BOOL = [c for c in FEATURES if train[c].dtype == 'bool']
    NUM = [c for c in FEATURES if train[c].dtype in ['int64', 'float64'] and c not in BOOL]
    CAT = [c for c in FEATURES if c not in BOOL and c not in NUM]
    print(f"{len(NUM)} numeric · {len(BOOL)} boolean · {len(CAT)} categorical")

    print("\n[2/5] Extracting digit keys and interaction pairs...")
    keys_tr = extract_digit_keys(train)
    keys_te = extract_digit_keys(test)
    KEYS = list(keys_tr.columns)

    both = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
    def pair_part(col):
        return pd.qcut(both[col[2:]], 10, labels=False, duplicates="drop") if col.startswith("q:") else both[col]

    pairs_df = pd.DataFrame({
        f"{a}*{b}".replace("q:", ""): pair_part(a).astype(str) + "|" + pair_part(b).astype(str)
        for a, b in PAIRS
    })
    PAIR_KEYS = list(pairs_df.columns)
    pairs_tr = pairs_df.iloc[:len(train)].reset_index(drop=True)
    pairs_te = pairs_df.iloc[len(train):].reset_index(drop=True)
    del both, pairs_df
    gc.collect()

    ALL_KEYS = FEATURES + KEYS
    GLM_KEYS = ALL_KEYS + PAIR_KEYS

    frame_tr = pd.concat([train[FEATURES], keys_tr, pairs_tr], axis=1)
    frame_te = pd.concat([test[FEATURES], keys_te, pairs_te], axis=1)

    print("\n[3/5] Factorizing categorical and integer levels for Target Encoding...")
    CODES, N_LEVELS = factorize_shared(frame_tr, frame_te, GLM_KEYS)
    print(f"Total encoded features for GLM: {len(GLM_KEYS)}")

    # Category conversion for XGBoost enable_categorical=True
    X_RAW = train[FEATURES].copy()
    T_RAW = test[FEATURES].copy()
    for c in CAT:
        categories = sorted(pd.concat([train[c], test[c]]).unique())
        X_RAW[c] = pd.Categorical(X_RAW[c], categories=categories)
        T_RAW[c] = pd.Categorical(T_RAW[c], categories=categories)
    X_RAW[BOOL] = X_RAW[BOOL].astype(int)
    T_RAW[BOOL] = T_RAW[BOOL].astype(int)

    to_logit = lambda E: logit(np.clip(E.to_numpy(np.float64), 1e-5, 1 - 1e-5))
    log_num = lambda df: np.log1p(df[NUM].to_numpy(float))

    seeds = [42, 2026]
    n_splits = 5

    oof_total = np.zeros(len(train))
    test_total = np.zeros(len(test))

    print(f"\n[4/5] Starting 2-Seed ({seeds}) x 5-Fold GLM-Boosted XGBoost Depth 2 GPU training...")

    for s_idx, seed in enumerate(seeds, 1):
        print(f"\n==================== SEED {seed} ({s_idx}/{len(seeds)}) ====================")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_seed = np.zeros(len(train))
        test_seed = np.zeros(len(test))

        for fold, (tr, va) in enumerate(skf.split(train, y), 1):
            # 1. Target Encoding for GLM
            E_tr, E_va, E_te = target_encode_fold(GLM_KEYS, CODES, N_LEVELS, tr, va, y, seed=seed + fold, m=50)

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

            # 3. XGBoost Depth 2 on top of GLM base_margin
            xgb_E_tr = E_tr[[f"te_{c}" for c in ALL_KEYS]]
            xgb_E_va = E_va[[f"te_{c}" for c in ALL_KEYS]]
            xgb_E_te = E_te[[f"te_{c}" for c in ALL_KEYS]]

            X_xgb_tr = pd.concat([X_RAW.iloc[tr].reset_index(drop=True), xgb_E_tr], axis=1)
            X_xgb_va = pd.concat([X_RAW.iloc[va].reset_index(drop=True), xgb_E_va], axis=1)
            X_xgb_te = pd.concat([T_RAW.reset_index(drop=True), xgb_E_te], axis=1)

            dtrain = xgb.DMatrix(X_xgb_tr, label=y[tr], enable_categorical=True)
            dtrain.set_base_margin(z_tr)

            dval = xgb.DMatrix(X_xgb_va, label=y[va], enable_categorical=True)
            dval.set_base_margin(z_va)

            dtest = xgb.DMatrix(X_xgb_te, enable_categorical=True)
            dtest.set_base_margin(z_te)

            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': 0.03,
                'max_depth': 2,              # 4-leaf equivalent
                'min_child_weight': 5.0,
                'subsample': 0.8,
                'colsample_bytree': 0.5,      # Half-feature subsampling
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'device': 'cuda',
                'random_state': seed + fold
            }

            evallist = [(dval, 'val')]
            booster = xgb.train(
                params,
                dtrain,
                evals=evallist,
                num_boost_round=10000,
                early_stopping_rounds=200,
                verbose_eval=False
            )

            p_va = booster.predict(dval)
            p_te = booster.predict(dtest)

            oof_seed[va] = p_va
            test_seed += p_te / n_splits

            boost_fold_auc = roc_auc_score(y[va], p_va)
            print(f"  [Seed {seed} Fold {fold}] GLM AUC: {glm_fold_auc:.5f} -> GLM + XGBoost Depth 2 AUC: {boost_fold_auc:.5f} (gain: {boost_fold_auc - glm_fold_auc:+.5f}, trees: {booster.best_iteration})")

            del E_tr, E_va, E_te, X_glm_tr, X_glm_va, X_glm_te, X_xgb_tr, X_xgb_va, X_xgb_te, dtrain, dval, dtest, booster
            gc.collect()

        seed_auc = roc_auc_score(y, oof_seed)
        print(f"\n>> Seed {seed} Single OOF ROC-AUC: {seed_auc:.5f}")

        oof_total += oof_seed / len(seeds)
        test_total += test_seed / len(seeds)

    overall_auc = roc_auc_score(y, oof_total)
    print("\n" + "=" * 70)
    print(f"[4/5] Exp #023 2-Seed Averaged GLM + XGBoost Depth 2 OOF ROC-AUC: {overall_auc:.5f}")
    print(f"  (Reference Exp #021 GLM + LightGBM: 0.95085 | Diff: {overall_auc - 0.95085:+.5f})")
    print(f"  (Reference Exp #019 Single XGBoost:  0.94994 | Gain: {overall_auc - 0.94994:+.5f})")
    print("=" * 70)

    # [5/5] Save Files & Direct Kaggle Submit
    sub_df = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_total
    })
    out_file = "submission_exp023_colab.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\n[5/5] Saved submission to {out_file} (Rows: {len(sub_df)})")
    print(sub_df.head())

    oof_df = pd.DataFrame({
        'id': train['id'],
        'oof_glm_xgb_d2': oof_total
    })
    oof_file = "oof_exp023_colab.csv"
    oof_df.to_csv(oof_file, index=False)
    print(f"  Saved OOF to {oof_file} (Rows: {len(oof_df)})")

    # Google Drive Sync
    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        dest_sub = os.path.join(drive_dir, out_file)
        dest_oof = os.path.join(drive_dir, oof_file)
        sub_df.to_csv(dest_sub, index=False)
        oof_df.to_csv(dest_oof, index=False)
        print(f"[DRIVE] Saved to Google Drive: {dest_sub} & {dest_oof}")

    # Direct Kaggle Submit
    print("\nSubmitting to Kaggle CLI from Colab...")
    sub_msg = f"Exp023: Colab L4 GPU GLM-Boosted XGBoost Depth 2 2-Seeds (Val AUC {overall_auc:.5f})"
    sub_cmd = [
        "kaggle", "competitions", "submit",
        "-c", "playground-series-s6e11",
        "-f", out_file,
        "-m", sub_msg
    ]
    try:
        res = subprocess.run(sub_cmd, capture_output=True, text=True, check=True)
        print("[SUBMISSION] Kaggle submit response:")
        print(res.stdout)
    except Exception as e:
        print(f"[WARNING] Direct submit failed: {e}")

    # Base64 output for submission CSV
    with open(out_file, "rb") as f:
        b64_sub = base64.b64encode(f.read()).decode("utf-8")
    print("\n===BASE64_SUBMISSION_START===")
    print(b64_sub)
    print("===BASE64_SUBMISSION_END===")

    # Base64 output for OOF CSV
    with open(oof_file, "rb") as f:
        b64_oof = base64.b64encode(f.read()).decode("utf-8")
    print("\n===BASE64_OOF_START===")
    print(b64_oof)
    print("===BASE64_OOF_END===")


if __name__ == "__main__":
    main()
