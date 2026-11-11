"""
004_credit-card-fraud: Exp #019 - Colab L4 GPU XGBoost Depth 2 with FLOATS6 Digit Keys In-Fold Target Encoding
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Key Upgrades:
  1. XGBoost Depth 2 (max_depth=2): The exact equivalent of 4-leaf shallow trees for XGBoost.
  2. FLOATS6 Digit Keys Target Encoding (42 keys):
     ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']
     Extracted cents, int, ten, dec, lead1, lead2, len.
  3. Strict In-Fold Target Encoding with sklearn.preprocessing.TargetEncoder(smooth=20.0) per fold.
  4. colsample_bytree=0.5: Random feature subsampling preventing overfitting on redundant digit keys.
  5. 2-Seed Averaging (Seeds: 42, 2026) for variance reduction.
  6. GPU Acceleration on Colab L4 GPU (tree_method='hist', device='cuda').
  7. Base64 export for BOTH test submission CSV AND 699,960-row OOF predictions CSV.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FLOATS6 = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']


def ensure_environment_and_data():
    """Verify packages, Kaggle credentials, and dataset on Colab VM."""
    print("[INIT] Verifying packages and datasets...")
    
    # 1. Packages
    try:
        import xgboost
        import sklearn
    except ImportError:
        print("  - Installing missing ML packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xgboost", "scikit-learn", "kaggle"], check=True)

    # 2. Kaggle auth
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

    # 3. Competition Data files
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
    """Extract string digit keys from the 6 float columns."""
    k = {}
    for c in FLOATS6:
        x = df[c].astype(float).values
        i = np.floor(x + 1e-9).astype(np.int64)
        s = np.abs(i).astype(str)
        k[f'cents_{c}'] = (np.round(x * 100).astype(np.int64) % 100).astype(str)
        k[f'int_{c}'] = i.astype(str)
        k[f'ten_{c}'] = np.floor(x / 10 + 1e-9).astype(np.int64).astype(str)
        k[f'dec_{c}'] = np.floor(x * 10 + 1e-6).astype(np.int64).astype(str)
        k[f'lead1_{c}'] = np.array([v[:1] for v in s])
        k[f'lead2_{c}'] = np.array([v[:2] for v in s])
        k[f'len_{c}'] = np.array([len(v) for v in s]).astype(str)
    return pd.DataFrame(k)


def main():
    print("=" * 80)
    print("  004 Credit Card Fraud: Exp #019 - Colab L4 GPU XGBoost Depth 2 with Digit-TE")
    print("=" * 80)

    data_dir = ensure_environment_and_data()

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    print("\n[1/5] Loading data and preparing features...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Train: {train.shape}, Test: {test.shape}")

    bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    for c in bool_cols:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)

    cat_cols = ['merchant_category', 'card_type', 'channel', 'device_type', 'auth_method']
    for c in cat_cols:
        train[c] = train[c].astype('category')
        test[c] = test[c].astype('category')

    base_features = [c for c in train.columns if c not in ['id', 'is_fraud']]
    y = train['is_fraud'].values

    print("\n[2/5] Extracting digit keys for FLOATS6...")
    train_keys = extract_digit_keys(train)
    test_keys = extract_digit_keys(test)
    te_cols = [f'dte_{c}' for c in train_keys.columns]
    print(f"Extracted {train_keys.shape[1]} digit keys.")

    seeds = [42, 2026]
    n_splits = 5

    oof_total = np.zeros(len(train))
    test_total = np.zeros(len(test))

    import xgboost as xgb

    print(f"\n[3/5] Starting 2-Seed ({seeds}) x 5-Fold XGBoost Depth 2 GPU training...")

    for s_idx, seed in enumerate(seeds, 1):
        print(f"\n==================== SEED {seed} ({s_idx}/{len(seeds)}) ====================")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_seed = np.zeros(len(train))
        test_seed = np.zeros(len(test))

        for fold, (trn_idx, val_idx) in enumerate(skf.split(train, y), 1):
            print(f"\n--- Seed {seed} | FOLD {fold}/{n_splits} ---")

            # 1. Target Encoding fitted strictly inside training split
            enc = TargetEncoder(target_type='binary', cv=5, smooth=20.0, random_state=seed + fold)
            te_trn = enc.fit_transform(train_keys.iloc[trn_idx], y[trn_idx])
            te_val = enc.transform(train_keys.iloc[val_idx])
            te_tst = enc.transform(test_keys)

            df_te_trn = pd.DataFrame(te_trn, columns=te_cols, index=train.iloc[trn_idx].index)
            df_te_val = pd.DataFrame(te_val, columns=te_cols, index=train.iloc[val_idx].index)
            df_te_tst = pd.DataFrame(te_tst, columns=te_cols, index=test.index)

            # Combine base features + target encoded digit keys
            X_tr = pd.concat([train.iloc[trn_idx][base_features], df_te_trn], axis=1)
            X_va = pd.concat([train.iloc[val_idx][base_features], df_te_val], axis=1)
            X_te = pd.concat([test[base_features], df_te_tst], axis=1)

            # 2. XGBoost Depth 2 GPU
            model_xgb = xgb.XGBClassifier(
                n_estimators=6000,
                learning_rate=0.03,
                max_depth=2,              # 4-leaf equivalent
                min_child_weight=5.0,
                subsample=0.8,
                colsample_bytree=0.5,      # Half-feature subsampling
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method='hist',
                device='cuda',
                eval_metric='auc',
                enable_categorical=True,
                random_state=seed + fold,
                early_stopping_rounds=150
            )

            model_xgb.fit(X_tr, y[trn_idx], eval_set=[(X_va, y[val_idx])], verbose=False)

            val_p = model_xgb.predict_proba(X_va)[:, 1]
            oof_seed[val_idx] = val_p
            test_seed += model_xgb.predict_proba(X_te)[:, 1] / n_splits

            fold_auc = roc_auc_score(y[val_idx], val_p)
            print(f"  XGBoost Depth 2 Fold {fold} AUC: {fold_auc:.5f} (iter: {model_xgb.best_iteration})")

        seed_auc = roc_auc_score(y, oof_seed)
        print(f"\n>> Seed {seed} Single OOF ROC-AUC: {seed_auc:.5f}")

        oof_total += oof_seed / len(seeds)
        test_total += test_seed / len(seeds)

    # [4/5] 2-Seed Averaged Evaluation
    overall_auc = roc_auc_score(y, oof_total)
    print("\n" + "=" * 60)
    print(f"[4/5] Exp #019 2-Seed Averaged XGBoost Depth 2 OOF ROC-AUC: {overall_auc:.5f}")
    print(f"  (Reference Exp #017 LightGBM Digit-TE: 0.94930 | Diff: {overall_auc - 0.94930:+.5f})")
    print(f"  (Reference Exp #012 XGBoost depth 3:   0.94768 | Gain: {overall_auc - 0.94768:+.5f})")
    print("=" * 60)

    # [5/5] Save Files & Direct Kaggle Submit
    sub_df = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_total
    })
    out_file = "submission_exp019_colab.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\n[5/5] Saved submission to {out_file} (Rows: {len(sub_df)})")
    print(sub_df.head())

    oof_df = pd.DataFrame({
        'id': train['id'],
        'oof_xgb_d2_te': oof_total
    })
    oof_file = "oof_exp019_colab.csv"
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
    sub_msg = f"Exp019: Colab L4 GPU XGBoost Depth 2 2-Seeds Digit-TE (Val AUC {overall_auc:.5f})"
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
