"""
004_credit-card-fraud: Exp #015 - Colab L4 GPU 3-GBDT with Original Dataset (2026) Integration
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Key Upgrades:
  1. Original Dataset (20,000 rows from uditjain13/credit-card-fraud-detection-2026) added to X_tr for all 3 GBDTs.
  2. Pure Out-of-Fold Validation on competition train.csv (zero leakage, fair comparison).
  3. 2-Seed Averaging (Seeds: 42, 2026) for variance reduction across 30 GPU training jobs.
  4. 3 Diverse GBDTs: LightGBM (4 leaves) + CatBoost GPU (depth 3) + XGBoost GPU (depth 3).
  5. Learning Rate 0.025 with up to 6,000 boosting rounds + L1/L2 regularization.
  6. Base64 recovery for BOTH submission CSV AND OOF predictions (enabling downstream logit blending).
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def ensure_environment_and_data():
    """Verify packages, Kaggle credentials, and datasets on Colab VM."""
    print("[INIT] Verifying packages and datasets...")
    
    # 1. Packages
    try:
        import lightgbm
        import catboost
        import xgboost
    except ImportError:
        print("  - Installing missing ML packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "catboost", "xgboost", "scikit-learn", "kaggle"], check=True)

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
        print("  - Downloading competition dataset from Kaggle...")
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "playground-series-s6e11", "-p", data_dir], check=True)
        for zip_file in glob.glob(os.path.join(data_dir, "*.zip")):
            subprocess.run(["unzip", "-q", "-o", zip_file, "-d", data_dir], check=True)

    # 4. Original Dataset files
    orig_dir = os.path.join(data_dir, "original")
    orig_file = os.path.join(orig_dir, "credit_card_fraud_2026.csv")
    if not os.path.exists(orig_file):
        print("  - Downloading original dataset (credit-card-fraud-detection-2026)...")
        os.makedirs(orig_dir, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", "uditjain13/credit-card-fraud-detection-2026", "-p", orig_dir], check=True)
        for zip_file in glob.glob(os.path.join(orig_dir, "*.zip")):
            subprocess.run(["unzip", "-q", "-o", zip_file, "-d", orig_dir], check=True)

    print(f"  - Dataset ready at: {data_dir} (Original: {orig_file})")
    return data_dir, orig_file


def create_features(df):
    """36 Engineered features from Exp003."""
    df = df.copy()
    
    # 1. Digit Keys & Amount Features
    df['amount_cents'] = np.round((df['amount_usd'] - np.floor(df['amount_usd'])) * 100).astype(int)
    df['amount_is_round'] = (df['amount_cents'] == 0).astype(int)
    df['amount_log1p'] = np.log1p(np.maximum(df['amount_usd'], 0))
    df['amount_int_mod10'] = (np.floor(df['amount_usd']).astype(int) % 10)
    
    # 2. Account & Velocity Ratios
    df['amount_to_balance_ratio'] = df['amount_usd'] / (np.maximum(df['account_balance_usd'], 0) + 1.0)
    df['velocity_per_txn'] = df['velocity_score'] / (df['txn_count_last_24h'] + 1.0)
    df['hours_per_card_age'] = df['hours_since_last_txn'] / (df['card_age_months'] + 1.0)
    
    # 3. Rare Risk Trigger Aggregation
    risk_bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    df['risk_flag_count'] = df[risk_bool_cols].sum(axis=1)
    df['is_multi_risk'] = (df['risk_flag_count'] >= 2).astype(int)
    
    # 4. High-Gain Feature Interactions
    df['cvv_is_nonzero'] = (df['cvv_retry_count'] > 0).astype(int)
    df['cvv_plus_risk_count'] = df['cvv_retry_count'] + df['risk_flag_count']
    df['velocity_x_merchant_risk'] = df['velocity_score'] * df['merchant_risk_score']
    
    return df


def main():
    print("=" * 80)
    print("  004 Credit Card Fraud: Exp #015 - Colab L4 GPU 3-GBDT with Original Dataset")
    print("=" * 80)

    data_dir, orig_file = ensure_environment_and_data()

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    print("\n[1/5] Loading data and engineering features...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_file)
    if 'transaction_id' in orig.columns:
        orig = orig.rename(columns={'transaction_id': 'id'})

    print(f"Raw shapes -> Train: {train.shape}, Test: {test.shape}, Original: {orig.shape}")

    bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    for c in bool_cols:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)
        orig[c] = orig[c].astype(int)

    train = create_features(train)
    test = create_features(test)
    orig = create_features(orig)

    target_col = "is_fraud"
    drop_cols = ["id", target_col]
    features = [c for c in train.columns if c not in drop_cols]
    print(f"Engineered features ({len(features)}): {features}")

    cat_cols = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']
    
    for c in cat_cols:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
        orig[c] = orig[c].astype("category")

    X = train[features]
    y = train[target_col].astype(int)
    X_test = test[features]

    X_orig = orig[features]
    y_orig = orig[target_col].astype(int)

    # For CatBoost: needs string categories
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    X_orig_cb = X_orig.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)
        X_orig_cb[c] = X_orig_cb[c].astype(str)

    seeds = [42, 2026]
    n_splits = 5

    oof_lgb_total = np.zeros(len(train))
    oof_cat_total = np.zeros(len(train))
    oof_xgb_total = np.zeros(len(train))

    test_lgb_total = np.zeros(len(test))
    test_cat_total = np.zeros(len(test))
    test_xgb_total = np.zeros(len(test))

    import lightgbm as lgb
    from catboost import CatBoostClassifier, Pool
    import xgboost as xgb

    print(f"\n[2/5] Starting 2-Seed ({seeds}) x 5-Fold Cross Validation with Original Data Added to Train...")

    for s_idx, seed in enumerate(seeds, 1):
        print(f"\n==================== SEED {seed} ({s_idx}/{len(seeds)}) ====================")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_lgb_seed = np.zeros(len(train))
        oof_cat_seed = np.zeros(len(train))
        oof_xgb_seed = np.zeros(len(train))

        test_lgb_seed = np.zeros(len(test))
        test_cat_seed = np.zeros(len(test))
        test_xgb_seed = np.zeros(len(test))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n--- Seed {seed} | FOLD {fold}/{n_splits} ---")
            
            # Combine Train Fold + Original Data (20,000 rows)
            X_tr = pd.concat([X.iloc[train_idx], X_orig], ignore_index=True)
            y_tr = pd.concat([y.iloc[train_idx], y_orig], ignore_index=True)
            
            # Val is pure competition train fold (zero leakage)
            X_va = X.iloc[val_idx]
            y_va = y.iloc[val_idx]

            # ----------------- 1. LightGBM (Tuned + Orig Data) -----------------
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.025,
                'num_leaves': 4,
                'max_depth': 3,
                'min_child_samples': 150,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': seed + fold,
                'n_jobs': -1,
                'verbose': -1
            }
            trn_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
            model_lgb = lgb.train(
                lgb_params,
                trn_data,
                num_boost_round=6000,
                valid_sets=[trn_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
            )
            val_pred_lgb = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
            oof_lgb_seed[val_idx] = val_pred_lgb
            test_lgb_seed += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
            print(f"  LGBM Fold {fold} AUC: {roc_auc_score(y_va, val_pred_lgb):.5f} (iter: {model_lgb.best_iteration})")

            # ----------------- 2. CatBoost GPU (Tuned + Orig Data) -----------------
            X_tr_cb = pd.concat([X_cb.iloc[train_idx], X_orig_cb], ignore_index=True)
            X_va_cb = X_cb.iloc[val_idx]
            cb_pool_tr = Pool(X_tr_cb, y_tr, cat_features=cat_cols)
            cb_pool_va = Pool(X_va_cb, y_va, cat_features=cat_cols)
            cb_pool_te = Pool(X_test_cb, cat_features=cat_cols)

            model_cat = CatBoostClassifier(
                iterations=6000,
                learning_rate=0.025,
                depth=3,
                l2_leaf_reg=3.0,
                loss_function='Logloss',
                eval_metric='AUC',
                task_type='GPU',
                random_seed=seed + fold,
                early_stopping_rounds=150,
                verbose=False
            )
            model_cat.fit(cb_pool_tr, eval_set=cb_pool_va, verbose=False)
            val_pred_cat = model_cat.predict_proba(cb_pool_va)[:, 1]
            oof_cat_seed[val_idx] = val_pred_cat
            test_cat_seed += model_cat.predict_proba(cb_pool_te)[:, 1] / n_splits
            print(f"  CatBoost GPU Fold {fold} AUC: {roc_auc_score(y_va, val_pred_cat):.5f} (iter: {model_cat.get_best_iteration()})")

            # ----------------- 3. XGBoost GPU (Tuned + Orig Data) -----------------
            model_xgb = xgb.XGBClassifier(
                n_estimators=6000,
                learning_rate=0.025,
                max_depth=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=5.0,
                tree_method='hist',
                device='cuda',
                eval_metric='auc',
                subsample=0.8,
                colsample_bytree=0.8,
                enable_categorical=True,
                random_state=seed + fold,
                early_stopping_rounds=150
            )
            model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            val_pred_xgb = model_xgb.predict_proba(X_va)[:, 1]
            oof_xgb_seed[val_idx] = val_pred_xgb
            test_xgb_seed += model_xgb.predict_proba(X_test)[:, 1] / n_splits
            print(f"  XGBoost GPU Fold {fold} AUC: {roc_auc_score(y_va, val_pred_xgb):.5f} (iter: {model_xgb.best_iteration})")

        print(f"\n>> Seed {seed} Single OOF Scores (with Orig Data):")
        print(f"  LGBM: {roc_auc_score(y, oof_lgb_seed):.5f}")
        print(f"  CAT:  {roc_auc_score(y, oof_cat_seed):.5f}")
        print(f"  XGB:  {roc_auc_score(y, oof_xgb_seed):.5f}")

        oof_lgb_total += oof_lgb_seed / len(seeds)
        oof_cat_total += oof_cat_seed / len(seeds)
        oof_xgb_total += oof_xgb_seed / len(seeds)

        test_lgb_total += test_lgb_seed / len(seeds)
        test_cat_total += test_cat_seed / len(seeds)
        test_xgb_total += test_xgb_seed / len(seeds)

    # [3/5] Evaluation & Optimization on 2-Seed Averaged OOF
    print("\n" + "=" * 60)
    print("[3/5] 2-Seed Averaged OOF Validation Results (with Orig Data):")
    auc_total_lgb = roc_auc_score(y, oof_lgb_total)
    auc_total_cat = roc_auc_score(y, oof_cat_total)
    auc_total_xgb = roc_auc_score(y, oof_xgb_total)

    print(f"  1. LightGBM (2-Seed Averaged) OOF ROC-AUC:     {auc_total_lgb:.5f}")
    print(f"  2. CatBoost GPU (2-Seed Averaged) OOF ROC-AUC: {auc_total_cat:.5f}")
    print(f"  3. XGBoost GPU (2-Seed Averaged) OOF ROC-AUC:  {auc_total_xgb:.5f}")

    # Optimize Ensemble Weights
    best_auc = 0.0
    best_weights = (0.4, 0.2, 0.4)
    
    for w_lgb in np.linspace(0.2, 0.6, 9):
        for w_cat in np.linspace(0.1, 0.4, 7):
            w_xgb = 1.0 - (w_lgb + w_cat)
            if w_xgb < 0.1 or w_xgb > 0.6:
                continue
            oof_blend = w_lgb * oof_lgb_total + w_cat * oof_cat_total + w_xgb * oof_xgb_total
            score = roc_auc_score(y, oof_blend)
            if score > best_auc:
                best_auc = score
                best_weights = (w_lgb, w_cat, w_xgb)

    w1, w2, w3 = best_weights
    oof_final = w1 * oof_lgb_total + w2 * oof_cat_total + w3 * oof_xgb_total
    test_final = w1 * test_lgb_total + w2 * test_cat_total + w3 * test_xgb_total

    print(f"  -> Optimal Weights: LGBM {w1:.2f} + CatBoost {w2:.2f} + XGBoost {w3:.2f}")
    print(f"  -> Exp #015 3-GBDT Ensemble OOF ROC-AUC: {best_auc:.5f} (Reference Exp012: 0.94771)")
    print("=" * 60)

    # [4/5] Save Submission & OOF Files
    sub_df = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_final
    })
    out_file = "submission_exp015_colab.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\n[4/5] Saved submission to {out_file} (Rows: {len(sub_df)})")
    print(sub_df.head())

    # Save OOF DataFrame for downstream blending
    oof_df = pd.DataFrame({
        'id': train['id'],
        'oof_lgb': oof_lgb_total,
        'oof_cat': oof_cat_total,
        'oof_xgb': oof_xgb_total,
        'oof_blend': oof_final
    })
    oof_file = "oof_exp015_colab.csv"
    oof_df.to_csv(oof_file, index=False)
    print(f"  Saved OOF to {oof_file} (Rows: {len(oof_df)})")

    # Google Drive Sync
    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        dest_sub = os.path.join(drive_dir, out_file)
        dest_oof = os.path.join(drive_dir, oof_file)
        sub_df.to_csv(dest_sub, index=False)
        oof_df.to_csv(dest_oof, index=False)
        print(f"[DRIVE] Automatically saved to Google Drive: {dest_sub} & {dest_oof}")

    # [5/5] Direct Kaggle Submission & Base64 Fallback
    print("\n[5/5] Submitting to Kaggle CLI from Colab...")
    sub_msg = f"Exp015: Colab L4 GPU Orig-Data 2-Seeds 3-GBDT (Val AUC {best_auc:.5f}, LGB {w1:.2f}+CAT {w2:.2f}+XGB {w3:.2f})"
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
