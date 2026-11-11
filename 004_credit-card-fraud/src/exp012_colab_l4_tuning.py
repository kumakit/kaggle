"""
004_credit-card-fraud: Exp #012 - Colab L4 GPU Hyperparameter Fine-Tuning & 2-Seed 3-GBDT
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Enhancements:
  - Fine-Tuned Regularized Shallow Trees (depth 3 / num_leaves 4)
  - Lower Learning Rate (lr=0.025) with up to 6,000 boosting rounds
  - L1/L2 Regularization (reg_alpha=0.1, reg_lambda=1.0, min_child_samples=150)
  - 2-Seed Averaging (Seeds: 42, 2026) for variance reduction
  - 3 Diverse GBDT Models (LightGBM + CatBoost GPU + XGBoost GPU)
  - Stratified 5-Fold Cross-Validation
  - Full 36 engineered features (cvv_plus_risk_count, velocity_x_merchant_risk, Digit Keys, Ratios)
  - Direct Kaggle CLI submission & Base64 console output
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
    """Verify packages, Kaggle credentials, and dataset on Colab VM."""
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

    # 3. Data files
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        candidates = ["/content/data", "004_credit-card-fraud/data", "/content/kaggle/004_credit-card-fraud/data"]
        for c in candidates:
            if os.path.exists(os.path.join(c, "train.csv")):
                data_dir = c
                break

    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("  - Downloading dataset from Kaggle...")
        os.makedirs("data", exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "playground-series-s6e11", "-p", "data"], check=True)
        for zip_file in glob.glob("data/*.zip"):
            subprocess.run(["unzip", "-q", "-o", zip_file, "-d", "data"], check=True)
        data_dir = "data"

    print(f"  - Dataset ready at: {data_dir}")
    return data_dir


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
    print("  004 Credit Card Fraud: Exp #012 - Colab L4 GPU Tuning (lr=0.025, 2-Seeds)")
    print("=" * 80)

    data_dir = ensure_environment_and_data()

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    print("\n[1/5] Loading data and engineering features...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Raw shapes -> Train: {train.shape}, Test: {test.shape}")

    bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    for c in bool_cols:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)

    train = create_features(train)
    test = create_features(test)

    target_col = "is_fraud"
    drop_cols = ["id", target_col]
    features = [c for c in train.columns if c not in drop_cols]
    print(f"Engineered features ({len(features)}): {features}")

    cat_cols = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']
    
    for c in cat_cols:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")

    X = train[features]
    y = train[target_col].astype(int)
    X_test = test[features]

    # For CatBoost: needs string categories
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

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

    print(f"\n[2/5] Starting 2-Seed ({seeds}) x 5-Fold Cross Validation (lr=0.025, max_iter=6000)...")

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
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            # ----------------- 1. LightGBM (Tuned) -----------------
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

            # ----------------- 2. CatBoost GPU (Tuned) -----------------
            X_tr_cb = X_cb.iloc[train_idx]
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

            # ----------------- 3. XGBoost GPU (Tuned) -----------------
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

        print(f"\n>> Seed {seed} Single OOF Scores:")
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
    print("[3/5] 2-Seed Averaged OOF Validation Results:")
    auc_total_lgb = roc_auc_score(y, oof_lgb_total)
    auc_total_cat = roc_auc_score(y, oof_cat_total)
    auc_total_xgb = roc_auc_score(y, oof_xgb_total)

    print(f"  1. LightGBM (2-Seed Averaged) OOF ROC-AUC:   {auc_total_lgb:.5f}")
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
    print(f"  -> Tuned Ensemble OOF ROC-AUC: {best_auc:.5f} (Gain over Exp006 0.94736: {best_auc - 0.94736:+.5f})")
    print("=" * 60)

    # [4/5] Save Submission & Drive Backup
    sub_df = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_final
    })
    out_file = "submission_exp012_colab.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\n[4/5] Saved submission to {out_file} (Rows: {len(sub_df)})")
    print(sub_df.head())

    # Google Drive Sync
    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        dest = os.path.join(drive_dir, out_file)
        sub_df.to_csv(dest, index=False)
        print(f"[DRIVE] Automatically saved to Google Drive: {dest}")

    # [5/5] Direct Kaggle Submission & Base64 Fallback
    print("\n[5/5] Submitting to Kaggle CLI from Colab...")
    sub_msg = f"Exp012: Colab L4 GPU Tuned 2-Seeds 3-GBDT (Val AUC {best_auc:.5f}, LGB {w1:.2f}+CAT {w2:.2f}+XGB {w3:.2f})"
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

    # Base64 output for local recovery
    with open(out_file, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("utf-8")
    print("\n===BASE64_SUBMISSION_START===")
    print(b64_content)
    print("===BASE64_SUBMISSION_END===")


if __name__ == "__main__":
    main()
