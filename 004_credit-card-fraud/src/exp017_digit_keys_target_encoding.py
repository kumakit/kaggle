"""
Exp #017: FLOATS6 Digit Keys with In-Fold Target Encoding (The 0.950+ Breakthrough)
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Features:
  - 6 Core Continuous Variables (FLOATS6):
    ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']
  - Digit Keys Extracted for each:
    * cents: rounded cents (mod 100)
    * int: integer floor
    * ten: integer / 10
    * dec: tenths (x * 10)
    * lead1: leading 1 digit
    * lead2: leading 2 digits
    * len: length of integer string
  - Nested In-Fold Target Encoding:
    sklearn.preprocessing.TargetEncoder(target_type='binary', cv=5, smooth=20.0)
    Fitted strictly inside training split of each fold (zero leakage).
  - Model: Shallow LightGBM (num_leaves: 4, colsample_bytree: 0.5, lr: 0.03, min_child_samples: 100)
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

FLOATS6 = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']

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
    train_path = "004_credit-card-fraud/data/train.csv"
    test_path = "004_credit-card-fraud/data/test.csv"
    output_dir = "004_credit-card-fraud/data"
    
    print("[1/4] Loading datasets...")
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
        
    y = train['is_fraud'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n[2/4] Extracting digit keys for FLOATS6...")
    train_keys = extract_digit_keys(train)
    test_keys = extract_digit_keys(test)
    print(f"Extracted {train_keys.shape[1]} digit key columns: {list(train_keys.columns[:6])} ...")
    
    # Base features to keep
    base_features = [c for c in train.columns if c not in ['id', 'is_fraud']]
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs = []
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 4,
        'max_depth': 3,
        'min_child_samples': 100,
        'colsample_bytree': 0.5,
        'subsample': 0.8,
        'subsample_freq': 1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    print("\n[3/4] Starting 5-Fold In-Fold Target Encoding & LightGBM training...")
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train, y), 1):
        # 1. Target Encoding fitted strictly on fold training data
        enc = TargetEncoder(target_type='binary', cv=5, smooth=20.0, random_state=42 + fold)
        
        te_train = enc.fit_transform(train_keys.iloc[trn_idx], y[trn_idx])
        te_val = enc.transform(train_keys.iloc[val_idx])
        te_test = enc.transform(test_keys)
        
        te_cols = [f'dte_{c}' for c in train_keys.columns]
        df_te_tr = pd.DataFrame(te_train, columns=te_cols, index=train.iloc[trn_idx].index)
        df_te_va = pd.DataFrame(te_val, columns=te_cols, index=train.iloc[val_idx].index)
        df_te_te = pd.DataFrame(te_test, columns=te_cols, index=test.index)
        
        # 2. Combine base features + target encoded digit keys
        X_tr = pd.concat([train.iloc[trn_idx][base_features], df_te_tr], axis=1)
        X_va = pd.concat([train.iloc[val_idx][base_features], df_te_va], axis=1)
        X_te = pd.concat([test[base_features], df_te_te], axis=1)
        
        trn_data = lgb.Dataset(X_tr, label=y[trn_idx])
        val_data = lgb.Dataset(X_va, label=y[val_idx], reference=trn_data)
        
        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=6000,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
        )
        
        val_p = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_p
        test_preds += model.predict(X_te, num_iteration=model.best_iteration) / 5
        
        fold_auc = roc_auc_score(y[val_idx], val_p)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.5f} (Best Iter: {model.best_iteration})")
        
    overall_auc = roc_auc_score(y, oof_preds)
    print("\n" + "=" * 60)
    print(f"[4/4] Exp #017 Digit-TE LightGBM 5-Fold OOF AUC: {overall_auc:.5f}")
    print(f"  (Reference Exp #002 Shallow Tree: 0.94724 | Gain: {overall_auc - 0.94724:+.5f})")
    print(f"  (Reference Exp #016 Blend:        0.94773 | Gain: {overall_auc - 0.94773:+.5f})")
    print("=" * 60)
    
    # Save OOF
    np.save(os.path.join(output_dir, "oof_exp017.npy"), oof_preds)
    
    # Save Submission
    sub = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_preds
    })
    sub_file = os.path.join(output_dir, "submission_exp017.csv")
    sub.to_csv(sub_file, index=False)
    print(f"Saved submission to {sub_file} (Rows: {len(sub)})")
    print(sub.head())

if __name__ == "__main__":
    main()
