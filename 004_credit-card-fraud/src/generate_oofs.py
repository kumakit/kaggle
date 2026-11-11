"""
Generate and save OOF predictions for Exp002, Exp003, Exp004, Exp008, Exp010.
This enables exact, mathematical ensemble optimization (Scipy / Nelder-Mead / Grid Search / Rank / Logit blends).
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

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
    train_path = "004_credit-card-fraud/data/train.csv"
    orig_path = "004_credit-card-fraud/data/original/credit_card_fraud_2026.csv"
    output_dir = "004_credit-card-fraud/data"
    
    print("[1/4] Loading train dataset...")
    train = pd.read_csv(train_path)
    orig = pd.read_csv(orig_path)
    if 'transaction_id' in orig.columns:
        orig = orig.rename(columns={'transaction_id': 'id'})
        
    bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    for c in bool_cols:
        train[c] = train[c].astype(int)
        orig[c] = orig[c].astype(int)
        
    y = train['is_fraud'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- MODEL 1: Exp002 (Shallow Trees, 4 leaves, raw 24 features) ---
    print("\n[2/4] Generating OOF for Exp #002 (Raw Features, 4-leaf LightGBM)...")
    cat_cols_raw = ['merchant_category', 'card_type', 'channel', 'device_type', 'auth_method']
    feature_cols_raw = [c for c in train.columns if c not in ['id', 'is_fraud']]
    
    X_raw = train[feature_cols_raw].copy()
    for c in cat_cols_raw:
        X_raw[c] = X_raw[c].astype('category')
        
    oof_exp002 = np.zeros(len(train))
    lgb_params_002 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 4,
        'max_depth': 3,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        X_tr, y_tr = X_raw.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X_raw.iloc[val_idx], y[val_idx]
        
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        
        m = lgb.train(
            lgb_params_002,
            trn_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        oof_exp002[val_idx] = m.predict(X_va)
        
    auc_002 = roc_auc_score(y, oof_exp002)
    print(f"  -> Exp #002 OOF ROC-AUC: {auc_002:.5f}")
    np.save(os.path.join(output_dir, "oof_exp002.npy"), oof_exp002)
    
    # --- MODEL 2: Exp003 (36 Engineered Features, 4-leaf LightGBM) ---
    print("\n[3/4] Generating OOF for Exp #003 (36 Engineered Features, 4-leaf LightGBM)...")
    train_eng = create_features(train)
    feature_cols_eng = [c for c in train_eng.columns if c not in ['id', 'is_fraud']]
    X_eng = train_eng[feature_cols_eng].copy()
    for c in cat_cols_raw:
        X_eng[c] = X_eng[c].astype('category')
        
    oof_exp003 = np.zeros(len(train))
    lgb_params_003 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 4,
        'max_depth': 3,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_eng, y), 1):
        X_tr, y_tr = X_eng.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X_eng.iloc[val_idx], y[val_idx]
        
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        
        m = lgb.train(
            lgb_params_003,
            trn_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        oof_exp003[val_idx] = m.predict(X_va)
        
    auc_003 = roc_auc_score(y, oof_exp003)
    print(f"  -> Exp #003 OOF ROC-AUC: {auc_003:.5f}")
    np.save(os.path.join(output_dir, "oof_exp003.npy"), oof_exp003)
    
    # Exp #004 OOF (40% Exp002 + 60% Exp003)
    oof_exp004 = 0.40 * oof_exp002 + 0.60 * oof_exp003
    auc_004 = roc_auc_score(y, oof_exp004)
    print(f"  -> Exp #004 (Blend 002+003) OOF ROC-AUC: {auc_004:.5f}")
    np.save(os.path.join(output_dir, "oof_exp004.npy"), oof_exp004)
    
    # --- MODEL 3: Exp010 (Original Data 20k + GLM Boosting, 4-leaf LightGBM) ---
    print("\n[4/4] Generating OOF for Exp #010 (Original Data + GLM Boosting)...")
    orig_eng = create_features(orig)
    
    # Preprocessor for GLM
    cat_cols_glm = ['merchant_category', 'card_type', 'channel', 'device_type', 'auth_method']
    num_cols_glm = [c for c in feature_cols_eng if c not in cat_cols_glm]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols_glm),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols_glm)
        ]
    )
    
    # LightGBM data
    X_lgb = train_eng[feature_cols_eng].copy()
    for c in cat_cols_glm:
        X_lgb[c] = X_lgb[c].astype('category')
        
    orig_lgb = orig_eng[feature_cols_eng].copy()
    for c in cat_cols_glm:
        orig_lgb[c] = orig_lgb[c].astype('category')
        
    oof_exp010 = np.zeros(len(train))
    lgb_params_010 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 4,
        'max_depth': 3,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_eng, y), 1):
        # 1. Prepare GLM data
        tr_df = train_eng.iloc[trn_idx]
        va_df = train_eng.iloc[val_idx]
        tr_combined = pd.concat([tr_df, orig_eng], ignore_index=True)
        
        y_tr_glm = tr_combined['is_fraud'].values
        y_va_glm = y[val_idx]
        
        X_tr_proc = preprocessor.fit_transform(tr_combined[feature_cols_eng])
        X_va_proc = preprocessor.transform(va_df[feature_cols_eng])
        
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_tr_proc, y_tr_glm)
        
        logit_tr = clf.decision_function(X_tr_proc)
        logit_va = clf.decision_function(X_va_proc)
        
        # 2. LightGBM with init_score
        X_tr_lgb = pd.concat([X_lgb.iloc[trn_idx], orig_lgb], ignore_index=True)
        X_va_lgb = X_lgb.iloc[val_idx]
        
        trn_data = lgb.Dataset(X_tr_lgb, label=y_tr_glm, init_score=logit_tr)
        val_data = lgb.Dataset(X_va_lgb, label=y_va_glm, init_score=logit_va, reference=trn_data)
        
        m = lgb.train(
            lgb_params_010,
            trn_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        val_raw = m.predict(X_va_lgb, raw_score=True)
        val_pred = expit(logit_va + val_raw)
        oof_exp010[val_idx] = val_pred
        
    auc_010 = roc_auc_score(y, oof_exp010)
    print(f"  -> Exp #010 OOF ROC-AUC: {auc_010:.5f}")
    np.save(os.path.join(output_dir, "oof_exp010.npy"), oof_exp010)
    
    print("\n" + "=" * 60)
    print("ALL OOFS GENERATED AND SAVED SUCCESSFULLY:")
    print(f"  Exp002 (4 leaves):              {auc_002:.5f}")
    print(f"  Exp003 (36 features):           {auc_003:.5f}")
    print(f"  Exp004 (Blend 002+003):         {auc_004:.5f}")
    print(f"  Exp010 (Orig Data + GLM Boost): {auc_010:.5f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
