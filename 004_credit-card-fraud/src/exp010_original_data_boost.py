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

def create_features(df):
    """36 Engineered features from Exp003/Exp008."""
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
    test_path = "004_credit-card-fraud/data/test.csv"
    orig_path = "004_credit-card-fraud/data/original/credit_card_fraud_2026.csv"
    output_dir = "004_credit-card-fraud/data"
    
    print("[1/5] Loading competition and original datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_path)
    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Original shape: {orig.shape}")
    
    # Align original columns (transaction_id -> id)
    if 'transaction_id' in orig.columns:
        orig = orig.rename(columns={'transaction_id': 'id'})
        
    bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    for c in bool_cols:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)
        orig[c] = orig[c].astype(int)
        
    # Apply feature engineering
    train = create_features(train)
    test = create_features(test)
    orig = create_features(orig)
    
    target_col = "is_fraud"
    drop_cols = ["id", target_col]
    features = [c for c in train.columns if c not in drop_cols]
    
    cat_cols = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']
    num_cols = [c for c in features if c not in cat_cols]
    
    # Categories for LightGBM
    train_lgb = train.copy()
    test_lgb = test.copy()
    orig_lgb = orig.copy()
    for c in cat_cols:
        train_lgb[c] = train_lgb[c].astype("category")
        test_lgb[c] = test_lgb[c].astype("category")
        orig_lgb[c] = orig_lgb[c].astype("category")
        
    y = train[target_col].astype(int)
    
    # Stratified 5-Fold CV on Competition Train Data Only (Prevent validation leakage)
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_boosted = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs = []
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 4,
        'max_depth': 3,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    print(f"\n[2/5] Starting 5-Fold Cross Validation with Original Data Added to Training Folds...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        # Training data: Train Fold + Original Data (20,000 rows)
        X_tr_comp = train.iloc[train_idx][features]
        y_tr_comp = y.iloc[train_idx]
        
        X_tr_full = pd.concat([X_tr_comp, orig[features]], ignore_index=True)
        y_tr_full = pd.concat([y_tr_comp, orig[target_col].astype(int)], ignore_index=True)
        
        # Validation data: Strictly Competition Fold
        X_va = train.iloc[val_idx][features]
        y_va = y.iloc[val_idx]
        X_te = test[features]
        
        # Step 1: Fit GLM on (Train Fold + Original Data)
        print(f"  [Step 1] Fitting GLM on {len(X_tr_full)} rows (Train Fold: {len(X_tr_comp)} + Original: {len(orig)})...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )
        
        X_tr_proc = preprocessor.fit_transform(X_tr_full)
        X_va_proc = preprocessor.transform(X_va)
        X_te_proc = preprocessor.transform(X_te)
        
        lr_model = LogisticRegression(C=0.1, max_iter=500, solver='lbfgs', random_state=42)
        lr_model.fit(X_tr_proc, y_tr_full)
        
        logit_tr = lr_model.decision_function(X_tr_proc)
        logit_va = lr_model.decision_function(X_va_proc)
        logit_te = lr_model.decision_function(X_te_proc)
        
        lr_fold_auc = roc_auc_score(y_va, logit_va)
        print(f"    -> GLM Fold {fold} AUC on Val: {lr_fold_auc:.5f}")
        
        # Step 2: LightGBM Boosting from GLM Logits
        print("  [Step 2] Boosting from GLM Logits with Shallow LightGBM...")
        X_tr_lgb_full = pd.concat([train_lgb.iloc[train_idx][features], orig_lgb[features]], ignore_index=True)
        X_va_lgb = train_lgb.iloc[val_idx][features]
        X_te_lgb = test_lgb[features]
        
        trn_data = lgb.Dataset(X_tr_lgb_full, label=y_tr_full, init_score=logit_tr)
        val_data = lgb.Dataset(X_va_lgb, label=y_va, reference=trn_data, init_score=logit_va)
        
        model_lgb = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=3000,
            valid_sets=[trn_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )
        
        val_raw = model_lgb.predict(X_va_lgb, raw_score=True)
        val_pred = expit(logit_va + val_raw)
        oof_boosted[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_va, val_pred)
        fold_aucs.append(fold_auc)
        print(f"    -> Boosted Fold {fold} AUC on Val: {fold_auc:.5f} (Best Iter: {model_lgb.best_iteration})")
        
        test_raw = model_lgb.predict(X_te_lgb, raw_score=True)
        test_preds += expit(logit_te + test_raw) / n_splits
        
    overall_auc = roc_auc_score(y, oof_boosted)
    print("\n" + "=" * 60)
    print("[3/5] Validation Results:")
    print(f"  Fold AUCs: {[round(a, 5) for a in fold_aucs]}")
    print(f"  Exp #010 (Original Data + GLM Boosting) OOF ROC-AUC: {overall_auc:.5f}")
    print(f"  (Reference Exp #008 without Orig Data: 0.94766 | Diff: {overall_auc - 0.94766:+.5f})")
    print("=" * 60)
    
    # [4/5] Prediction Distribution Check
    print(f"\n[4/5] Prediction Distribution Check:")
    print(f"  OOF Pred Mean: {oof_boosted.mean():.5f}, Std: {oof_boosted.std():.5f}, Min: {oof_boosted.min():.5f}, Max: {oof_boosted.max():.5f}")
    print(f"  Test Pred Mean: {test_preds.mean():.5f}, Std: {test_preds.std():.5f}, Min: {test_preds.min():.5f}, Max: {test_preds.max():.5f}")
    
    # Save submission
    sub = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_preds
    })
    sub_file = os.path.join(output_dir, "submission_exp010.csv")
    sub.to_csv(sub_file, index=False)
    print(f"\n[5/5] Submission saved to {sub_file} (Rows: {len(sub)})")
    print(sub.head())

if __name__ == "__main__":
    main()
