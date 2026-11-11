import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def create_base_features(df):
    """Base features from Exp003 that proved highly effective."""
    df = df.copy()
    
    # Digit & Amount
    df['amount_cents'] = np.round((df['amount_usd'] - np.floor(df['amount_usd'])) * 100).astype(int)
    df['amount_is_round'] = (df['amount_cents'] == 0).astype(int)
    df['amount_log1p'] = np.log1p(np.maximum(df['amount_usd'], 0))
    
    # Ratios
    df['amount_to_balance_ratio'] = df['amount_usd'] / (np.maximum(df['account_balance_usd'], 0) + 1.0)
    df['velocity_per_txn'] = df['velocity_score'] / (df['txn_count_last_24h'] + 1.0)
    
    # Risk flags aggregation
    risk_bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    df['risk_flag_count'] = df[risk_bool_cols].sum(axis=1)
    
    # High-Gain Interactions
    df['cvv_plus_risk_count'] = df['cvv_retry_count'] + df['risk_flag_count']
    df['velocity_x_merchant_risk'] = df['velocity_score'] * df['merchant_risk_score']
    
    # Category Interactions for Target Encoding
    df['cat_x_auth'] = df['merchant_category'].astype(str) + "_" + df['auth_method'].astype(str)
    df['channel_x_device'] = df['channel'].astype(str) + "_" + df['device_type'].astype(str)
    df['merchant_x_card'] = df['merchant_category'].astype(str) + "_" + df['card_type'].astype(str)
    
    return df

def target_encode(train_s, y_train, val_s, test_s, m=20.0):
    """
    Smoothed target encoding without leakage:
    TE(c) = (count_c * mean_c + m * global_mean) / (count_c + m)
    """
    global_mean = y_train.mean()
    stats = pd.DataFrame({'cat': train_s, 'target': y_train}).groupby('cat')['target'].agg(['count', 'mean'])
    stats['smooth'] = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    mapping = stats['smooth'].to_dict()
    
    val_encoded = val_s.map(mapping).fillna(global_mean).values
    test_encoded = test_s.map(mapping).fillna(global_mean).values
    train_encoded = train_s.map(mapping).fillna(global_mean).values
    
    return train_encoded, val_encoded, test_encoded

def main():
    train_path = "004_credit-card-fraud/data/train.csv"
    test_path = "004_credit-card-fraud/data/test.csv"
    output_dir = "004_credit-card-fraud/data"
    
    print("[1/5] Loading data and engineering base features...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Preprocessing bools
    bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    for c in bool_cols:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)
        
    train = create_base_features(train)
    test = create_base_features(test)
    
    target_col = "is_fraud"
    y = train[target_col].astype(int)
    
    # Target encoding candidates
    te_cols = [
        'merchant_category', 'card_type', 'auth_method', 'channel', 
        'device_type', 'cvv_retry_count',
        'cat_x_auth', 'channel_x_device', 'merchant_x_card'
    ]
    
    # Base categorical columns for LightGBM
    cat_cols = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']
    for c in cat_cols:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
        
    # Drop intermediate interaction strings before GBDT
    intermediate_strings = ['cat_x_auth', 'channel_x_device', 'merchant_x_card']
    
    drop_cols = ["id", target_col] + intermediate_strings
    base_features = [c for c in train.columns if c not in drop_cols]
    
    print(f"Base Features ({len(base_features)}): {base_features}")
    print(f"Target Encoding Columns ({len(te_cols)}): {te_cols}")
    
    # Stratified 5-Fold CV
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train))
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
    
    print(f"\n[2/5] Starting 5-Fold CV with Nested Out-of-Fold Target Encoding...")
    feature_importances = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, y), 1):
        X_tr = train.iloc[train_idx][base_features].copy()
        y_tr = y.iloc[train_idx]
        X_va = train.iloc[val_idx][base_features].copy()
        y_va = y.iloc[val_idx]
        X_te = test[base_features].copy()
        
        # Apply nested target encoding for each fold
        for col in te_cols:
            tr_s = train.iloc[train_idx][col]
            va_s = train.iloc[val_idx][col]
            te_s = test[col]
            
            _, va_enc, test_enc = target_encode(tr_s, y_tr, va_s, te_s, m=20.0)
            X_tr[f'{col}_te'] = _
            X_va[f'{col}_te'] = va_enc
            X_te[f'{col}_te'] = test_enc
            
        if feature_importances is None:
            feature_importances = pd.DataFrame(index=X_tr.columns)
            
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        
        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=3000,
            valid_sets=[trn_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )
        
        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_va, val_pred)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.5f} (Best Iteration: {model.best_iteration})")
        
        feature_importances[f'fold_{fold}'] = model.feature_importance(importance_type='gain')
        test_preds += model.predict(X_te, num_iteration=model.best_iteration) / n_splits
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n[3/5] Validation Results:")
    print(f"  Fold AUCs: {[round(a, 5) for a in fold_aucs]}")
    print(f"  Exp #005 Overall OOF ROC-AUC: {overall_auc:.5f}")
    print(f"  (Reference Exp #003: 0.94717 | Diff: {overall_auc - 0.94717:+.5f})")
    print(f"  (Reference Exp #002: 0.94727 | Diff: {overall_auc - 0.94727:+.5f})")
    print(f"  (Reference Exp #001: 0.94652 | Diff: {overall_auc - 0.94652:+.5f})")
    
    # Feature Importance Top 15
    feature_importances['mean_gain'] = feature_importances.mean(axis=1)
    top_features = feature_importances.sort_values(by='mean_gain', ascending=False)
    print("\nTop 15 Features by Gain:")
    for rank, (feat, row) in enumerate(top_features.head(15).iterrows(), 1):
        print(f"  {rank:2d}. {feat:<32}: {row['mean_gain']:.1f}")
        
    # Check predictions summary
    print(f"\n[4/5] Prediction Distribution Check:")
    print(f"  OOF Pred Mean: {oof_preds.mean():.5f}, Std: {oof_preds.std():.5f}, Min: {oof_preds.min():.5f}, Max: {oof_preds.max():.5f}")
    print(f"  Test Pred Mean: {test_preds.mean():.5f}, Std: {test_preds.std():.5f}, Min: {test_preds.min():.5f}, Max: {test_preds.max():.5f}")
    
    # Save OOF and Submission
    oof_file = os.path.join(output_dir, "oof_exp005.npy")
    np.save(oof_file, oof_preds)
    print(f"OOF predictions saved to {oof_file}")
    
    sub = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_preds
    })
    sub_file = os.path.join(output_dir, "submission_exp005.csv")
    sub.to_csv(sub_file, index=False)
    print(f"\n[5/5] Submission saved to {sub_file} (Rows: {len(sub)})")
    print(sub.head())

if __name__ == "__main__":
    main()
