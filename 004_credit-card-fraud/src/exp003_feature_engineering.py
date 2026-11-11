import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def create_features(df):
    """
    Exp003 Feature Engineering:
    - Digit Keys (cents, round amount flag, log amount, integer mod 10)
    - Account & Transaction Ratios
    - Risk Flag Aggregation (Rare trigger sums)
    - Interaction features around cvv_retry_count & merchant risk
    """
    df = df.copy()
    
    # 1. Digit Keys & Amount Features
    # Cents: fractional part as integer (e.g. 61.16 -> 16)
    df['amount_cents'] = np.round((df['amount_usd'] - np.floor(df['amount_usd'])) * 100).astype(int)
    # Round amount flag (.00 cents)
    df['amount_is_round'] = (df['amount_cents'] == 0).astype(int)
    # Amount log1p
    df['amount_log1p'] = np.log1p(np.maximum(df['amount_usd'], 0))
    # Integer part mod 10
    df['amount_int_mod10'] = (np.floor(df['amount_usd']).astype(int) % 10)
    
    # 2. Account & Velocity Ratios
    # Amount to balance ratio
    df['amount_to_balance_ratio'] = df['amount_usd'] / (np.maximum(df['account_balance_usd'], 0) + 1.0)
    # Velocity per transaction
    df['velocity_per_txn'] = df['velocity_score'] / (df['txn_count_last_24h'] + 1.0)
    # Hours since last txn per age
    df['hours_per_card_age'] = df['hours_since_last_txn'] / (df['card_age_months'] + 1.0)
    
    # 3. Rare Risk Trigger Aggregation
    risk_bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    df['risk_flag_count'] = df[risk_bool_cols].sum(axis=1)
    df['is_multi_risk'] = (df['risk_flag_count'] >= 2).astype(int)
    
    # 4. High-Gain Feature Interactions
    # CVV retry flag
    df['cvv_is_nonzero'] = (df['cvv_retry_count'] > 0).astype(int)
    # Combined CVV + Risk count
    df['cvv_plus_risk_count'] = df['cvv_retry_count'] + df['risk_flag_count']
    # Interaction: High velocity + High merchant risk
    df['velocity_x_merchant_risk'] = df['velocity_score'] * df['merchant_risk_score']
    
    return df

def main():
    train_path = "004_credit-card-fraud/data/train.csv"
    test_path = "004_credit-card-fraud/data/test.csv"
    output_dir = "004_credit-card-fraud/data"
    
    print("[1/5] Loading data and engineering features...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Original Train shape: {train.shape}, Test shape: {test.shape}")
    
    # Preprocessing bools
    bool_cols = [
        'is_foreign_transaction', 'is_new_merchant', 'used_vpn', 
        'ip_country_mismatch', 'billing_shipping_mismatch', 'is_ai_generated_scam_attempt'
    ]
    for c in bool_cols:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)
        
    # Feature Engineering
    train = create_features(train)
    test = create_features(test)
    print(f"Engineered Train shape: {train.shape}, Test shape: {test.shape}")
    
    target_col = "is_fraud"
    drop_cols = ["id", target_col]
    features = [c for c in train.columns if c not in drop_cols]
    
    # Categoricals
    cat_cols = ['merchant_category', 'card_type', 'auth_method', 'channel', 'device_type']
    for c in cat_cols:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
        
    X = train[features]
    y = train[target_col].astype(int)
    X_test = test[features]
    
    print(f"Total Features ({len(features)}): {features}")
    
    # Stratified 5-Fold CV
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_aucs = []
    feature_importances = pd.DataFrame(index=features)
    
    # Same tree parameters as Exp002 for direct comparison
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
    
    print(f"\n[2/5] Starting 5-Fold CV with Engineered Features...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
        
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
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.5f} (Best Iteration: {model.best_iteration})")
        
        feature_importances[f'fold_{fold}'] = model.feature_importance(importance_type='gain')
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n[3/5] Validation Results:")
    print(f"  Fold AUCs: {[round(a, 5) for a in fold_aucs]}")
    print(f"  Exp #003 Overall OOF ROC-AUC: {overall_auc:.5f}")
    print(f"  (Reference Exp #002: 0.94727 | Diff: {overall_auc - 0.94727:+.5f})")
    print(f"  (Reference Exp #001: 0.94652 | Diff: {overall_auc - 0.94652:+.5f})")
    
    # Feature Importance Top 15
    feature_importances['mean_gain'] = feature_importances.mean(axis=1)
    top_features = feature_importances.sort_values(by='mean_gain', ascending=False)
    print("\nTop 15 Features by Gain:")
    for rank, (feat, row) in enumerate(top_features.head(15).iterrows(), 1):
        print(f"  {rank:2d}. {feat:<28}: {row['mean_gain']:.1f}")
        
    # Check predictions summary
    print(f"\n[4/5] Prediction Distribution Check:")
    print(f"  OOF Pred Mean: {oof_preds.mean():.5f}, Std: {oof_preds.std():.5f}, Min: {oof_preds.min():.5f}, Max: {oof_preds.max():.5f}")
    print(f"  Test Pred Mean: {test_preds.mean():.5f}, Std: {test_preds.std():.5f}, Min: {test_preds.min():.5f}, Max: {test_preds.max():.5f}")
    
    # Save submission
    sub = pd.DataFrame({
        'id': test['id'],
        'is_fraud': test_preds
    })
    sub_file = os.path.join(output_dir, "submission_exp003.csv")
    sub.to_csv(sub_file, index=False)
    print(f"\n[5/5] Submission saved to {sub_file} (Rows: {len(sub)})")
    print("Sample lines:")
    print(sub.head())

if __name__ == "__main__":
    main()
