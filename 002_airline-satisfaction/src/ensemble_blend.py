"""
002_airline-satisfaction: LightGBM + CatBoost Ensemble Blend (Stratified 5-Fold CV)
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #003: 異なる決定木アルゴリズム（LightGBM + CatBoost）の 50:50 アンサンブル
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier


def find_data_dir() -> str:
    candidates = [
        "002_airline-satisfaction/data",
        "data",
        "../data",
        "/kaggle/input/competitions/playground-series-s6e10",
        "/kaggle/input/playground-series-s6e10",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    matches = glob.glob("**/playground-series-s6e10/train.csv", recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError("train.csv が見つかりませんでした。")


service_cols = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    # 1. 遅延関連
    dep_delay = X["Departure Delay in Minutes"].fillna(0)
    arr_delay = X["Arrival Delay in Minutes"].fillna(0)
    X["Total_Delay"] = dep_delay + arr_delay
    X["Has_Delay"] = (X["Total_Delay"] > 0).astype(int)
    X["Delay_Diff"] = arr_delay - dep_delay

    # 2. サービス統計量
    X["Service_Mean"] = X[service_cols].mean(axis=1)
    X["Service_Min"] = X[service_cols].min(axis=1)
    X["Service_Max"] = X[service_cols].max(axis=1)
    X["Service_Std"] = X[service_cols].std(axis=1)
    X["Service_Sum"] = X[service_cols].sum(axis=1)

    # 3. 複合スコア
    X["Comfort_Score"] = X["Seat comfort"] + X["Leg room service"] + X["Cleanliness"]
    X["Digital_Score"] = X["Inflight wifi service"] + X["Ease of Online booking"] + X["Online boarding"]

    # 4. カテゴリ交差
    X["Class_Travel"] = X["Class"].astype(str) + "_" + X["Type of Travel"].astype(str)

    drop_cols = [c for c in ["id", "satisfaction"] if c in X.columns]
    X = X.drop(columns=drop_cols)
    return X


def main():
    data_dir = find_data_dir()
    print(f"[1/4] データ取得元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    y = train_df["satisfaction"].astype(int).values
    X = engineer_features(train_df)
    X_test = engineer_features(test_df)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # LightGBM用（category型）
    X_lgb = X.copy()
    X_test_lgb = X_test.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    # CatBoost用（string型）
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    print(f"[2/4] 特徴量作成完了: {X.shape[1]} 列 (カテゴリ: {len(cat_cols)}個)")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_oof = np.zeros(len(train_df))
    cb_oof = np.zeros(len(train_df))
    lgb_test_preds = np.zeros(len(test_df))
    cb_test_preds = np.zeros(len(test_df))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "n_estimators": 1000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    print(f"\n[3/4] Stratified {n_splits}-Fold CV (LightGBM + CatBoost) 学習開始...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        y_tr, y_val = y[train_idx], y[val_idx]

        # --- A. LightGBM ---
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_lgb.iloc[train_idx], y_tr,
            eval_set=[(X_lgb.iloc[val_idx], y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_val = model_lgb.predict_proba(X_lgb.iloc[val_idx])[:, 1]
        lgb_oof[val_idx] = lgb_val
        lgb_test_preds += model_lgb.predict_proba(X_test_lgb)[:, 1] / n_splits

        # --- B. CatBoost ---
        model_cb = CatBoostClassifier(
            iterations=500,
            learning_rate=0.08,
            depth=6,
            eval_metric="AUC",
            random_seed=42 + fold,
            thread_count=-1,
            verbose=0,
            cat_features=cat_cols,
        )
        model_cb.fit(
            X_cb.iloc[train_idx], y_tr,
            eval_set=(X_cb.iloc[val_idx], y_val),
            early_stopping_rounds=30,
        )
        cb_val = model_cb.predict_proba(X_cb.iloc[val_idx])[:, 1]
        cb_oof[val_idx] = cb_val
        cb_test_preds += model_cb.predict_proba(X_test_cb)[:, 1] / n_splits

        # Fold Blend
        fold_blend = 0.5 * lgb_val + 0.5 * cb_val
        print(f"  - Fold {fold}/{n_splits} | LGB: {roc_auc_score(y_val, lgb_val):.5f} | CB: {roc_auc_score(y_val, cb_val):.5f} | Blend: {roc_auc_score(y_val, fold_blend):.5f}")

    # 全体 OOF スコア
    overall_lgb = roc_auc_score(y, lgb_oof)
    overall_cb = roc_auc_score(y, cb_oof)
    final_blend_oof = 0.5 * lgb_oof + 0.5 * cb_oof
    overall_blend = roc_auc_score(y, final_blend_oof)

    print("=" * 50)
    print(f"全体 LightGBM 単体 OOF AUC : {overall_lgb:.5f}")
    print(f"全体 CatBoost 単体 OOF AUC : {overall_cb:.5f}")
    print(f"🌟 全体 Blend (50:50) OOF AUC : {overall_blend:.5f}")
    print(f"単体ベストからの向上         : {overall_blend - max(overall_lgb, overall_cb):+.5f}")
    print("=" * 50)

    # 提出ファイル
    final_test_preds = 0.5 * lgb_test_preds + 0.5 * cb_test_preds
    sub_df["satisfaction"] = final_test_preds
    sub_df.to_csv("submission.csv", index=False)
    print("\n[4/4] 提出ファイル作成完了: submission.csv (Blend 50:50)")


if __name__ == "__main__":
    main()
