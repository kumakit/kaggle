"""
002_airline-satisfaction: Feature Engineering & LightGBM Tuning (Stratified 5-Fold CV)
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #002: ドメイン特徴量（遅延差分・サービス統計量・カテゴリ交差）+ チューニング
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


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
    X["Delay_Diff"] = arr_delay - dep_delay  # 到着遅延 - 出発遅延

    # 2. サービス評価の心理統計量
    X["Service_Mean"] = X[service_cols].mean(axis=1)
    X["Service_Min"] = X[service_cols].min(axis=1)   # 最低評価（不満の引き金）
    X["Service_Max"] = X[service_cols].max(axis=1)
    X["Service_Std"] = X[service_cols].std(axis=1)
    X["Service_Sum"] = X[service_cols].sum(axis=1)

    # 3. 複合スコア
    X["Comfort_Score"] = X["Seat comfort"] + X["Leg room service"] + X["Cleanliness"]
    X["Digital_Score"] = X["Inflight wifi service"] + X["Ease of Online booking"] + X["Online boarding"]

    # 4. カテゴリ掛け合わせ
    X["Class_Travel"] = X["Class"].astype(str) + "_" + X["Type of Travel"].astype(str)

    drop_cols = [c for c in ["id", "satisfaction"] if c in X.columns]
    X = X.drop(columns=drop_cols)

    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("category")

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

    print(f"[2/4] 特徴量作成完了: {X.shape[1]} 列 (元は21列)")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_scores = []

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

    print(f"\n[3/4] Stratified {n_splits}-Fold CV 学習開始...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        val_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_pred
        score = roc_auc_score(y_val, val_pred)
        fold_scores.append(score)
        print(f"  - Fold {fold}/{n_splits} ROC AUC: {score:.5f}")

        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    overall_auc = roc_auc_score(y, oof_preds)
    print("=" * 45)
    print(f"新 OOF ROC AUC (手元検証): {overall_auc:.5f} (ベースライン: 0.95828)")
    print(f"改善スコア差分: {overall_auc - 0.95828:+.5f}")
    print("=" * 45)

    sub_df["satisfaction"] = test_preds
    sub_df.to_csv("submission_fe.csv", index=False)
    print("\n[4/4] 提出ファイル作成完了: submission_fe.csv")


if __name__ == "__main__":
    main()
