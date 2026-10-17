"""
002_airline-satisfaction: Flight Distance as Route ID + Target Encoding (Stratified 5-Fold CV)
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #004: MCP Discussion で発掘した知見（Flight Distance ＝ 路線ID）の実装
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
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


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    dep_delay = X["Departure Delay in Minutes"].fillna(0)
    arr_delay = X["Arrival Delay in Minutes"].fillna(0)
    X["Total_Delay"] = dep_delay + arr_delay
    X["Has_Delay"] = (X["Total_Delay"] > 0).astype(int)
    X["Delay_Diff"] = arr_delay - dep_delay

    X["Service_Mean"] = X[service_cols].mean(axis=1)
    X["Service_Min"] = X[service_cols].min(axis=1)
    X["Service_Max"] = X[service_cols].max(axis=1)
    X["Service_Std"] = X[service_cols].std(axis=1)
    X["Service_Sum"] = X[service_cols].sum(axis=1)

    X["Comfort_Score"] = X["Seat comfort"] + X["Leg room service"] + X["Cleanliness"]
    X["Digital_Score"] = X["Inflight wifi service"] + X["Ease of Online booking"] + X["Online boarding"]
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

    print("[2/4] 特徴量作成中（ドメイン特徴量 + 路線頻度）...")
    X = base_features(train_df)
    X_test = base_features(test_df)

    # Flight Distance の全体出現頻度（路線規模）
    all_dist = pd.concat([train_df["Flight Distance"], test_df["Flight Distance"]], axis=0)
    dist_counts = all_dist.value_counts()
    X["dist_freq"] = train_df["Flight Distance"].map(dist_counts).astype(float)
    X_test["dist_freq"] = test_df["Flight Distance"].map(dist_counts).astype(float)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    print(f"  - 特徴量数: {X.shape[1]} 列")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "n_estimators": 1200,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    print(f"\n[3/4] Stratified {n_splits}-Fold CV (In-Fold Target Encoding) 学習開始...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr = X.iloc[train_idx].copy()
        X_va = X.iloc[val_idx].copy()
        X_te = X_test.copy()
        y_tr, y_val = y[train_idx], y[val_idx]

        # In-Fold Target Encoding (データ漏洩防止)
        te = TargetEncoder(smooth="auto", cv=5, random_state=42)
        dist_tr = train_df.iloc[train_idx][["Flight Distance"]]
        dist_va = train_df.iloc[val_idx][["Flight Distance"]]
        dist_te = test_df[["Flight Distance"]]

        X_tr["dist_te"] = te.fit_transform(dist_tr, y_tr)
        X_va["dist_te"] = te.transform(dist_va)
        X_te["dist_te"] = te.transform(dist_te)

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        val_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_pred
        score = roc_auc_score(y_val, val_pred)
        print(f"  - Fold {fold}/{n_splits} ROC AUC: {score:.5f}")

        test_preds += model.predict_proba(X_te)[:, 1] / n_splits

    overall_auc = roc_auc_score(y, oof_preds)
    print("=" * 50)
    print(f"新 OOF ROC AUC (路線特徴量 + TE) : {overall_auc:.5f}")
    print(f"前回ベスト (0.95853) からの差分    : {overall_auc - 0.95853:+.5f}")
    print("=" * 50)

    sub_df["satisfaction"] = test_preds
    sub_df.to_csv("submission.csv", index=False)
    print("\n[4/4] 提出ファイル作成完了: submission.csv")


if __name__ == "__main__":
    main()
