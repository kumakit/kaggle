"""
002_airline-satisfaction: Plan A (Zero Signals + Online Boarding Ratio) + Weighted Ensemble
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #006: 策A（0点VIPフラグ + Online boarding比率）+ 重み最適化アンサンブル
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
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


def engineer_features_plan_a(df: pd.DataFrame, dist_counts: pd.Series) -> pd.DataFrame:
    X = df.copy()

    # 1. 遅延関連
    dep_delay = X["Departure Delay in Minutes"].fillna(0)
    arr_delay = X["Arrival Delay in Minutes"].fillna(0)
    X["Total_Delay"] = dep_delay + arr_delay
    X["Has_Delay"] = (X["Total_Delay"] > 0).astype(int)
    X["Delay_Diff"] = arr_delay - dep_delay
    X["Log_Total_Delay"] = np.log1p(X["Total_Delay"])

    # 2. サービス評価の心理統計量
    X["Service_Mean"] = X[service_cols].mean(axis=1)
    X["Service_Min"] = X[service_cols].min(axis=1)
    X["Service_Max"] = X[service_cols].max(axis=1)
    X["Service_Std"] = X[service_cols].std(axis=1)
    X["Service_Sum"] = X[service_cols].sum(axis=1)

    # ★ 3. 策A: 0点（未評価・該当なし）の強力なシグナル
    X["Zero_Count"] = (X[service_cols] == 0).sum(axis=1)
    X["Many_Zeros"] = (X["Zero_Count"] >= 3).astype(int)   # 満足度90%超のVIPフラグ
    X["One_Zero"] = (X["Zero_Count"] == 1).astype(int)     # 満足度32%の不満フラグ

    # ★ 4. 策A: 最強特徴量 Online boarding の相対位置
    X["Online_Boarding_Ratio"] = X["Online boarding"] / (X["Service_Mean"] + 1e-5)

    # 5. 複合スコア
    X["Comfort_Score"] = X["Seat comfort"] + X["Leg room service"] + X["Cleanliness"]
    X["Digital_Score"] = X["Inflight wifi service"] + X["Ease of Online booking"] + X["Online boarding"]
    X["Class_Travel"] = X["Class"].astype(str) + "_" + X["Type of Travel"].astype(str)

    # 6. 路線規模（出現頻度）
    X["dist_freq"] = X["Flight Distance"].map(dist_counts).astype(float)

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

    all_dist = pd.concat([train_df["Flight Distance"], test_df["Flight Distance"]], axis=0)
    dist_counts = all_dist.value_counts()

    print("[2/4] 特徴量作成中 (策A: ゼロ値シグナル + 最強特徴量比率)...")
    X = engineer_features_plan_a(train_df, dist_counts)
    X_test = engineer_features_plan_a(test_df, dist_counts)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    X_lgb, X_test_lgb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    X_cb, X_test_cb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    print(f"  - 特徴量拡張完了: {X.shape[1]} 列 (元は21列)")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_oof = np.zeros(len(train_df))
    cb_oof = np.zeros(len(train_df))
    lgb_test_preds = np.zeros(len(test_df))
    cb_test_preds = np.zeros(len(test_df))

    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1,
        "n_estimators": 1200, "random_state": 42, "n_jobs": -1, "verbose": -1,
        "subsample": 0.9, "colsample_bytree": 0.9
    }

    print(f"\n[3/4] Stratified {n_splits}-Fold CV (Plan A + Ensemble) 学習開始...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        y_tr, y_val = y[train_idx], y[val_idx]

        # In-Fold Target Encoding
        te = TargetEncoder(smooth="auto", cv=5, random_state=42)
        dist_tr = train_df.iloc[train_idx][["Flight Distance"]]
        dist_va = train_df.iloc[val_idx][["Flight Distance"]]
        dist_te = test_df[["Flight Distance"]]

        te_tr = te.fit_transform(dist_tr, y_tr)
        te_va = te.transform(dist_va)
        te_test = te.transform(dist_te)

        # LightGBM
        X_tr_l = X_lgb.iloc[train_idx].copy()
        X_va_l = X_lgb.iloc[val_idx].copy()
        X_te_l = X_test_lgb.copy()
        X_tr_l["dist_te"], X_va_l["dist_te"], X_te_l["dist_te"] = te_tr, te_va, te_test

        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(
            X_tr_l, y_tr,
            eval_set=[(X_va_l, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        lgb_val = model_lgb.predict_proba(X_va_l)[:, 1]
        lgb_oof[val_idx] = lgb_val
        lgb_test_preds += model_lgb.predict_proba(X_te_l)[:, 1] / n_splits

        # CatBoost
        X_tr_c = X_cb.iloc[train_idx].copy()
        X_va_c = X_cb.iloc[val_idx].copy()
        X_te_c = X_test_cb.copy()
        X_tr_c["dist_te"], X_va_c["dist_te"], X_te_c["dist_te"] = te_tr, te_va, te_test

        model_cb = CatBoostClassifier(
            iterations=600, learning_rate=0.08, depth=6,
            eval_metric="AUC", random_seed=42 + fold, thread_count=-1, verbose=0,
            cat_features=cat_cols
        )
        model_cb.fit(
            X_tr_c, y_tr,
            eval_set=(X_va_c, y_val),
            early_stopping_rounds=30
        )
        cb_val = model_cb.predict_proba(X_va_c)[:, 1]
        cb_oof[val_idx] = cb_val
        cb_test_preds += model_cb.predict_proba(X_te_c)[:, 1] / n_splits

        print(f"  - Fold {fold}/{n_splits} | LGB: {roc_auc_score(y_val, lgb_val):.5f} | CB: {roc_auc_score(y_val, cb_val):.5f}")

    # 重み最適化探索
    auc_lgb = roc_auc_score(y, lgb_oof)
    auc_cb = roc_auc_score(y, cb_oof)
    print("\n" + "=" * 50)
    print(f"全体 LightGBM 単体 OOF AUC : {auc_lgb:.5f}")
    print(f"全体 CatBoost 単体 OOF AUC : {auc_cb:.5f}")

    best_w = 1.0
    best_score = auc_lgb
    print("\n=== 重み比率探索結果 ===")
    for w in np.linspace(0.5, 1.0, 11):
        blend = w * lgb_oof + (1 - w) * cb_oof
        score = roc_auc_score(y, blend)
        print(f"  LGB {w:4.2f} : CB {1-w:4.2f} | OOF AUC: {score:.5f}")
        if score > best_score:
            best_score = score
            best_w = w

    print(f"\n🌟 自動決定された最適比率: LightGBM {best_w:.2f} : CatBoost {1-best_w:.2f}")
    print(f"🌟 最適ブレンド OOF AUC   : {best_score:.5f}")
    print("=" * 50)

    # 提出ファイル
    final_pred = best_w * lgb_test_preds + (1 - best_w) * cb_test_preds
    sub_df["satisfaction"] = final_pred
    sub_df.to_csv("submission.csv", index=False)
    print("\n[4/4] 提出ファイル作成完了: submission.csv")


if __name__ == "__main__":
    main()
