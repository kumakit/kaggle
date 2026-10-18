"""
002_airline-satisfaction: Plan B - Route Profile Residuals + Weighted Ensemble
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #007 (策B): 路線ID + 路線プロファイル残差 (Seat, Entertainment, Wifi, Boarding, Service_Mean) + 重み最適化アンサンブル
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

# 路線プロファイル残差を計算する対象（主要機材・サービス指標）
residual_targets = [
    "Seat comfort",
    "Inflight entertainment",
    "Inflight wifi service",
    "Online boarding",
    "Service_Mean"
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


def add_route_residuals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_dist: pd.Series,
    test_dist: pd.Series,
    smoothing_m: float = 10.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    路線（Flight Distance）ごとのサービス評価の平滑化平均（ベイズ平均）を求め、
    各乗客の評価との差分（残差: 個人評価 - 路線平均）を特徴量として追加する。
    """
    # train と test の特徴量を結合して路線平均を計算（目的変数は含まないため安全）
    full_dist = pd.concat([train_dist, test_dist], axis=0)

    for col in residual_targets:
        full_val = pd.concat([X_train[col], X_test[col]], axis=0)
        global_mean = full_val.mean()

        # 路線ごとの件数と平均
        stats = pd.DataFrame({"dist": full_dist, "val": full_val}).groupby("dist")["val"].agg(["count", "mean"])
        # 平滑化平均: (count * mean + m * global_mean) / (count + m)
        smooth_mean = (stats["count"] * stats["mean"] + smoothing_m * global_mean) / (stats["count"] + smoothing_m)

        tr_route_mean = train_dist.map(smooth_mean).fillna(global_mean)
        te_route_mean = test_dist.map(smooth_mean).fillna(global_mean)

        # 残差特徴量
        feat_name = f"res_{col.replace(' ', '_')}"
        X_train[feat_name] = X_train[col] - tr_route_mean
        X_test[feat_name] = X_test[col] - te_route_mean

    return X_train, X_test


def main():
    data_dir = find_data_dir()
    print(f"[1/4] データ取得元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    y = train_df["satisfaction"].astype(int).values

    print("[2/4] 特徴量作成中（ドメイン特徴量 + 路線頻度 + 路線プロファイル残差）...")
    X = base_features(train_df)
    X_test = base_features(test_df)

    # 路線頻度カウント
    all_dist = pd.concat([train_df["Flight Distance"], test_df["Flight Distance"]], axis=0)
    dist_counts = all_dist.value_counts()
    X["dist_freq"] = train_df["Flight Distance"].map(dist_counts).astype(float)
    X_test["dist_freq"] = test_df["Flight Distance"].map(dist_counts).astype(float)

    # 策B: 路線プロファイル残差の追加
    X, X_test = add_route_residuals(
        X, X_test,
        train_df["Flight Distance"],
        test_df["Flight Distance"],
        smoothing_m=10.0
    )

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    X_lgb, X_test_lgb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    X_cb, X_test_cb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    residual_names = [f"res_{c.replace(' ', '_')}" for c in residual_targets]
    print(f"  - 特徴量拡張完了: {X.shape[1]} 列")
    print(f"  - 残差特徴量: {residual_names}")

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

    print(f"\n[3/4] Stratified {n_splits}-Fold CV (LightGBM + CatBoost) 学習開始...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        y_tr, y_val = y[train_idx], y[val_idx]

        # In-Fold Target Encoding (路線IDに対する満足度のターゲットエンコーディング)
        te = TargetEncoder(smooth="auto", cv=5, random_state=42)
        dist_tr = train_df.iloc[train_idx][["Flight Distance"]]
        dist_va = train_df.iloc[val_idx][["Flight Distance"]]
        dist_te = test_df[["Flight Distance"]]

        te_tr = te.fit_transform(dist_tr, y_tr)
        te_va = te.transform(dist_va)
        te_test = te.transform(dist_te)

        # LGB
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

        # CB
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

        print(f"  - Fold {fold}/{n_splits} | LGB AUC: {roc_auc_score(y_val, lgb_val):.5f} | CB AUC: {roc_auc_score(y_val, cb_val):.5f}")

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

    print(f"\n🌟 最適比率: LightGBM {best_w:.2f} : CatBoost {1-best_w:.2f}")
    print(f"🌟 最適ブレンド OOF AUC : {best_score:.5f} (単体ベストより {best_score - auc_lgb:+.5f})")
    print("=" * 50)

    # 提出ファイル
    final_pred = best_w * lgb_test_preds + (1 - best_w) * cb_test_preds
    sub_df["satisfaction"] = final_pred
    sub_df.to_csv("submission.csv", index=False)
    print("\n[4/4] 提出ファイル作成完了: submission.csv (策B: 路線残差 + 最適重みブレンド)")


if __name__ == "__main__":
    main()
