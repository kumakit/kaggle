"""
002_airline-satisfaction: CatBoost Deep Tuning + Weighted Ensemble
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #008: 研ぎ澄まされた最強特徴量（実験 #005） + CatBoost 本格深化（1000木・低学習率） + 最適重み探索
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

# GPU 利用可能性の自動検知
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False


def find_data_dir() -> str:
    candidates = [
        "/kaggle/input/competitions/playground-series-s6e10",
        "/kaggle/input/playground-series-s6e10",
        "002_airline-satisfaction/data",
        "data",
        "../data",
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
    print(f"  - アクセラレータ環境: {'GPU (CUDA 有効)' if HAS_GPU else 'CPU'}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    y = train_df["satisfaction"].astype(int).values

    print("[2/4] 特徴量作成中（実験 #005 の最強コア特徴量セット）...")
    X = base_features(train_df)
    X_test = base_features(test_df)

    # 路線頻度カウント（Flight Distance = 路線ID）
    all_dist = pd.concat([train_df["Flight Distance"], test_df["Flight Distance"]], axis=0)
    dist_counts = all_dist.value_counts()
    X["dist_freq"] = train_df["Flight Distance"].map(dist_counts).astype(float)
    X_test["dist_freq"] = test_df["Flight Distance"].map(dist_counts).astype(float)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # LightGBM 用カテゴリ型変換
    X_lgb, X_test_lgb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    # CatBoost 用文字列型変換
    X_cb, X_test_cb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    print(f"  - 特徴量数: {X.shape[1]} 列（過学習要因のない研ぎ澄まされた特徴量空間）")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_oof = np.zeros(len(train_df))
    cb_oof = np.zeros(len(train_df))
    lgb_test_preds = np.zeros(len(test_df))
    cb_test_preds = np.zeros(len(test_df))

    # LightGBM パラメータ (安定の精密設定)
    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1,
        "n_estimators": 1200, "random_state": 42, "n_jobs": -1, "verbose": -1,
        "subsample": 0.9, "colsample_bytree": 0.9
    }

    # CatBoost パラメータ (本格深化チューニング: 1000木・学習率0.05・早期終了50)
    cb_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "verbose": 0,
        "cat_features": cat_cols,
    }
    if HAS_GPU:
        cb_params["task_type"] = "GPU"
    else:
        cb_params["thread_count"] = -1

    print(f"\n[3/4] Stratified {n_splits}-Fold CV (LightGBM + CatBoost深化) 学習開始...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        y_tr, y_val = y[train_idx], y[val_idx]

        # In-Fold Target Encoding (データ漏洩のない安全な路線平均ターゲット)
        te = TargetEncoder(smooth="auto", cv=5, random_state=42)
        dist_tr = train_df.iloc[train_idx][["Flight Distance"]]
        dist_va = train_df.iloc[val_idx][["Flight Distance"]]
        dist_te = test_df[["Flight Distance"]]

        te_tr = te.fit_transform(dist_tr, y_tr)
        te_va = te.transform(dist_va)
        te_test = te.transform(dist_te)

        # --- 1. LightGBM ---
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

        # --- 2. CatBoost (深化チューニング) ---
        X_tr_c = X_cb.iloc[train_idx].copy()
        X_va_c = X_cb.iloc[val_idx].copy()
        X_te_c = X_test_cb.copy()
        X_tr_c["dist_te"], X_va_c["dist_te"], X_te_c["dist_te"] = te_tr, te_va, te_test

        model_cb = CatBoostClassifier(**cb_params, random_seed=42 + fold)
        model_cb.fit(
            X_tr_c, y_tr,
            eval_set=(X_va_c, y_val),
            early_stopping_rounds=50
        )
        cb_val = model_cb.predict_proba(X_va_c)[:, 1]
        cb_oof[val_idx] = cb_val
        cb_test_preds += model_cb.predict_proba(X_te_c)[:, 1] / n_splits

        fold_lgb_auc = roc_auc_score(y_val, lgb_val)
        fold_cb_auc = roc_auc_score(y_val, cb_val)
        print(f"  - Fold {fold}/{n_splits} | LGB AUC: {fold_lgb_auc:.5f} | CB深化 AUC: {fold_cb_auc:.5f} (ベスト木数: {model_cb.get_best_iteration()})")

    # --- 最適重み比率探索 ---
    auc_lgb = roc_auc_score(y, lgb_oof)
    auc_cb = roc_auc_score(y, cb_oof)
    print("\n" + "=" * 55)
    print(f"全体 LightGBM 単体 OOF AUC : {auc_lgb:.5f}")
    print(f"全体 CatBoost深化 単体 OOF AUC : {auc_cb:.5f}")

    best_w = 1.0
    best_score = max(auc_lgb, auc_cb)
    print("\n=== 重み比率探索 (Grid Search) ===")
    for w in np.linspace(0.3, 0.9, 13):
        blend = w * lgb_oof + (1 - w) * cb_oof
        score = roc_auc_score(y, blend)
        print(f"  LGB {w:4.2f} : CB {1-w:4.2f} | OOF AUC: {score:.5f}")
        if score > best_score:
            best_score = score
            best_w = w

    print(f"\n🌟 最適比率: LightGBM {best_w:.2f} : CatBoost深化 {1-best_w:.2f}")
    print(f"🌟 最適ブレンド OOF AUC : {best_score:.5f} (LightGBM単体より {best_score - auc_lgb:+.5f})")
    print("=" * 55)

    # 提出ファイル
    final_pred = best_w * lgb_test_preds + (1 - best_w) * cb_test_preds
    sub_df["satisfaction"] = final_pred
    sub_df.to_csv("submission.csv", index=False)
    print("\n[4/4] 提出ファイル作成完了: submission.csv (CatBoost深化 + 最適重みブレンド)")


if __name__ == "__main__":
    main()
