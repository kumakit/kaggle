"""
002_airline-satisfaction: 3-GBDT Ensemble (LightGBM + CatBoost + XGBoost)
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #009: 3大GBDTアンサンブル (多様性を注入したXGBoost + 絶妙なCatBoost + 精密LightGBM)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

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


def find_best_3d_weights(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """
    3モデルの予測確率に対して、OOF AUC を最大化する最適な重み (w1, w2, w3) を探索する。
    制約: w1 + w2 + w3 = 1, w_i >= 0
    """
    print("\n=== 3モデル最適ブレンド重み探索 (Coarse-to-Fine Grid Search) ===")
    start_t = time.time()
    best_auc = -1.0
    best_w = (1.0, 0.0, 0.0)

    # 1. 粗い探索 (step = 0.05, 231通り)
    step1 = 0.05
    steps1 = int(round(1.0 / step1))
    for i in range(steps1 + 1):
        w1 = round(i * step1, 2)
        for j in range(steps1 + 1 - i):
            w2 = round(j * step1, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -1e-5:
                continue
            blend = w1 * p1 + w2 * p2 + w3 * p3
            score = roc_auc_score(y_true, blend)
            if score > best_auc:
                best_auc = score
                best_w = (w1, w2, w3)

    # 2. 精密探索 (best_w の周辺 ±0.06 を step = 0.01 で微小探索)
    bw1, bw2, bw3 = best_w
    fine_step = 0.01
    w1_range = np.arange(max(0.0, bw1 - 0.06), min(1.0, bw1 + 0.06) + 1e-5, fine_step)
    w2_range = np.arange(max(0.0, bw2 - 0.06), min(1.0, bw2 + 0.06) + 1e-5, fine_step)

    for w1 in w1_range:
        w1 = round(w1, 2)
        for w2 in w2_range:
            w2 = round(w2, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0.0:
                continue
            blend = w1 * p1 + w2 * p2 + w3 * p3
            score = roc_auc_score(y_true, blend)
            if score > best_auc:
                best_auc = score
                best_w = (w1, w2, w3)

    elapsed = time.time() - start_t
    print(f"  - 探索完了 ({elapsed:.1f} 秒)")
    return best_w, best_auc


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

    # 路線頻度カウント
    all_dist = pd.concat([train_df["Flight Distance"], test_df["Flight Distance"]], axis=0)
    dist_counts = all_dist.value_counts()
    X["dist_freq"] = train_df["Flight Distance"].map(dist_counts).astype(float)
    X_test["dist_freq"] = test_df["Flight Distance"].map(dist_counts).astype(float)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    # 各モデル向け特徴量フォーマット
    # 1. LightGBM: category型
    X_lgb, X_test_lgb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    # 2. CatBoost: string型
    X_cb, X_test_cb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    # 3. XGBoost: category型 (enable_categorical=True)
    X_xgb, X_test_xgb = X.copy(), X_test.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].astype("category")
        X_test_xgb[c] = X_test_xgb[c].astype("category")

    print(f"  - 特徴量数: {X.shape[1]} 列")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_oof = np.zeros(len(train_df))
    cb_oof = np.zeros(len(train_df))
    xgb_oof = np.zeros(len(train_df))

    lgb_test_preds = np.zeros(len(test_df))
    cb_test_preds = np.zeros(len(test_df))
    xgb_test_preds = np.zeros(len(test_df))

    # --- 1. LightGBM パラメータ (Leaf-wise 精密エース) ---
    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1,
        "n_estimators": 1200, "random_state": 42, "n_jobs": -1, "verbose": -1,
        "subsample": 0.9, "colsample_bytree": 0.9
    }

    # --- 2. CatBoost パラメータ (Symmetric Tree 多様性サポーター) ---
    cb_params = {
        "iterations": 700, "learning_rate": 0.07, "depth": 6,
        "eval_metric": "AUC", "early_stopping_rounds": 40,
        "verbose": 0, "cat_features": cat_cols,
    }
    if HAS_GPU:
        cb_params["task_type"] = "GPU"
    else:
        cb_params["thread_count"] = -1

    # --- 3. XGBoost パラメータ (Depth-wise 多様性注入・堅牢バランサー) ---
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 6,
        "n_estimators": 1000,
        "subsample": 0.8,
        "colsample_bytree": 0.7,      # ★重要工夫: 特徴量サンプリング70%で多様性を強制注入
        "reg_alpha": 1.0,             # L1正則化
        "reg_lambda": 5.0,            # L2正則化
        "random_state": 42,
        "enable_categorical": True,
    }
    if HAS_GPU:
        xgb_params["device"] = "cuda"
    else:
        xgb_params["n_jobs"] = -1

    print(f"\n[3/4] Stratified {n_splits}-Fold CV (3大GBDT: LightGBM + CatBoost + XGBoost) 学習開始...")
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

        # -----------------------------
        # 1. LightGBM
        # -----------------------------
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

        # -----------------------------
        # 2. CatBoost
        # -----------------------------
        X_tr_c = X_cb.iloc[train_idx].copy()
        X_va_c = X_cb.iloc[val_idx].copy()
        X_te_c = X_test_cb.copy()
        X_tr_c["dist_te"], X_va_c["dist_te"], X_te_c["dist_te"] = te_tr, te_va, te_test

        model_cb = CatBoostClassifier(**cb_params, random_seed=42 + fold)
        model_cb.fit(
            X_tr_c, y_tr,
            eval_set=(X_va_c, y_val),
            early_stopping_rounds=40
        )
        cb_val = model_cb.predict_proba(X_va_c)[:, 1]
        cb_oof[val_idx] = cb_val
        cb_test_preds += model_cb.predict_proba(X_te_c)[:, 1] / n_splits

        # -----------------------------
        # 3. XGBoost
        # -----------------------------
        X_tr_x = X_xgb.iloc[train_idx].copy()
        X_va_x = X_xgb.iloc[val_idx].copy()
        X_te_x = X_test_xgb.copy()
        X_tr_x["dist_te"], X_va_x["dist_te"], X_te_x["dist_te"] = te_tr, te_va, te_test

        model_xgb = xgb.XGBClassifier(**xgb_params, random_seed=42 + fold)
        model_xgb.fit(
            X_tr_x, y_tr,
            eval_set=[(X_va_x, y_val)],
            verbose=False
        )
        xgb_val = model_xgb.predict_proba(X_va_x)[:, 1]
        xgb_oof[val_idx] = xgb_val
        xgb_test_preds += model_xgb.predict_proba(X_te_x)[:, 1] / n_splits

        print(
            f"  - Fold {fold}/{n_splits} | "
            f"LGB: {roc_auc_score(y_val, lgb_val):.5f} | "
            f"CB: {roc_auc_score(y_val, cb_val):.5f} | "
            f"XGB: {roc_auc_score(y_val, xgb_val):.5f}"
        )

    # --- 単体 OOF AUC 評価 ---
    auc_lgb = roc_auc_score(y, lgb_oof)
    auc_cb = roc_auc_score(y, cb_oof)
    auc_xgb = roc_auc_score(y, xgb_oof)

    print("\n" + "=" * 60)
    print("【単体モデル OOF AUC】")
    print(f"  1. LightGBM 単体 OOF AUC : {auc_lgb:.5f}")
    print(f"  2. CatBoost 単体 OOF AUC : {auc_cb:.5f}")
    print(f"  3. XGBoost  単体 OOF AUC : {auc_xgb:.5f}")

    # モデル間相関（多様性の確認）
    preds_df = pd.DataFrame({"LGB": lgb_oof, "CB": cb_oof, "XGB": xgb_oof})
    print("\n【モデル予測値の相関行列 (多様性チェック)】")
    print(preds_df.corr().round(4))

    # 3モデル最適重み探索
    best_weights, best_score = find_best_3d_weights(y, lgb_oof, cb_oof, xgb_oof)
    w_lgb, w_cb, w_xgb = best_weights

    print("\n" + "=" * 60)
    print(f"🌟 最適比率: LightGBM {w_lgb:.2f} : CatBoost {w_cb:.2f} : XGBoost {w_xgb:.2f}")
    print(f"🌟 3大GBDT最適ブレンド OOF AUC : {best_score:.5f} (LightGBM単体より {best_score - auc_lgb:+.5f})")
    print("=" * 60)

    # 提出ファイル
    final_pred = w_lgb * lgb_test_preds + w_cb * cb_test_preds + w_xgb * xgb_test_preds
    sub_df["satisfaction"] = final_pred
    sub_df.to_csv("submission.csv", index=False)
    print("\n[4/4] 提出ファイル作成完了: submission.csv (3大GBDT最適ブレンド)")


if __name__ == "__main__":
    main()
