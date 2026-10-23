"""
003_store-sales: 3-GBDT Diversity Ensemble (LightGBM + CatBoost GPU + XGBoost GPU) (Experiment #005)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク: RTX 4060 Laptop GPU をフル活用した 3大GBDT多様性アンサンブル ＋ 前年同期ラグ ＋ ゼロフィルター
評価指標: RMSLE (Root Mean Squared Logarithmic Error)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_data_dir() -> str:
    candidates = [
        "/kaggle/input/store-sales-time-series-forecasting",
        "/kaggle/input/competitions/store-sales-time-series-forecasting",
        "003_store-sales/data",
        "data",
        "../data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    matches_local = glob.glob("**/train.csv", recursive=True)
    if matches_local:
        return os.path.dirname(matches_local[0])
    raise FileNotFoundError("train.csv が見つかりませんでした。")


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_clipped = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped))))


def main():
    print("=" * 75)
    print("  003 Store Sales: 実験 #005 (3大GBDT多様性アンサンブル with RTX 4060 GPU)")
    print("=" * 75)

    data_dir = find_data_dir()
    print(f"\n[1/5] データ読み込み元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    # 直近21日間に売上が全くないペア（Zero Series）を抽出
    recent_train = train_df[train_df["date"] >= "2017-07-25"]
    recent_sum = recent_train.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
    zero_pairs = set(zip(recent_sum[recent_sum["sales"] == 0]["store_nbr"], recent_sum[recent_sum["sales"] == 0]["family"]))
    print(f"  - 直近21日間で売上ゼロのペア数: {len(zero_pairs)} / 1,782 ペア")

    # 全体結合 & 特徴量作成
    test_df["sales"] = np.nan
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df["log_sales"] = np.log1p(all_df["sales"])

    # カレンダー特徴量
    all_df["year"] = all_df["date"].dt.year
    all_df["month"] = all_df["date"].dt.month
    all_df["day"] = all_df["date"].dt.day
    all_df["dayofweek"] = all_df["date"].dt.dayofweek
    all_df["is_weekend"] = all_df["dayofweek"].isin([5, 6]).astype(int)
    all_df["dayofyear"] = all_df["date"].dt.dayofyear
    all_df["weekofyear"] = all_df["date"].dt.isocalendar().week.astype(int)
    all_df["is_month_start"] = all_df["date"].dt.is_month_start.astype(int)
    all_df["is_month_end"] = all_df["date"].dt.is_month_end.astype(int)
    all_df["is_wage_day"] = ((all_df["day"] == 15) | (all_df["is_month_end"] == 1)).astype(int)
    all_df["is_wage_day_after"] = ((all_df["day"] == 16) | (all_df["is_month_start"] == 1)).astype(int)

    # ラグ & ローリング特徴量
    all_df = all_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    group = all_df.groupby(["store_nbr", "family"])["log_sales"]

    print("\n[2/5] 時系列ラグ & ローリング特徴量構築（前年同期 364日ラグ含む）...")
    for lag in [16, 21, 28, 35, 364]:  # 364日前（ちょうど52週前の同じ曜日！）
        all_df[f"sales_lag_{lag}"] = group.shift(lag)

    # 欠損補間 (364日前が存在しない初期データは直近ラグで補完)
    all_df["sales_lag_364"] = all_df["sales_lag_364"].fillna(all_df["sales_lag_28"])

    lag16 = group.shift(16)
    all_df["sales_rolling_mean_7"] = lag16.groupby([all_df["store_nbr"], all_df["family"]]).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    all_df["sales_rolling_mean_14"] = lag16.groupby([all_df["store_nbr"], all_df["family"]]).transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    all_df["sales_rolling_mean_28"] = lag16.groupby([all_df["store_nbr"], all_df["family"]]).transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )

    train_feat = all_df[all_df["date"] <= "2017-08-15"].copy()
    test_feat = all_df[all_df["date"] >= "2017-08-16"].copy()

    features = [
        "store_nbr", "onpromotion", "year", "month", "day", "dayofweek", "is_weekend",
        "dayofyear", "weekofyear", "is_month_start", "is_month_end", "is_wage_day", "is_wage_day_after",
        "sales_lag_16", "sales_lag_21", "sales_lag_28", "sales_lag_35", "sales_lag_364",
        "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_28"
    ]

    val_start_date = "2017-07-31"
    val_end_date = "2017-08-15"
    train_start_date = "2016-01-01"

    trn_mask = (train_feat["date"] >= train_start_date) & (train_feat["date"] < val_start_date)
    val_mask = (train_feat["date"] >= val_start_date) & (train_feat["date"] <= val_end_date)

    # 手元検証用直近0売上ペア
    pre_val = train_df[(train_df["date"] >= "2017-07-10") & (train_df["date"] < val_start_date)]
    zero_pre_val = pre_val.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
    val_zero_pairs = set(zip(zero_pre_val[zero_pre_val["sales"] == 0]["store_nbr"], zero_pre_val[zero_pre_val["sales"] == 0]["family"]))

    families = train_df["family"].unique()
    print(f"\n[3/5] 33商品ファミリーごとに 3大GBDT (LightGBM + CatBoost GPU + XGBoost GPU) を学習中...")

    val_lgb_preds, val_cb_preds, val_xgb_preds = [], [], []
    val_trues = []
    test_predictions = []

    # GPUの利用可能性確認
    has_gpu = False
    try:
        test_cb = cb.CatBoostRegressor(iterations=1, task_type="GPU", verbose=0)
        test_cb.fit(np.random.randn(10, 2), np.random.randn(10))
        has_gpu = True
        print("  ⚡ NVIDIA GPU (CUDA) 加速が有効です！")
    except Exception as e:
        print(f"  ℹ️ GPU未検出のためCPUモードで実行します: {e}")

    cb_task_type = "GPU" if has_gpu else "CPU"
    xgb_device = "cuda" if has_gpu else "cpu"

    for fam in families:
        fam_trn_mask = trn_mask & (train_feat["family"] == fam)
        fam_val_mask = val_mask & (train_feat["family"] == fam)
        fam_tst_mask = (test_feat["family"] == fam)

        X_trn = train_feat.loc[fam_trn_mask, features].copy()
        y_trn = np.log1p(train_feat.loc[fam_trn_mask, "sales"].values)
        X_v = train_feat.loc[fam_val_mask, features].copy()
        y_v = train_feat.loc[fam_val_mask, "sales"].values
        X_tst = test_feat.loc[fam_tst_mask, features].copy()

        # 1. LightGBM (Leaf-wise: スパイク需要キャッチ)
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            random_state=42,
            verbose=-1,
            n_estimators=500,
        )
        # category型指定
        X_trn_lgb = X_trn.copy()
        X_v_lgb = X_v.copy()
        X_tst_lgb = X_tst.copy()
        X_trn_lgb["store_nbr"] = X_trn_lgb["store_nbr"].astype("category")
        X_v_lgb["store_nbr"] = X_v_lgb["store_nbr"].astype("category")
        X_tst_lgb["store_nbr"] = X_tst_lgb["store_nbr"].astype("category")

        lgb_model.fit(
            X_trn_lgb,
            y_trn,
            eval_set=[(X_v_lgb, np.log1p(y_v))],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        lgb_pred_v = lgb_model.predict(X_v_lgb)
        lgb_pred_tst = lgb_model.predict(X_tst_lgb)

        # 2. CatBoost (Symmetric Trees: 頑健・平滑化予測)
        cb_model = cb.CatBoostRegressor(
            iterations=500,
            learning_rate=0.06,
            depth=6,
            loss_function="RMSE",
            task_type=cb_task_type,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=30,
        )
        cb_model.fit(X_trn, y_trn, eval_set=(X_v, np.log1p(y_v)), verbose=0)
        cb_pred_v = cb_model.predict(X_v)
        cb_pred_tst = cb_model.predict(X_tst)

        # 3. XGBoost (Depth-wise: 正則化木)
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=xgb_device,
            random_state=42,
            early_stopping_rounds=30,
        )
        xgb_model.fit(X_trn, y_trn, eval_set=[(X_v, np.log1p(y_v))], verbose=False)
        xgb_pred_v = xgb_model.predict(X_v)
        xgb_pred_tst = xgb_model.predict(X_tst)

        # 3大GBDTアンサンブル（50% LightGBM + 30% CatBoost + 20% XGBoost）
        ens_pred_v = np.clip(np.expm1(0.50 * lgb_pred_v + 0.30 * cb_pred_v + 0.20 * xgb_pred_v), 0, None)
        ens_pred_tst = np.clip(np.expm1(0.50 * lgb_pred_tst + 0.30 * cb_pred_tst + 0.20 * xgb_pred_tst), 0, None)

        # ゼロフィルター適用
        val_stores = train_feat.loc[fam_val_mask, "store_nbr"].values
        for i, st in enumerate(val_stores):
            if (st, fam) in val_zero_pairs:
                ens_pred_v[i] = 0.0

        tst_stores = test_feat.loc[fam_tst_mask, "store_nbr"].values
        for i, st in enumerate(tst_stores):
            if (st, fam) in zero_pairs:
                ens_pred_tst[i] = 0.0

        val_lgb_preds.extend(ens_pred_v)
        val_trues.extend(y_v)

        fam_test_df = test_feat.loc[fam_tst_mask, ["id"]].copy()
        fam_test_df["sales"] = ens_pred_tst
        test_predictions.append(fam_test_df)

    ens_rmsle = calculate_rmsle(np.array(val_trues), np.array(val_lgb_preds))
    print("\n" + "=" * 70)
    print(f"  🏆 [実験 #005] 3大GBDTアンサンブル 全体手元検証 RMSLE: {ens_rmsle:.5f}")
    print(f"  (参考: #004 単体LGB: 0.40444 ➔ 本番LB 0.45505)")
    print("=" * 70)

    # 提出ファイル作成 (id順ソート)
    final_sub = pd.concat(test_predictions, ignore_index=True)
    final_sub = final_sub.sort_values("id").reset_index(drop=True)

    out_path = "003_store-sales/submission_exp005.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_sub.to_csv(out_path, index=False)
    print(f"\n[5/5] 提出ファイルを保存しました: {out_path}")
    print(f"  - 先頭10行:\n{final_sub.head(10)}")


if __name__ == "__main__":
    main()
