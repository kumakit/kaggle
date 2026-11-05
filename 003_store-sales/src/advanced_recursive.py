"""
003_store-sales: Advanced Recursive Multi-step Forecasting (Experiment #010)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク:
  1. 再帰的推論エンジン (実験 #008 成功コア) をベースに進化
  2. ドメイン精密化: 給与日 (Quincena) 金曜前倒し判定 & 元旦 (1/1) 休業日除外
  3. 特売平常比 (Promotion Intensity 7d / 28d) の新規投入
  4. 直近ボラティリティ (sales_rolling_std_7) の追加
  5. ハードウェア最適化: Intel N100 (4コア/4スレッド CPU 最適化)
評価指標: RMSLE (Root Mean Squared Logarithmic Error)
"""

import os
import sys
import glob
import warnings
from datetime import timedelta
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
warnings.filterwarnings("ignore", category=UserWarning)


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


def create_wage_day_features(dates: pd.Series):
    """
    エクアドルの給与日 (15日/月末) の週末金曜前倒し判定 (Quincena)
    """
    day = dates.dt.day
    is_month_end = dates.dt.is_month_end
    dow = dates.dt.dayofweek  # 0:月 ... 4:金, 5:土, 6:日

    is_wage = (day == 15) | is_month_end
    is_friday = (dow == 4)
    next_day = dates + pd.Timedelta(days=1)
    next_2day = dates + pd.Timedelta(days=2)

    wage_sat = (next_day.dt.day == 15) | next_day.dt.is_month_end
    wage_sun = (next_2day.dt.day == 15) | next_2day.dt.is_month_end
    is_weekend = dow.isin([5, 6])

    # 週末の場合は直前金曜日に前倒し支給
    actual_wage_day = (is_wage & ~is_weekend) | (is_friday & (wage_sat | wage_sun))
    actual_wage_day_after = actual_wage_day.shift(1).fillna(False)

    return actual_wage_day.astype(int), actual_wage_day_after.astype(int)


def main():
    print("=" * 75)
    print("  003 Store Sales: 実験 #010 (進化版再帰予測 ＆ ドメイン精密化 ＆ 特売平常比)")
    print("  実行環境最適化: Intel N100 (4 Cores CPU Mode)")
    print("=" * 75)

    data_dir = find_data_dir()
    print(f"\n[1/5] データ読み込み元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    # ゼロ売上系列抽出
    recent_train = train_df[train_df["date"] >= "2017-07-25"]
    recent_sum = recent_train.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
    zero_pairs = set(zip(recent_sum[recent_sum["sales"] == 0]["store_nbr"], recent_sum[recent_sum["sales"] == 0]["family"]))
    print(f"  - テスト用 直近21日間売上ゼロのペア数: {len(zero_pairs)} / 1,782 ペア")

    pre_val = train_df[(train_df["date"] >= "2017-07-10") & (train_df["date"] < "2017-07-31")]
    zero_pre_val = pre_val.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
    val_zero_pairs = set(zip(zero_pre_val[zero_pre_val["sales"] == 0]["store_nbr"], zero_pre_val[zero_pre_val["sales"] == 0]["family"]))

    # 全体結合
    test_df["sales"] = np.nan
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df["log_sales"] = np.log1p(all_df["sales"])

    # 1. カレンダー特徴量
    all_df["year"] = all_df["date"].dt.year
    all_df["month"] = all_df["date"].dt.month
    all_df["day"] = all_df["date"].dt.day
    all_df["dayofweek"] = all_df["date"].dt.dayofweek
    all_df["is_weekend"] = all_df["dayofweek"].isin([5, 6]).astype(int)
    all_df["dayofyear"] = all_df["date"].dt.dayofyear
    all_df["weekofyear"] = all_df["date"].dt.isocalendar().week.astype(int)
    all_df["is_month_start"] = all_df["date"].dt.is_month_start.astype(int)
    all_df["is_month_end"] = all_df["date"].dt.is_month_end.astype(int)

    # ドメイン補正 1: 給与日金曜前倒し (Quincena)
    all_df["is_wage_day"], all_df["is_wage_day_after"] = create_wage_day_features(all_df["date"])

    # 系列ソート
    all_df = all_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    group_promo = all_df.groupby(["store_nbr", "family"])["onpromotion"]

    print("\n[2/5] 特売先行・後行 ＆ 【新規】特売平常比 (Promotion Intensity) 構築...")
    all_df["promo_lag_1"] = group_promo.shift(1).fillna(0)
    all_df["promo_lag_2"] = group_promo.shift(2).fillna(0)
    all_df["promo_lag_7"] = group_promo.shift(7).fillna(0)
    all_df["promo_lead_1"] = group_promo.shift(-1).bfill().fillna(0)
    all_df["promo_lead_2"] = group_promo.shift(-2).bfill().fillna(0)
    all_df["promo_diff_1"] = all_df["onpromotion"] - all_df["promo_lag_1"]
    all_df["promo_rolling_mean_7"] = group_promo.transform(lambda x: x.rolling(7, min_periods=1).mean())
    all_df["promo_rolling_mean_28"] = group_promo.transform(lambda x: x.rolling(28, min_periods=1).mean())

    # 新規投入: 特売平常比 (平常時の特売に対する比率)
    all_df["promo_intensity_7"] = all_df["onpromotion"] / (all_df["promo_rolling_mean_7"] + 1.0)
    all_df["promo_intensity_28"] = all_df["onpromotion"] / (all_df["promo_rolling_mean_28"] + 1.0)

    # 過去売上ラグ（学習用: 確定している過去実績から一括作成）
    group_sales = all_df.groupby(["store_nbr", "family"])["log_sales"]
    for lag in [1, 2, 3, 7, 14, 21, 28, 364]:
        all_df[f"sales_lag_{lag}"] = group_sales.shift(lag)

    all_df["sales_lag_364"] = all_df["sales_lag_364"].fillna(all_df["sales_lag_28"])

    # 直近移動平均 & ボラティリティ（前日までの実績から作成）
    lag1 = group_sales.shift(1)
    all_df["sales_rolling_mean_7"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    all_df["sales_rolling_mean_14"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    all_df["sales_rolling_mean_28"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    all_df["sales_rolling_std_7"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(
        lambda x: x.rolling(7, min_periods=1).std()
    ).fillna(0)

    features = [
        "store_nbr", "onpromotion", "year", "month", "day", "dayofweek", "is_weekend",
        "dayofyear", "weekofyear", "is_month_start", "is_month_end", "is_wage_day", "is_wage_day_after",
        "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_7", "sales_lag_14",
        "sales_lag_21", "sales_lag_28", "sales_lag_364",
        "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_28", "sales_rolling_std_7",
        "promo_lag_1", "promo_lag_2", "promo_lag_7",
        "promo_lead_1", "promo_lead_2",
        "promo_diff_1", "promo_rolling_mean_7",
        "promo_intensity_7", "promo_intensity_28"
    ]
    print(f"  - 特徴量数: {len(features)} 列")

    # 33ファミリー別の学習モデル格納辞書
    families = train_df["family"].unique()
    val_start_date = pd.to_datetime("2017-07-31")
    val_end_date = pd.to_datetime("2017-08-15")
    train_start_date = pd.to_datetime("2016-01-01")
    test_start_date = pd.to_datetime("2017-08-16")
    test_end_date = pd.to_datetime("2017-08-31")

    # ドメイン補正 2: 元旦 (1/1) 休業日の外れ値ノイズを除外
    is_new_year = (all_df["month"] == 1) & (all_df["day"] == 1)
    trn_mask = (all_df["date"] >= train_start_date) & (all_df["date"] < val_start_date) & (~is_new_year)

    print(f"\n[3/5] 33商品ファミリーごとに 3大GBDT 学習開始 (Intel N100 最適化)...")
    trained_models = {}

    for fam in families:
        fam_trn_mask = trn_mask & (all_df["family"] == fam)
        fam_train_df = all_df.loc[fam_trn_mask].dropna(subset=features)

        X_trn = fam_train_df[features].copy()
        y_trn = fam_train_df["log_sales"].values

        # 1. LightGBM (Intel N100: n_jobs=4)
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            random_state=42,
            verbose=-1,
            n_estimators=350,
            n_jobs=4,
        )
        X_trn_lgb = X_trn.copy()
        X_trn_lgb["store_nbr"] = X_trn_lgb["store_nbr"].astype("category")
        lgb_model.fit(X_trn_lgb, y_trn)

        # 2. CatBoost (Intel N100: thread_count=4)
        cb_model = cb.CatBoostRegressor(
            iterations=350,
            learning_rate=0.06,
            depth=6,
            loss_function="RMSE",
            task_type="CPU",
            thread_count=4,
            random_seed=42,
            verbose=0,
        )
        cb_model.fit(X_trn, y_trn, verbose=0)

        # 3. XGBoost (Intel N100: hist, n_jobs=4)
        xgb_model = xgb.XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cpu",
            n_jobs=4,
            random_state=42,
        )
        xgb_model.fit(X_trn, y_trn, verbose=False)

        trained_models[fam] = {
            "lgb": lgb_model,
            "cb": cb_model,
            "xgb": xgb_model,
        }

    print("  - 全33ファミリーの学習が完了しました。")

    # =========================================================================
    # 再帰的推論関数 (Recursive Roll-forward Engine)
    # =========================================================================
    def run_recursive_inference(df_source: pd.DataFrame, start_date, end_date, is_val: bool):
        pivot_sales = df_source.pivot(index="date", columns=["store_nbr", "family"], values="log_sales").copy()
        dates = pd.date_range(start_date, end_date)
        step_predictions = []
        active_zero_pairs = val_zero_pairs if is_val else zero_pairs

        for current_date in dates:
            day_mask = (df_source["date"] == current_date)
            day_df = df_source.loc[day_mask].copy()

            # 直前ラグを動的に抽出して注入
            for lag in [1, 2, 3, 7, 14, 21, 28, 364]:
                target_d = current_date - timedelta(days=lag)
                alt_d = current_date - timedelta(days=28)
                lag_series = pivot_sales.loc[target_d] if target_d in pivot_sales.index else pivot_sales.loc[alt_d]
                day_df[f"sales_lag_{lag}"] = day_df.set_index(["store_nbr", "family"]).index.map(lag_series).values

            # 直近ローリング平均 & 標準偏差 (前日〜7日前, 14日前, 28日前)
            d_1 = current_date - timedelta(days=1)
            for window in [7, 14, 28]:
                d_win = current_date - timedelta(days=window)
                sub_pivot = pivot_sales.loc[d_win:d_1]
                mean_series = sub_pivot.mean(axis=0)
                day_df[f"sales_rolling_mean_{window}"] = day_df.set_index(["store_nbr", "family"]).index.map(mean_series).values

            # 直近7日標準偏差
            d_win_7 = current_date - timedelta(days=7)
            sub_pivot_7 = pivot_sales.loc[d_win_7:d_1]
            std_series_7 = sub_pivot_7.std(axis=0).fillna(0)
            day_df["sales_rolling_std_7"] = day_df.set_index(["store_nbr", "family"]).index.map(std_series_7).values

            # 33ファミリーごとに推論
            day_preds_dict = {}
            for fam in families:
                fam_mask = (day_df["family"] == fam)
                if not fam_mask.any():
                    continue

                X_day = day_df.loc[fam_mask, features].copy()
                X_day_lgb = X_day.copy()
                X_day_lgb["store_nbr"] = X_day_lgb["store_nbr"].astype("category")

                # 3大GBDT アンサンブル予測 (50:30:20)
                p_lgb = trained_models[fam]["lgb"].predict(X_day_lgb)
                p_cb = trained_models[fam]["cb"].predict(X_day)
                p_xgb = trained_models[fam]["xgb"].predict(X_day)

                p_log = 0.50 * p_lgb + 0.30 * p_cb + 0.20 * p_xgb
                p_sales = np.clip(np.expm1(p_log), 0, None)

                # ゼロ売上フィルター
                for idx, (st, row_pred) in enumerate(zip(day_df.loc[fam_mask, "store_nbr"], p_sales)):
                    if (st, fam) in active_zero_pairs:
                        p_sales[idx] = 0.0

                for idx_orig, p_val in zip(day_df.loc[fam_mask].index, p_sales):
                    day_preds_dict[idx_orig] = p_val

            day_df["pred_sales"] = pd.Series(day_preds_dict)
            step_predictions.append(day_df[["id", "date", "store_nbr", "family", "sales", "pred_sales"]])

            # ピボットテーブルに当日の予測値（log_sales）を書き戻し（翌日以降のラグとして利用！）
            for _, row in day_df.iterrows():
                st = row["store_nbr"]
                fm = row["family"]
                p_log_val = np.log1p(row["pred_sales"])
                pivot_sales.loc[current_date, (st, fm)] = p_log_val

        return pd.concat(step_predictions, ignore_index=True)

    print("\n[4/5] 再帰的推論の実行...")
    # A. 手元検証 (2017-07-31 〜 2017-08-15)
    print("  - 手元検証データ（直近16日間）に対する再帰的推論中...")
    val_result = run_recursive_inference(all_df, val_start_date, val_end_date, is_val=True)
    val_rmsle = calculate_rmsle(val_result["sales"].values, val_result["pred_sales"].values)

    print("\n" + "=" * 70)
    print(f"  🏆 実験 #010 進化版再帰予測 手元検証 RMSLE: {val_rmsle:.5f}")
    print(f"     (参考: 実験 #008 自己ベスト: 0.39956, 実験 #006 Direct: 0.40219)")
    print("=" * 70)

    # B. 本番テスト推論 (2017-08-16 〜 2017-08-31)
    print("\n  - テストデータ（2017-08-16 〜 2017-08-31）に対する再帰的推論中...")
    test_result = run_recursive_inference(all_df, test_start_date, test_end_date, is_val=False)

    sub_final = test_result[["id", "pred_sales"]].rename(columns={"pred_sales": "sales"}).sort_values("id").reset_index(drop=True)
    out_path = "003_store-sales/submission_exp010.csv"
    sub_final.to_csv(out_path, index=False)
    print(f"\n[5/5] 提出ファイルを保存しました: {out_path}")
    print(f"  - 行数: {len(sub_final)} 行")
    print(f"  - 先頭10行:\n{sub_final.head(10)}")


if __name__ == "__main__":
    main()
