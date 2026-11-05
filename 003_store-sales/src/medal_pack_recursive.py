"""
003_store-sales: Medal Pack Recursive Forecasting (Experiment #011)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク:
  1. ゼロ売上フィルターの誤爆救済 (2017年実績ゼロの67系列のみに限定、学用品・書籍の誤爆解除)
  2. 店舗52 (2017-04-20開店) のオープン前架空ゼロ売上データの除外
  3. アンバト市設立記念日 (8/24) 地方祝日フラグ (店舗23, 50)
  4. 2シード・アベレージング (Seed 42 + Seed 2026) によるバリアンス極小化
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
    dow = dates.dt.dayofweek

    is_wage = (day == 15) | is_month_end
    is_friday = (dow == 4)
    next_day = dates + pd.Timedelta(days=1)
    next_2day = dates + pd.Timedelta(days=2)

    wage_sat = (next_day.dt.day == 15) | next_day.dt.is_month_end
    wage_sun = (next_2day.dt.day == 15) | next_2day.dt.is_month_end
    is_weekend = dow.isin([5, 6])

    actual_wage_day = (is_wage & ~is_weekend) | (is_friday & (wage_sat | wage_sun))
    actual_wage_day_after = actual_wage_day.shift(1).fillna(False)

    return actual_wage_day.astype(int), actual_wage_day_after.astype(int)


def main():
    print("=" * 75)
    print("  003 Store Sales: 実験 #011 (メダル圏突破パック ＆ 2シードアベレージング)")
    print("  実行環境最適化: Intel N100 (4 Cores CPU Mode)")
    print("=" * 75)

    data_dir = find_data_dir()
    print(f"\n[1/5] データ読み込み元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    # 1. 安全なゼロ売上系列抽出 (誤爆救済)
    # 本番用: 2017年を通じて一度も売れていない正真正銘の完全休止系列
    t_2017 = train_df[train_df["date"] >= "2017-01-01"]
    zero_sum_tst = t_2017.groupby(["store_nbr", "family"])["sales"].sum()
    safe_zero_pairs_tst = set(zero_sum_tst[zero_sum_tst == 0].index)
    print(f"  - 【安全化】テスト用 2017年完全ゼロ系列数: {len(safe_zero_pairs_tst)} / 1,782 ペア (従来124件から57件を救済！)")

    # 手元検証用: 2017-01-01 〜 2017-07-30 の完全ゼロ系列
    v_2017 = train_df[(train_df["date"] >= "2017-01-01") & (train_df["date"] < "2017-07-31")]
    zero_sum_val = v_2017.groupby(["store_nbr", "family"])["sales"].sum()
    safe_zero_pairs_val = set(zero_sum_val[zero_sum_val == 0].index)

    # 全体結合
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

    # ドメイン補正: 給与日金曜前倒し
    all_df["is_wage_day"], all_df["is_wage_day_after"] = create_wage_day_features(all_df["date"])

    # ドメイン補正: アンバト市設立記念日 (8/24) 地方祝日 (店舗23, 50)
    all_df["is_ambato_holiday"] = ((all_df["month"] == 8) & (all_df["day"] == 24) & (all_df["store_nbr"].isin([23, 50]))).astype(int)

    # 系列ソート
    all_df = all_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    group_promo = all_df.groupby(["store_nbr", "family"])["onpromotion"]

    print("\n[2/5] 特売先行・後行 ＆ 特売平常比 (Promotion Intensity) 構築...")
    all_df["promo_lag_1"] = group_promo.shift(1).fillna(0)
    all_df["promo_lag_2"] = group_promo.shift(2).fillna(0)
    all_df["promo_lag_7"] = group_promo.shift(7).fillna(0)
    all_df["promo_lead_1"] = group_promo.shift(-1).bfill().fillna(0)
    all_df["promo_lead_2"] = group_promo.shift(-2).bfill().fillna(0)
    all_df["promo_diff_1"] = all_df["onpromotion"] - all_df["promo_lag_1"]
    all_df["promo_rolling_mean_7"] = group_promo.transform(lambda x: x.rolling(7, min_periods=1).mean())
    all_df["promo_rolling_mean_28"] = group_promo.transform(lambda x: x.rolling(28, min_periods=1).mean())
    all_df["promo_intensity_7"] = all_df["onpromotion"] / (all_df["promo_rolling_mean_7"] + 1.0)
    all_df["promo_intensity_28"] = all_df["onpromotion"] / (all_df["promo_rolling_mean_28"] + 1.0)

    # 過去売上ラグ
    group_sales = all_df.groupby(["store_nbr", "family"])["log_sales"]
    for lag in [1, 2, 3, 7, 14, 21, 28, 364]:
        all_df[f"sales_lag_{lag}"] = group_sales.shift(lag)
    all_df["sales_lag_364"] = all_df["sales_lag_364"].fillna(all_df["sales_lag_28"])

    # 移動平均 & ボラティリティ
    lag1 = group_sales.shift(1)
    all_df["sales_rolling_mean_7"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(7, min_periods=1).mean())
    all_df["sales_rolling_mean_14"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(14, min_periods=1).mean())
    all_df["sales_rolling_mean_28"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(28, min_periods=1).mean())
    all_df["sales_rolling_std_7"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)

    features = [
        "store_nbr", "onpromotion", "year", "month", "day", "dayofweek", "is_weekend",
        "dayofyear", "weekofyear", "is_month_start", "is_month_end", "is_wage_day", "is_wage_day_after", "is_ambato_holiday",
        "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_7", "sales_lag_14",
        "sales_lag_21", "sales_lag_28", "sales_lag_364",
        "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_28", "sales_rolling_std_7",
        "promo_lag_1", "promo_lag_2", "promo_lag_7",
        "promo_lead_1", "promo_lead_2",
        "promo_diff_1", "promo_rolling_mean_7",
        "promo_intensity_7", "promo_intensity_28"
    ]
    print(f"  - 特徴量数: {len(features)} 列")

    families = train_df["family"].unique()
    val_start_date = pd.to_datetime("2017-07-31")
    val_end_date = pd.to_datetime("2017-08-15")
    train_start_date = pd.to_datetime("2016-01-01")
    test_start_date = pd.to_datetime("2017-08-16")
    test_end_date = pd.to_datetime("2017-08-31")

    # 学習データ純化マスク:
    # 1. 2016年以降
    # 2. 元旦 (1/1) 休業日除外
    # 3. 【新規】店舗52 オープン前 (2017-04-20以前) の架空ゼロ売上除外！
    is_new_year = (all_df["month"] == 1) & (all_df["day"] == 1)
    is_pre_open_st52 = (all_df["store_nbr"] == 52) & (all_df["date"] < "2017-04-20")
    trn_mask = (all_df["date"] >= train_start_date) & (all_df["date"] < val_start_date) & (~is_new_year) & (~is_pre_open_st52)
    print(f"  - 【純化】店舗52オープン前売上ゼロを除外しました。")

    # =========================================================================
    # 再帰的推論関数 (Recursive Roll-forward Engine)
    # =========================================================================
    def run_recursive_inference(df_source: pd.DataFrame, start_date, end_date, models_dict, is_val: bool):
        pivot_sales = df_source.pivot(index="date", columns=["store_nbr", "family"], values="log_sales").copy()
        dates = pd.date_range(start_date, end_date)
        step_predictions = []
        active_zero_pairs = safe_zero_pairs_val if is_val else safe_zero_pairs_tst

        for current_date in dates:
            day_mask = (df_source["date"] == current_date)
            day_df = df_source.loc[day_mask].copy()

            for lag in [1, 2, 3, 7, 14, 21, 28, 364]:
                target_d = current_date - timedelta(days=lag)
                alt_d = current_date - timedelta(days=28)
                lag_series = pivot_sales.loc[target_d] if target_d in pivot_sales.index else pivot_sales.loc[alt_d]
                day_df[f"sales_lag_{lag}"] = day_df.set_index(["store_nbr", "family"]).index.map(lag_series).values

            d_1 = current_date - timedelta(days=1)
            for window in [7, 14, 28]:
                d_win = current_date - timedelta(days=window)
                mean_series = pivot_sales.loc[d_win:d_1].mean(axis=0)
                day_df[f"sales_rolling_mean_{window}"] = day_df.set_index(["store_nbr", "family"]).index.map(mean_series).values

            d_win_7 = current_date - timedelta(days=7)
            std_series_7 = pivot_sales.loc[d_win_7:d_1].std(axis=0).fillna(0)
            day_df["sales_rolling_std_7"] = day_df.set_index(["store_nbr", "family"]).index.map(std_series_7).values

            day_preds_dict = {}
            for fam in families:
                fam_mask = (day_df["family"] == fam)
                if not fam_mask.any():
                    continue

                X_day = day_df.loc[fam_mask, features].copy()
                X_day_lgb = X_day.copy()
                X_day_lgb["store_nbr"] = X_day_lgb["store_nbr"].astype("category")

                p_lgb = models_dict[fam]["lgb"].predict(X_day_lgb)
                p_cb = models_dict[fam]["cb"].predict(X_day)
                p_xgb = models_dict[fam]["xgb"].predict(X_day)

                p_log = 0.50 * p_lgb + 0.30 * p_cb + 0.20 * p_xgb
                p_sales = np.clip(np.expm1(p_log), 0, None)

                # 安全なゼロフィルター (完全ゼロのみクリップ)
                for idx, (st, row_pred) in enumerate(zip(day_df.loc[fam_mask, "store_nbr"], p_sales)):
                    if (st, fam) in active_zero_pairs:
                        p_sales[idx] = 0.0

                for idx_orig, p_val in zip(day_df.loc[fam_mask].index, p_sales):
                    day_preds_dict[idx_orig] = p_val

            day_df["pred_sales"] = pd.Series(day_preds_dict)
            step_predictions.append(day_df[["id", "date", "store_nbr", "family", "sales", "pred_sales"]])

            for _, row in day_df.iterrows():
                st = row["store_nbr"]
                fm = row["family"]
                pivot_sales.loc[current_date, (st, fm)] = np.log1p(row["pred_sales"])

        return pd.concat(step_predictions, ignore_index=True)

    # =========================================================================
    # 2シード・アベレージング (Seed 42 + Seed 2026)
    # =========================================================================
    seeds = [42, 2026]
    val_preds_seeds = []
    test_preds_seeds = []

    print(f"\n[3/5] 2シード・アベレージング学習 ＆ 再帰推論 (Seeds: {seeds})...")

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n--- [Seed {seed_idx}/{len(seeds)}: seed={seed}] 学習開始 ---")
        models_seed = {}

        for fam in families:
            fam_trn_mask = trn_mask & (all_df["family"] == fam)
            fam_train_df = all_df.loc[fam_trn_mask].dropna(subset=features)

            X_trn = fam_train_df[features].copy()
            y_trn = fam_train_df["log_sales"].values

            # LightGBM
            lgb_m = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                random_state=seed,
                verbose=-1,
                n_estimators=350,
                n_jobs=4,
            )
            X_lgb = X_trn.copy()
            X_lgb["store_nbr"] = X_lgb["store_nbr"].astype("category")
            lgb_m.fit(X_lgb, y_trn)

            # CatBoost
            cb_m = cb.CatBoostRegressor(
                iterations=350,
                learning_rate=0.06,
                depth=6,
                loss_function="RMSE",
                task_type="CPU",
                thread_count=4,
                random_seed=seed,
                verbose=0,
            )
            cb_m.fit(X_trn, y_trn, verbose=0)

            # XGBoost
            xgb_m = xgb.XGBRegressor(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cpu",
                n_jobs=4,
                random_state=seed,
            )
            xgb_m.fit(X_trn, y_trn, verbose=False)

            models_seed[fam] = {"lgb": lgb_m, "cb": cb_m, "xgb": xgb_m}

        print(f"  - Seed {seed}: 手元検証推論中...")
        val_res_seed = run_recursive_inference(all_df, val_start_date, val_end_date, models_seed, is_val=True)
        val_preds_seeds.append(val_res_seed["pred_sales"].values)

        print(f"  - Seed {seed}: テスト推論中...")
        test_res_seed = run_recursive_inference(all_df, test_start_date, test_end_date, models_seed, is_val=False)
        test_preds_seeds.append(test_res_seed["pred_sales"].values)

    print("\n[4/5] 2シードのブレンド (Seed Averaging) ＆ 検証スコア算出...")
    # 2シード予測値の単純平均
    final_val_sales = np.mean(val_preds_seeds, axis=0)
    final_test_sales = np.mean(test_preds_seeds, axis=0)

    val_true = val_res_seed["sales"].values
    val_rmsle = calculate_rmsle(val_true, final_val_sales)

    print("\n" + "=" * 70)
    print(f"  🏆 実験 #011 メダルパック (2シード平均) 手元検証 RMSLE: {val_rmsle:.5f}")
    print(f"     (参考: 実験 #010: 0.39958, 実験 #008 自己ベスト: 0.39956)")
    print("=" * 70)

    # 提出用DataFrame作成
    sub_final = test_res_seed[["id"]].copy()
    sub_final["sales"] = final_test_sales
    sub_final = sub_final.sort_values("id").reset_index(drop=True)

    out_path = "003_store-sales/submission_exp011.csv"
    sub_final.to_csv(out_path, index=False)
    print(f"\n[5/5] 提出ファイルを保存しました: {out_path}")
    print(f"  - 行数: {len(sub_final)} 行")
    print(f"  - 先頭10行:\n{sub_final.head(10)}")


if __name__ == "__main__":
    main()
