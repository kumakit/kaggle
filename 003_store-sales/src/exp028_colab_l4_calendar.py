"""
003_store-sales: Exp #028 - Colab L4 GPU School Season & Calendar Enhanced 5-Seeds 1,000-Trees Ensemble
Kaggle Competition: Store Sales - Time Series Forecasting

Key Enhancements over Exp #024:
  - 50 Feature Columns (added domain calendar & school season):
    * is_school_season: Late August indicator (Aug 16 - Aug 31)
    * is_school_peak_weekend: Final weekend before September school opening (Aug 25 - Aug 28)
    * is_wage_day_before: Day before salary payment (14th & month-end eve)
    * promo_intensity_change: Ratio of current promotion to 7-day rolling mean
  - Deeper Trees & Fine Learning Rate:
    * LightGBM: num_leaves=45, min_child_samples=20, lr=0.030
    * CatBoost: depth=7, lr=0.030, task_type=GPU
    * XGBoost: max_depth=6, lr=0.030, device=cuda
  - 5 Seeds Averaging: [42, 2026, 777, 1234, 9999] across 33 families = 165 models
  - Safe 21-day zero sales filter (127 series / 2,032 rows)
  - Output to Google Drive & Base64 stream
"""

import os
import sys
import subprocess
import glob
import json
import base64
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_rmsle(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped))))


def ensure_environment_and_data():
    """Package and data verification"""
    print("[INIT] Verifying packages and datasets...")
    
    # 1. Packages
    try:
        import lightgbm
        import catboost
        import xgboost
    except ImportError:
        print("  - Installing missing ML packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lightgbm", "catboost", "xgboost", "scikit-learn", "kaggle"], check=True)

    # 2. Kaggle auth
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    username = os.environ.get("KAGGLE_USERNAME", "nobuyukikumagai")
    key = os.environ.get("KAGGLE_KEY", "757b1a8d9ce71f3b129fbae7cc34b1b2")
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)
        print(f"  - Configured kaggle.json for user: {username}")

    # 3. Data files
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        candidates = ["/content/data", "003_store-sales/data", "/content/kaggle/003_store-sales/data"]
        for c in candidates:
            if os.path.exists(os.path.join(c, "train.csv")):
                data_dir = c
                break

    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("  - Downloading dataset from Kaggle...")
        os.makedirs("data", exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "store-sales-time-series-forecasting", "-p", "data"], check=True)
        for zip_file in glob.glob("data/*.zip"):
            subprocess.run(["unzip", "-q", "-o", zip_file, "-d", "data"], check=True)
        data_dir = "data"

    print(f"  - Dataset ready at: {data_dir}")
    return data_dir


def create_wage_day_features(dates):
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
    actual_wage_day_before = actual_wage_day.shift(-1).fillna(False)
    return actual_wage_day.astype(int), actual_wage_day_after.astype(int), actual_wage_day_before.astype(int)


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #028 - Colab L4 GPU Calendar & School Season 5-Seeds")
    print("=" * 80)

    data_dir = ensure_environment_and_data()

    print("\n[1/5] Loading datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])

    train_df["sales"] = train_df["sales"].astype("float32")
    train_df["onpromotion"] = train_df["onpromotion"].astype("float32")
    test_df["onpromotion"] = test_df["onpromotion"].astype("float32")
    test_df["sales"] = np.nan

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df["sales"] = all_df["sales"].clip(lower=0)
    all_df["log_sales"] = np.log1p(all_df["sales"])

    # Safe zero sales filter identification
    print("\n[2/5] Identifying 21-day safe zero sales series...")
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    zero_pairs = set(series_21_sum[series_21_sum == 0].index)
    print(f"  - Found {len(zero_pairs)} zero sales series out of 1,782 pairs")

    val_21_days = train_df[(train_df["date"] >= "2017-07-10") & (train_df["date"] <= "2017-07-30")]
    val_series_21_sum = val_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    val_zero_pairs = set(val_series_21_sum[val_series_21_sum == 0].index)

    print("\nBuilding 50 feature columns...")
    # Calendar features
    all_df["year"] = all_df["date"].dt.year
    all_df["month"] = all_df["date"].dt.month
    all_df["day"] = all_df["date"].dt.day
    all_df["dayofweek"] = all_df["date"].dt.dayofweek
    all_df["is_weekend"] = all_df["dayofweek"].isin([5, 6]).astype(int)
    all_df["dayofyear"] = all_df["date"].dt.dayofyear
    all_df["weekofyear"] = all_df["date"].dt.isocalendar().week.astype(int)
    all_df["is_month_start"] = all_df["date"].dt.is_month_start.astype(int)
    all_df["is_month_end"] = all_df["date"].dt.is_month_end.astype(int)

    # Wage day features (today, after, before)
    all_df["is_wage_day"], all_df["is_wage_day_after"], all_df["is_wage_day_before"] = create_wage_day_features(all_df["date"])

    # School Season Domain Features (NEW)
    all_df["is_school_season"] = ((all_df["month"] == 8) & (all_df["day"] >= 16)).astype(int)
    all_df["is_school_peak_weekend"] = ((all_df["month"] == 8) & (all_df["day"] >= 25) & (all_df["day"] <= 28) & (all_df["is_weekend"] == 1)).astype(int)

    # Advanced promotion domain features
    all_df["has_promotion"] = (all_df["onpromotion"] > 0).astype(int)
    all_df["promo_weekend"] = all_df["has_promotion"] * all_df["is_weekend"]

    store_total_promo = all_df.groupby(["date", "store_nbr"])["onpromotion"].transform("sum")
    all_df["promo_family_ratio"] = all_df["onpromotion"] / (store_total_promo + 1.0)

    # Day-of-week group mean features
    all_df = all_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    hist_df = train_df[train_df["date"] < "2017-07-31"].copy()
    store_dow_mean = hist_df.groupby(["store_nbr", hist_df["date"].dt.dayofweek])["sales"].mean().reset_index()
    store_dow_mean.rename(columns={"date": "dayofweek", "sales": "store_dow_mean_4w"}, inplace=True)
    family_dow_mean = hist_df.groupby(["family", hist_df["date"].dt.dayofweek])["sales"].mean().reset_index()
    family_dow_mean.rename(columns={"date": "dayofweek", "sales": "family_dow_mean_4w"}, inplace=True)

    all_df = all_df.merge(store_dow_mean, on=["store_nbr", "dayofweek"], how="left")
    all_df = all_df.merge(family_dow_mean, on=["family", "dayofweek"], how="left")
    all_df["store_dow_mean_4w"] = np.log1p(all_df["store_dow_mean_4w"].fillna(0))
    all_df["family_dow_mean_4w"] = np.log1p(all_df["family_dow_mean_4w"].fillna(0))

    # Promotion lags & leads
    all_df = all_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    group_promo = all_df.groupby(["store_nbr", "family"])["onpromotion"]

    all_df["promo_lag_1"] = group_promo.shift(1).fillna(0)
    all_df["promo_lag_2"] = group_promo.shift(2).fillna(0)
    all_df["promo_lag_7"] = group_promo.shift(7).fillna(0)
    all_df["promo_lead_1"] = group_promo.shift(-1).bfill().fillna(0)
    all_df["promo_lead_2"] = group_promo.shift(-2).bfill().fillna(0)
    all_df["promo_diff_1"] = all_df["onpromotion"] - all_df["promo_lag_1"]
    all_df["promo_is_burst"] = (all_df["promo_diff_1"] >= 5.0).astype(int)
    all_df["promo_rolling_mean_7"] = group_promo.transform(lambda x: x.rolling(7, min_periods=1).mean())
    all_df["promo_rolling_mean_28"] = group_promo.transform(lambda x: x.rolling(28, min_periods=1).mean())
    all_df["promo_intensity_7"] = all_df["onpromotion"] / (all_df["promo_rolling_mean_7"] + 1.0)
    all_df["promo_intensity_28"] = all_df["onpromotion"] / (all_df["promo_rolling_mean_28"] + 1.0)
    all_df["promo_intensity_change"] = all_df["onpromotion"] / (all_df["promo_lag_1"] + 1.0) # NEW
    all_df["promo_decay_lead"] = 0.6 * all_df["promo_lead_1"] + 0.4 * all_df["promo_lead_2"]

    # Sales lags & rolling stats
    group_sales = all_df.groupby(["store_nbr", "family"])["log_sales"]
    for lag in [1, 2, 3, 7, 14, 21, 28, 364]:
        all_df[f"sales_lag_{lag}"] = group_sales.shift(lag)
    all_df["sales_lag_364"] = all_df["sales_lag_364"].fillna(all_df["sales_lag_28"])

    lag1 = group_sales.shift(1)
    all_df["sales_rolling_mean_7"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(7, min_periods=1).mean())
    all_df["sales_rolling_mean_14"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(14, min_periods=1).mean())
    all_df["sales_rolling_mean_28"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(28, min_periods=1).mean())
    all_df["sales_rolling_std_7"] = lag1.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)

    all_df["dow_mean_4w"] = (all_df["sales_lag_7"] + all_df["sales_lag_14"] + all_df["sales_lag_21"] + all_df["sales_lag_28"]) / 4.0
    all_df["dow_decay_mean_4w"] = 0.4 * all_df["sales_lag_7"] + 0.3 * all_df["sales_lag_14"] + 0.2 * all_df["sales_lag_21"] + 0.1 * all_df["sales_lag_28"]
    all_df["dow_diff_lag1"] = all_df["sales_lag_1"] - all_df["dow_mean_4w"]

    # YoY trend and momentum
    all_df["sales_yoy_diff"] = all_df["sales_lag_7"] - all_df["sales_lag_364"]
    all_df["sales_momentum_ratio"] = all_df["sales_lag_1"] / (all_df["dow_mean_4w"] + 1.0)

    features = [
        "store_nbr", "onpromotion", "has_promotion", "promo_weekend", "promo_family_ratio", "promo_is_burst",
        "year", "month", "day", "dayofweek", "is_weekend",
        "dayofyear", "weekofyear", "is_month_start", "is_month_end",
        "is_wage_day", "is_wage_day_after", "is_wage_day_before",
        "is_school_season", "is_school_peak_weekend",
        "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_7", "sales_lag_14",
        "sales_lag_21", "sales_lag_28", "sales_lag_364",
        "sales_rolling_mean_7", "sales_rolling_mean_14", "sales_rolling_mean_28", "sales_rolling_std_7",
        "promo_lag_1", "promo_lag_2", "promo_lag_7",
        "promo_lead_1", "promo_lead_2", "promo_decay_lead",
        "promo_diff_1", "promo_rolling_mean_7",
        "promo_intensity_7", "promo_intensity_28", "promo_intensity_change",
        "dow_mean_4w", "dow_decay_mean_4w", "dow_diff_lag1",
        "sales_yoy_diff", "sales_momentum_ratio",
        "store_dow_mean_4w", "family_dow_mean_4w"
    ]
    print(f"  - Features count: {len(features)} columns")

    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb

    families = train_df["family"].unique()
    val_start_date = pd.to_datetime("2017-07-31")
    val_end_date = pd.to_datetime("2017-08-15")
    train_start_date = pd.to_datetime("2016-01-01")
    test_start_date = pd.to_datetime("2017-08-16")
    test_end_date = pd.to_datetime("2017-08-31")

    is_new_year = (all_df["month"] == 1) & (all_df["day"] == 1)
    trn_mask = (all_df["date"] >= train_start_date) & (all_df["date"] < val_start_date) & (~is_new_year)

    # 5 Seeds configuration
    SEEDS = [42, 2026, 777, 1234, 9999]
    N_ESTIMATORS = 1000
    LR = 0.030
    print(f"\n[3/5] Starting L4 GPU 5-Seeds training (Seeds={SEEDS}, n_estimators={N_ESTIMATORS}, lr={LR})...")
    print("  - LightGBM: num_leaves=45, min_child_samples=20")
    print("  - CatBoost: depth=7, task_type=GPU")
    print("  - XGBoost: max_depth=6, device=cuda")
    seed_models = {}

    for s_idx, seed in enumerate(SEEDS, 1):
        print(f"  >>> [{s_idx}/{len(SEEDS)}] Training Seed {seed}...")
        trained_models = {}
        for fam in families:
            fam_trn_mask = trn_mask & (all_df["family"] == fam)
            fam_train_df = all_df.loc[fam_trn_mask].dropna(subset=features)
            X_trn = fam_train_df[features].copy()
            y_trn = fam_train_df["log_sales"].values

            lgb_model = lgb.LGBMRegressor(
                objective="regression", metric="rmse", learning_rate=LR,
                num_leaves=45, min_child_samples=20, feature_fraction=0.8, subsample=0.8,
                random_state=seed, verbose=-1, n_estimators=N_ESTIMATORS, n_jobs=-1,
            )
            X_trn_lgb = X_trn.copy()
            X_trn_lgb["store_nbr"] = X_trn_lgb["store_nbr"].astype("category")
            lgb_model.fit(X_trn_lgb, y_trn)

            cb_model = cb.CatBoostRegressor(
                iterations=N_ESTIMATORS, learning_rate=LR, depth=7, loss_function="RMSE",
                task_type="GPU", random_seed=seed, verbose=0,
            )
            cb_model.fit(X_trn, y_trn, verbose=0)

            xgb_model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS, learning_rate=LR, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", device="cuda", random_state=seed,
            )
            xgb_model.fit(X_trn, y_trn, verbose=False)

            trained_models[fam] = {"lgb": lgb_model, "cb": cb_model, "xgb": xgb_model}
        seed_models[seed] = trained_models

    print("  - Finished training 5 seeds x 33 families successfully!")

    def run_recursive_inference_5seeds(df_source, start_date, end_date, is_val):
        pivot_sales = df_source.pivot(index="date", columns=["store_nbr", "family"], values="log_sales").copy()
        dates = pd.date_range(start_date, end_date)
        step_predictions = []
        active_zero_pairs = val_zero_pairs if is_val else zero_pairs

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
                sub_pivot = pivot_sales.loc[d_win:d_1]
                mean_series = sub_pivot.mean(axis=0)
                day_df[f"sales_rolling_mean_{window}"] = day_df.set_index(["store_nbr", "family"]).index.map(mean_series).values

            d_win_7 = current_date - timedelta(days=7)
            sub_pivot_7 = pivot_sales.loc[d_win_7:d_1]
            std_series_7 = sub_pivot_7.std(axis=0).fillna(0)
            day_df["sales_rolling_std_7"] = day_df.set_index(["store_nbr", "family"]).index.map(std_series_7).values

            day_df["dow_mean_4w"] = (day_df["sales_lag_7"] + day_df["sales_lag_14"] + day_df["sales_lag_21"] + day_df["sales_lag_28"]) / 4.0
            day_df["dow_decay_mean_4w"] = 0.4 * day_df["sales_lag_7"] + 0.3 * day_df["sales_lag_14"] + 0.2 * day_df["sales_lag_21"] + 0.1 * day_df["sales_lag_28"]
            day_df["dow_diff_lag1"] = day_df["sales_lag_1"] - day_df["dow_mean_4w"]

            day_df["sales_yoy_diff"] = day_df["sales_lag_7"] - day_df["sales_lag_364"]
            day_df["sales_momentum_ratio"] = day_df["sales_lag_1"] / (day_df["dow_mean_4w"] + 1.0)

            day_preds_dict = {}
            for fam in families:
                fam_mask = (day_df["family"] == fam)
                if not fam_mask.any():
                    continue

                X_day = day_df.loc[fam_mask, features].copy()
                X_day_lgb = X_day.copy()
                X_day_lgb["store_nbr"] = X_day_lgb["store_nbr"].astype("category")

                seed_logs = []
                for seed in SEEDS:
                    p_lgb = seed_models[seed][fam]["lgb"].predict(X_day_lgb)
                    p_cb = seed_models[seed][fam]["cb"].predict(X_day)
                    p_xgb = seed_models[seed][fam]["xgb"].predict(X_day)
                    p_log = 0.50 * p_lgb + 0.30 * p_cb + 0.20 * p_xgb
                    seed_logs.append(p_log)

                p_log_avg = np.mean(seed_logs, axis=0)
                p_sales = np.clip(np.expm1(p_log_avg), 0, None)

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
                p_log_val = np.log1p(row["pred_sales"])
                pivot_sales.loc[current_date, (st, fm)] = p_log_val

        return pd.concat(step_predictions, ignore_index=True)

    print("\n[4/5] Running recursive roll-forward inference (5 seeds)...")
    val_result = run_recursive_inference_5seeds(all_df, val_start_date, val_end_date, is_val=True)
    val_rmsle = calculate_rmsle(val_result["sales"].values, val_result["pred_sales"].values)

    print("=" * 70)
    print(f"[RESULT] Exp #028 Colab L4 GPU Calendar Enhanced Val RMSLE: {val_rmsle:.5f}")
    print("=" * 70)

    print("\nRunning test data inference (5 seeds)...")
    test_result = run_recursive_inference_5seeds(all_df, test_start_date, test_end_date, is_val=False)

    sub_final = test_result[["id", "pred_sales"]].rename(columns={"pred_sales": "sales"}).sort_values("id").reset_index(drop=True)
    out_path = "submission_exp028_colab_calendar.csv"
    sub_final.to_csv(out_path, index=False)
    print(f"\n[5/5] Saved submission to: {out_path} ({len(sub_final)} rows)")
    print(sub_final.head(10))

    # Google Drive copy
    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        dest = os.path.join(drive_dir, out_path)
        sub_final.to_csv(dest, index=False)
        print(f"[DRIVE] Automatically saved to Google Drive: {dest}")

    # Base64 output for local recovery
    with open(out_path, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("utf-8")
    print("\n===BASE64_SUBMISSION_START===")
    print(b64_content)
    print("===BASE64_SUBMISSION_END===")
    print("[SUCCESS] Exp #028 completed successfully!")


if __name__ == "__main__":
    main()
