"""
003_store-sales: Exp #038 - Colab L4 GPU Regionally Disentangled School Season & Earthquake/Holiday 5-Seeds Ensemble
Kaggle Competition: Store Sales - Time Series Forecasting

Key Enhancements over Exp #032 & Exp #028:
  - 60 Feature Columns (Regionally Disentangled Calendar & Domain):
    * is_region_sierra: Sierra / Amazonia stores (34 stores in Quito, Ambato, Cuenca, etc.)
    * is_school_sierra: Late August school season for Sierra stores (Aug 16 - Aug 31)
    * is_school_peak_weekend_sierra: Pre-school peak weekend for Sierra stores (Aug 25 - Aug 28)
    * is_region_costa: Coastal stores (20 stores in Guayaquil, Manta, Machala, etc.)
    * is_school_costa: April-May school season for Coastal stores (Apr 15 - May 15)
    * is_school_peak_costa: Peak period for Coastal school season (Apr 25 - May 5)
    * earthquake_decay: Exponential decay from 2016-04-16 earthquake (tau=30 days)
    * is_manabi: Epicenter province indicator (Stores 52, 53, 54)
    * earthquake_decay_manabi: Localized earthquake impact intensity
    * is_holiday_national: Effective nationwide holidays (transferred handled, work days excluded)
    * is_holiday_regional: State-level holiday matching store state
    * is_holiday_local: City-level local holiday matching store city (e.g., Fundacion de Ambato on Aug 24)
    * is_holiday_effective: Any active holiday affecting the store
    * is_holiday_eve: Day before any active holiday (pre-holiday grocery shopping surge)
  - Zero-Filter Architectural Fix:
    * Sierra stores in SCHOOL AND OFFICE SUPPLIES are strictly PROTECTED from the 21-day zero filter,
      allowing GBDT models to naturally forecast back-to-school surges without artificial suppression!
  - 5 Seeds Averaging: [42, 2026, 777, 1234, 9999] across 33 families = 165 model triples (495 models)
  - Deeper Trees: LightGBM (num_leaves=45), CatBoost (depth=7, GPU), XGBoost (max_depth=6, CUDA), n_estimators=1000, lr=0.030
  - Output to Google Drive & Base64 stream for instant local synchronization
"""

import os
import sys
import subprocess
import glob
import json
import base64
import time
import threading
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)


def start_heartbeat(interval_sec=15):
    """Keep WebSocket connection alive by periodic flushing"""
    def beat():
        while True:
            time.sleep(interval_sec)
            print("[HEARTBEAT] GPU computation active...", flush=True)
    t = threading.Thread(target=beat, daemon=True)
    t.start()


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
    start_heartbeat(interval_sec=15)
    print("=" * 80)
    print("  003 Store Sales: Exp #038 - Colab L4 GPU Regional School & Domain 5-Seeds")
    print("=" * 80)

    data_dir = ensure_environment_and_data()

    print("\n[1/5] Loading datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])
    stores_df = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    holidays_df = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=["date"])

    train_df["sales"] = train_df["sales"].astype("float32")
    train_df["onpromotion"] = train_df["onpromotion"].astype("float32")
    test_df["onpromotion"] = test_df["onpromotion"].astype("float32")
    test_df["sales"] = np.nan

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df["sales"] = all_df["sales"].clip(lower=0)
    all_df["log_sales"] = np.log1p(all_df["sales"])

    # Merge store metadata
    all_df = all_df.merge(stores_df[["store_nbr", "city", "state", "type", "cluster"]], on="store_nbr", how="left")

    # Define geographical partitions
    sierra_states = {'Pichincha', 'Santo Domingo de los Tsachilas', 'Azuay', 'Cotopaxi', 'Tungurahua', 'Chimborazo', 'Imbabura', 'Bolivar', 'Pastaza', 'Loja'}
    costa_states = {'Guayas', 'Manabi', 'Los Rios', 'El Oro', 'Santa Elena', 'Esmeraldas'}

    # Safe zero sales filter identification WITH Sierra School supplies protection!
    print("\n[2/5] Identifying 21-day safe zero sales series with Sierra school protection...")
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    raw_zero_pairs = set(series_21_sum[series_21_sum == 0].index)

    sierra_stores = set(stores_df[stores_df["state"].isin(sierra_states)]["store_nbr"])
    sierra_school_pairs = {(st, "SCHOOL AND OFFICE SUPPLIES") for st in sierra_stores}
    zero_pairs = raw_zero_pairs - sierra_school_pairs
    print(f"  - Initial zero pairs: {len(raw_zero_pairs)}, after Sierra school protection: {len(zero_pairs)} pairs")

    val_21_days = train_df[(train_df["date"] >= "2017-07-10") & (train_df["date"] <= "2017-07-30")]
    val_series_21_sum = val_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    val_raw_zero_pairs = set(val_series_21_sum[val_series_21_sum == 0].index)
    val_zero_pairs = val_raw_zero_pairs - sierra_school_pairs

    print("\nBuilding 60 feature columns (Regionally Disentangled School & Domain)...")
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

    # Region indicators
    all_df["is_region_sierra"] = all_df["state"].isin(sierra_states).astype(int)
    all_df["is_region_costa"] = all_df["state"].isin(costa_states).astype(int)

    # Sierra School Season (August-September)
    all_df["is_school_sierra"] = (all_df["is_region_sierra"] == 1) & (all_df["month"] == 8) & (all_df["day"] >= 16)
    all_df["is_school_sierra"] = all_df["is_school_sierra"].astype(int)
    all_df["is_school_peak_weekend_sierra"] = (all_df["is_school_sierra"] == 1) & (all_df["day"] >= 25) & (all_df["day"] <= 28) & (all_df["is_weekend"] == 1)
    all_df["is_school_peak_weekend_sierra"] = all_df["is_school_peak_weekend_sierra"].astype(int)

    # Costa School Season (April-May)
    all_df["is_school_costa"] = (all_df["is_region_costa"] == 1) & (((all_df["month"] == 4) & (all_df["day"] >= 15)) | ((all_df["month"] == 5) & (all_df["day"] <= 15)))
    all_df["is_school_costa"] = all_df["is_school_costa"].astype(int)
    all_df["is_school_peak_costa"] = (all_df["is_region_costa"] == 1) & (((all_df["month"] == 4) & (all_df["day"] >= 25)) | ((all_df["month"] == 5) & (all_df["day"] <= 5)))
    all_df["is_school_peak_costa"] = all_df["is_school_peak_costa"].astype(int)

    # Earthquake Domain Features
    eq_date = pd.to_datetime("2016-04-16")
    days_since_eq = (all_df["date"] - eq_date).dt.days
    all_df["earthquake_decay"] = np.where(days_since_eq >= 0, np.exp(-days_since_eq / 30.0), 0.0).astype("float32")
    all_df["is_manabi"] = (all_df["state"] == "Manabi").astype(int)
    all_df["earthquake_decay_manabi"] = (all_df["earthquake_decay"] * all_df["is_manabi"]).astype("float32")

    # Holiday Domain Features
    eff_holidays = holidays_df[~holidays_df["transferred"]].copy()
    act_holidays = eff_holidays[eff_holidays["type"] != "Work Day"]

    nat_dates = set(act_holidays[act_holidays["locale"] == "National"]["date"])
    reg_pairs = set(zip(act_holidays[act_holidays["locale"] == "Regional"]["locale_name"], act_holidays[act_holidays["locale"] == "Regional"]["date"]))
    loc_pairs = set(zip(act_holidays[act_holidays["locale"] == "Local"]["locale_name"], act_holidays[act_holidays["locale"] == "Local"]["date"]))

    all_df["is_holiday_national"] = all_df["date"].isin(nat_dates).astype(int)
    all_df["is_holiday_regional"] = [1 if (st, dt) in reg_pairs else 0 for st, dt in zip(all_df["state"], all_df["date"])]
    all_df["is_holiday_local"] = [1 if (ct, dt) in loc_pairs else 0 for ct, dt in zip(all_df["city"], all_df["date"])]
    all_df["is_holiday_effective"] = (all_df["is_holiday_national"] | all_df["is_holiday_regional"] | all_df["is_holiday_local"]).astype(int)

    # Holiday Eve feature (pre-holiday grocery shopping surge)
    all_df = all_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    all_df["is_holiday_eve"] = all_df.groupby(["store_nbr", "family"])["is_holiday_effective"].shift(-1).fillna(0).astype(int)

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
    all_df["promo_intensity_change"] = all_df["onpromotion"] / (all_df["promo_lag_1"] + 1.0)
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
        "is_region_sierra", "is_school_sierra", "is_school_peak_weekend_sierra",
        "is_region_costa", "is_school_costa", "is_school_peak_costa",
        "earthquake_decay", "is_manabi", "earthquake_decay_manabi",
        "is_holiday_national", "is_holiday_regional", "is_holiday_local", "is_holiday_effective", "is_holiday_eve",
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
    print(f"  - Features count: {len(features)} columns (added regionally disentangled school season)")

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
        print(f"  >>> [{s_idx}/{len(SEEDS)}] Training Seed {seed}...", flush=True)
        trained_models = {}
        for f_idx, fam in enumerate(families, 1):
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
            if f_idx % 5 == 0 or f_idx == len(families):
                print(f"    - [{f_idx:2d}/{len(families)}] {fam} done", flush=True)
        seed_models[seed] = trained_models

    print("  - Finished training 5 seeds x 33 families successfully!", flush=True)

    def run_recursive_inference_5seeds(df_source, start_date, end_date, is_val):
        pivot_sales = df_source.pivot(index="date", columns=["store_nbr", "family"], values="log_sales").copy()
        dates = pd.date_range(start_date, end_date)
        step_predictions = []
        active_zero_pairs = val_zero_pairs if is_val else zero_pairs

        for current_date in dates:
            print(f"    - Recursive step: {current_date.strftime('%Y-%m-%d')}", flush=True)
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
    print(f"[RESULT] Exp #038 Colab L4 GPU Regional School Val RMSLE: {val_rmsle:.5f}")
    print("=" * 70)

    # Save validation predictions
    val_out_path = "val_preds_exp038_regional.csv"
    val_result[["id", "date", "store_nbr", "family", "sales", "pred_sales"]].to_csv(val_out_path, index=False)
    print(f"  - Saved validation predictions to: {val_out_path}")

    print("\nRunning test data inference (5 seeds)...")
    test_result = run_recursive_inference_5seeds(all_df, test_start_date, test_end_date, is_val=False)

    sub_final = test_result[["id", "pred_sales"]].rename(columns={"pred_sales": "sales"}).sort_values("id").reset_index(drop=True)
    out_path = "submission_exp038_colab_regional.csv"
    sub_final.to_csv(out_path, index=False)
    print(f"\n[5/5] Saved submission to: {out_path} ({len(sub_final)} rows)")
    print(sub_final.head(10))

    # Google Drive copy
    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        dest = os.path.join(drive_dir, out_path)
        sub_final.to_csv(dest, index=False)
        print(f"[DRIVE] Automatically saved to Google Drive: {dest}")
        dest_val = os.path.join(drive_dir, val_out_path)
        val_result.to_csv(dest_val, index=False)
        print(f"[DRIVE] Automatically saved val predictions to: {dest_val}")

    # Base64 output for local recovery
    with open(out_path, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("utf-8")
    print("\n===BASE64_SUBMISSION_START===")
    print(b64_content)
    print("===BASE64_SUBMISSION_END===")

    print(f"\n===VAL_RMSLE==={val_rmsle:.5f}===VAL_RMSLE===")
    print("[SUCCESS] Exp #038 completed successfully!")


if __name__ == "__main__":
    main()
