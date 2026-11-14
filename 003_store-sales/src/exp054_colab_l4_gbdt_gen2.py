"""
003_store-sales: Exp #054 - Gen-2 Direct Multi-Horizon GBDT Ensemble (LightGBM + CatBoost GPU + XGBoost CUDA)
Kaggle Competition: Store Sales - Time Series Forecasting
Designed for Google Colab L4 GPU (VRAM 24GB / RAM 53GB) Ephemeral Execution

Why Gen-2 Replaces Recursive GBDT (Exp #036 / #038):
  - Zero Recursive Drift: Predicts target day d = T + h directly using strictly observed cutoff <= T
    sales history plus horizon h (1..16) conditioning — eliminating 16-day autoregressive error cascade.
  - Rich Exogenous & Cross-Sectional Signal Missing in Exp #036/#038:
    * 90-day cutoff rolling mean/std/slope + days_since_nonzero & activity ratios
    * Target-aligned Same-DOW 1..4 & 8-week lags + 1Y (-364d), 53w (-371d), and 2Y (-728d) YoY anchors
    * Store traffic (`transactions.csv` cutoff + rolling + same-DOW) & Oil macro (`dcoilwtico` + 14d delta)
    * Cross-sectional Store total & Family chain-wide demand + Store/Family target promo sums
    * Granular Regional/Local/National holidays, eve, workday, Jan 1 closure, regional school & earthquake
  - Early-Stopped Multi-Engine Diversity: Global LightGBM + CatBoost GPU + XGBoost CUDA across 3 seeds
    with exact Val (2017-08-01..15) early stopping + Google Drive & Base64 stdout sync.
"""

import argparse
import base64
import gc
import glob
import json
import os
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CAT_COLS = ["store_nbr", "family_idx", "cluster", "city_idx", "state_idx", "type_idx", "dayofweek"]


def start_heartbeat(interval_sec: int = 20) -> None:
    """Keep WebSocket alive during long GBDT fits."""
    def beat():
        while True:
            time.sleep(interval_sec)
            print("[HEARTBEAT] L4 GPU Gen-2 GBDT active...", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def ensure_environment_and_data() -> str:
    """Verify required GBDT libraries, Kaggle auth, and competition CSV files."""
    print("[INIT] Verifying packages and datasets...")
    try:
        import lightgbm  # noqa: F401
        import catboost  # noqa: F401
        import xgboost  # noqa: F401
    except ImportError:
        print("  - Installing missing ML packages (lightgbm, catboost, xgboost)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "lightgbm", "catboost", "xgboost", "scikit-learn", "kaggle"],
            check=True,
        )

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    username = os.environ.get("KAGGLE_USERNAME", "nobuyukikumagai")
    key = os.environ.get("KAGGLE_KEY", "757b1a8d9ce71f3b129fbae7cc34b1b2")
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json_path, 0o600)

    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        candidates = ["003_store-sales/data", "/content/data", "/content/kaggle/003_store-sales/data"]
        for c in candidates:
            if os.path.exists(os.path.join(c, "train.csv")):
                data_dir = c
                break

    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("  - Downloading dataset from Kaggle...")
        os.makedirs("data", exist_ok=True)
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "store-sales-time-series-forecasting", "-p", "data"],
            check=True,
        )
        for zip_file in glob.glob("data/*.zip"):
            import zipfile
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall("data")
        data_dir = "data"

    print(f"  - Datasets ready at: {data_dir}")
    return data_dir


def rolling_mean_matrix(mat: np.ndarray, window: int) -> np.ndarray:
    """Fast causal trailing rolling mean along axis=0 (inclusive of t) via cumulative sum."""
    csum = np.cumsum(mat, axis=0, dtype=np.float64)
    out = np.empty_like(mat, dtype=np.float32)
    out[:window] = (csum[:window] / np.arange(1, window + 1)[:, None]).astype(np.float32)
    out[window:] = ((csum[window:] - csum[:-window]) / window).astype(np.float32)
    return out


def rolling_std_matrix(mat: np.ndarray, window: int) -> np.ndarray:
    """Fast causal trailing rolling std along axis=0 via E[X^2] - E[X]^2."""
    m1 = rolling_mean_matrix(mat, window)
    m2 = rolling_mean_matrix(mat * mat, window)
    return np.sqrt(np.clip(m2 - m1 * m1, 0.0, None)).astype(np.float32)


def precompute_matrices(data_dir: str):
    """Build dense date x series matrices for instant vectorized multi-horizon slicing."""
    print("\n[DATA] Building vectorized Gen-2 multi-source matrices...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(data_dir, "stores.csv"))
    oil = pd.read_csv(os.path.join(data_dir, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(data_dir, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"), parse_dates=["date"])

    full_dates = pd.date_range("2013-01-01", "2017-08-31", freq="1D")
    n_dates = len(full_dates)
    date2idx = {d: i for i, d in enumerate(full_dates)}

    piv_sales = train.pivot(index="date", columns=["store_nbr", "family"], values="sales").reindex(full_dates).fillna(0.0)
    series_cols = piv_sales.columns
    n_series = len(series_cols)

    piv_promo = (
        pd.concat([
            train.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
            test.pivot(index="date", columns=["store_nbr", "family"], values="onpromotion"),
        ])
        .reindex(full_dates)
        .fillna(0.0)
    )
    sales_log = np.log1p(np.clip(piv_sales.values, 0.0, None)).astype(np.float32)
    promo_log = np.log1p(np.clip(piv_promo.values, 0.0, None)).astype(np.float32)

    # Precomputed causal cutoff rolling stats on sales_log (T, 1782)
    roll7 = rolling_mean_matrix(sales_log, 7)
    roll14 = rolling_mean_matrix(sales_log, 14)
    roll28 = rolling_mean_matrix(sales_log, 28)
    roll60 = rolling_mean_matrix(sales_log, 60)
    roll90 = rolling_mean_matrix(sales_log, 90)
    rstd7 = rolling_std_matrix(sales_log, 7)
    rstd28 = rolling_std_matrix(sales_log, 28)

    nz = (sales_log > 0.0).astype(np.float32)
    nz_ratio28 = rolling_mean_matrix(nz, 28)
    nz_ratio90 = rolling_mean_matrix(nz, 90)
    days_since_nz = np.zeros_like(sales_log, dtype=np.float32)
    curr = np.full(n_series, 91.0, dtype=np.float32)
    for t in range(n_dates):
        active = sales_log[t] > 0.0
        curr = np.where(active, 0.0, np.minimum(curr + 1.0, 91.0))
        days_since_nz[t] = curr

    # Promo causal rolling & target promo cross-sections
    promo_roll7 = rolling_mean_matrix(promo_log, 7)
    promo_roll28 = rolling_mean_matrix(promo_log, 28)

    store_ids = np.array([s for s, _ in series_cols], dtype=np.int32)
    fam_names = [f for _, f in series_cols]
    families = sorted(train["family"].unique())
    fam2idx = {f: i for i, f in enumerate(families)}
    fam_ids = np.array([fam2idx[f] for f in fam_names], dtype=np.int32)

    # Cross-sectional store & family aggregates -> shape (T, 54) and (T, 33) -> mapped to (T, 1782)
    store_mean_sales = np.zeros((n_dates, 54), dtype=np.float32)
    store_promo_sum = np.zeros((n_dates, 54), dtype=np.float32)
    for s in range(1, 55):
        mask = store_ids == s
        store_mean_sales[:, s - 1] = sales_log[:, mask].mean(axis=1)
        store_promo_sum[:, s - 1] = promo_log[:, mask].sum(axis=1)
    store_roll7_m = rolling_mean_matrix(store_mean_sales, 7)[:, store_ids - 1]
    store_roll28_m = rolling_mean_matrix(store_mean_sales, 28)[:, store_ids - 1]
    store_promo_m = store_promo_sum[:, store_ids - 1]

    fam_mean_sales = np.zeros((n_dates, len(families)), dtype=np.float32)
    fam_promo_sum = np.zeros((n_dates, len(families)), dtype=np.float32)
    for fi in range(len(families)):
        mask = fam_ids == fi
        fam_mean_sales[:, fi] = sales_log[:, mask].mean(axis=1)
        fam_promo_sum[:, fi] = promo_log[:, mask].sum(axis=1)
    fam_roll28_m = rolling_mean_matrix(fam_mean_sales, 28)[:, fam_ids]
    fam_sales_series = fam_mean_sales[:, fam_ids]
    fam_promo_m = fam_promo_sum[:, fam_ids]

    # Store transactions (normalized log1p + rolling 7/28) mapped to series
    piv_tx = transactions.pivot(index="date", columns="store_nbr", values="transactions").reindex(full_dates).fillna(0.0)
    tx_log = np.log1p(np.clip(piv_tx.values, 0.0, None)).astype(np.float32)
    tx_roll7 = rolling_mean_matrix(tx_log, 7)[:, store_ids - 1]
    tx_roll28 = rolling_mean_matrix(tx_log, 28)[:, store_ids - 1]
    tx_series = tx_log[:, store_ids - 1]

    # Oil macro
    oil_s = oil.set_index("date")["dcoilwtico"].reindex(full_dates).interpolate().ffill().bfill()
    oil_vals = oil_s.values.astype(np.float32)
    oil_diff14 = np.zeros_like(oil_vals)
    oil_diff14[14:] = oil_vals[14:] - oil_vals[:-14]

    # Holidays
    stores_map = stores.set_index("store_nbr")
    active_hol = holidays[~holidays["transferred"]].copy()
    nat_dates = set(active_hol[(active_hol["locale"] == "National") & (active_hol["type"] != "Work Day")]["date"])
    work_dates = set(active_hol[active_hol["type"] == "Work Day"]["date"])
    is_nat_hol = np.array([1.0 if d in nat_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_work_day = np.array([1.0 if d in work_dates else 0.0 for d in full_dates], dtype=np.float32)
    is_jan1 = np.array([1.0 if (d.month == 1 and d.day == 1) else 0.0 for d in full_dates], dtype=np.float32)

    reg_map = active_hol[active_hol["locale"] == "Regional"].groupby("locale_name")["date"].apply(set).to_dict()
    loc_map = active_hol[active_hol["locale"] == "Local"].groupby("locale_name")["date"].apply(set).to_dict()
    is_local_hol = np.zeros((n_dates, n_series), dtype=np.float32)
    for j, (s, _) in enumerate(series_cols):
        matched_dates = reg_map.get(stores_map.loc[s, "state"], set()) | loc_map.get(stores_map.loc[s, "city"], set())
        for d in matched_dates:
            if d in date2idx:
                is_local_hol[date2idx[d], j] = 1.0
    is_eff_hol = np.maximum(is_nat_hol[:, None], is_local_hol).astype(np.float32)
    is_hol_eve = np.zeros_like(is_eff_hol)
    is_hol_eve[:-1] = is_eff_hol[1:]

    # Regional / Calendar / Wage day / Earthquake
    is_wage = np.array(
        [
            1.0
            if (
                ((d.day == 15 or d.is_month_end) and d.dayofweek not in (5, 6))
                or (d.dayofweek == 4 and (
                    ((d + pd.Timedelta(days=1)).day == 15 or (d + pd.Timedelta(days=1)).is_month_end)
                    or ((d + pd.Timedelta(days=2)).day == 15 or (d + pd.Timedelta(days=2)).is_month_end)
                ))
            )
            else 0.0
            for d in full_dates
        ],
        dtype=np.float32,
    )
    is_wage_before = np.roll(is_wage, -1)
    is_wage_before[-1] = 0.0
    is_wage_after = np.roll(is_wage, 1)
    is_wage_after[0] = 0.0

    eq_start = pd.Timestamp("2016-04-16")
    eq_decay = np.array(
        [np.exp(-max(0, (d - eq_start).days) / 30.0) if d >= eq_start else 0.0 for d in full_dates],
        dtype=np.float32,
    )

    sierra_states = {"Pichincha", "Tungurahua", "Azuay", "Cotopaxi", "Chimborazo", "Imbabura", "Loja", "Bolivar", "Pastaza"}
    cities = sorted(stores["city"].unique())
    states = sorted(stores["state"].unique())
    types = sorted(stores["type"].unique())
    city2idx = {c: i for i, c in enumerate(cities)}
    state2idx = {s: i for i, s in enumerate(states)}
    type2idx = {t: i for i, t in enumerate(types)}

    static_meta = {
        "store_nbr": (store_ids - 1).astype(np.int32),
        "family_idx": fam_ids.astype(np.int32),
        "cluster": np.array([stores_map.loc[s, "cluster"] - 1 for s, _ in series_cols], dtype=np.int32),
        "city_idx": np.array([city2idx[stores_map.loc[s, "city"]] for s, _ in series_cols], dtype=np.int32),
        "state_idx": np.array([state2idx[stores_map.loc[s, "state"]] for s, _ in series_cols], dtype=np.int32),
        "type_idx": np.array([type2idx[stores_map.loc[s, "type"]] for s, _ in series_cols], dtype=np.int32),
        "is_sierra": np.array([1.0 if stores_map.loc[s, "state"] in sierra_states else 0.0 for s, _ in series_cols], dtype=np.float32),
        "is_manabi": np.array([1.0 if stores_map.loc[s, "state"] == "Manabi" else 0.0 for s, _ in series_cols], dtype=np.float32),
    }

    date_arrays = {
        "month": full_dates.month.values.astype(np.float32),
        "day": full_dates.day.values.astype(np.float32),
        "dayofweek": full_dates.dayofweek.values.astype(np.int32),
        "dayofyear": full_dates.dayofyear.values.astype(np.float32),
        "is_weekend": full_dates.dayofweek.isin([5, 6]).astype(np.float32),
        "is_school_sierra_cal": np.array([1.0 if (d.month == 8 and d.day >= 16) else 0.0 for d in full_dates], dtype=np.float32),
        "is_school_costa_cal": np.array([1.0 if ((d.month == 4 and d.day >= 15) or (d.month == 5 and d.day <= 15)) else 0.0 for d in full_dates], dtype=np.float32),
        "oil_vals": oil_vals,
        "oil_diff14": oil_diff14,
        "is_nat_hol": is_nat_hol,
        "is_work_day": is_work_day,
        "is_jan1": is_jan1,
        "is_wage": is_wage,
        "is_wage_before": is_wage_before,
        "is_wage_after": is_wage_after,
        "eq_decay": eq_decay,
    }

    print(f"  - Matrices ready: T={n_dates}, S={n_series}")
    return {
        "n_dates": n_dates,
        "n_series": n_series,
        "series_cols": series_cols,
        "date2idx": date2idx,
        "test_df": test,
        "sales_log": sales_log,
        "promo_log": promo_log,
        "roll7": roll7,
        "roll14": roll14,
        "roll28": roll28,
        "roll60": roll60,
        "roll90": roll90,
        "rstd7": rstd7,
        "rstd28": rstd28,
        "nz_ratio28": nz_ratio28,
        "nz_ratio90": nz_ratio90,
        "days_since_nz": days_since_nz,
        "promo_roll7": promo_roll7,
        "promo_roll28": promo_roll28,
        "store_roll7_m": store_roll7_m,
        "store_roll28_m": store_roll28_m,
        "store_promo_m": store_promo_m,
        "fam_roll28_m": fam_roll28_m,
        "fam_sales_series": fam_sales_series,
        "fam_promo_m": fam_promo_m,
        "tx_series": tx_series,
        "tx_roll7": tx_roll7,
        "tx_roll28": tx_roll28,
        "is_local_hol": is_local_hol,
        "is_eff_hol": is_eff_hol,
        "is_hol_eve": is_hol_eve,
        "static_meta": static_meta,
        "date_arrays": date_arrays,
    }


def materialize_horizon_frame(mats: dict, cutoffs: np.ndarray, horizons: np.ndarray, with_target: bool = True):
    """Build flat (len(cutoffs)*len(horizons)*S, F) DataFrame with exact zero-leak cutoff alignment."""
    S = mats["n_series"]
    n_c = len(cutoffs)
    n_h = len(horizons)
    n_dates = mats["n_dates"]

    # Grid ordering: (cutoff, horizon, series)
    c_grid = np.repeat(cutoffs, n_h * S)
    h_grid = np.tile(np.repeat(horizons, S), n_c)
    s_grid = np.tile(np.arange(S, dtype=np.int32), n_c * n_h)
    d_grid = c_grid + h_grid
    d_next = np.minimum(d_grid + 1, n_dates - 1)

    # Target-aligned exact weekday offsets strictly <= cutoff (<= c_grid)
    sdow_base = c_grid + ((h_grid - 1) % 7 + 1)

    data = {}
    sm = mats["static_meta"]
    da = mats["date_arrays"]

    # Categoricals (int32)
    data["store_nbr"] = sm["store_nbr"][s_grid]
    data["family_idx"] = sm["family_idx"][s_grid]
    data["cluster"] = sm["cluster"][s_grid]
    data["city_idx"] = sm["city_idx"][s_grid]
    data["state_idx"] = sm["state_idx"][s_grid]
    data["type_idx"] = sm["type_idx"][s_grid]
    data["dayofweek"] = da["dayofweek"][d_grid]

    # Horizon & cutoff recent sales
    data["horizon"] = h_grid.astype(np.float32)
    data["sales_c0"] = mats["sales_log"][c_grid, s_grid]
    data["sales_c1"] = mats["sales_log"][c_grid - 1, s_grid]
    data["sales_c2"] = mats["sales_log"][c_grid - 2, s_grid]
    data["sales_c7"] = mats["sales_log"][c_grid - 7, s_grid]

    # Cutoff rolling features & momentum
    data["roll_mean_7"] = mats["roll7"][c_grid, s_grid]
    data["roll_mean_14"] = mats["roll14"][c_grid, s_grid]
    data["roll_mean_28"] = mats["roll28"][c_grid, s_grid]
    data["roll_mean_60"] = mats["roll60"][c_grid, s_grid]
    data["roll_mean_90"] = mats["roll90"][c_grid, s_grid]
    data["roll_std_7"] = mats["rstd7"][c_grid, s_grid]
    data["roll_std_28"] = mats["rstd28"][c_grid, s_grid]
    data["trend_7_28"] = data["roll_mean_7"] - data["roll_mean_28"]
    data["trend_28_90"] = data["roll_mean_28"] - data["roll_mean_90"]
    data["nz_ratio28"] = mats["nz_ratio28"][c_grid, s_grid]
    data["nz_ratio90"] = mats["nz_ratio90"][c_grid, s_grid]
    data["days_since_nz"] = mats["days_since_nz"][c_grid, s_grid]

    # Target-aligned exact weekday lags (<= cutoff)
    sd1 = mats["sales_log"][sdow_base - 7, s_grid]
    sd2 = mats["sales_log"][sdow_base - 14, s_grid]
    sd3 = mats["sales_log"][sdow_base - 21, s_grid]
    sd4 = mats["sales_log"][sdow_base - 28, s_grid]
    sd8 = mats["sales_log"][sdow_base - 56, s_grid]
    sdow_m4 = 0.25 * (sd1 + sd2 + sd3 + sd4)
    sdow_m8 = 0.125 * (
        sd1 + sd2 + sd3 + sd4
        + mats["sales_log"][sdow_base - 35, s_grid]
        + mats["sales_log"][sdow_base - 42, s_grid]
        + mats["sales_log"][sdow_base - 49, s_grid]
        + sd8
    )
    data["sdow_lag1"] = sd1
    data["sdow_lag2"] = sd2
    data["sdow_lag4"] = sd4
    data["sdow_mean4"] = sdow_m4
    data["sdow_mean8"] = sdow_m8
    data["sdow_decay4"] = 0.4 * sd1 + 0.3 * sd2 + 0.2 * sd3 + 0.1 * sd4
    data["sdow_std4"] = np.sqrt(0.25 * ((sd1 - sdow_m4) ** 2 + (sd2 - sdow_m4) ** 2 + (sd3 - sdow_m4) ** 2 + (sd4 - sdow_m4) ** 2))
    data["sdow_diff_roll28"] = sdow_m4 - data["roll_mean_28"]

    # Exact 1Y & 2Y same-weekday YoY (both <= cutoff)
    data["yoy_364"] = mats["sales_log"][d_grid - 364, s_grid]
    data["yoy_371"] = mats["sales_log"][d_grid - 371, s_grid]
    data["yoy_728"] = mats["sales_log"][d_grid - 728, s_grid]
    data["yoy_growth"] = sdow_m4 - data["yoy_364"]

    # Store traffic & cross-sectional macro signals
    data["tx_c0"] = mats["tx_series"][c_grid, s_grid]
    data["tx_roll7"] = mats["tx_roll7"][c_grid, s_grid]
    data["tx_roll28"] = mats["tx_roll28"][c_grid, s_grid]
    data["tx_sdow1"] = mats["tx_series"][sdow_base - 7, s_grid]
    data["store_roll7"] = mats["store_roll7_m"][c_grid, s_grid]
    data["store_roll28"] = mats["store_roll28_m"][c_grid, s_grid]
    data["store_trend"] = data["store_roll7"] - data["store_roll28"]
    data["fam_roll28"] = mats["fam_roll28_m"][c_grid, s_grid]
    data["fam_sdow1"] = mats["fam_sales_series"][sdow_base - 7, s_grid]
    data["fam_yoy364"] = mats["fam_sales_series"][d_grid - 364, s_grid]

    # Known target-date promotions & cross-sections
    p_d = mats["promo_log"][d_grid, s_grid]
    p_c0 = mats["promo_log"][c_grid, s_grid]
    p_r28 = mats["promo_roll28"][c_grid, s_grid]
    data["promo_target"] = p_d
    data["promo_d_prev1"] = mats["promo_log"][d_grid - 1, s_grid]
    data["promo_d_next1"] = mats["promo_log"][d_next, s_grid]
    data["promo_c0"] = p_c0
    data["promo_roll7_d"] = mats["promo_roll7"][d_grid, s_grid]
    data["promo_diff_28"] = p_d - p_r28
    data["store_promo_sum"] = mats["store_promo_m"][d_grid, s_grid]
    data["fam_promo_sum"] = mats["fam_promo_m"][d_grid, s_grid]

    # Macro oil & calendar/holiday/school/earthquake covariates at target date d_grid
    data["oil_val"] = da["oil_vals"][d_grid]
    data["oil_diff14"] = da["oil_diff14"][d_grid]
    data["is_nat_hol"] = da["is_nat_hol"][d_grid]
    data["is_local_hol"] = mats["is_local_hol"][d_grid, s_grid]
    data["is_eff_hol"] = mats["is_eff_hol"][d_grid, s_grid]
    data["is_hol_eve"] = mats["is_hol_eve"][d_grid, s_grid]
    data["is_work_day"] = da["is_work_day"][d_grid]
    data["is_jan1"] = da["is_jan1"][d_grid]
    data["is_wage"] = da["is_wage"][d_grid]
    data["is_wage_before"] = da["is_wage_before"][d_grid]
    data["is_wage_after"] = da["is_wage_after"][d_grid]
    data["month"] = da["month"][d_grid]
    data["day"] = da["day"][d_grid]
    data["dayofyear"] = da["dayofyear"][d_grid]
    data["is_weekend"] = da["is_weekend"][d_grid]
    data["is_sierra_school"] = da["is_school_sierra_cal"][d_grid] * sm["is_sierra"][s_grid]
    data["is_costa_school"] = da["is_school_costa_cal"][d_grid] * (1.0 - sm["is_sierra"][s_grid])
    data["eq_decay_local"] = da["eq_decay"][d_grid] * (1.0 + sm["is_manabi"][s_grid])

    df = pd.DataFrame(data)
    y = mats["sales_log"][d_grid, s_grid] if with_target else None
    return df, y, c_grid, h_grid, s_grid


def train_and_evaluate(mats: dict, smoke_test: bool = False):
    """Train 3-Engine x 3-Seed Direct Multi-Horizon GBDTs on L4 GPU + optimize Val blend."""
    import lightgbm as lgb
    import xgboost as xgb

    if not smoke_test:
        import catboost as cb

    d2i = mats["date2idx"]

    # Regular 6D cutoffs (coprime to 7) + August & recent-month seasonal reinforcement (<= 2017-07-15)
    base_cuts = pd.date_range("2015-05-01", "2017-07-15", freq="6D")
    aug15_cuts = pd.date_range("2015-08-01", "2015-08-31", freq="3D")
    aug16_cuts = pd.date_range("2016-08-01", "2016-08-31", freq="3D")
    recent_cuts = pd.date_range("2017-06-15", "2017-07-15", freq="3D")
    all_cuts = sorted(set(base_cuts) | set(aug15_cuts) | set(aug16_cuts) | set(recent_cuts))
    if smoke_test:
        all_cuts = all_cuts[-6:]

    train_cutoffs = np.array([d2i[d] for d in all_cuts], dtype=np.int32)
    val_cutoffs = np.array([d2i[pd.Timestamp("2017-07-31")]], dtype=np.int32)
    test_cutoffs = np.array([d2i[pd.Timestamp("2017-08-15")]], dtype=np.int32)

    print(f"\n[BUILD] Materializing Train ({len(train_cutoffs)} cutoffs x 16H), Val (15H), Test (16H)...")
    X_trn, y_trn, _, _, _ = materialize_horizon_frame(mats, train_cutoffs, np.arange(1, 17, dtype=np.int32), True)
    X_val, y_val, _, _, val_s = materialize_horizon_frame(mats, val_cutoffs, np.arange(1, 16, dtype=np.int32), True)
    X_tst, _, _, tst_h, tst_s = materialize_horizon_frame(mats, test_cutoffs, np.arange(1, 17, dtype=np.int32), False)
    print(f"  - Shapes: X_trn={X_trn.shape} | X_val={X_val.shape} | X_tst={X_tst.shape}")

    val_dead_mask = mats["days_since_nz"][val_cutoffs[0], val_s] >= 21.0

    # Prepare category-dtype views for LightGBM / XGBoost (CatBoost keeps int32 DataFrame)
    X_trn_cat = X_trn.copy()
    X_val_cat = X_val.copy()
    X_tst_cat = X_tst.copy()
    for col in CAT_COLS:
        cats = np.union1d(X_trn[col].unique(), np.union1d(X_val[col].unique(), X_tst[col].unique()))
        dtype = pd.CategoricalDtype(categories=cats)
        X_trn_cat[col] = X_trn_cat[col].astype(dtype)
        X_val_cat[col] = X_val_cat[col].astype(dtype)
        X_tst_cat[col] = X_tst_cat[col].astype(dtype)

    seeds = [42] if smoke_test else [42, 2026, 777]
    lgb_est = 30 if smoke_test else 2500
    xgb_est = 30 if smoke_test else 2500
    cb_est = 30 if smoke_test else 3500

    preds_val = {"lgb": [], "xgb": [], "cb": []}
    preds_tst = {"lgb": [], "xgb": [], "cb": []}

    t0 = time.time()
    for seed in seeds:
        print(f"\n--- Seed {seed} [{time.time() - t0:.0f}s] ---")

        # 1. LightGBM Global
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=lgb_est,
            learning_rate=0.05,
            num_leaves=95,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            force_col_wise=True,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_trn_cat,
            y_trn,
            eval_set=[(X_val_cat, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
        )
        pv_lgb = np.clip(lgb_model.predict(X_val_cat), 0.0, None)
        pt_lgb = np.clip(lgb_model.predict(X_tst_cat), 0.0, None)
        preds_val["lgb"].append(pv_lgb)
        preds_tst["lgb"].append(pt_lgb)
        print(f"  [LGB] best_iter={lgb_model.best_iteration_} | Val Raw: {np.sqrt(np.mean((pv_lgb - y_val)**2)):.5f}", flush=True)
        del lgb_model
        gc.collect()

        # 2. XGBoost CUDA Global
        xgb_device = "cpu" if smoke_test else "cuda"
        xgb_model = xgb.XGBRegressor(
            n_estimators=xgb_est,
            learning_rate=0.04,
            max_depth=8,
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            tree_method="hist",
            device=xgb_device,
            enable_categorical=True,
            early_stopping_rounds=80,
            eval_metric="rmse",
            random_state=seed,
        )
        xgb_model.fit(X_trn_cat, y_trn, eval_set=[(X_val_cat, y_val)], verbose=False)
        pv_xgb = np.clip(xgb_model.predict(X_val_cat), 0.0, None)
        pt_xgb = np.clip(xgb_model.predict(X_tst_cat), 0.0, None)
        preds_val["xgb"].append(pv_xgb)
        preds_tst["xgb"].append(pt_xgb)
        print(f"  [XGB] best_iter={xgb_model.best_iteration} | Val Raw: {np.sqrt(np.mean((pv_xgb - y_val)**2)):.5f}", flush=True)
        del xgb_model
        gc.collect()

        # 3. CatBoost GPU Global
        if not smoke_test:
            trn_pool = cb.Pool(X_trn, y_trn, cat_features=CAT_COLS)
            val_pool = cb.Pool(X_val, y_val, cat_features=CAT_COLS)
            tst_pool = cb.Pool(X_tst, cat_features=CAT_COLS)
            cb_model = cb.CatBoostRegressor(
                iterations=cb_est,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=5.0,
                loss_function="RMSE",
                task_type="GPU",
                random_seed=seed,
                early_stopping_rounds=100,
                verbose=500,
            )
            cb_model.fit(trn_pool, eval_set=val_pool)
            pv_cb = np.clip(cb_model.predict(val_pool), 0.0, None)
            pt_cb = np.clip(cb_model.predict(tst_pool), 0.0, None)
            preds_val["cb"].append(pv_cb)
            preds_tst["cb"].append(pt_cb)
            print(f"  [CB]  best_iter={cb_model.get_best_iteration()} | Val Raw: {np.sqrt(np.mean((pv_cb - y_val)**2)):.5f}", flush=True)
            del cb_model, trn_pool, val_pool, tst_pool
            gc.collect()

    # Seed averages per engine
    active_engines = ["lgb", "xgb"] if smoke_test else ["lgb", "xgb", "cb"]
    val_avg = {k: np.mean(preds_val[k], axis=0) for k in active_engines}
    tst_avg = {k: np.mean(preds_tst[k], axis=0) for k in active_engines}

    print("\n=== Engine Multi-Seed Val RMSLE ===")
    for k in active_engines:
        raw_s = np.sqrt(np.mean((val_avg[k] - y_val) ** 2))
        z_v = np.where(val_dead_mask, 0.0, val_avg[k])
        z_s = np.sqrt(np.mean((z_v - y_val) ** 2))
        print(f"  {k.upper():4s}: Val Raw = {raw_s:.5f} | Val ZMask = {z_s:.5f}")

    # Optimal non-negative convex weights across engines on Val via SLSQP
    from scipy.optimize import minimize

    M_val = np.vstack([val_avg[k] for k in active_engines])  # (K, N)
    M_tst = np.vstack([tst_avg[k] for k in active_engines])  # (K, N)

    def obj(w):
        pred = np.dot(w, M_val)
        return np.mean((pred - y_val) ** 2)

    w0 = np.ones(len(active_engines)) / len(active_engines)
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(active_engines),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    w_opt = res.x
    ens_val = np.clip(np.dot(w_opt, M_val), 0.0, None)
    ens_tst = np.clip(np.dot(w_opt, M_tst), 0.0, None)
    ens_val_z = np.where(val_dead_mask, 0.0, ens_val)

    print(f"\n[EVAL] Optimal Engine Weights ({active_engines}): {np.round(w_opt, 4)}")
    print(f"[EVAL] Gen-2 GBDT ENSEMBLE Val Raw RMSLE:   {np.sqrt(np.mean((ens_val - y_val)**2)):.5f}")
    print(f"[EVAL] Gen-2 GBDT ENSEMBLE Val ZMask RMSLE: {np.sqrt(np.mean((ens_val_z - y_val)**2)):.5f}")

    # Reconstruct test.csv order using exact (date, store_nbr, family)
    test_dates = pd.date_range("2017-08-16", "2017-08-31")
    series_cols = mats["series_cols"]
    rows_df = pd.DataFrame({
        "date": test_dates[tst_h - 1],
        "store_nbr": [series_cols[s][0] for s in tst_s],
        "family": [series_cols[s][1] for s in tst_s],
        "sales": np.clip(np.expm1(ens_tst), 0.0, None),
    })
    test_sub = pd.merge(mats["test_df"][["id", "date", "store_nbr", "family"]], rows_df, on=["date", "store_nbr", "family"], how="left")
    sub = pd.DataFrame({"id": test_sub["id"], "sales": test_sub["sales"]})

    assert len(sub) == 28512, f"Expected 28512 rows, got {len(sub)}"
    assert sub["sales"].isna().sum() == 0, "NaNs found in predictions!"

    if smoke_test:
        print("\n[SMOKE TEST PASSED] Direct Multi-Horizon GBDT pipeline verified cleanly!")
        return sub

    out_path = "003_store-sales/submission_exp054_colab_l4_gbdt_gen2.csv"
    os.makedirs("003_store-sales", exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Saved to: {out_path} ({len(sub)} rows, Mean={sub['sales'].mean():.2f})")

    drive_dir = "/content/drive/MyDrive"
    if os.path.exists(drive_dir):
        drive_path = os.path.join(drive_dir, "submission_exp054_colab_l4_gbdt_gen2.csv")
        sub.to_csv(drive_path, index=False)
        print(f"  - Synchronized to Google Drive: {drive_path}")

    with open(out_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"\n===BASE64_SUBMISSION_START===\n{b64}\n===BASE64_SUBMISSION_END===")
    return sub


def main():
    parser = argparse.ArgumentParser(description="Exp #054: Gen-2 Direct Multi-Horizon GBDT on Colab L4 GPU.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  003 Store Sales: Exp #054 - Gen-2 Direct Multi-Horizon GBDT (Colab L4 GPU)")
    print("=" * 80)

    if not args.smoke_test:
        start_heartbeat(20)
    data_dir = ensure_environment_and_data()
    mats = precompute_matrices(data_dir)
    train_and_evaluate(mats, smoke_test=args.smoke_test)
    print("\n[COMPLETE] Exp #054 finished successfully!")


if __name__ == "__main__":
    main()
