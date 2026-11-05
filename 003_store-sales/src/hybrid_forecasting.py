"""
003_store-sales: Hybrid Trend + GBDT Residual Forecasting (Experiment #009)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク:
  1. 第1段階: Ridge回帰による系列ごとの決定論的成長トレンド外挿
  2. 第2段階: 3大GBDT (LGB 50% + CB 30% + XGB 20%) による再帰的残差学習
  3. ドメイン補正: 給与日 (Quincena) 金曜前倒し判定 & 元旦 (1/1) 休業日除外
  4. ハードウェア最適化: Intel N100 (4コア/4スレッド CPU) 最適化設定
評価指標: RMSLE (Root Mean Squared Logarithmic Error)
"""

import os
import sys
import glob
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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

    # 週末の場合は前倒し金曜日に支給
    actual_wage_day = (is_wage & ~is_weekend) | (is_friday & (wage_sat | wage_sun))
    actual_wage_day_after = actual_wage_day.shift(1).fillna(False)

    return actual_wage_day.astype(int), actual_wage_day_after.astype(int)


def main():
    print("=" * 75)
    print("  003 Store Sales: 実験 #009 (ハイブリッド Trend + GBDT 残差再帰予測)")
    print("  実行環境最適化: Intel N100 (4 Cores CPU Mode)")
    print("=" * 75)

    data_dir = find_data_dir()
    print(f"\n[1/6] データ読み込み元: {data_dir}")

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

    # ドメイン補正: 給与日金曜前倒し (Quincena)
    all_df["is_wage_day"], all_df["is_wage_day_after"] = create_wage_day_features(all_df["date"])

    # 系列ソート
    all_df = all_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    group_promo = all_df.groupby(["store_nbr", "family"])["onpromotion"]

    print("\n[2/6] 特売先行・後行特徴量の構築...")
    all_df["promo_lag_1"] = group_promo.shift(1).fillna(0)
    all_df["promo_lag_2"] = group_promo.shift(2).fillna(0)
    all_df["promo_lag_7"] = group_promo.shift(7).fillna(0)
    all_df["promo_lead_1"] = group_promo.shift(-1).bfill().fillna(0)
    all_df["promo_lead_2"] = group_promo.shift(-2).bfill().fillna(0)
    all_df["promo_diff_1"] = all_df["onpromotion"] - all_df["promo_lag_1"]
    all_df["promo_rolling_mean_7"] = group_promo.transform(lambda x: x.rolling(7, min_periods=1).mean())

    # =========================================================================
    # 第1段階: Ridge 回帰による系列別中長期成長トレンド学習 & 外挿
    # =========================================================================
    print("\n[3/6] 第1段階: 系列ごとの線形成長トレンド (Ridge回帰) フィッティング...")
    base_date = pd.to_datetime("2016-01-01")
    all_df["time_step"] = (all_df["date"] - base_date).dt.days
    all_df["time_step_norm"] = all_df["time_step"] / 365.25

    # トレンド用特徴量: 時間経過 t, t^2, 曜日周期 (sin/cos), 月周期 (sin/cos)
    t_feat = pd.DataFrame(index=all_df.index)
    t_feat["t"] = all_df["time_step_norm"]
    t_feat["t_sq"] = all_df["time_step_norm"] ** 2
    t_feat["dow_sin"] = np.sin(2 * np.pi * all_df["dayofweek"] / 7)
    t_feat["dow_cos"] = np.cos(2 * np.pi * all_df["dayofweek"] / 7)
    t_feat["month_sin"] = np.sin(2 * np.pi * all_df["month"] / 12)
    t_feat["month_cos"] = np.cos(2 * np.pi * all_df["month"] / 12)

    val_start_date = pd.to_datetime("2017-07-31")
    train_mask = (all_df["date"] >= base_date) & (all_df["date"] < val_start_date)
    # 元旦 (1/1) 休業日の外れ値ノイズを除外
    is_new_year = (all_df["month"] == 1) & (all_df["day"] == 1)
    clean_train_mask = train_mask & (~is_new_year)

    trend_preds = np.zeros(len(all_df), dtype=np.float32)

    # 1,782系列ごとに高速フィッティング (NumPy解析解)
    all_df["series_id"] = all_df["store_nbr"].astype(str) + "_" + all_df["family"]
    series_list = all_df["series_id"].unique()

    X_trend_all = t_feat.values
    y_all_log = all_df["log_sales"].values

    for sid in series_list:
        s_mask = (all_df["series_id"] == sid)
        s_trn_idx = np.where(s_mask & clean_train_mask)[0]
        s_all_idx = np.where(s_mask)[0]

        X_s_trn = X_trend_all[s_trn_idx]
        y_s_trn = y_all_log[s_trn_idx]

        ridge = Ridge(alpha=10.0)
        ridge.fit(X_s_trn, y_s_trn)
        trend_preds[s_all_idx] = ridge.predict(X_trend_all[s_all_idx])

    all_df["pred_trend_log"] = trend_preds
    # 残差算出: residual = log_sales - trend
    all_df["residual"] = all_df["log_sales"] - all_df["pred_trend_log"]
    print("  - 全1,782系列のRidgeトレンドフィッティング完了！")

    # =========================================================================
    # 第2段階: 残差に対するラグ & 移動平均特徴量構築
    # =========================================================================
    print("\n[4/6] 第2段階: 残差ラグ (Residual Lags) & 3大GBDT学習 (Intel N100 最適化)...")
    group_res = all_df.groupby(["store_nbr", "family"])["residual"]
    for lag in [1, 2, 3, 7, 14, 21, 28, 364]:
        all_df[f"res_lag_{lag}"] = group_res.shift(lag)
    all_df["res_lag_364"] = all_df["res_lag_364"].fillna(all_df["res_lag_28"])

    lag1_res = group_res.shift(1)
    all_df["res_rolling_mean_7"] = lag1_res.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(7, min_periods=1).mean())
    all_df["res_rolling_mean_14"] = lag1_res.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(14, min_periods=1).mean())
    all_df["res_rolling_mean_28"] = lag1_res.groupby([all_df["store_nbr"], all_df["family"]]).transform(lambda x: x.rolling(28, min_periods=1).mean())

    features = [
        "store_nbr", "onpromotion", "year", "month", "day", "dayofweek", "is_weekend",
        "dayofyear", "weekofyear", "is_month_start", "is_month_end", "is_wage_day", "is_wage_day_after",
        "res_lag_1", "res_lag_2", "res_lag_3", "res_lag_7", "res_lag_14",
        "res_lag_21", "res_lag_28", "res_lag_364",
        "res_rolling_mean_7", "res_rolling_mean_14", "res_rolling_mean_28",
        "promo_lag_1", "promo_lag_2", "promo_lag_7",
        "promo_lead_1", "promo_lead_2",
        "promo_diff_1", "promo_rolling_mean_7"
    ]
    print(f"  - 特徴量数: {len(features)} 列")

    # 33ファミリー別の学習モデル辞書
    families = train_df["family"].unique()
    trained_models = {}

    for fam in families:
        fam_trn_mask = clean_train_mask & (all_df["family"] == fam)
        fam_train_df = all_df.loc[fam_trn_mask].dropna(subset=features)

        X_trn = fam_train_df[features].copy()
        y_trn_res = fam_train_df["residual"].values

        # 1. LightGBM (Intel N100: n_jobs=4)
        lgb_model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            random_state=42,
            verbose=-1,
            n_estimators=300,
            n_jobs=4,
        )
        X_trn_lgb = X_trn.copy()
        X_trn_lgb["store_nbr"] = X_trn_lgb["store_nbr"].astype("category")
        lgb_model.fit(X_trn_lgb, y_trn_res)

        # 2. CatBoost (Intel N100: thread_count=4)
        cb_model = cb.CatBoostRegressor(
            iterations=300,
            learning_rate=0.06,
            depth=6,
            loss_function="RMSE",
            task_type="CPU",
            thread_count=4,
            random_seed=42,
            verbose=0,
        )
        cb_model.fit(X_trn, y_trn_res, verbose=0)

        # 3. XGBoost (Intel N100: hist, n_jobs=4)
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cpu",
            n_jobs=4,
            random_state=42,
        )
        xgb_model.fit(X_trn, y_trn_res, verbose=False)

        trained_models[fam] = {
            "lgb": lgb_model,
            "cb": cb_model,
            "xgb": xgb_model,
        }

    print("  - 全33ファミリーの 3大GBDT 残差学習完了！")

    # =========================================================================
    # 第3段階: 再帰的ハイブリッド推論ループ (Recursive Roll-forward Engine)
    # =========================================================================
    def run_recursive_hybrid_inference(df_source: pd.DataFrame, start_date, end_date, is_val: bool):
        # ピボットテーブル: 行=date, 列=(store_nbr, family), 値=residual
        pivot_res = df_source.pivot(index="date", columns=["store_nbr", "family"], values="residual").copy()

        dates = pd.date_range(start_date, end_date)
        step_predictions = []
        active_zero_pairs = val_zero_pairs if is_val else zero_pairs

        for current_date in dates:
            day_mask = (df_source["date"] == current_date)
            day_df = df_source.loc[day_mask].copy()

            # 残差ラグとローリング平均を動的代入
            for lag in [1, 2, 3, 7, 14, 21, 28, 364]:
                target_d = current_date - timedelta(days=lag)
                alt_d = current_date - timedelta(days=28)
                lag_series = pivot_res.loc[target_d] if target_d in pivot_res.index else pivot_res.loc[alt_d]
                day_df[f"res_lag_{lag}"] = day_df.set_index(["store_nbr", "family"]).index.map(lag_series).values

            d_1 = current_date - timedelta(days=1)
            for window in [7, 14, 28]:
                d_win = current_date - timedelta(days=window)
                mean_series = pivot_res.loc[d_win:d_1].mean(axis=0)
                day_df[f"res_rolling_mean_{window}"] = day_df.set_index(["store_nbr", "family"]).index.map(mean_series).values

            day_preds_dict = {}
            for fam in families:
                fam_mask = (day_df["family"] == fam)
                if not fam_mask.any():
                    continue

                X_day = day_df.loc[fam_mask, features].copy()
                X_day_lgb = X_day.copy()
                X_day_lgb["store_nbr"] = X_day_lgb["store_nbr"].astype("category")

                # 3大GBDT 残差予測 (50:30:20)
                p_res_lgb = trained_models[fam]["lgb"].predict(X_day_lgb)
                p_res_cb = trained_models[fam]["cb"].predict(X_day)
                p_res_xgb = trained_models[fam]["xgb"].predict(X_day)
                pred_res = 0.50 * p_res_lgb + 0.30 * p_res_cb + 0.20 * p_res_xgb

                # トレンド値と合算: z_final = z_trend + r_pred
                trend_val = day_df.loc[fam_mask, "pred_trend_log"].values
                pred_log_sales = trend_val + pred_res
                pred_sales = np.clip(np.expm1(pred_log_sales), 0, None)

                # ゼロ売上フィルター
                for idx, (st, val) in enumerate(zip(day_df.loc[fam_mask, "store_nbr"], pred_sales)):
                    if (st, fam) in active_zero_pairs:
                        pred_sales[idx] = 0.0

                for idx_orig, p_val in zip(day_df.loc[fam_mask].index, pred_sales):
                    day_preds_dict[idx_orig] = p_val

            day_df["pred_sales"] = pd.Series(day_preds_dict)
            step_predictions.append(day_df[["id", "date", "store_nbr", "family", "sales", "pred_sales", "pred_trend_log"]])

            # ピボットテーブルに更新された残差をフィードバック！
            # 確定予測売上から実際の残差を逆算して保存
            for _, row in day_df.iterrows():
                st = row["store_nbr"]
                fm = row["family"]
                actual_log_pred = np.log1p(row["pred_sales"])
                updated_res = actual_log_pred - row["pred_trend_log"]
                pivot_res.loc[current_date, (st, fm)] = updated_res

        return pd.concat(step_predictions, ignore_index=True)

    print("\n[5/6] 再帰的ハイブリッド推論の実行...")
    # A. 手元検証 (2017-07-31 〜 2017-08-15)
    print("  - 手元検証データ（直近16日間）に対する再帰的推論中...")
    val_result = run_recursive_hybrid_inference(all_df, val_start_date, pd.to_datetime("2017-08-15"), is_val=True)
    val_rmsle = calculate_rmsle(val_result["sales"].values, val_result["pred_sales"].values)

    print("\n" + "=" * 70)
    print(f"  🏆 実験 #009 ハイブリッド推論 手元検証 RMSLE: {val_rmsle:.5f}")
    print(f"     (参考: 実験 #008 再帰単体: 0.39956, 実験 #006 Direct: 0.40219)")
    print("=" * 70)

    # B. 本番テスト推論 (2017-08-16 〜 2017-08-31)
    print("\n  - テストデータ（2017-08-16 〜 2017-08-31）に対する再帰的推論中...")
    test_result = run_recursive_hybrid_inference(all_df, pd.to_datetime("2017-08-16"), pd.to_datetime("2017-08-31"), is_val=False)

    sub_final = test_result[["id", "pred_sales"]].rename(columns={"pred_sales": "sales"}).sort_values("id").reset_index(drop=True)
    out_path = "003_store-sales/submission_exp009.csv"
    sub_final.to_csv(out_path, index=False)
    print(f"\n[6/6] 提出ファイルを保存しました: {out_path}")
    print(f"  - 行数: {len(sub_final)} 行")
    print(f"  - 先頭10行:\n{sub_final.head(10)}")


if __name__ == "__main__":
    main()
