"""
003_store-sales: Feature Engineering & Model (Experiment #002)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク: カレンダー拡張 ＋ 給与日 ＋ 祝日振替 ＋ 店舗メタ情報 ＋ 原油価格の統合モデル
評価指標: RMSLE (Root Mean Squared Logarithmic Error)
検証設計: 直近16日間 (2017-07-31 〜 2017-08-15) を手元検証（Validation）とする Time-based Split
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Windows環境でのUTF-8出力確保
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_data_dir() -> str:
    """ローカルまたはKaggle Notebook環境のデータディレクトリを自動検出する"""
    candidates = [
        "003_store-sales/data",
        "data",
        "../data",
        "/kaggle/input/competitions/store-sales-time-series-forecasting",
        "/kaggle/input/store-sales-time-series-forecasting",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    matches = glob.glob("**/store-sales-time-series-forecasting/train.csv", recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError("train.csv が見つかりませんでした。データディレクトリを確認してください。")


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE (Root Mean Squared Logarithmic Error) を計算する"""
    y_pred_clipped = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped))))


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, data_dir: str):
    """ドメイン知識に基づく多角的特徴量エンジニアリング"""
    print("  [特徴量構築 1/4] カレンダー & 給与日特徴量作成...")
    for df in [train_df, test_df]:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df["dayofyear"] = df["date"].dt.dayofyear
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

        # エクアドル給与日（毎月15日と月末）
        df["is_wage_day"] = ((df["day"] == 15) | (df["is_month_end"] == 1)).astype(int)
        # 給与日翌日（16日と月初1日：買い出しピーク）
        df["is_wage_day_after"] = ((df["day"] == 16) | (df["is_month_start"] == 1)).astype(int)

        # 2016年4月16日の大地震 (M7.8) 影響期間フラグ (1ヶ月間)
        df["is_earthquake"] = ((df["date"] >= "2016-04-16") & (df["date"] <= "2016-05-16")).astype(int)

    # 2. 店舗メタ情報 (stores.csv)
    print("  [特徴量構築 2/4] 店舗メタ情報 (stores.csv) 結合...")
    stores_path = os.path.join(data_dir, "stores.csv")
    if os.path.exists(stores_path):
        stores_df = pd.read_csv(stores_path)
        stores_df = stores_df.rename(columns={"type": "store_type"})
        train_df = train_df.merge(stores_df, on="store_nbr", how="left")
        test_df = test_df.merge(stores_df, on="store_nbr", how="left")

    # 3. 祝日・イベント情報 (holidays_events.csv)
    print("  [特徴量構築 3/4] 祝日振替・イベント情報 (holidays_events.csv) 結合...")
    holidays_path = os.path.join(data_dir, "holidays_events.csv")
    if os.path.exists(holidays_path):
        holidays_df = pd.read_csv(holidays_path)
        holidays_df["date"] = pd.to_datetime(holidays_df["date"])

        # 振替休日処理: transferred == True の日は休日ではない
        real_holidays = holidays_df[~holidays_df["transferred"]].copy()
        # Work Day (平日出勤) は休日扱いから除外
        real_holidays = real_holidays[real_holidays["type"] != "Work Day"]

        # 全国祝日フラグ
        nat_holidays = real_holidays[real_holidays["locale"] == "National"][["date"]].drop_duplicates()
        nat_holidays["is_national_holiday"] = 1

        train_df = train_df.merge(nat_holidays, on="date", how="left")
        test_df = test_df.merge(nat_holidays, on="date", how="left")

        train_df["is_national_holiday"] = train_df["is_national_holiday"].fillna(0).astype(int)
        test_df["is_national_holiday"] = test_df["is_national_holiday"].fillna(0).astype(int)

    # 4. 原油価格情報 (oil.csv)
    print("  [特徴量構築 4/4] 原油価格 (oil.csv) 欠損補間 & 結合...")
    oil_path = os.path.join(data_dir, "oil.csv")
    if os.path.exists(oil_path):
        oil_df = pd.read_csv(oil_path)
        oil_df["date"] = pd.to_datetime(oil_df["date"])
        # 日付を全期間でリインデックスして線形補間
        all_dates = pd.date_range(start=oil_df["date"].min(), end="2017-08-31")
        oil_full = pd.DataFrame({"date": all_dates}).merge(oil_df, on="date", how="left")
        oil_full["dcoilwtico"] = oil_full["dcoilwtico"].interpolate(method="linear").bfill()
        oil_full["oil_7d_mean"] = oil_full["dcoilwtico"].rolling(7, min_periods=1).mean()

        train_df = train_df.merge(oil_full[["date", "dcoilwtico", "oil_7d_mean"]], on="date", how="left")
        test_df = test_df.merge(oil_full[["date", "dcoilwtico", "oil_7d_mean"]], on="date", how="left")

    return train_df, test_df


def main():
    print("=" * 65)
    print("  003 Store Sales: 実験 #002 (ドメイン特徴量拡張モデル)")
    print("=" * 65)

    data_dir = find_data_dir()
    print(f"\n[1/5] データ読み込み元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    # 特徴量エンジニアリング
    print("\n[2/5] 特徴量エンジニアリング実行中...")
    train_df, test_df = build_features(train_df, test_df, data_dir)

    # カテゴリ変数の型変換
    categorical_cols = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")

    ignore_cols = ["id", "date", "sales"]
    features = [c for c in train_df.columns if c not in ignore_cols]

    print(f"\n  - 拡張後の総特徴量数: {len(features)} 列")
    print(f"  - 特徴量一覧: {features}")

    # 時系列バリデーション分割 (Time-based Split)
    val_start_date = "2017-07-31"
    val_end_date = "2017-08-15"
    train_start_date = "2016-01-01"

    print(f"\n[3/5] 時系列バリデーション分割:")
    print(f"  - 学習データ: {train_start_date} 〜 2017-07-30")
    print(f"  - 検証データ: {val_start_date} 〜 {val_end_date} (16日間)")

    trn_mask = (train_df["date"] >= train_start_date) & (train_df["date"] < val_start_date)
    val_mask = (train_df["date"] >= val_start_date) & (train_df["date"] <= val_end_date)

    X_train, y_train = train_df.loc[trn_mask, features], train_df.loc[trn_mask, "sales"].values
    X_val, y_val = train_df.loc[val_mask, features], train_df.loc[val_mask, "sales"].values
    X_test = test_df[features]

    print(f"  - 学習サンプル数: {len(X_train):,} 件")
    print(f"  - 検証サンプル数: {len(X_val):,} 件")

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # LightGBM 回帰モデル学習
    print("\n[4/5] LightGBM 回帰モデル学習中...")
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_estimators": 1000,
    }

    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=100)]

    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=callbacks,
    )

    # 手元検証スコア評価
    val_pred_log = model.predict(X_val)
    val_pred = np.clip(np.expm1(val_pred_log), 0, None)

    val_rmsle = calculate_rmsle(y_val, val_pred)
    print("\n" + "=" * 55)
    print(f"  [実験 #002] 手元検証スコア (Val RMSLE): {val_rmsle:.5f}")
    print(f"  (参考: 実験 #001 ベースライン: 0.41440)")
    print("=" * 55)

    # 特徴量重要度 Top 10
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n特徴量重要度 Top 10:")
    for feat, imp in importances.head(10).items():
        print(f"  - {feat:20s}: {imp:6d}")

    # テストデータ予測 & 提出ファイル作成
    print("\n[5/5] テストデータ予測と提出ファイル作成...")
    test_pred_log = model.predict(X_test)
    test_pred = np.clip(np.expm1(test_pred_log), 0, None)

    sub_df["sales"] = test_pred
    out_path = "003_store-sales/submission_exp002.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub_df.to_csv(out_path, index=False)

    print(f"  - 提出ファイルを保存しました: {out_path}")
    print(f"  - 先頭5行:\n{sub_df.head()}")


if __name__ == "__main__":
    main()
