"""
003_store-sales: Lag & Rolling Features Model (Experiment #003)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク: テスト16日間を考慮したリーク完全ゼロの過去実績（Lag & Rolling）特徴量モデル
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
    """RMSLE (Root Mean Squared Logarithmic Error) を計算する"""
    y_pred_clipped = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped))))


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    店舗×商品ファミリーごとのラグ特徴量およびローリング移動平均を作成する
    重要: テスト期間は16日間のため、未来リークを防ぐため最小ラグは16日(>=16)とする！
    """
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # 目的変数の対数売上に対してラグを計算（スケール安定化）
    group = df.groupby(["store_nbr", "family"])["log_sales"]

    # 1. ラグ特徴量 (16日前、21日前=3週前同曜日、28日前=4週前同曜日、35日前=5週前同曜日)
    print("  - ラグ特徴量作成中 (Lag 16, 21, 28, 35)...")
    for lag in [16, 21, 28, 35]:
        df[f"sales_lag_{lag}"] = group.shift(lag)

    # 2. ローリング移動平均 & 標準偏差 (16日前起点の7日・14日・28日移動平均)
    print("  - ローリング移動平均作成中 (Rolling 7, 14, 28 from Lag 16)...")
    # shift(16) した系列に対してローリング計算
    lag16_series = group.shift(16)
    df["sales_rolling_mean_7"] = lag16_series.groupby([df["store_nbr"], df["family"]]).transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["sales_rolling_mean_14"] = lag16_series.groupby([df["store_nbr"], df["family"]]).transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )
    df["sales_rolling_mean_28"] = lag16_series.groupby([df["store_nbr"], df["family"]]).transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    df["sales_rolling_std_7"] = lag16_series.groupby([df["store_nbr"], df["family"]]).transform(
        lambda x: x.rolling(7, min_periods=1).std()
    ).fillna(0)

    return df


def main():
    print("=" * 65)
    print("  003 Store Sales: 実験 #003 (過去実績ラグ & 移動平均モデル)")
    print("=" * 65)

    data_dir = find_data_dir()
    print(f"\n[1/5] データ読み込み元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    train_df["date"] = pd.to_datetime(train_df["date"])
    test_df["date"] = pd.to_datetime(test_df["date"])

    # ラグ計算のために train と test を縦結合
    # testの sales は NaN
    test_df["sales"] = np.nan
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    # 目的変数の対数売上列（log1p）
    all_df["log_sales"] = np.log1p(all_df["sales"])

    # 2. カレンダー & 給与日特徴量
    print("\n[2/5] カレンダー & 給与日特徴量作成...")
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

    # 3. ラグ & ローリング特徴量
    print("\n[3/5] 時系列ラグ & ローリング特徴量構築...")
    all_df = build_lag_features(all_df)

    # 4. カテゴリ変数の型変換
    categorical_cols = ["store_nbr", "family"]
    for col in categorical_cols:
        all_df[col] = all_df[col].astype("category")

    # train と test に再分離
    train_feat = all_df[all_df["date"] <= "2017-08-15"].copy()
    test_feat = all_df[all_df["date"] >= "2017-08-16"].copy()

    # 特徴量リストの決定
    ignore_cols = ["id", "date", "sales", "log_sales"]
    features = [c for c in all_df.columns if c not in ignore_cols]

    print(f"\n  - 総特徴量数: {len(features)} 列")
    print(f"  - 特徴量一覧: {features}")

    # 時系列バリデーション分割 (Time-based Split)
    val_start_date = "2017-07-31"
    val_end_date = "2017-08-15"
    # ラグ生成に必要なため 2016-01-01 以降を学習データに使用
    train_start_date = "2016-01-01"

    print(f"\n[4/5] 時系列バリデーション分割:")
    print(f"  - 学習期間: {train_start_date} 〜 2017-07-30")
    print(f"  - 検証期間: {val_start_date} 〜 {val_end_date} (直近16日間)")

    trn_mask = (train_feat["date"] >= train_start_date) & (train_feat["date"] < val_start_date)
    val_mask = (train_feat["date"] >= val_start_date) & (train_feat["date"] <= val_end_date)

    X_train, y_train = train_feat.loc[trn_mask, features], train_feat.loc[trn_mask, "sales"].values
    X_val, y_val = train_feat.loc[val_mask, features], train_feat.loc[val_mask, "sales"].values
    X_test = test_feat[features]

    print(f"  - 学習サンプル数: {len(X_train):,} 件")
    print(f"  - 検証サンプル数: {len(X_val):,} 件")

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # 5. LightGBM 回帰モデル学習
    print("\n[5/5] LightGBM 回帰モデル学習中...")
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
    print(f"  🏆 [実験 #003] 手元検証スコア (Val RMSLE): {val_rmsle:.5f}")
    print(f"  (参考: #001 ベースライン: 0.41440 / #002 ドメイン: 0.41426)")
    print("=" * 55)

    # 特徴量重要度 Top 15
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n特徴量重要度 Top 15:")
    for feat, imp in importances.head(15).items():
        print(f"  - {feat:22s}: {imp:6d}")

    # テストデータ予測 & 提出ファイル作成
    test_pred_log = model.predict(X_test)
    test_pred = np.clip(np.expm1(test_pred_log), 0, None)

    sub_df["sales"] = test_pred
    out_path = "003_store-sales/submission_exp003.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub_df.to_csv(out_path, index=False)

    print(f"\n  - 提出ファイルを保存しました: {out_path}")
    print(f"  - 先頭5行:\n{sub_df.head()}")


if __name__ == "__main__":
    main()
