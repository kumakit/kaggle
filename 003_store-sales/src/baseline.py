"""
003_store-sales: Baseline Script (LightGBM + Time-based Validation)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク: 店舗・商品ファミリー別 日次売上数量（sales）の時系列回帰予測
評価指標: RMSLE (Root Mean Squared Logarithmic Error)
検証設計: 直近16日間 (2017-07-31 〜 2017-08-15) を手元検証（Validation）とする Time-based Split
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


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
    if os.path.exists("/kaggle/input"):
        input_contents = os.listdir("/kaggle/input")
        raise FileNotFoundError(
            f"/kaggle/input 配下に train.csv が見つかりませんでした。現在の中身: {input_contents}。"
            '右上の [+ Add Input] から "store-sales-time-series-forecasting" を追加してください。'
        )
    raise FileNotFoundError("train.csv が見つかりませんでした。データディレクトリを確認してください。")


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE (Root Mean Squared Logarithmic Error) を計算する"""
    y_pred_clipped = np.clip(y_pred, 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped))))


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """日付列からカレンダー特徴量（曜日・日・月・年・週末フラグ）を抽出する"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    return df


def main():
    print("=" * 60)
    print("  003 Store Sales - Time Series Forecasting: 初回ベースライン")
    print("=" * 60)

    # 1. データ読み込み
    data_dir = find_data_dir()
    print(f"\n[1/5] データ読み込み元: {data_dir}")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    print(f"  - train.csv: {train_df.shape} (期間: {train_df['date'].min()} 〜 {train_df['date'].max()})")
    print(f"  - test.csv:  {test_df.shape} (期間: {test_df['date'].min()} 〜 {test_df['date'].max()})")

    # 2. カレンダー特徴量作成
    print("\n[2/5] カレンダー特徴量作成...")
    train_df = create_calendar_features(train_df)
    test_df = create_calendar_features(test_df)

    # カテゴリ変数の型変換
    categorical_cols = ["store_nbr", "family"]
    for col in categorical_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")

    features = ["store_nbr", "family", "onpromotion", "year", "month", "day", "dayofweek", "is_weekend", "dayofyear"]
    print(f"  - 使用特徴量 ({len(features)}列): {features}")

    # 3. 時系列バリデーション分割 (Time-based Split)
    # テスト期間と同じ「直近16日間」を手元検証データとする
    val_start_date = "2017-07-31"
    val_end_date = "2017-08-15"

    # 学習期間: 直近のトレンドを重視するため 2016-01-01 以降を使用（高速化と最新傾向の学習）
    train_start_date = "2016-01-01"

    print(f"\n[3/5] 時系列バリデーション分割:")
    print(f"  - 学習データ期間: {train_start_date} 〜 2017-07-30")
    print(f"  - 検証データ期間: {val_start_date} 〜 {val_end_date} (テストと同じ16日間)")

    trn_mask = (train_df["date"] >= train_start_date) & (train_df["date"] < val_start_date)
    val_mask = (train_df["date"] >= val_start_date) & (train_df["date"] <= val_end_date)

    X_train, y_train = train_df.loc[trn_mask, features], train_df.loc[trn_mask, "sales"].values
    X_val, y_val = train_df.loc[val_mask, features], train_df.loc[val_mask, "sales"].values
    X_test = test_df[features]

    print(f"  - 学習サンプル数: {len(X_train):,} 件")
    print(f"  - 検証サンプル数: {len(X_val):,} 件")

    # 目的変数の対数変換 (RMSLE最適化の鉄則)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    # 4. LightGBM 回帰モデルの学習
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

    # 手元検証スコアの算出
    val_pred_log = model.predict(X_val)
    val_pred = np.expm1(val_pred_log)
    val_pred = np.clip(val_pred, 0, None)  # 売上は0以上

    val_rmsle = calculate_rmsle(y_val, val_pred)
    print(f"\n" + "=" * 50)
    print(f"  🏆 手元検証スコア (Validation RMSLE): {val_rmsle:.5f}")
    print(f"=" * 50)

    # 特徴量重要度 Top 5
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n特徴量重要度:")
    for feat, imp in importances.items():
        print(f"  - {feat:15s}: {imp:6d}")

    # 5. テストデータ予測 & 提出ファイル作成
    print("\n[5/5] テストデータ予測と提出ファイル作成...")
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log)
    test_pred = np.clip(test_pred, 0, None)

    sub_df["sales"] = test_pred
    out_path = "003_store-sales/submission_baseline.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub_df.to_csv(out_path, index=False)

    print(f"  - 提出ファイルを保存しました: {out_path}")
    print(f"  - 先頭5行:\n{sub_df.head()}")
    print("\n[完了] Kaggle Notebookにコピーして提出、またはローカルから提出可能です！")


if __name__ == "__main__":
    main()
