"""
002_airline-satisfaction: Baseline Script (LightGBM + Stratified 5-Fold CV)
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


def find_data_dir() -> str:
    """ローカルまたはKaggle環境のデータディレクトリを自動検出する"""
    candidates = [
        "002_airline-satisfaction/data",
        "data",
        "../data",
        "/kaggle/input/competitions/playground-series-s6e10",
        "/kaggle/input/playground-series-s6e10",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    # glob fallback
    matches = glob.glob("**/playground-series-s6e10/train.csv", recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError("train.csv が見つかりませんでした。データディレクトリを確認してください。")


def main():
    data_dir = find_data_dir()
    print(f"[1/5] データディレクトリ: {data_dir}")

    # 1. データ読み込み
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sub_path = os.path.join(data_dir, "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(sub_path)

    print(f"  - 訓練データ: {train_df.shape}")
    print(f"  - テストデータ: {test_df.shape}")

    # 2. 前処理
    target_col = "satisfaction"
    id_col = "id"

    # 目的変数を 0/1 に変換
    y = train_df[target_col].astype(int).values
    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # カテゴリ変数を category 型に変換（LightGBMが最適に自動分割）
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"  - カテゴリ特徴量 ({len(cat_cols)}個): {cat_cols}")

    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # 3. 層化5分割交差検証（Stratified 5-Fold CV）
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_scores = []

    print(f"\n[2/5] Stratified {n_splits}-Fold CV による LightGBM 学習開始...")

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        # 検証データの予測確率（クラス1 = 満足の確率）
        val_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_pred

        score = roc_auc_score(y_val, val_pred)
        fold_scores.append(score)
        print(f"  - Fold {fold}/{n_splits} ROC AUC: {score:.5f}")

        # テストデータの予測（5モデルの平均をとる）
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n[3/5] 手元検証（CV）結果:")
    print(f"  - 各Foldスコア: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"  - 全体 OOF ROC AUC: {overall_auc:.5f} (期待本番スコアの目安)")

    # 4. 提出ファイル作成
    sub_df[target_col] = test_preds
    output_path = "submission_baseline.csv"
    sub_df.to_csv(output_path, index=False)

    print(f"\n[4/5] 提出ファイル出力完了: {output_path}")
    print(f"  - 行数: {len(sub_df)} 行 (テストデータと一致)")
    print(f"  - 欠損値: {sub_df[target_col].isnull().sum()} 件")
    print(f"  - 予測確率の範囲: 最小 {sub_df[target_col].min():.4f} 〜 最大 {sub_df[target_col].max():.4f}")
    print("\n[5/5] サンプルプレビュー:")
    print(sub_df.head())


if __name__ == "__main__":
    main()
