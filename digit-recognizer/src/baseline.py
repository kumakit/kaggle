"""
Kaggle Digit Recognizer: ベースライン実行スクリプト (scikit-learn Random Forest)

使い方:
  uv run python src/baseline.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("=== Kaggle Digit Recognizer Baseline ===")

    # 1. データパスの探索
    import glob

    candidate_paths = [
        "/kaggle/input/competitions/digit-recognizer",
        "/kaggle/input/digit-recognizer",
        "../input/digit-recognizer",
        "./digit-recognizer/data",
        "./digit-recognizer",
        "./data",
        "."
    ]

    data_dir = None
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, "train.csv")):
            data_dir = p
            break

    if data_dir is None:
        found = glob.glob("/kaggle/input/**/train.csv", recursive=True)
        if found:
            data_dir = os.path.dirname(found[0])

    if data_dir is None:
        print("【エラー】train.csv が見つかりませんでした。")
        print("Kaggleのコンペページからデータをダウンロードして ./data またはカレントディレクトリに配置してください。")
        sys.exit(1)

    print(f"データディレクトリ: {data_dir}")

    # 2. データの読み込み
    print("データを読み込み中...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"訓練データ: {train_df.shape}, テストデータ: {test_df.shape}")

    # 3. 前処理 (正規化)
    print("前処理 (0.0〜1.0へ正規化) を実行中...")
    y = train_df['label'].values
    X = train_df.drop('label', axis=1).values / 255.0
    X_test = test_df.values / 255.0

    # 4. 訓練/検証分割 (80% / 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. モデル学習 (Random Forest)
    print("ランダムフォレストモデル (n_estimators=100) を学習中...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 6. 検証
    y_val_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"=== 検証データ 正解率 (Accuracy): {acc:.4f} ({acc * 100:.2f}%) ===")

    # 7. 提出ファイルの出力
    print("本番テストデータを予測中...")
    test_preds = rf.predict(X_test)
    submission_df = pd.DataFrame({
        'ImageId': range(1, len(test_preds) + 1),
        'Label': test_preds
    })

    out_path = "submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"提出ファイルを保存しました: {out_path}")
    print(f"行数: {len(submission_df)} (28000行ならOK), 欠損値: {submission_df.isnull().sum().sum()}")

if __name__ == "__main__":
    main()
