"""
Kaggle Digit Recognizer: モデル比較実験スクリプト (Phase 4)
- ロジスティック回帰 (Logistic Regression) vs ランダムフォレスト (Random Forest)
- 同じ検証データでの評価と誤答パターンの分析

使い方:
  uv run python 001_digit-recognizer/src/model_comparison.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("=== Kaggle Digit Recognizer: Model Comparison (Phase 4) ===")

    # 1. パス検索
    candidate_paths = [
        "/kaggle/input/competitions/digit-recognizer",
        "/kaggle/input/digit-recognizer",
        "../input/digit-recognizer",
        "./001_digit-recognizer/data",
        "./001_digit-recognizer",
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
        sys.exit(1)

    print(f"データディレクトリ: {data_dir}")

    # 2. 読み込みと前処理
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    y = train_df['label'].values
    X = train_df.drop('label', axis=1).values / 255.0

    # 公平な比較のため同じシード値で分割
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"訓練データ: {X_train.shape}, 検証データ: {X_val.shape}")

    # 3. モデル1: ロジスティック回帰
    print("\n--- 1. ロジスティック回帰 (Logistic Regression) ---")
    lr = LogisticRegression(max_iter=200, solver='lbfgs', random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_val)
    lr_acc = accuracy_score(y_val, lr_preds)
    print(f"ロジスティック回帰 検証正解率: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

    # 4. モデル2: ランダムフォレスト
    print("\n--- 2. ランダムフォレスト (Random Forest 100木) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_preds)
    print(f"ランダムフォレスト 検証正解率: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

    # 5. 比較まとめ
    print("\n=== 比較結果まとめ ===")
    print(f"{'モデル':<35} | {'検証正解率':<12} | {'誤答数 (8,400問中)'}")
    print("-" * 65)
    print(f"{'Logistic Regression (線形モデル)':<35} | {lr_acc:.4f} ({lr_acc*100:.2f}%) | {int((1-lr_acc)*len(y_val))} 問")
    print(f"{'Random Forest (非線形アンサンブル)':<35} | {rf_acc:.4f} ({rf_acc*100:.2f}%) | {int((1-rf_acc)*len(y_val))} 問")
    diff = rf_acc - lr_acc
    print(f"\n差分: ランダムフォレストが +{diff*100:.2f}% ({int(diff*len(y_val))}問多く正解)")

if __name__ == "__main__":
    main()
