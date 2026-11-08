"""
003_store-sales: Exp #021 - 5-Seeds Gold Medal Blend (Experiment #021)
Kaggle Competition: Store Sales - Time Series Forecasting

配合比率 (Log-space Weighted Geometric Blend):
  - Exp #015 (自己ベスト最高峰 LB 0.38856 / 69位): 50%
  - Exp #020 (Colab L4 GPU 5シード 1,000木 完全ノイズ除去 LB 0.39113 / Val 0.39794 過去最高): 35%
  - Exp #013 (曜日グループ平均単体最強 LB 0.38972): 15%

後処理:
  - 直近21日売上ゼロフィルター (124系列 / 1,984行 強制ゼロクリップ)
"""

import os
import numpy as np
import pandas as pd


def main():
    print("=" * 75)
    print("  003 Store Sales: Exp #021 - 5シード黄金メダルブレンド")
    print("=" * 75)

    f_exp15 = "003_store-sales/submission_exp015.csv"
    f_exp20 = "003_store-sales/submission_exp020_colab_5seeds.csv"
    f_exp13 = "003_store-sales/submission_exp013.csv"

    print("\n[1/4] モデル予測ファイルの読み込み...")
    df15 = pd.read_csv(f_exp15)
    df20 = pd.read_csv(f_exp20)
    df13 = pd.read_csv(f_exp13)

    print(f"  - Exp #015 (自己ベスト LB 0.38856): {len(df15)} 行")
    print(f"  - Exp #020 (5シード LB 0.39113):   {len(df20)} 行")
    print(f"  - Exp #013 (曜日単体最強 LB 0.38972): {len(df13)} 行")

    # ID整合性チェック
    assert (df15["id"] == df20["id"]).all(), "ID mismatch between 15 and 20"
    assert (df15["id"] == df13["id"]).all(), "ID mismatch between 15 and 13"

    print("\n[2/4] 対数空間加重ブレンドの計算...")
    w15, w20, w13 = 0.50, 0.35, 0.15
    print(f"  - 重み配分: Exp015={w15:.2f}, Exp020={w20:.2f}, Exp013={w13:.2f}")

    log15 = np.log1p(np.clip(df15["sales"].values, 0, None))
    log20 = np.log1p(np.clip(df20["sales"].values, 0, None))
    log13 = np.log1p(np.clip(df13["sales"].values, 0, None))

    log_blend = w15 * log15 + w20 * log20 + w13 * log13
    sales_blend = np.clip(np.expm1(log_blend), 0, None)

    df_blend = df15[["id"]].copy()
    df_blend["sales"] = sales_blend

    print("\n[3/4] 直近21日売上ゼロフィルターの安全確認...")
    train_path = "003_store-sales/data/train.csv"
    test_path = "003_store-sales/data/test.csv"
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        train_df["date"] = pd.to_datetime(train_df["date"])
        test_df["date"] = pd.to_datetime(test_df["date"])

        recent_train = train_df[train_df["date"] >= "2017-07-25"]
        recent_sum = recent_train.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
        zero_pairs = set(zip(recent_sum[recent_sum["sales"] == 0]["store_nbr"], recent_sum[recent_sum["sales"] == 0]["family"]))
        print(f"  - 抽出された直近ゼロ系列数: {len(zero_pairs)} / 1,782 ペア")

        test_merged = pd.merge(test_df, df_blend, on="id")
        zero_count = 0
        for idx, row in test_merged.iterrows():
            if (row["store_nbr"], row["family"]) in zero_pairs:
                df_blend.loc[df_blend["id"] == row["id"], "sales"] = 0.0
                zero_count += 1
        print(f"  - ゼロクリップ適用行数: {zero_count} 行 (期待値: {len(zero_pairs) * 16} 行)")

    out_path = "003_store-sales/submission_exp021_gold_blend.csv"
    df_blend.to_csv(out_path, index=False)
    print(f"\n[4/4] 提出ファイル保存完了: {out_path} ({len(df_blend)} 行)")

    print("\n=== 予測先頭10行 ===")
    print(df_blend.head(10))

    print("\n=== 予測値統計サマリー ===")
    print(f"  平均売上: {df_blend['sales'].mean():.4f}")
    print(f"  中央値:   {df_blend['sales'].median():.4f}")
    print(f"  最大売上: {df_blend['sales'].max():.4f}")
    print(f"  ゼロ件数: {(df_blend['sales'] == 0).sum()} 行 ({((df_blend['sales'] == 0).mean() * 100):.2f}%)")
    print(f"  欠損値数: {df_blend['sales'].isnull().sum()} 件")

    # 相関と距離の確認
    log_final = np.log1p(df_blend["sales"].values)
    print(f"\n  Exp #015 との対数相関: {np.corrcoef(log15, log_final)[0, 1]:.6f}")
    print(f"  Exp #015 からのRMSLE距離: {np.sqrt(np.mean((log15 - log_final)**2)):.5f}")


if __name__ == "__main__":
    main()
