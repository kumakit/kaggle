"""
003_store-sales: Top Models Log-space Weighted Blending (Experiment #012)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク:
  3大トップモデルの対数空間加重ブレンド (Log-space Weighted Blending)
    - Exp #010 (本番最強: LB 0.38968 / 76位): 重み 60%
    - Exp #011 (手元最強: Val 0.39765 / 2シード平均): 重み 30%
    - Exp #008 (堅牢アンカー: LB 0.39084 / 82位): 重み 10%
  後処理: 直近21日ゼロ売上フィルター (124系列) の厳格適用
評価指標: RMSLE (Root Mean Squared Logarithmic Error)
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 75)
    print("  003 Store Sales: 実験 #012 (3大トップモデル 対数空間加重ブレンド)")
    print("=" * 75)

    base_dir = "003_store-sales"
    path_sub10 = os.path.join(base_dir, "submission_exp010.csv")
    path_sub11 = os.path.join(base_dir, "submission_exp011.csv")
    path_sub8 = os.path.join(base_dir, "submission_exp008.csv")

    for p in [path_sub10, path_sub11, path_sub8]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} が見つかりませんでした。")

    print("\n[1/4] 3大トップモデルの予測ファイルを読み込み...")
    sub10 = pd.read_csv(path_sub10)
    sub11 = pd.read_csv(path_sub11)
    sub8 = pd.read_csv(path_sub8)

    print(f"  - Exp #010 (LB 0.38968 / 本番最高): 平均売上={sub10['sales'].mean():.2f}")
    print(f"  - Exp #011 (Val 0.39765 / 手元最高): 平均売上={sub11['sales'].mean():.2f}")
    print(f"  - Exp #008 (LB 0.39084 / アンカー): 平均売上={sub8['sales'].mean():.2f}")

    # 重み設定 (60% : 30% : 10%)
    w10, w11, w8 = 0.60, 0.30, 0.10
    print(f"\n[2/4] 対数空間加重平均 (Log-space Blending: {w10*100:.0f}% : {w11*100:.0f}% : {w8*100:.0f}%)...")

    log10 = np.log1p(np.clip(sub10["sales"].values, 0, None))
    log11 = np.log1p(np.clip(sub11["sales"].values, 0, None))
    log8 = np.log1p(np.clip(sub8["sales"].values, 0, None))

    log_blend = w10 * log10 + w11 * log11 + w8 * log8
    sales_blend = np.clip(np.expm1(log_blend), 0, None)

    # ゼロ売上フィルターの厳格適用 (直近21日間売上ゼロの124ペア)
    print("\n[3/4] 直近21日ゼロ売上フィルター (124系列) の厳格適用...")
    train_path = os.path.join(base_dir, "data", "train.csv")
    test_path = os.path.join(base_dir, "data", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df["date"] = pd.to_datetime(train_df["date"])

    recent_train = train_df[train_df["date"] >= "2017-07-25"]
    recent_sum = recent_train.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
    zero_pairs = set(zip(recent_sum[recent_sum["sales"] == 0]["store_nbr"], recent_sum[recent_sum["sales"] == 0]["family"]))
    print(f"  - ゼロクリップ対象ペア数: {len(zero_pairs)} ペア (1,984行)")

    # test.csv と結合して系列判定
    sub_df = pd.DataFrame({"id": sub10["id"], "sales": sales_blend})
    test_merged = pd.merge(test_df[["id", "store_nbr", "family"]], sub_df, on="id")

    zero_mask = [((st, fam) in zero_pairs) for st, fam in zip(test_merged["store_nbr"], test_merged["family"])]
    sales_blend[zero_mask] = 0.0

    print(f"  - クリップ後のゼロ売上件数: {(sales_blend == 0).sum()} 件")
    print(f"  - ブレンド後 平均売上: {np.mean(sales_blend):.2f}")

    # 出力ファイル作成
    final_sub = pd.DataFrame({
        "id": sub10["id"],
        "sales": sales_blend
    }).sort_values("id").reset_index(drop=True)

    out_path = os.path.join(base_dir, "submission_exp012.csv")
    final_sub.to_csv(out_path, index=False)

    print(f"\n[4/4] 提出ファイルを保存しました: {out_path}")
    print(f"  - 行数: {len(final_sub)} 行")
    print(f"  - 先頭10行:\n{final_sub.head(10)}")


if __name__ == "__main__":
    main()
