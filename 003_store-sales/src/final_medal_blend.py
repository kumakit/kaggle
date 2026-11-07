"""
003_store-sales: Final Medal Blend (Experiment #017)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク:
  1. 現自己ベスト Exp #015 (LB 0.38856 / 69位) を主軸 (60%) に据える
  2. Colab GPU 2シード 800木 Exp #016 (LB 0.39040) を配合 (25%)
  3. Colab GPU 1,000木深層 Exp #014 (LB 0.39019) を配合 (15%)
  4. 直近21日ゼロ売上フィルター (124系列) の厳格適用
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
    print("  003 Store Sales: 実験 #017 (メダル圏突入 ファイナル・ブレンド)")
    print("=" * 75)

    base_dir = "003_store-sales"
    path_sub15 = os.path.join(base_dir, "submission_exp015.csv")
    path_sub16 = os.path.join(base_dir, "submission_colab_2seeds.csv")
    path_sub14 = os.path.join(base_dir, "submission_colab_gpu.csv")

    for p in [path_sub15, path_sub16, path_sub14]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} が見つかりませんでした。")

    print("\n[1/4] 3大トップモデルの予測ファイルを読み込み...")
    sub15 = pd.read_csv(path_sub15)
    sub16 = pd.read_csv(path_sub16)
    sub14 = pd.read_csv(path_sub14)

    print(f"  - Exp #015 (LB 0.38856 / 現自己ベスト 69位): 平均売上={sub15['sales'].mean():.2f}")
    print(f"  - Exp #016 (LB 0.39040 / Colab 2シード 800木): 平均売上={sub16['sales'].mean():.2f}")
    print(f"  - Exp #014 (LB 0.39019 / Colab 1,000木深層): 平均売上={sub14['sales'].mean():.2f}")

    # 重み設定 (60% : 25% : 15%)
    w15, w16, w14 = 0.60, 0.25, 0.15
    print(f"\n[2/4] 対数空間加重ブレンド ({w15*100:.0f}% : {w16*100:.0f}% : {w14*100:.0f}%)...")

    log15 = np.log1p(np.clip(sub15["sales"].values, 0, None))
    log16 = np.log1p(np.clip(sub16["sales"].values, 0, None))
    log14 = np.log1p(np.clip(sub14["sales"].values, 0, None))

    log_blend = w15 * log15 + w16 * log16 + w14 * log14
    sales_blend = np.clip(np.expm1(log_blend), 0, None)

    # 直近21日ゼロ売上フィルター (124系列)
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

    sub_df = pd.DataFrame({"id": sub15["id"], "sales": sales_blend})
    test_merged = pd.merge(test_df[["id", "store_nbr", "family"]], sub_df, on="id")
    zero_mask = [((st, fam) in zero_pairs) for st, fam in zip(test_merged["store_nbr"], test_merged["family"])]
    sales_blend[zero_mask] = 0.0

    print(f"  - クリップ後のゼロ売上件数: {(sales_blend == 0).sum()} 件")
    print(f"  - ブレンド後 平均売上: {np.mean(sales_blend):.2f}")

    # 保存
    final_sub = pd.DataFrame({
        "id": sub15["id"],
        "sales": sales_blend
    }).sort_values("id").reset_index(drop=True)

    out_path = os.path.join(base_dir, "submission_exp017.csv")
    final_sub.to_csv(out_path, index=False)
    print(f"\n[4/4] 提出ファイルを保存しました: {out_path} ({len(final_sub)} 行)")
    print(final_sub.head(10))


if __name__ == "__main__":
    main()
