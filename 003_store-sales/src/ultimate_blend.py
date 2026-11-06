"""
003_store-sales: Ultimate Blending (Experiment #015)
Kaggle Competition: Store Sales - Time Series Forecasting

タスク:
  1. 歴代最高峰モデル群の対数空間加重ブレンド (Log-space Ultimate Blending)
     - Exp #012 (現自己ベスト: LB 0.38908 / 71位): 重み 40%
     - Exp #014 (Colab T4 GPU 1,000木大規模モデル: LB 0.39019): 重み 25%
     - Exp #013 (曜日グループ平均単体モデル: LB 0.38972): 重み 20%
     - Exp #010 (特売平常比・Quincena本番最強: LB 0.38968): 重み 15%
     (※ submission_colab_gpu.csv が未配置の場合は Exp #012: 50%, Exp #013: 30%, Exp #010: 20% で自動ブレンド)
  2. 直近21日ゼロ売上フィルター (124系列) の厳格適用
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
    print("  003 Store Sales: 実験 #015 (メダル圏突破 アルティメット・ブレンド)")
    print("=" * 75)

    base_dir = "003_store-sales"
    path_sub12 = os.path.join(base_dir, "submission_exp012.csv")
    path_sub13 = os.path.join(base_dir, "submission_exp013.csv")
    path_sub10 = os.path.join(base_dir, "submission_exp010.csv")
    path_sub14 = os.path.join(base_dir, "submission_colab_gpu.csv")

    has_sub14 = os.path.exists(path_sub14)

    print("\n[1/4] トップモデル予測ファイルを読み込み...")
    sub12 = pd.read_csv(path_sub12)
    sub13 = pd.read_csv(path_sub13)
    sub10 = pd.read_csv(path_sub10)
    print(f"  - Exp #012 (LB 0.38908 / 現自己ベスト): 平均売上={sub12['sales'].mean():.2f}")
    print(f"  - Exp #013 (LB 0.38972 / 曜日ベースライン): 平均売上={sub13['sales'].mean():.2f}")
    print(f"  - Exp #010 (LB 0.38968 / 特売平常比): 平均売上={sub10['sales'].mean():.2f}")

    if has_sub14:
        sub14 = pd.read_csv(path_sub14)
        print(f"  - Exp #014 (LB 0.39019 / Colab T4 GPU 1,000木): 平均売上={sub14['sales'].mean():.2f}")
        # 4大ブレンド (40% : 25% : 20% : 15%)
        w12, w14, w13, w10 = 0.40, 0.25, 0.20, 0.15
        print(f"\n[2/4] 4大モデル対数空間加重ブレンド ({w12*100:.0f}% : {w14*100:.0f}% : {w13*100:.0f}% : {w10*100:.0f}%)...")
        log12 = np.log1p(np.clip(sub12["sales"].values, 0, None))
        log14 = np.log1p(np.clip(sub14["sales"].values, 0, None))
        log13 = np.log1p(np.clip(sub13["sales"].values, 0, None))
        log10 = np.log1p(np.clip(sub10["sales"].values, 0, None))
        log_blend = w12 * log12 + w14 * log14 + w13 * log13 + w10 * log10
    else:
        # 3大ブレンド (50% : 30% : 20%)
        w12, w13, w10 = 0.50, 0.30, 0.20
        print(f"\n[2/4] 3大モデル対数空間加重ブレンド ({w12*100:.0f}% : {w13*100:.0f}% : {w10*100:.0f}%)...")
        log12 = np.log1p(np.clip(sub12["sales"].values, 0, None))
        log13 = np.log1p(np.clip(sub13["sales"].values, 0, None))
        log10 = np.log1p(np.clip(sub10["sales"].values, 0, None))
        log_blend = w12 * log12 + w13 * log13 + w10 * log10

    sales_blend = np.clip(np.expm1(log_blend), 0, None)

    # ゼロ売上フィルター (直近21日間完全ゼロの124系列)
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

    sub_df = pd.DataFrame({"id": sub12["id"], "sales": sales_blend})
    test_merged = pd.merge(test_df[["id", "store_nbr", "family"]], sub_df, on="id")
    zero_mask = [((st, fam) in zero_pairs) for st, fam in zip(test_merged["store_nbr"], test_merged["family"])]
    sales_blend[zero_mask] = 0.0

    print(f"  - クリップ後のゼロ売上件数: {(sales_blend == 0).sum()} 件")
    print(f"  - ブレンド後 平均売上: {np.mean(sales_blend):.2f}")

    # 保存
    final_sub = pd.DataFrame({
        "id": sub12["id"],
        "sales": sales_blend
    }).sort_values("id").reset_index(drop=True)

    out_path = os.path.join(base_dir, "submission_exp015.csv")
    final_sub.to_csv(out_path, index=False)
    print(f"\n[4/4] 提出ファイルを保存しました: {out_path} ({len(final_sub)} 行)")
    print(final_sub.head(10))


if __name__ == "__main__":
    main()
