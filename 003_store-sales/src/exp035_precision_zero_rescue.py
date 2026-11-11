"""
003_store-sales: Exp #035 - Precision Zero Rescue & Back-to-School Sierra Un-clip
Kaggle Competition: Store Sales - Time Series Forecasting

Key Breakthrough & Discovery:
  - Discovery: The 21-day zero filter (127 pairs / 2,032 rows) mistakenly clipped 
    active Sierra back-to-school stores (Stores 1, 2, 6, 23, 37) in SCHOOL AND OFFICE SUPPLIES.
  - In 2016, Store 6 exploded to 18 units/day during Aug 25-31! Test data has onpromotion=14 days!
  - Forcing sales=0 caused severe RMSLE penalty ((log(1+18) - 0)^2 = 8.7).
  
Rescue Logic:
  1. Base model: Exp #030 (PB 0.38793, Dual Math Optimal)
  2. For Sierra school supply stores (Stores 1, 2, 6, 23, 37):
     - Restore 2016 historical sales trajectory adjusted for 2017 promotions.
  3. Strict True Dead Filter:
     - Only clip pairs that have ZERO sales for >90 consecutive days AND onpromotion == 0 in test.
     - Series with onpromotion > 0 in test are NEVER clipped to 0!
"""

import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #035 - Precision Zero Rescue & Sierra School Un-clip")
    print("=" * 80)

    print("\n[1/4] Loading baseline Exp #030 (PB 0.38793) and datasets...")
    sub030 = pd.read_csv("003_store-sales/submission_exp030_dual_optimal_blend.csv")
    test = pd.read_csv("003_store-sales/data/test.csv", parse_dates=["date"])
    train = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])
    stores = pd.read_csv("003_store-sales/data/stores.csv")

    pred_df = test[["id", "date", "store_nbr", "family", "onpromotion"]].copy()
    pred_df["sales"] = sub030["sales"].values

    print("\n[2/4] Rescuing mistakenly clipped Sierra Back-to-School Stores...")
    # Sierra stores in SCHOOL AND OFFICE SUPPLIES: Store 6, 2, 23, 1, 37
    rescued_stores = [6, 2, 23, 1, 37]
    for st in rescued_stores:
        mask_2016 = (train["store_nbr"] == st) & (train["family"] == "SCHOOL AND OFFICE SUPPLIES") & \
                    (train["date"] >= "2016-08-16") & (train["date"] <= "2016-08-31")
        hist_sales = train.loc[mask_2016].sort_values("date")["sales"].values
        
        mask_test = (pred_df["store_nbr"] == st) & (pred_df["family"] == "SCHOOL AND OFFICE SUPPLIES")
        # In test, dates are 2017-08-16 to 2017-08-31 (16 days, exact match)
        if len(hist_sales) == 16:
            # 2017 has promotion, scale by promotion intensity (at least 1.0x 2016 sales)
            promo_ratio = np.where(pred_df.loc[mask_test, "onpromotion"].values > 0, 1.2, 1.0)
            restored_sales = hist_sales * promo_ratio
            pred_df.loc[mask_test, "sales"] = restored_sales
            print(f"  - Store {st:2d}: Rescued 16 days! Total restored sales: {restored_sales.sum():.1f} (2016: {hist_sales.sum():.1f})")

    print("\n[3/4] Verifying and applying Strict True Dead Zero Filter (>90 days & promo == 0)...")
    # Identify pairs that had 0 sales in last 90 days of training
    last_90 = train[train["date"] >= "2017-05-15"]
    sum_90 = last_90.groupby(["store_nbr", "family"])["sales"].sum()
    dead_pairs = set(sum_90[sum_90 == 0].index)
    print(f"  - Strict 90-day dead pairs: {len(dead_pairs)} pairs (vs 127 in 21-day rule)")

    # For dead pairs, only clip if onpromotion == 0 in test
    dead_clipped_count = 0
    for idx, row in pred_df.iterrows():
        pair = (row["store_nbr"], row["family"])
        if pair in dead_pairs and row["onpromotion"] == 0:
            pred_df.at[idx, "sales"] = 0.0
            dead_clipped_count += 1

    print(f"  - Applied strict dead zero filter: {dead_clipped_count} rows")

    out_path = "003_store-sales/submission_exp035_precision_rescue.csv"
    sub_final = pred_df[["id", "sales"]].copy()
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    log_030 = np.log1p(sub030["sales"].values)
    log_final = np.log1p(sub_final["sales"].values)
    diff_rmsle = np.sqrt(np.mean((log_final - log_030)**2))
    corr = np.corrcoef(log_final, log_030)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp030 PB {sub030['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp030 PB {sub030['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp030 PB {sub030['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp030 PB {(sub030['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #030 (PB 0.38793): {diff_rmsle:.5f}")
    print(f"  Correlation with Exp #030 (PB 0.38793): {corr:.6f}")


if __name__ == "__main__":
    main()
