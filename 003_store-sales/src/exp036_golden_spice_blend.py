"""
003_store-sales: Exp #036 - Golden Center & Back-to-School Sierra Precision Spice (Exp030 85% + True Rescue 15%)
Kaggle Competition: Store Sales - Time Series Forecasting

Formulation & Rationale:
  - Base: Exp #030 (Personal Best 0.38793, Dual Math Optimal Blend) at 85% weight
  - Spice: True Back-to-School Sierra Un-clip Model (recovering Stores 1, 2, 6, 23, 37 in SCHOOL SUPPLIES) at 15% weight
  - Geometric interpolation in log-space:
      log(1 + y) = 0.85 * log(1 + y_030) + 0.15 * log(1 + y_rescue)
  - Mathematical Property:
      Asymmetric risk hedge: significantly mitigates severe RMSLE penalties if school supplies explode (>10 units/day)
      while introducing near-zero variance penalty (delta < 0.2 in squared error) if sales remain near zero.
"""

import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #036 - Golden Center & Back-to-School Sierra Precision Spice")
    print("=" * 80)

    print("\n[1/4] Loading baseline Exp #030 (PB 0.38793) and datasets...")
    sub030 = pd.read_csv("003_store-sales/submission_exp030_dual_optimal_blend.csv")
    test = pd.read_csv("003_store-sales/data/test.csv", parse_dates=["date"])
    train = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])

    # Build True Rescued Prediction
    pred_rescue = test[["id", "date", "store_nbr", "family", "onpromotion"]].copy()
    pred_rescue["sales"] = sub030["sales"].values

    rescued_stores = [6, 2, 23, 1, 37]
    rescued_pairs = {(st, "SCHOOL AND OFFICE SUPPLIES") for st in rescued_stores}

    print("\n[2/4] Restoring Sierra Back-to-School historical patterns (Stores 1, 2, 6, 23, 37)...")
    for st in rescued_stores:
        mask_2016 = (train["store_nbr"] == st) & (train["family"] == "SCHOOL AND OFFICE SUPPLIES") & \
                    (train["date"] >= "2016-08-16") & (train["date"] <= "2016-08-31")
        hist_sales = train.loc[mask_2016].sort_values("date")["sales"].values
        mask_test = (pred_rescue["store_nbr"] == st) & (pred_rescue["family"] == "SCHOOL AND OFFICE SUPPLIES")
        
        if len(hist_sales) == 16:
            promo_ratio = np.where(pred_rescue.loc[mask_test, "onpromotion"].values > 0, 1.2, 1.0)
            restored_sales = hist_sales * promo_ratio
            pred_rescue.loc[mask_test, "sales"] = restored_sales
            print(f"  - Store {st:2d}: Fully restored 16 days! Total = {restored_sales.sum():.1f}")

    # Strict Dead Filter (excluding rescued_pairs from dead clipping)
    last_90 = train[train["date"] >= "2017-05-15"]
    sum_90 = last_90.groupby(["store_nbr", "family"])["sales"].sum()
    dead_pairs = set(sum_90[sum_90 == 0].index) - rescued_pairs

    dead_clipped_count = 0
    for idx, row in pred_rescue.iterrows():
        pair = (row["store_nbr"], row["family"])
        if pair in dead_pairs and row["onpromotion"] == 0:
            pred_rescue.at[idx, "sales"] = 0.0
            dead_clipped_count += 1
    print(f"  - Applied strict dead zero filter to {dead_clipped_count} rows")

    print("\n[3/4] Blending in log-space: Exp #030 (85%) + True Rescue (15%)...")
    log030 = np.log1p(sub030["sales"].values)
    log_rescue = np.log1p(pred_rescue["sales"].values)

    log_blend = 0.85 * log030 + 0.15 * log_rescue
    sales_final = np.clip(np.expm1(log_blend), 0, None)

    out_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": sales_final
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    log_final = np.log1p(sales_final)
    diff_rmsle = np.sqrt(np.mean((log_final - log030)**2))
    corr = np.corrcoef(log_final, log030)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp030 PB {sub030['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp030 PB {sub030['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp030 PB {sub030['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp030 PB {(sub030['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #030 (PB 0.38793): {diff_rmsle:.6f}")
    print(f"  Correlation with Exp #030 (PB 0.38793): {corr:.7f}")


if __name__ == "__main__":
    main()
