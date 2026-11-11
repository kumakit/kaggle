"""
003_store-sales: Exp #034 - Super Dual Optimal Blend (Exp #030 PB 50% + Exp #033 7-Models 50%)
Kaggle Competition: Store Sales - Time Series Forecasting

Dual Optimal Formulation:
  - Exp #030: Current Personal Best (0.38793, Dual SLSQP Golden Blend)
  - Exp #033: 7-Model 33 Family-Adaptive SLSQP Blend (integrating Exp #032 Earthquake & Holiday)
  - Log-space 50:50 geometric mean integration to achieve ultimate flat minimum noise cancellation.
  - Safe 21-day zero sales filter (127 series / 2,032 rows).
"""

import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #034 - Super Dual Optimal Blend")
    print("=" * 80)

    print("\n[1/4] Loading Exp #030 (PB 0.38793) and Exp #033 (7-Models)...")
    sub030 = pd.read_csv("003_store-sales/submission_exp030_dual_optimal_blend.csv")
    sub033 = pd.read_csv("003_store-sales/submission_exp033_7models_blend.csv")
    test = pd.read_csv("003_store-sales/data/test.csv")

    print(f"  - Exp #030: {len(sub030)} rows")
    print(f"  - Exp #033: {len(sub033)} rows")

    log030 = np.log1p(sub030["sales"].values)
    log033 = np.log1p(sub033["sales"].values)

    print("\n[2/4] Blending in log-space: Exp #030 (50%) + Exp #033 (50%)...")
    log_blend = 0.50 * log030 + 0.50 * log033
    pred_sales = np.clip(np.expm1(log_blend), 0, None)

    print("\n[3/4] Applying 21-day safe zero sales filter...")
    train_df = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    zero_pairs = set(series_21_sum[series_21_sum == 0].index)
    zero_mask = test.set_index(["store_nbr", "family"]).index.isin(zero_pairs)
    pred_sales[zero_mask] = 0.0
    print(f"  - Zero clip applied rows: {zero_mask.sum()} rows (expected: 2032 rows)")

    out_path = "003_store-sales/submission_exp034_super_dual_blend.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": pred_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    log_final = np.log1p(sub_final["sales"].values)
    diff_rmsle = np.sqrt(np.mean((log_final - log030)**2))
    corr = np.corrcoef(log_final, log030)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp030 PB {sub030['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp030 PB {sub030['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp030 PB {sub030['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp030 PB {(sub030['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #030 (PB 0.38793): {diff_rmsle:.5f}")
    print(f"  Correlation with Exp #030 (PB 0.38793): {corr:.6f}")


if __name__ == "__main__":
    main()
