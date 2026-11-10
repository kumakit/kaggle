"""
003_store-sales: Exp #030 - Dual Mathematical Optimal 50:50 Golden Blend
Kaggle Competition: Store Sales - Time Series Forecasting

Methodology:
  - Ensemble of Ensembles: Combines the two top mathematically-optimized models
    * Exp #027 (5-Model 33-Family SLSQP Optimal Blend: LB 0.38794 / PB)
    * Exp #029 (6-Model 33-Family SLSQP Optimal Blend with Exp028: LB 0.38797)
  - Log-space equal weighted geometric mean (50:50)
  - Cancels residual parameter exploration noise between the two independent SLSQP solutions.
  - Safe 21-day zero sales filter (127 series / 2,032 rows).
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #030 - Dual Mathematical Optimal 50:50 Golden Blend")
    print("=" * 80)

    print("\n[1/4] Loading Top-2 Math Optimal Submissions...")
    f_027 = "003_store-sales/submission_exp027_adaptive_blend.csv"
    f_029 = "003_store-sales/submission_exp029_6models_blend.csv"

    sub027 = pd.read_csv(f_027)
    sub029 = pd.read_csv(f_029)

    print(f"  - Exp #027 (PB LB 0.38794): {len(sub027)} rows")
    print(f"  - Exp #029 (LB 0.38797):    {len(sub029)} rows")

    log_027 = np.log1p(sub027["sales"].values)
    log_029 = np.log1p(sub029["sales"].values)

    corr = np.corrcoef(log_027, log_029)[0, 1]
    diff_rmsle = np.sqrt(np.mean((log_027 - log_029)**2))
    print(f"  - Correlation: {corr:.6f}")
    print(f"  - Distance RMSLE: {diff_rmsle:.5f}")

    print("\n[2/4] Calculating 50:50 log-space geometric mean...")
    blend_log = 0.50 * log_027 + 0.50 * log_029
    pred_sales = np.clip(np.expm1(blend_log), 0, None)

    print("\n[3/4] Applying 21-day safe zero sales filter...")
    test = pd.read_csv("003_store-sales/data/test.csv")
    train_df = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    zero_pairs = set(series_21_sum[series_21_sum == 0].index)
    zero_mask = test.set_index(["store_nbr", "family"]).index.isin(zero_pairs)
    pred_sales[zero_mask] = 0.0
    print(f"  - Zero clip applied: {zero_mask.sum()} rows (expected: 2032 rows)")

    out_path = "003_store-sales/submission_exp030_dual_optimal_blend.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": pred_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    pred_log_clipped = np.log1p(sub_final["sales"].values)
    rmsle_vs_027 = np.sqrt(np.mean((pred_log_clipped - log_027)**2))
    rmsle_vs_029 = np.sqrt(np.mean((pred_log_clipped - log_029)**2))

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (Exp027: {sub027['sales'].mean():.4f}, Exp029: {sub029['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f}")
    print(f"  Max sales:    {sub_final['sales'].max():.4f}")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()}")
    print(f"\n  Distance RMSLE vs Exp #027 (PB 0.38794): {rmsle_vs_027:.5f}")
    print(f"  Distance RMSLE vs Exp #029 (0.38797):    {rmsle_vs_029:.5f}")

if __name__ == "__main__":
    main()
