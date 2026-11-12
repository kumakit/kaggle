"""
003_store-sales: Exp #041 - Regional School Disentangled Micro-Spice Blend (Exp036 PB 92.5% + Exp038 Regional 7.5%)
Kaggle Competition: Store Sales - Time Series Forecasting

Rationale & Design:
  - Exp #036: Current Personal Best (0.38788, Rank 79)
  - Exp #038: Colab L4 GPU Regional School Disentangled 5-Seeds Ensemble (Val 0.39116)
  - Insight from Exp #039 (25% spice -> LB 0.38878):
    * 25% spice injected excessive variance from the single model into the ensemble flat basin.
    * Shift RMSLE for Exp #036 PB was 0.00698 (Corr 0.999996).
    * Setting alpha = 7.5% precisely matches this golden shift (Shift RMSLE = 0.00720, Corr = 0.9999958).
  - Mathematical Blending:
    log(1 + y) = 0.925 * log(1 + y_036) + 0.075 * log(1 + y_038)
"""

import os
import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #041 - Regional Micro-Spice Blend (92.5% PB + 7.5% Regional)")
    print("=" * 80)

    sub036_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub038_path = "003_store-sales/submission_exp038_colab_regional.csv"

    print("\n[1/3] Loading Exp #036 (PB 0.38788) and Exp #038 (Regional School L4)...")
    sub036 = pd.read_csv(sub036_path)
    sub038 = pd.read_csv(sub038_path)
    test = pd.read_csv("003_store-sales/data/test.csv")

    log036 = np.log1p(sub036["sales"].values)
    log038 = np.log1p(sub038["sales"].values)

    corr = np.corrcoef(log036, log038)[0, 1]
    diff_init = np.sqrt(np.mean((log038 - log036) ** 2))
    print(f"  - Correlation between Exp036 and Exp038: {corr:.6f}")
    print(f"  - Initial RMSLE difference: {diff_init:.5f}")

    print("\n[2/3] Blending in log-space: Exp #036 (92.5%) + Exp #038 (7.5%)...")
    alpha = 0.075
    log_blend = (1.0 - alpha) * log036 + alpha * log038
    sales_final = np.clip(np.expm1(log_blend), 0, None)

    out_path = "003_store-sales/submission_exp041_regional_micro_spice.csv"
    sub = pd.DataFrame({
        "id": test["id"],
        "sales": sales_final
    })
    sub.to_csv(out_path, index=False)
    print(f"\n[3/3] Output saved to: {out_path} ({len(sub)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub.head(10))

    shift_rmsle = np.sqrt(np.mean((log_blend - log036) ** 2))
    blend_corr = np.corrcoef(log_blend, log036)[0, 1]
    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs Exp036 PB {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs Exp036 PB {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs Exp036 PB {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 PB {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (PB 0.38788): {shift_rmsle:.6f}")
    print(f"  Correlation with Exp #036 (PB 0.38788): {blend_corr:.7f}")


if __name__ == "__main__":
    main()
