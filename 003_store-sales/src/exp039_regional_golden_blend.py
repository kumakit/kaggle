"""
003_store-sales: Exp #039 - Regional School Disentangled Golden Blend (Exp036 PB 75% + Exp038 Regional 25%)
Kaggle Competition: Store Sales - Time Series Forecasting

Rationale & Design:
  - Exp #036: Current Personal Best (0.38788, Rank 65, Sierra precision spice)
  - Exp #038: Colab L4 GPU Regionally Disentangled School Season 5-Seeds 1000-Trees Ensemble
    * Explicitly models Costa (Apr-May) vs Sierra (Aug-Sep) school calendars across 60 features.
    * Features native protection for Sierra school supplies from artificial zero-clipping.
  - Blending Formulation:
    * 75% Exp #036 (PB anchor) + 25% Exp #038 (native regional school GBDT)
    * Geometric interpolation in log-space:
        log(1 + y) = 0.75 * log(1 + y_036) + 0.25 * log(1 + y_038)
"""

import os
import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #039 - Regional School Disentangled Golden Blend")
    print("=" * 80)

    sub036_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub038_path = "003_store-sales/submission_exp038_colab_regional.csv"

    if not os.path.exists(sub038_path):
        print(f"[WAIT] {sub038_path} not found yet. Please run this script after Exp #038 completes!")
        return

    print("\n[1/3] Loading Exp #036 (PB 0.38788) and Exp #038 (Regional School L4)...")
    sub036 = pd.read_csv(sub036_path)
    sub038 = pd.read_csv(sub038_path)
    test = pd.read_csv("003_store-sales/data/test.csv")

    log036 = np.log1p(sub036["sales"].values)
    log038 = np.log1p(sub038["sales"].values)

    corr = np.corrcoef(log036, log038)[0, 1]
    diff_init = np.sqrt(np.mean((log038 - log036)**2))
    print(f"  - Correlation between Exp036 and Exp038: {corr:.6f}")
    print(f"  - Initial RMSLE difference: {diff_init:.5f}")

    print("\n[2/3] Blending in log-space: Exp #036 (75%) + Exp #038 (25%)...")
    log_blend = 0.75 * log036 + 0.25 * log038
    sales_final = np.clip(np.expm1(log_blend), 0, None)

    out_path = "003_store-sales/submission_exp039_regional_golden.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": sales_final
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[3/3] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    log_final = np.log1p(sales_final)
    diff_rmsle = np.sqrt(np.mean((log_final - log036)**2))
    corr_final = np.corrcoef(log_final, log036)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp036 PB {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp036 PB {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp036 PB {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp036 PB {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (PB 0.38788): {diff_rmsle:.6f}")
    print(f"  Correlation with Exp #036 (PB 0.38788): {corr_final:.7f}")


if __name__ == "__main__":
    main()
