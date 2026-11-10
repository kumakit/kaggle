"""
003_store-sales: Exp #031 - Micro Transactions Spice Tuning (Exp030 98.5% + Exp022 1.5%)
Kaggle Competition: Store Sales - Time Series Forecasting

Methodology:
  - Takes the all-time personal best Exp #030 (LB 0.38793) as anchor (98.5%).
  - Injects a 1.5% micro-spice of Exp #022 (Transactions 2-Stage model, diversity distance RMSLE 0.136).
  - Designed for final submission before the 9:00 AM daily limit reset.
  - Safe 21-day zero sales filter (127 series / 2,032 rows).
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #031 - Micro Transactions Spice Tuning")
    print("=" * 80)

    print("\n[1/4] Loading Exp #030 and Exp #022...")
    f_030 = "003_store-sales/submission_exp030_dual_optimal_blend.csv"
    f_022 = "003_store-sales/submission_exp022_colab_2stage.csv"

    sub030 = pd.read_csv(f_030)
    sub022 = pd.read_csv(f_022)

    log_030 = np.log1p(sub030["sales"].values)
    log_022 = np.log1p(sub022["sales"].values)

    w_030 = 0.985
    w_022 = 0.015

    print(f"\n[2/4] Blending: Exp030 ({w_030*100:.1f}%) + Exp022 ({w_022*100:.1f}%)...")
    blend_log = w_030 * log_030 + w_022 * log_022
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

    out_path = "003_store-sales/submission_exp031_spice_micro.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": pred_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    pred_log_clipped = np.log1p(sub_final["sales"].values)
    diff_rmsle = np.sqrt(np.mean((pred_log_clipped - log_030)**2))
    corr = np.corrcoef(pred_log_clipped, log_030)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (Exp030: {sub030['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f}")
    print(f"  Max sales:    {sub_final['sales'].max():.4f}")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()}")
    print(f"\n  Shift RMSLE from Exp #030 (PB 0.38793): {diff_rmsle:.5f}")
    print(f"  Correlation with Exp #030: {corr:.6f}")

if __name__ == "__main__":
    main()
