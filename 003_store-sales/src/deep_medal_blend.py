"""
003_store-sales: Exp #025 - Deep 5-Seeds Medal Breakthrough Blend
Kaggle Competition: Store Sales - Time Series Forecasting

Blend Configuration (Log-space Weighted Geometric Mean):
  - Exp #015 (Anchor / PB LB 0.38856):                                       45%
  - Exp #024 (Colab L4 GPU Deeper Trees 5-Seeds 1000-Trees / Val 0.39729):    25%
  - Exp #020 (Colab L4 GPU 5-Seeds 1000-Trees / Val 0.39794 / LB 0.39113):    15%
  - Exp #013 (Day-of-Week Group Mean Single Best / LB 0.38972):              12%
  - Exp #022 (Stage 1 Transactions Model / Diversity Spice):                   3%
  - Post-processing: Safe 21-day zero sales filter (124 series / 1,984 rows)
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 75)
    print("  003 Store Sales: Exp #025 - Deep 5-Seeds Medal Breakthrough Blend")
    print("=" * 75)

    print("\n[1/4] Loading model submission files...")
    f_015 = "003_store-sales/submission_exp015.csv"
    f_024 = "003_store-sales/submission_exp024_colab_deep.csv"
    f_020 = "003_store-sales/submission_exp020_colab_5seeds.csv"
    f_013 = "003_store-sales/submission_exp013.csv"
    f_022 = "003_store-sales/submission_exp022_colab_2stage.csv"

    sub_015 = pd.read_csv(f_015)
    sub_024 = pd.read_csv(f_024)
    sub_020 = pd.read_csv(f_020)
    sub_013 = pd.read_csv(f_013)
    sub_022 = pd.read_csv(f_022)

    print(f"  - Exp #015 (PB LB 0.38856):       {len(sub_015)} rows")
    print(f"  - Exp #024 (Colab L4 Deep 5S):    {len(sub_024)} rows")
    print(f"  - Exp #020 (Colab L4 5Seeds):     {len(sub_020)} rows")
    print(f"  - Exp #013 (DOW Group Mean):      {len(sub_013)} rows")
    print(f"  - Exp #022 (Transactions Spice):  {len(sub_022)} rows")

    log_015 = np.log1p(sub_015["sales"].values)
    log_024 = np.log1p(sub_024["sales"].values)
    log_020 = np.log1p(sub_020["sales"].values)
    log_013 = np.log1p(sub_013["sales"].values)
    log_022 = np.log1p(sub_022["sales"].values)

    print("\n[2/4] Calculating log-space weighted blend...")
    w_015 = 0.45
    w_024 = 0.25
    w_020 = 0.15
    w_013 = 0.12
    w_022 = 0.03

    blend_log = (
        w_015 * log_015 +
        w_024 * log_024 +
        w_020 * log_020 +
        w_013 * log_013 +
        w_022 * log_022
    )

    pred_sales = np.clip(np.expm1(blend_log), 0, None)

    print("\n[3/4] Applying 21-day safe zero sales filter...")
    train_df = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    zero_pairs = set(series_21_sum[series_21_sum == 0].index)
    print(f"  - Safe zero series identified: {len(zero_pairs)} / 1,782 pairs")

    test_df = pd.read_csv("003_store-sales/data/test.csv")
    zero_mask = test_df.set_index(["store_nbr", "family"]).index.isin(zero_pairs)
    clipped_count = zero_mask.sum()
    pred_sales[zero_mask] = 0.0
    print(f"  - Zero clip applied rows: {clipped_count} rows")

    out_path = "003_store-sales/submission_exp025_deep_medal_blend.csv"
    sub_final = pd.DataFrame({
        "id": sub_015["id"],
        "sales": pred_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f}")
    print(f"  Median sales: {sub_final['sales'].median():.4f}")
    print(f"  Max sales:    {sub_final['sales'].max():.4f}")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()}")

    pred_log_final = np.log1p(sub_final["sales"].values)
    corr_015 = np.corrcoef(pred_log_final, log_015)[0, 1]
    rmsle_015 = np.sqrt(np.mean((pred_log_final - log_015)**2))
    print(f"\n  Correlation with Exp #015: {corr_015:.6f}")
    print(f"  Distance RMSLE with Exp #015: {rmsle_015:.5f}")

if __name__ == "__main__":
    main()
