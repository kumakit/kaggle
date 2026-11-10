"""
003_store-sales: Exp #026 - School Season Domain Boost (Sierra / Amazonia Regions)
Kaggle Competition: Store Sales - Time Series Forecasting

Key Rationale:
  - Ecuador Sierra & Amazonia regions start school year in September.
  - Historical data (2015, 2016) shows SCHOOL AND OFFICE SUPPLIES surges 2.5x to 3.2x in late August.
  - In 2017, early August already reached 50,169 sales, but models (Exp #025) decayed to 37,533 (0.76x)
    due to autoregressive roll-forward dampening.
  - Dynamic curve post-processing boost applied only to Sierra/Amazonia stores for SCHOOL AND OFFICE SUPPLIES:
    * Aug 16 - Aug 25: 1.25x
    * Aug 26 - Aug 28 (Peak pre-school weekend): 1.50x
    * Aug 29 - Aug 31 (Final rush): 1.35x
  - Preserves 100% of non-school predictions and respects 21-day safe zero filter.
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 75)
    print("  003 Store Sales: Exp #026 - School Season Domain Boost")
    print("=" * 75)

    print("\n[1/4] Loading Exp #025 submission and metadata...")
    sub025_path = "003_store-sales/submission_exp025_deep_medal_blend.csv"
    test_path = "003_store-sales/data/test.csv"
    stores_path = "003_store-sales/data/stores.csv"

    sub025 = pd.read_csv(sub025_path)
    test = pd.read_csv(test_path, parse_dates=["date"])
    stores = pd.read_csv(stores_path)

    merged = sub025.merge(test, on="id").merge(stores, on="store_nbr")
    print(f"  - Loaded submission: {len(merged)} rows")

    print("\n[2/4] Identifying Sierra & Amazonia regions for SCHOOL AND OFFICE SUPPLIES...")
    sierra_states = {
        "Pichincha", "Tungurahua", "Azuay", "Pastaza",
        "Cotopaxi", "Imbabura", "Chimborazo", "Loja", "Bolivar"
    }

    target_mask = (merged["family"] == "SCHOOL AND OFFICE SUPPLIES") & (merged["state"].isin(sierra_states))
    target_count = target_mask.sum()
    print(f"  - Target rows to boost: {target_count} / {len(merged)} rows (31 stores x 16 days)")
    
    curr_target_sum = merged.loc[target_mask, "sales"].sum()
    print(f"  - Pre-boost target sales sum: {curr_target_sum:.1f} (mean: {curr_target_sales_mean:.2f})" if 'curr_target_sales_mean' in locals() else f"  - Pre-boost target sales sum: {curr_target_sum:.1f}")

    print("\n[3/4] Applying dynamic curve boost...")
    boosted_sales = merged["sales"].copy()
    c_mults = []
    for d in merged.loc[target_mask, "date"]:
        if d >= pd.to_datetime("2017-08-26") and d <= pd.to_datetime("2017-08-28"):
            c_mults.append(1.50)
        elif d >= pd.to_datetime("2017-08-29"):
            c_mults.append(1.35)
        else:
            c_mults.append(1.25)

    boosted_sales.loc[target_mask] *= np.array(c_mults)
    new_target_sum = boosted_sales.loc[target_mask].sum()
    print(f"  - Post-boost target sales sum: {new_target_sum:.1f} (+{new_target_sum - curr_target_sum:.1f}, +{(new_target_sum/curr_target_sum - 1)*100:.1f}%)")

    # Safe zero filter verification
    train_df = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    zero_pairs = set(series_21_sum[series_21_sum == 0].index)
    zero_mask = test.set_index(["store_nbr", "family"]).index.isin(zero_pairs)
    boosted_sales[zero_mask] = 0.0
    print(f"  - Zero clip verified: {zero_mask.sum()} rows")

    out_path = "003_store-sales/submission_exp026_school_boost.csv"
    sub_final = pd.DataFrame({
        "id": sub025["id"],
        "sales": boosted_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Saved submission to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    print("\n=== Statistics Comparison ===")
    print(f"  Exp #025 Mean: {sub025['sales'].mean():.4f} -> Exp #026 Mean: {sub_final['sales'].mean():.4f}")
    print(f"  Exp #025 Max:  {sub025['sales'].max():.4f} -> Exp #026 Max:  {sub_final['sales'].max():.4f}")
    print(f"  Exp #025 Zeros: {(sub025['sales'] == 0).sum()} -> Exp #026 Zeros: {(sub_final['sales'] == 0).sum()}")

    log_025 = np.log1p(sub025["sales"])
    log_026 = np.log1p(sub_final["sales"])
    shift_rmsle = np.sqrt(np.mean((log_026 - log_025)**2))
    corr = np.corrcoef(log_026, log_025)[0, 1]
    print(f"\n  Shift RMSLE from Exp #025: {shift_rmsle:.5f}")
    print(f"  Correlation with Exp #025: {corr:.6f}")

if __name__ == "__main__":
    main()
