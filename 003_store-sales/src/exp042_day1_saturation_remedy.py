"""
003_store-sales: Exp #042 - Day 1 Mega-Promo Saturation Remedy & Zero Un-clip
Kaggle Competition: Store Sales - Time Series Forecasting

Key Discoveries & Formulations:
  1. Day 1 (2017-08-16) Unprecedented Mega-Promotion:
     - 30,539 promotions across company (2x normal Wednesday of 16,000).
     - GBDT decision trees suffer from extrapolation saturation when onpromotion exceeds historical maximums.
  2. Day 1 Zero-Filter Bug:
     - 25 rows on 2017-08-16 had onpromotion=14~16 but were clipped to 0.0 by the 21-day rule:
       * LAWN AND GARDEN: 14 stores with onpromotion=16 clipped to 0.0
       * SCHOOL AND OFFICE SUPPLIES: 11 stores with onpromotion=14 clipped to 0.0
     - Each 0 prediction suffers severe penalty (log(1+25) - log(1))^2 = 10.6 per row.
  3. Remedies:
     - Component A: Un-clip Day 1 promo zeros:
       * For SCHOOL stores: restore natural GBDT prediction from Exp #038 (avg ~10 units)
       * For LAWN stores: restore conservative active store median (~10-15 units)
     - Component B: Gentle saturation boost for extreme promo families on Day 1:
       * Target: SCHOOL AND OFFICE SUPPLIES, LAWN AND GARDEN, LINGERIE
       * log(1+y) += beta * log(promo / hist_max) with beta=0.15
"""

import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #042 - Day 1 Mega-Promo Saturation Remedy & Zero Un-clip")
    print("=" * 80)

    sub036_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub038_path = "003_store-sales/submission_exp038_colab_regional.csv"
    test_path = "003_store-sales/data/test.csv"
    train_path = "003_store-sales/data/train.csv"

    print("\n[1/4] Loading Exp #036 (PB 0.38788), Exp #038, and datasets...")
    sub036 = pd.read_csv(sub036_path)
    sub038 = pd.read_csv(sub038_path)
    test = pd.read_csv(test_path)
    train = pd.read_csv(train_path)

    test["pred_036"] = sub036["sales"].values
    test["pred_038"] = sub038["sales"].values
    test["sales_new"] = sub036["sales"].values

    # Historical max promotion in train
    hist_max_promo = train.groupby(["store_nbr", "family"])["onpromotion"].max().reset_index()
    hist_max_promo.rename(columns={"onpromotion": "hist_max_p"}, inplace=True)
    test = pd.merge(test, hist_max_promo, on=["store_nbr", "family"], how="left").fillna(0)

    print("\n[2/4] Component A: Un-clipping 25 Day 1 Promo-Zero rows...")
    day1_mask = test["date"] == "2017-08-16"
    promo_zero_mask = day1_mask & (test["onpromotion"] > 0) & (test["sales_new"] < 0.001)

    print(f"  - Found {promo_zero_mask.sum()} rows matching Day 1 promo zeros")

    # Un-clip SCHOOL: Use Exp038 prediction (clipped to minimum 5.0, maximum 25.0)
    school_mask = promo_zero_mask & (test["family"] == "SCHOOL AND OFFICE SUPPLIES")
    school_restored = np.clip(test.loc[school_mask, "pred_038"].values, 5.0, 25.0)
    # If pred_038 was also 0 for any store, set to conservative 8.0
    school_restored = np.where(school_restored < 5.0, 8.0, school_restored)
    test.loc[school_mask, "sales_new"] = school_restored
    print(f"  - Rescued SCHOOL ({school_mask.sum()} stores): mean={school_restored.mean():.2f}")

    # Un-clip LAWN AND GARDEN: Active stores median on Day 1 was 27.5. Set conservative 12.0
    lawn_mask = promo_zero_mask & (test["family"] == "LAWN AND GARDEN")
    lawn_restored = np.full(lawn_mask.sum(), 12.0)
    test.loc[lawn_mask, "sales_new"] = lawn_restored
    print(f"  - Rescued LAWN AND GARDEN ({lawn_mask.sum()} stores): value=12.0")

    print("\n[3/4] Component B: Gentle Saturation Boost for Extreme Promo Families on Day 1...")
    # Target families with massive promotion surplus on Day 1
    target_fams = ["SCHOOL AND OFFICE SUPPLIES", "LAWN AND GARDEN", "LINGERIE"]
    beta = 0.15  # Conservative elasticity

    for fam in target_fams:
        fam_day1_mask = day1_mask & (test["family"] == fam) & (test["onpromotion"] > test["hist_max_p"]) & (test["sales_new"] > 0)
        if fam_day1_mask.sum() > 0:
            ratio = (test.loc[fam_day1_mask, "onpromotion"].values + 1.0) / (test.loc[fam_day1_mask, "hist_max_p"].values + 1.0)
            log_orig = np.log1p(test.loc[fam_day1_mask, "sales_new"].values)
            log_boosted = log_orig + beta * np.log(np.clip(ratio, 1.0, 5.0))
            new_val = np.expm1(log_boosted)
            test.loc[fam_day1_mask, "sales_new"] = new_val
            print(f"  - Boosted {fam:26s} ({fam_day1_mask.sum():2d} stores): mean ratio {ratio.mean():.2f}x -> sales boosted {new_val.mean() / np.expm1(log_orig).mean():.3f}x")

    out_path = "003_store-sales/submission_exp042_day1_saturation_remedy.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": test["sales_new"]
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    # Validation
    assert len(sub_final) == 28512
    assert sub_final["sales"].isna().sum() == 0
    assert (sub_final["sales"] < 0).sum() == 0

    log_036 = np.log1p(sub036["sales"].values)
    log_final = np.log1p(sub_final["sales"].values)
    diff_rmsle = np.sqrt(np.mean((log_final - log_036) ** 2))
    corr = np.corrcoef(log_final, log_036)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp036 PB {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp036 PB {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp036 PB {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp036 PB {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (PB 0.38788): {diff_rmsle:.6f}")
    print(f"  Correlation with Exp #036 (PB 0.38788): {corr:.7f}")


if __name__ == "__main__":
    main()
