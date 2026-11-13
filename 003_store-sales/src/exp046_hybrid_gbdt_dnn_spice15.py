"""
003_store-sales: Exp #046 - Heterogeneous GBDT + PyTorch DNN Hybrid Blend (85% Exp036 + 15% DNN Spice)
Kaggle Competition: Store Sales - Time Series Forecasting

Rationale & Design:
  - Exp #036: GBDT Multi-Layer Ensemble (LB 0.38788)
  - Exp #043: Colab L4 GPU PyTorch Direct Multi-Horizon ResNet 5-Seeds (Val 0.41263, Corr 0.998094)
  - Exp #044 (5% DNN): LB 0.38632 (-0.00156)
  - Exp #045 (10% DNN): LB 0.38491 (-0.00141, Rank 63, only 0.00013 behind Top 10% Rank 62 0.38478)
  - Exp #046 Hypothesis:
    * Testing alpha = 0.15 (85% Exp036 GBDT + 15% Exp043 PyTorch DNN) in log-space.
    * Preserves exact 0.0 predictions from Exp #036 (2,001 rows) to protect zero-sales filter integrity.
    * Expected to break decisively past Rank 62 (0.38478) into the Top 10% Bronze Medal zone.
"""

import argparse
import numpy as np
import pandas as pd


def run_blend(alpha: float) -> str:
    pct = int(round(alpha * 100))
    gbdt_pct = 100 - pct
    print("=" * 80)
    print(f"  003 Store Sales: Exp #046 - Hybrid GBDT ({gbdt_pct}%) + PyTorch DNN ({pct}%) Blend")
    print("=" * 80)

    sub036_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub043_path = "003_store-sales/submission_exp043_colab_l4_dnn.csv"
    sub045_path = "003_store-sales/submission_exp045_hybrid_dnn_spice10.csv"

    print(f"\n[1/3] Loading Exp #036 (0.38788), Exp #043 (DNN), and Exp #045 (PB 0.38491)...")
    sub036 = pd.read_csv(sub036_path)
    sub043 = pd.read_csv(sub043_path)
    sub045 = pd.read_csv(sub045_path)
    test = pd.read_csv("003_store-sales/data/test.csv")

    assert (sub036["id"].values == sub043["id"].values).all(), "ID mismatch!"
    assert (sub036["id"].values == sub045["id"].values).all(), "Exp045 ID mismatch!"
    assert (sub036["id"].values == test["id"].values).all(), "Test ID mismatch!"

    log036 = np.log1p(sub036["sales"].values)
    log043 = np.log1p(sub043["sales"].values)
    log045 = np.log1p(sub045["sales"].values)

    print(f"\n[2/3] Blending in log-space ({gbdt_pct}% Exp036 + {pct}% DNN) with Zero-Filter Preservation...")
    log_blend = (1.0 - alpha) * log036 + alpha * log043

    # Preserve Exp #036 exact zeros (2,001 rows)
    zero_mask = sub036["sales"].values == 0.0
    log_blend = np.where(zero_mask, 0.0, log_blend)
    sales_final = np.clip(np.expm1(log_blend), 0, None)

    out_path = f"003_store-sales/submission_exp046_hybrid_dnn_spice{pct}.csv"
    sub = pd.DataFrame({
        "id": test["id"],
        "sales": sales_final,
    })
    sub.to_csv(out_path, index=False)
    print(f"\n[3/3] Output saved to: {out_path} ({len(sub)} rows)")

    # Validation checks
    assert len(sub) == 28512
    assert sub["sales"].isna().sum() == 0
    assert (sub["sales"] < 0).sum() == 0
    assert ((sub["sales"].values == 0.0) == zero_mask).all(), "Zero mask mismatch!"

    shift_from_036 = np.sqrt(np.mean((log_blend - log036) ** 2))
    shift_from_045 = np.sqrt(np.mean((log_blend - log045) ** 2))
    corr_036 = np.corrcoef(log_blend, log036)[0, 1]
    corr_045 = np.corrcoef(log_blend, log045)[0, 1]

    print("\n=== Top 10 Predictions ===")
    print(sub.head(10))

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs Exp045 PB {sub045['sales'].mean():.4f} / Exp036 {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs Exp045 PB {sub045['sales'].median():.4f} / Exp036 {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs Exp045 PB {sub045['sales'].max():.4f} / Exp036 {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (0.38788): {shift_from_036:.6f} (Corr: {corr_036:.7f})")
    print(f"  Shift RMSLE from Exp #045 (0.38491): {shift_from_045:.6f} (Corr: {corr_045:.7f})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Exp #046: Zero-preserved log-space GBDT + PyTorch DNN hybrid blend.")
    parser.add_argument("--alpha", type=float, default=0.15, help="Weight for Exp #043 PyTorch DNN (default: 0.15)")
    args = parser.parse_args()
    run_blend(args.alpha)


if __name__ == "__main__":
    main()
