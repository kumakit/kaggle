"""
003_store-sales: Exp #045 - Heterogeneous GBDT + PyTorch DNN Hybrid Blend (90% Exp036 + 10% DNN Spice)
Kaggle Competition: Store Sales - Time Series Forecasting

Rationale & Design:
  - Exp #036: GBDT Multi-Layer Ensemble (LB 0.38788)
  - Exp #043: Colab L4 GPU PyTorch Direct Multi-Horizon ResNet 5-Seeds (Val 0.41263, Corr 0.998094)
  - Exp #044: 95% Exp036 + 5% Exp043 jumped LB from 0.38788 to 0.38632 (-0.00156)!
  - Exp #045 Hypothesis:
    * Testing alpha = 0.10 (90% Exp036 GBDT + 10% Exp043 PyTorch DNN) in log-space.
    * Preserves exact 0.0 predictions from Exp #036 (2,001 rows) to avoid dead-series noise.
    * Checks whether doubling DNN spice pushes past the Top 10% Bronze cutoff (0.38478).
"""

import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #045 - Hybrid GBDT (90%) + PyTorch DNN (10%) Blend")
    print("=" * 80)

    sub036_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub043_path = "003_store-sales/submission_exp043_colab_l4_dnn.csv"
    sub044_path = "003_store-sales/submission_exp044_hybrid_dnn_spice5.csv"

    print("\n[1/3] Loading Exp #036 (0.38788), Exp #043 (DNN), and Exp #044 (PB 0.38632)...")
    sub036 = pd.read_csv(sub036_path)
    sub043 = pd.read_csv(sub043_path)
    sub044 = pd.read_csv(sub044_path)
    test = pd.read_csv("003_store-sales/data/test.csv")

    assert (sub036["id"].values == sub043["id"].values).all(), "ID mismatch!"
    assert (sub036["id"].values == sub044["id"].values).all(), "Exp044 ID mismatch!"
    assert (sub036["id"].values == test["id"].values).all(), "Test ID mismatch!"

    log036 = np.log1p(sub036["sales"].values)
    log043 = np.log1p(sub043["sales"].values)
    log044 = np.log1p(sub044["sales"].values)

    print("\n[2/3] Blending in log-space (90% Exp036 + 10% DNN) with Zero-Filter Preservation...")
    alpha = 0.10
    log_blend = (1.0 - alpha) * log036 + alpha * log043

    # Preserve Exp #036 exact zeros (2,001 rows)
    zero_mask = sub036["sales"].values == 0.0
    log_blend = np.where(zero_mask, 0.0, log_blend)
    sales_final = np.clip(np.expm1(log_blend), 0, None)

    out_path = "003_store-sales/submission_exp045_hybrid_dnn_spice10.csv"
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
    shift_from_044 = np.sqrt(np.mean((log_blend - log044) ** 2))
    corr_036 = np.corrcoef(log_blend, log036)[0, 1]
    corr_044 = np.corrcoef(log_blend, log044)[0, 1]

    print("\n=== Top 10 Predictions ===")
    print(sub.head(10))

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs Exp044 PB {sub044['sales'].mean():.4f} / Exp036 {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs Exp044 PB {sub044['sales'].median():.4f} / Exp036 {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs Exp044 PB {sub044['sales'].max():.4f} / Exp036 {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (0.38788): {shift_from_036:.6f} (Corr: {corr_036:.7f})")
    print(f"  Shift RMSLE from Exp #044 (0.38632): {shift_from_044:.6f} (Corr: {corr_044:.7f})")


if __name__ == "__main__":
    main()
