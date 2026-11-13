"""
003_store-sales: Exp #044 - Heterogeneous GBDT + PyTorch DNN Hybrid Blend (95% PB + 5% DNN Spice)
Kaggle Competition: Store Sales - Time Series Forecasting

Rationale & Design:
  - Exp #036: Current Personal Best GBDT Multi-Layer Ensemble (PB 0.38788)
  - Exp #043: Colab L4 GPU PyTorch Direct Multi-Horizon ResNet 5-Seeds (Val 0.41263, Corr 0.998094)
  - Krogh-Vedelsby Diversity Injection:
    * Blending 5% of continuous neural network outputs into the tree-based PB anchor.
    * Preserves exact 0.0 predictions from Exp #036 to protect zero-sales filter integrity.
    * Achieves Shift RMSLE = 0.00783 (Corr = 0.9999953), matching the golden perturbation regime.
"""

import numpy as np
import pandas as pd


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #044 - Hybrid GBDT (95%) + PyTorch DNN (5%) Blend")
    print("=" * 80)

    sub036_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub043_path = "003_store-sales/submission_exp043_colab_l4_dnn.csv"

    print("\n[1/3] Loading Exp #036 (PB 0.38788) and Exp #043 (PyTorch Multi-Horizon DNN)...")
    sub036 = pd.read_csv(sub036_path)
    sub043 = pd.read_csv(sub043_path)
    test = pd.read_csv("003_store-sales/data/test.csv")

    assert (sub036["id"].values == sub043["id"].values).all(), "ID mismatch!"
    assert (sub036["id"].values == test["id"].values).all(), "Test ID mismatch!"

    log036 = np.log1p(sub036["sales"].values)
    log043 = np.log1p(sub043["sales"].values)

    corr = np.corrcoef(log036, log043)[0, 1]
    diff_init = np.sqrt(np.mean((log043 - log036) ** 2))
    print(f"  - Initial Correlation (GBDT PB vs DNN): {corr:.6f}")
    print(f"  - Initial RMSLE difference:             {diff_init:.5f}")

    print("\n[2/3] Blending in log-space (95% PB + 5% DNN) with Zero-Filter Preservation...")
    alpha = 0.05
    log_blend = (1.0 - alpha) * log036 + alpha * log043

    # Preserve Exp #036 exact zeros
    zero_mask = sub036["sales"].values == 0.0
    log_blend = np.where(zero_mask, 0.0, log_blend)
    sales_final = np.clip(np.expm1(log_blend), 0, None)

    out_path = "003_store-sales/submission_exp044_hybrid_dnn_spice5.csv"
    sub = pd.DataFrame({
        "id": test["id"],
        "sales": sales_final
    })
    sub.to_csv(out_path, index=False)
    print(f"\n[3/3] Output saved to: {out_path} ({len(sub)} rows)")

    # Validation checks
    assert len(sub) == 28512
    assert sub["sales"].isna().sum() == 0
    assert (sub["sales"] < 0).sum() == 0

    shift_rmsle = np.sqrt(np.mean((log_blend - log036) ** 2))
    blend_corr = np.corrcoef(log_blend, log036)[0, 1]

    print("\n=== Top 10 Predictions ===")
    print(sub.head(10))

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs Exp036 PB {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs Exp036 PB {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs Exp036 PB {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 PB {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (PB 0.38788): {shift_rmsle:.6f}")
    print(f"  Correlation with Exp #036 (PB 0.38788): {blend_corr:.7f}")


if __name__ == "__main__":
    main()
