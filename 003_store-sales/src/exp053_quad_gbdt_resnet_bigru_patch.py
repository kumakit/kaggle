"""
003_store-sales: Exp #053 - Four-System Orthogonal Quad Hybrid Blend
Kaggle Competition: Store Sales - Time Series Forecasting

Four Orthogonal Architecture Families (Zero-Filter Preserved from Exp #036):
  1. Exp #036: GBDT Multi-Layer Ensemble (LB 0.38788, 2,001 zero rows)
  2. Exp #043: PyTorch Direct Multi-Horizon ResNet MLP 5-Seeds (Val 0.41263)
  3. Exp #050: Colab L4 GPU Seq2Horizon Dilated-TCN + BiGRU + CrossAttn 5-Seeds (Val 0.38926)
     -> Triad Exp #051 (36% + 39% + 25%) = LB 0.37839 (Rank 26/633, Top 5% Silver Zone!)
  4. Exp #052: Colab L4 GPU RevIN + DLinear + Patch-Transformer + 2Y-YoY 5-Seeds
     * Record Single-Model Val Raw RMSLE = 0.38636 (-0.00290 vs Exp050, -0.00480 vs best GBDT Exp038!)
     * Correlation with Exp #051 Triad = 0.999285 (Shift RMSLE = 0.09364)

Exp #053 Default (gamma = 0.25 onto Exp #051):
  - Effective Log-Space Weights: 27.0% GBDT + 29.25% ResNet + 18.75% TCN-BiGRU + 25.0% Patch-DLinear-Transformer
"""

import argparse
import numpy as np
import pandas as pd


def run_quad_blend(gamma: float = 0.25, exp_num: int = 53) -> str:
    """Blend Exp #051 Triad PB with Exp #052 RevIN+DLinear+Transformer while preserving Exp #036 zeros."""
    w_51 = 1.0 - gamma
    w_36 = 0.36 * w_51
    w_43 = 0.39 * w_51
    w_50 = 0.25 * w_51
    w_52 = gamma

    print("=" * 80)
    print(
        f"  003 Store Sales: Exp #{exp_num:03d} - Quad Blend "
        f"({w_36*100:.2f}% GBDT + {w_43*100:.2f}% ResNet + {w_50*100:.2f}% BiGRU + {w_52*100:.2f}% Patch-Trans)"
    )
    print("=" * 80)

    sub036 = pd.read_csv("003_store-sales/submission_exp036_golden_spice.csv")
    sub043 = pd.read_csv("003_store-sales/submission_exp043_colab_l4_dnn.csv")
    sub050 = pd.read_csv("003_store-sales/submission_exp050_colab_l4_bigru_tcn.csv")
    sub051 = pd.read_csv("003_store-sales/submission_exp051_triad_bigru25.csv")
    sub052 = pd.read_csv("003_store-sales/submission_exp052_colab_l4_patch_dlinear.csv")
    test = pd.read_csv("003_store-sales/data/test.csv")

    for name, df in [("Exp043", sub043), ("Exp050", sub050), ("Exp051", sub051), ("Exp052", sub052), ("Test", test)]:
        assert (sub036["id"].values == df["id"].values).all(), f"ID mismatch on {name}!"

    log036 = np.log1p(sub036["sales"].values)
    log043 = np.log1p(sub043["sales"].values)
    log050 = np.log1p(sub050["sales"].values)
    log051 = np.log1p(sub051["sales"].values)
    log052 = np.log1p(sub052["sales"].values)

    log_blend = w_36 * log036 + w_43 * log043 + w_50 * log050 + w_52 * log052

    zero_mask = sub036["sales"].values == 0.0
    log_blend = np.where(zero_mask, 0.0, log_blend)
    sales_final = np.clip(np.expm1(log_blend), 0.0, None)

    pct = int(round(gamma * 100))
    out_path = f"003_store-sales/submission_exp{exp_num:03d}_quad_patch{pct}.csv"
    sub = pd.DataFrame({"id": test["id"], "sales": sales_final})
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Output saved to: {out_path} ({len(sub)} rows)")

    assert len(sub) == 28512
    assert sub["sales"].isna().sum() == 0
    assert (sub["sales"] < 0).sum() == 0
    assert ((sub["sales"].values == 0.0) == zero_mask).all(), "Zero mask mismatch!"

    shift_051 = np.sqrt(np.mean((log_blend - log051) ** 2))
    shift_036 = np.sqrt(np.mean((log_blend - log036) ** 2))
    corr_051 = np.corrcoef(log_blend, log051)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs Exp051 {sub051['sales'].mean():.4f} / Exp036 {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs Exp051 {sub051['sales'].median():.4f} / Exp036 {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs Exp051 {sub051['sales'].max():.4f} / Exp036 {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #051 (0.37839): {shift_051:.6f} (Corr: {corr_051:.7f})")
    print(f"  Shift RMSLE from Exp #036 (0.38788): {shift_036:.6f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Exp #053+: Quad hybrid blend (GBDT + ResNet + BiGRU + Patch-Transformer).")
    parser.add_argument("--gamma", type=float, default=0.25, help="Weight on Exp #052 Patch-DLinear-Transformer (default: 0.25)")
    parser.add_argument("--exp", type=int, default=53, help="Experiment number (default: 53)")
    args = parser.parse_args()
    run_quad_blend(args.gamma, args.exp)


if __name__ == "__main__":
    main()
