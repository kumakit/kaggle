"""
003_store-sales: Exp #051 - Three-System Orthogonal Triad Hybrid Blend
Kaggle Competition: Store Sales - Time Series Forecasting

Architecture Triad (Zero-Filter Preserved from Exp #036):
  1. Exp #036: GBDT Multi-Layer Ensemble (LB 0.38788, 2,001 zero rows)
  2. Exp #043: PyTorch Direct Multi-Horizon ResNet MLP 5-Seeds (Val 0.41263)
     -> Combined in Exp #049 (48% Exp036 + 52% Exp043) = LB 0.37932 (Rank 32/633)
  3. Exp #050: Colab L4 GPU Seq2Horizon Dilated-TCN + 2-Layer BiGRU + CrossAttn 5-Seeds
     * Val Raw RMSLE = 0.38926 (-0.02337 better than Exp #043!)
     * Correlation with Exp #049 = 0.999043 (Shift RMSLE = 0.10838)

Exp #051 Default (beta = 0.25 onto Exp #049):
  - Effective Log-Space Weights: 36% GBDT + 39% ResNet MLP + 25% TCN-BiGRU
  - Exploits Krogh-Vedelsby three-way ambiguity across Trees, Tabular ResNet, and Temporal Sequence NN.
"""

import argparse
import numpy as np
import pandas as pd


def run_triad_blend(beta: float = 0.25, exp_num: int = 51) -> str:
    """Blend Exp #049 (48% GBDT + 52% ResNet) with Exp #050 (TCN+BiGRU) while preserving Exp #036 zeros."""
    w_049 = 1.0 - beta
    w_036 = 0.48 * w_049
    w_043 = 0.52 * w_049
    w_050 = beta

    print("=" * 80)
    print(
        f"  003 Store Sales: Exp #{exp_num:03d} - Triad Blend "
        f"({w_036*100:.1f}% GBDT + {w_043*100:.1f}% ResNet + {w_050*100:.1f}% TCN-BiGRU)"
    )
    print("=" * 80)

    sub036 = pd.read_csv("003_store-sales/submission_exp036_golden_spice.csv")
    sub043 = pd.read_csv("003_store-sales/submission_exp043_colab_l4_dnn.csv")
    sub049 = pd.read_csv("003_store-sales/submission_exp049_hybrid_dnn_spice52.csv")
    sub050 = pd.read_csv("003_store-sales/submission_exp050_colab_l4_bigru_tcn.csv")
    test = pd.read_csv("003_store-sales/data/test.csv")

    for name, df in [("Exp043", sub043), ("Exp049", sub049), ("Exp050", sub050), ("Test", test)]:
        assert (sub036["id"].values == df["id"].values).all(), f"ID mismatch on {name}!"

    log036 = np.log1p(sub036["sales"].values)
    log043 = np.log1p(sub043["sales"].values)
    log049 = np.log1p(sub049["sales"].values)
    log050 = np.log1p(sub050["sales"].values)

    log_blend = w_036 * log036 + w_043 * log043 + w_050 * log050

    zero_mask = sub036["sales"].values == 0.0
    log_blend = np.where(zero_mask, 0.0, log_blend)
    sales_final = np.clip(np.expm1(log_blend), 0.0, None)

    pct = int(round(beta * 100))
    out_path = f"003_store-sales/submission_exp{exp_num:03d}_triad_bigru{pct}.csv"
    sub = pd.DataFrame({"id": test["id"], "sales": sales_final})
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Output saved to: {out_path} ({len(sub)} rows)")

    assert len(sub) == 28512
    assert sub["sales"].isna().sum() == 0
    assert (sub["sales"] < 0).sum() == 0
    assert ((sub["sales"].values == 0.0) == zero_mask).all(), "Zero mask mismatch!"

    shift_049 = np.sqrt(np.mean((log_blend - log049) ** 2))
    shift_036 = np.sqrt(np.mean((log_blend - log036) ** 2))
    corr_049 = np.corrcoef(log_blend, log049)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs Exp049 {sub049['sales'].mean():.4f} / Exp036 {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs Exp049 {sub049['sales'].median():.4f} / Exp036 {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs Exp049 {sub049['sales'].max():.4f} / Exp036 {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #049 (0.37932): {shift_049:.6f} (Corr: {corr_049:.7f})")
    print(f"  Shift RMSLE from Exp #036 (0.38788): {shift_036:.6f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Exp #051+: Triad hybrid blend (GBDT + ResNet + TCN-BiGRU).")
    parser.add_argument("--beta", type=float, default=0.25, help="Weight on Exp #050 TCN-BiGRU (default: 0.25)")
    parser.add_argument("--exp", type=int, default=51, help="Experiment number (default: 51)")
    args = parser.parse_args()
    run_triad_blend(args.beta, args.exp)


if __name__ == "__main__":
    main()
