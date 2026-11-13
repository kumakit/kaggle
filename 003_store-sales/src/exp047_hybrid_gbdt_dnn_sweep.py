"""
003_store-sales: Exp #047+ - Heterogeneous GBDT + PyTorch DNN Hybrid Blend Ratio Sweep
Kaggle Competition: Store Sales - Time Series Forecasting

Rationale & Design:
  - Exp #036: GBDT Multi-Layer Ensemble (LB 0.38788)
  - Exp #043: Colab L4 GPU PyTorch Direct Multi-Horizon ResNet 5-Seeds (Val 0.41263, Corr 0.998094)
  - Observed LB progression (with Exp #036 exact 2,001 zero rows preserved):
      * alpha = 0.00 (Exp036): 0.38788
      * alpha = 0.05 (Exp044): 0.38632 (-0.00156)
      * alpha = 0.10 (Exp045): 0.38491 (-0.00141)
      * alpha = 0.15 (Exp046): 0.38366 (-0.00125, Rank 54/624 - Top 10% Bronze Zone!)
  - Exp #047 Default:
      * alpha = 0.25 (75% Exp036 GBDT + 25% Exp043 PyTorch DNN) in log-space.
      * Quadratic MSE fit projects ~0.38162 LB around alpha = 0.25.
"""

import argparse
import numpy as np
import pandas as pd


def run_blend(
    alpha: float,
    exp_num: int,
    anchor_path: str = "003_store-sales/submission_exp046_hybrid_dnn_spice15.csv",
    anchor_label: str = "Exp #046 (0.38366)",
) -> str:
    """Blend Exp #036 GBDT and Exp #043 PyTorch DNN in log-space while preserving Exp #036 zeros."""
    pct = int(round(alpha * 100))
    gbdt_pct = 100 - pct
    print("=" * 80)
    print(f"  003 Store Sales: Exp #{exp_num:03d} - Hybrid GBDT ({gbdt_pct}%) + PyTorch DNN ({pct}%) Blend")
    print("=" * 80)

    sub036_path = "003_store-sales/submission_exp036_golden_spice.csv"
    sub043_path = "003_store-sales/submission_exp043_colab_l4_dnn.csv"

    print(f"\n[1/3] Loading Exp #036 (0.38788), Exp #043 (DNN), and {anchor_label}...")
    sub036 = pd.read_csv(sub036_path)
    sub043 = pd.read_csv(sub043_path)
    sub_anchor = pd.read_csv(anchor_path)
    test = pd.read_csv("003_store-sales/data/test.csv")

    assert (sub036["id"].values == sub043["id"].values).all(), "ID mismatch!"
    assert (sub036["id"].values == sub_anchor["id"].values).all(), "Anchor ID mismatch!"
    assert (sub036["id"].values == test["id"].values).all(), "Test ID mismatch!"

    log036 = np.log1p(sub036["sales"].values)
    log043 = np.log1p(sub043["sales"].values)
    log_anchor = np.log1p(sub_anchor["sales"].values)

    print(f"\n[2/3] Blending in log-space ({gbdt_pct}% Exp036 + {pct}% DNN) with Zero-Filter Preservation...")
    log_blend = (1.0 - alpha) * log036 + alpha * log043

    zero_mask = sub036["sales"].values == 0.0
    log_blend = np.where(zero_mask, 0.0, log_blend)
    sales_final = np.clip(np.expm1(log_blend), 0, None)

    out_path = f"003_store-sales/submission_exp{exp_num:03d}_hybrid_dnn_spice{pct}.csv"
    sub = pd.DataFrame({"id": test["id"], "sales": sales_final})
    sub.to_csv(out_path, index=False)
    print(f"\n[3/3] Output saved to: {out_path} ({len(sub)} rows)")

    assert len(sub) == 28512
    assert sub["sales"].isna().sum() == 0
    assert (sub["sales"] < 0).sum() == 0
    assert ((sub["sales"].values == 0.0) == zero_mask).all(), "Zero mask mismatch!"

    shift_036 = np.sqrt(np.mean((log_blend - log036) ** 2))
    shift_anchor = np.sqrt(np.mean((log_blend - log_anchor) ** 2))
    corr_036 = np.corrcoef(log_blend, log036)[0, 1]
    corr_anchor = np.corrcoef(log_blend, log_anchor)[0, 1]

    print("\n=== Top 10 Predictions ===")
    print(sub.head(10))

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs {anchor_label} {sub_anchor['sales'].mean():.4f} / Exp036 {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs {anchor_label} {sub_anchor['sales'].median():.4f} / Exp036 {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs {anchor_label} {sub_anchor['sales'].max():.4f} / Exp036 {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (0.38788): {shift_036:.6f} (Corr: {corr_036:.7f})")
    print(f"  Shift RMSLE from {anchor_label}: {shift_anchor:.6f} (Corr: {corr_anchor:.7f})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Zero-preserved log-space GBDT + PyTorch DNN hybrid sweep.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Weight for Exp #043 PyTorch DNN (default: 0.25)")
    parser.add_argument("--exp", type=int, default=47, help="Experiment number for filename/banner (default: 47)")
    parser.add_argument(
        "--anchor-path",
        type=str,
        default="003_store-sales/submission_exp046_hybrid_dnn_spice15.csv",
        help="Reference submission CSV for comparison",
    )
    parser.add_argument("--anchor-label", type=str, default="Exp #046 (0.38366)", help="Label for anchor submission")
    args = parser.parse_args()
    run_blend(args.alpha, args.exp, args.anchor_path, args.anchor_label)


if __name__ == "__main__":
    main()
