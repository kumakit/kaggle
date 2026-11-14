"""
003_store-sales: Exp #055 - Five-System Orthogonal Quintet Hybrid Blend
Kaggle Competition: Store Sales - Time Series Forecasting

Five Orthogonal Modeling Pillars (Zero-Filter Preserved from Exp #036):
  1. Exp #036: Gen-1 Recursive GBDT Multi-Layer Ensemble (LB 0.38788, 2,001 zero rows)
  2. Exp #043: PyTorch Direct Multi-Horizon ResNet MLP 5-Seeds (Val 0.41263)
  3. Exp #050: Colab L4 GPU Seq2Horizon Dilated-TCN + BiGRU + CrossAttn 5-Seeds (Val 0.38926)
  4. Exp #052: Colab L4 GPU RevIN + DLinear + Patch-Transformer 5-Seeds (Val 0.38636)
     -> Quad Exp #053 = LB 0.37784 (Rank 25/632)
  5. Exp #054: Colab L4 GPU Gen-2 Direct Multi-Horizon GBDT (XGB CUDA + CatBoost GPU 3-Seeds)
     * Eliminates 16-day recursive error accumulation with 72 rich features (Val Raw 0.39284)
     * Correlation with Exp #053 Quad = 0.999391 (Shift RMSLE = 0.09184)

Exp #055 Default (delta = 0.20 onto Exp #053):
  - Effective Log-Space Weights:
    21.60% Gen-1 GBDT + 23.40% ResNet + 15.00% TCN-BiGRU + 20.00% Patch-Trans + 20.00% Gen-2 Direct GBDT
"""

import argparse
import numpy as np
import pandas as pd


def run_quintet_blend(delta: float = 0.20, exp_num: int = 55) -> str:
    """Blend Exp #053 Quad PB with Exp #054 Gen-2 Direct GBDT while preserving Exp #036 zeros."""
    w_53 = 1.0 - delta
    w_36 = 0.27 * w_53
    w_43 = 0.2925 * w_53
    w_50 = 0.1875 * w_53
    w_52 = 0.25 * w_53
    w_54 = delta

    print("=" * 80)
    print(
        f"  003 Store Sales: Exp #{exp_num:03d} - Quintet Blend "
        f"({w_36*100:.2f}% GBDT1 + {w_43*100:.2f}% ResNet + {w_50*100:.2f}% BiGRU + "
        f"{w_52*100:.2f}% Patch + {w_54*100:.2f}% GBDT2)"
    )
    print("=" * 80)

    sub036 = pd.read_csv("003_store-sales/submission_exp036_golden_spice.csv")
    sub043 = pd.read_csv("003_store-sales/submission_exp043_colab_l4_dnn.csv")
    sub050 = pd.read_csv("003_store-sales/submission_exp050_colab_l4_bigru_tcn.csv")
    sub052 = pd.read_csv("003_store-sales/submission_exp052_colab_l4_patch_dlinear.csv")
    sub053 = pd.read_csv("003_store-sales/submission_exp053_quad_patch25.csv")
    sub054 = pd.read_csv("003_store-sales/submission_exp054_colab_l4_gbdt_gen2.csv")
    test = pd.read_csv("003_store-sales/data/test.csv")

    for name, df in [("Exp043", sub043), ("Exp050", sub050), ("Exp052", sub052), ("Exp053", sub053), ("Exp054", sub054), ("Test", test)]:
        assert (sub036["id"].values == df["id"].values).all(), f"ID mismatch on {name}!"

    log036 = np.log1p(sub036["sales"].values)
    log043 = np.log1p(sub043["sales"].values)
    log050 = np.log1p(sub050["sales"].values)
    log052 = np.log1p(sub052["sales"].values)
    log053 = np.log1p(sub053["sales"].values)
    log054 = np.log1p(sub054["sales"].values)

    log_blend = w_36 * log036 + w_43 * log043 + w_50 * log050 + w_52 * log052 + w_54 * log054

    zero_mask = sub036["sales"].values == 0.0
    log_blend = np.where(zero_mask, 0.0, log_blend)
    sales_final = np.clip(np.expm1(log_blend), 0.0, None)

    pct = int(round(delta * 100))
    out_path = f"003_store-sales/submission_exp{exp_num:03d}_quintet_gbdt{pct}.csv"
    sub = pd.DataFrame({"id": test["id"], "sales": sales_final})
    sub.to_csv(out_path, index=False)
    print(f"\n[SAVE] Output saved to: {out_path} ({len(sub)} rows)")

    assert len(sub) == 28512
    assert sub["sales"].isna().sum() == 0
    assert (sub["sales"] < 0).sum() == 0
    assert ((sub["sales"].values == 0.0) == zero_mask).all(), "Zero mask mismatch!"

    shift_053 = np.sqrt(np.mean((log_blend - log053) ** 2))
    shift_036 = np.sqrt(np.mean((log_blend - log036) ** 2))
    corr_053 = np.corrcoef(log_blend, log053)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub['sales'].mean():.4f} (vs Exp053 {sub053['sales'].mean():.4f} / Exp036 {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub['sales'].median():.4f} (vs Exp053 {sub053['sales'].median():.4f} / Exp036 {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub['sales'].max():.4f} (vs Exp053 {sub053['sales'].max():.4f} / Exp036 {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub['sales'] == 0).sum()} (vs Exp036 {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #053 (0.37784): {shift_053:.6f} (Corr: {corr_053:.7f})")
    print(f"  Shift RMSLE from Exp #036 (0.38788): {shift_036:.6f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Exp #055+: 5-System Quintet hybrid blend.")
    parser.add_argument("--delta", type=float, default=0.20, help="Weight on Exp #054 Gen-2 GBDT (default: 0.20)")
    parser.add_argument("--exp", type=int, default=55, help="Experiment number (default: 55)")
    args = parser.parse_args()
    run_quintet_blend(args.delta, args.exp)


if __name__ == "__main__":
    main()
