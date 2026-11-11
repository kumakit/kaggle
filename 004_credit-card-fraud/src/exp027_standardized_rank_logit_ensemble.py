"""
004_credit-card-fraud: Exp #027 - Standardized Rank-Logit Hybrid Ensemble
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Theoretical Breakthrough:
  1. Rank Normalization:
     Every model's raw probability is normalized to a uniform percentile rank:
     r = (rank(p) - 0.5) / N  in (0, 1)
     -> Eliminates calibration bias and probability scale mismatch between models.
  2. Inverse Sigmoid Logit Mapping:
     z = logit(clip(r, 1e-6, 1 - 1e-6))
     -> Restores high resolution to extreme values (tail events) while maintaining uniform distribution properties.
  3. Standardization:
     Z = (z - mean(z)) / std(z)
     -> Guarantees exact equal scale weighting across all participating model streams.
  4. Final Optimal Blend:
     Z_final = 0.65 * Z_pool + 0.15 * Z_22 + 0.12 * Z_23 + 0.08 * Z_19
  5. Target: Break 0.9522+ towards 0.9525-0.9530 (Medal Zone / Top 10%)
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit

def to_standardized_rank_logit(series, eps=1e-6):
    n = len(series)
    r = (series.rank().values - 0.5) / n
    z = logit(np.clip(r, eps, 1.0 - eps))
    return (z - z.mean()) / z.std()

def main():
    print("=" * 70)
    print("  Exp #027: Standardized Rank-Logit Hybrid Ensemble")
    print("=" * 70)

    # 1. Load Submissions
    sub_pool = pd.read_csv("004_credit-card-fraud/data/yashteotia/submission_pooled.csv")
    sub_22 = pd.read_csv("004_credit-card-fraud/data/submission_exp022.csv")
    sub_23 = pd.read_csv("004_credit-card-fraud/data/submission_exp023_colab.csv")
    sub_19 = pd.read_csv("004_credit-card-fraud/data/submission_exp019_colab.csv")

    p_pool = sub_pool["is_fraud"]
    p_22 = sub_22["is_fraud"]
    p_23 = sub_23["is_fraud"]
    p_19 = sub_19["is_fraud"]

    print("Computing Standardized Rank-Logit representations...")
    Z_pool = to_standardized_rank_logit(p_pool)
    Z_22 = to_standardized_rank_logit(p_22)
    Z_23 = to_standardized_rank_logit(p_23)
    Z_19 = to_standardized_rank_logit(p_19)

    # 2. Weighted Sum
    w_pool = 0.65
    w_22 = 0.15
    w_23 = 0.12
    w_19 = 0.08

    Z_final = w_pool * Z_pool + w_22 * Z_22 + w_23 * Z_23 + w_19 * Z_19

    # 3. Final Probability Mapping (expit)
    final_pred = expit(Z_final)

    print(f"\nFinal Predictions:")
    print(f"  min={final_pred.min():.6f}, max={final_pred.max():.6f}, mean={final_pred.mean():.6f}, std={final_pred.std():.6f}")

    sub_df = pd.DataFrame({
        "id": sub_pool["id"],
        "is_fraud": final_pred
    })

    out_file = "004_credit-card-fraud/data/submission_exp027.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\nSaved submission to {out_file} (Rows: {len(sub_df)})")
    print("Head:\n", sub_df.head())


if __name__ == "__main__":
    main()
