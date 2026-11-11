"""
004_credit-card-fraud: Exp #026 - Fine-Tuned Grand Pooled Ensemble
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Constituents & Theoretical Weights:
  1. Pooled Stack 78 Models (65%): GPT-2, EBM, CatBoost, LightGBM, Bagged GBDTs (CV 0.95214, LB 0.95200)
  2. Our Exp023 GLM-XGBoost Depth 2 (15%): Our highest single OOF (0.95116), 2 Seeds on L4 GPU
  3. Our Exp022 Grand Blend (12%): Our best LB baseline (0.95144), balancing GLM-LGB & XGB
  4. Our Exp019 Pure XGBoost Depth 2 (8%): The ultimate diversity anchor (Correlation 0.9792 with Pooled Stack)

Mathematical Synergy:
  - Exp025 (75% Pooled + 25% Exp022) achieved LB 0.95212, beating pure Pooled Stack (0.95200).
  - Injecting Exp023 (single best OOF 0.95116) and Exp019 (pure tree without GLM, corr 0.9792) provides mathematically verified diversity (OOF gain +0.00015).
  - Target: LB 0.9523 ~ 0.9528+ (Pushing deep towards Top 10% / Medal Zone)
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit

def main():
    print("=" * 70)
    print("  Exp #026: Fine-Tuned Grand Pooled Ensemble")
    print("=" * 70)

    # 1. Load Submissions
    sub_pool = pd.read_csv("004_credit-card-fraud/data/yashteotia/submission_pooled.csv")
    sub_23 = pd.read_csv("004_credit-card-fraud/data/submission_exp023_colab.csv")
    sub_22 = pd.read_csv("004_credit-card-fraud/data/submission_exp022.csv")
    sub_19 = pd.read_csv("004_credit-card-fraud/data/submission_exp019_colab.csv")

    p_pool = sub_pool["is_fraud"].values
    p_23 = sub_23["is_fraud"].values
    p_22 = sub_22["is_fraud"].values
    p_19 = sub_19["is_fraud"].values

    eps = 1e-7
    L_pool = logit(np.clip(p_pool, eps, 1 - eps))
    L_23 = logit(np.clip(p_23, eps, 1 - eps))
    L_22 = logit(np.clip(p_22, eps, 1 - eps))
    L_19 = logit(np.clip(p_19, eps, 1 - eps))

    # Optimal Multilateral Weights
    w_pool = 0.65
    w_23 = 0.15
    w_22 = 0.12
    w_19 = 0.08

    final_pred = expit(w_pool * L_pool + w_23 * L_23 + w_22 * L_22 + w_19 * L_19)

    print(f"Blended Predictions (65% Pooled + 15% Exp023 + 12% Exp022 + 8% Exp019):")
    print(f"  min={final_pred.min():.6f}, max={final_pred.max():.6f}, mean={final_pred.mean():.6f}, std={final_pred.std():.6f}")

    sub_df = pd.DataFrame({
        "id": sub_pool["id"],
        "is_fraud": final_pred
    })

    out_file = "004_credit-card-fraud/data/submission_exp026.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\nSaved submission to {out_file} (Rows: {len(sub_df)})")
    print("Head:\n", sub_df.head())


if __name__ == "__main__":
    main()
