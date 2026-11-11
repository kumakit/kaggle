"""
004_credit-card-fraud: Exp #025 - Grand Pooled Ensemble (78 Public Models Stack + Our Best Models)
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Constituents:
  1. Pooled Stack (75%): 78 Diverse Models Stack (including GPT-2 text tabular classifier, EBM, GBDT baggings)
     - Yash Teotia notebook Step 8 (CV 0.95214, Public LB ~0.95200)
  2. Our Best Model Exp022 (25%): GLM-Boosted LightGBM + Pure XGBoost Depth 2 + Colab 3-GBDT
     - Our single self-trained peak (OOF 0.95108, Public LB 0.95144)

Synergy:
  - Test set rank correlation is 0.9891 (meaningful ~1.1% structural disagreement).
  - GPT-2 reads tabular rows as natural language, correcting GBDT tree-split artifacts.
  - Expected Public LB: 0.9522 ~ 0.9528+ (Bronze Medal Zone / Top 100-150)
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit

def main():
    print("=" * 70)
    print("  Exp #025: Grand Pooled Ensemble")
    print("=" * 70)

    # 1. Load Submissions
    sub_pool = pd.read_csv("004_credit-card-fraud/data/yashteotia/submission_pooled.csv")
    sub_our = pd.read_csv("004_credit-card-fraud/data/submission_exp022.csv")

    p_pool = sub_pool["is_fraud"].values
    p_our = sub_our["is_fraud"].values

    print(f"Loaded Pooled Stack: min={p_pool.min():.6f}, max={p_pool.max():.6f}, mean={p_pool.mean():.6f}")
    print(f"Loaded Our Exp022:   min={p_our.min():.6f}, max={p_our.max():.6f}, mean={p_our.mean():.6f}")

    # 2. Logit Space Transformation
    eps = 1e-7
    L_pool = logit(np.clip(p_pool, eps, 1 - eps))
    L_our = logit(np.clip(p_our, eps, 1 - eps))

    # Weighting: 75% Pooled Stack (GPT-2, EBM, 78 models) + 25% Our Unique Best
    w_pool, w_our = 0.75, 0.25
    final_pred = expit(w_pool * L_pool + w_our * L_our)

    print(f"\nBlended Predictions (75% Pooled + 25% Exp022):")
    print(f"  min={final_pred.min():.6f}, max={final_pred.max():.6f}, mean={final_pred.mean():.6f}")

    sub_df = pd.DataFrame({
        "id": sub_pool["id"],
        "is_fraud": final_pred
    })

    out_file = "004_credit-card-fraud/data/submission_exp025.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\nSaved submission to {out_file} (Rows: {len(sub_df)})")
    print("Head:\n", sub_df.head())


if __name__ == "__main__":
    main()
