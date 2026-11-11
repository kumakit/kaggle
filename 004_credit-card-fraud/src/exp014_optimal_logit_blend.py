"""
Exp #014: Mathematically Optimized 4-Model Logit Blend
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Features & Methodology:
  - 4 Diverse Component Models:
    1. Exp #002: Shallow Trees (4 leaves, raw features, LB 0.94768)
    2. Exp #003: Engineered Risk Features (36 features, LB 0.94798)
    3. Exp #010: Original Data 20k + GLM Boosting (LB 0.94796)
    4. Exp #012: Colab L4 GPU 2-Seed 3-GBDT (LGB+CAT+XGB, LB 0.94780)
  - Mathematical Optimization in Logit Space:
    - Reconstructed OOF scores proved Logit-space averaging beats probability averaging (+0.000238 AUC).
    - Grid search on OOF yielded optimal ratio:
        * Exp002: 34% (prevents tree overfitting, high stability)
        * Exp003: 17% (captures interaction spikes)
        * Exp010: 34% (original dataset linear backbone)
        * Exp012: 15% (algorithm diversity from CatBoost GPU & XGBoost GPU)
"""

import os
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import spearmanr

def main():
    sub2_path = "004_credit-card-fraud/data/submission_exp002.csv"
    sub3_path = "004_credit-card-fraud/data/submission_exp003.csv"
    sub10_path = "004_credit-card-fraud/data/submission_exp010.csv"
    sub12_path = "004_credit-card-fraud/data/submission_exp012_colab.csv"
    sub13_path = "004_credit-card-fraud/data/submission_exp013.csv"
    output_path = "004_credit-card-fraud/data/submission_exp014.csv"
    
    print("[1/4] Loading prediction files...")
    sub2 = pd.read_csv(sub2_path)
    sub3 = pd.read_csv(sub3_path)
    sub10 = pd.read_csv(sub10_path)
    sub12 = pd.read_csv(sub12_path)
    sub13 = pd.read_csv(sub13_path)
    
    p2 = sub2['is_fraud'].values
    p3 = sub3['is_fraud'].values
    p10 = sub10['is_fraud'].values
    p12 = sub12['is_fraud'].values
    p13 = sub13['is_fraud'].values
    
    print("\n[2/4] Transforming predictions to logit space...")
    eps = 1e-15
    l2 = logit(np.clip(p2, eps, 1 - eps))
    l3 = logit(np.clip(p3, eps, 1 - eps))
    l10 = logit(np.clip(p10, eps, 1 - eps))
    l12 = logit(np.clip(p12, eps, 1 - eps))
    
    # OOF Mathematically Optimized Weights
    w2 = 0.34   # Exp002: Shallow Raw Trees
    w3 = 0.17   # Exp003: Engineered Risk Features
    w10 = 0.34  # Exp010: Original Data + GLM Boosting
    w12 = 0.15  # Exp012: Tuned 2-Seed 3-GBDT (Colab L4 GPU)
    
    assert abs((w2 + w3 + w10 + w12) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    l_blend = w2 * l2 + w3 * l3 + w10 * l10 + w12 * l12
    p_blend = expit(l_blend)
    
    print("\n[3/4] Prediction Diagnostics:")
    print(f"  Exp002 (34%): Mean: {p2.mean():.6f}, Std: {p2.std():.6f}")
    print(f"  Exp003 (17%): Mean: {p3.mean():.6f}, Std: {p3.std():.6f}")
    print(f"  Exp010 (34%): Mean: {p10.mean():.6f}, Std: {p10.std():.6f}")
    print(f"  Exp012 (15%): Mean: {p12.mean():.6f}, Std: {p12.std():.6f}")
    print(f"  Exp013 (Prev Best LB 0.94819): Mean: {p13.mean():.6f}, Std: {p13.std():.6f}")
    print(f"  Exp014 Optimized Logit Blend: Mean: {p_blend.mean():.6f}, Std: {p_blend.std():.6f}")
    
    rho13, _ = spearmanr(p13, p_blend)
    print(f"  Spearman rank correlation vs Exp013: {rho13:.6f}")
    
    sub14 = pd.DataFrame({
        'id': sub2['id'],
        'is_fraud': p_blend
    })
    sub14.to_csv(output_path, index=False)
    print(f"\n[4/4] Exp #014 submission saved to {output_path} (Rows: {len(sub14)})")
    print(sub14.head())

if __name__ == "__main__":
    main()
