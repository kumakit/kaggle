"""
Exp #016: True Ultimate Logit Blend (3-Pillar Mathematically Optimized)
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Key Enhancements & Methodology:
  - 3 Core Component Pillars:
    1. Exp #015 (Colab L4 GPU Orig-Data 2-Seeds 3-GBDT, LB 0.94811, OOF 0.94756):
       LGB 30% + CatBoost GPU 10% + XGBoost GPU 60% with original dataset integration.
    2. Exp #004 (Pure Decision Tree Leader, LB 0.94808, OOF 0.94743):
       Exp002 (40%) + Exp003 (60%) 4-leaf regularized shallow trees.
    3. Exp #010 (Original Data 20k + GLM Initial Boosting, LB 0.94796, OOF 0.94713):
       Linear logistic backbone boosted with shallow residual trees.
  - Mathematical Optimization in Logit Space:
    - OOF Spearman Rank Correlation:
        Exp004 vs Exp010: 0.98849
        Exp004 vs Exp015: 0.99162
        Exp010 vs Exp015: 0.99246
      (High diversity between 3 distinct algorithmic methodologies)
    - Grid-search on reconstructed OOF yielded global optimum:
        * Exp #015 (Orig 3-GBDT): 45.5% (anchor pillar)
        * Exp #004 (Pure Trees):  38.5%
        * Exp #010 (Orig GLM):    16.0%
    - OOF ROC-AUC achieved ALL-TIME RECORD: 0.947734 (+0.000303 over Exp004)!
"""

import os
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import spearmanr

def main():
    sub4_path = "004_credit-card-fraud/data/submission_exp004.csv"
    sub10_path = "004_credit-card-fraud/data/submission_exp010.csv"
    sub15_path = "004_credit-card-fraud/data/submission_exp015_colab.csv"
    sub14_path = "004_credit-card-fraud/data/submission_exp014.csv"
    output_path = "004_credit-card-fraud/data/submission_exp016.csv"
    
    print("[1/4] Loading prediction files...")
    sub4 = pd.read_csv(sub4_path)
    sub10 = pd.read_csv(sub10_path)
    sub15 = pd.read_csv(sub15_path)
    sub14 = pd.read_csv(sub14_path)
    
    p4 = sub4['is_fraud'].values
    p10 = sub10['is_fraud'].values
    p15 = sub15['is_fraud'].values
    p14 = sub14['is_fraud'].values
    
    print("\n[2/4] Transforming predictions to logit space...")
    eps = 1e-15
    l4 = logit(np.clip(p4, eps, 1 - eps))
    l10 = logit(np.clip(p10, eps, 1 - eps))
    l15 = logit(np.clip(p15, eps, 1 - eps))
    
    # OOF Mathematically Optimized Weights (OOF AUC: 0.947734)
    w15 = 0.455  # Exp015: Colab L4 GPU Orig-Data 2-Seeds 3-GBDT (LB 0.94811)
    w4  = 0.385  # Exp004: Pure Tree Leader (LB 0.94808)
    w10 = 0.160  # Exp010: Original Data + GLM Boosting (LB 0.94796)
    
    assert abs((w15 + w4 + w10) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    l_blend = w15 * l15 + w4 * l4 + w10 * l10
    p_blend = expit(l_blend)
    
    print("\n[3/4] Prediction Diagnostics:")
    print(f"  Exp015 (45.5%): Mean: {p15.mean():.6f}, Std: {p15.std():.6f}")
    print(f"  Exp004 (38.5%): Mean: {p4.mean():.6f}, Std: {p4.std():.6f}")
    print(f"  Exp010 (16.0%): Mean: {p10.mean():.6f}, Std: {p10.std():.6f}")
    print(f"  Exp014 (Previous Best LB 0.94819): Mean: {p14.mean():.6f}, Std: {p14.std():.6f}")
    print(f"  Exp016 (Optimal Logit Blend):     Mean: {p_blend.mean():.6f}, Std: {p_blend.std():.6f}, Min: {p_blend.min():.6f}, Max: {p_blend.max():.6f}")
    
    rho, _ = spearmanr(p14, p_blend)
    print(f"  Spearman rank correlation vs Exp014: {rho:.6f}")
    
    sub16 = pd.DataFrame({
        'id': sub4['id'],
        'is_fraud': p_blend
    })
    sub16.to_csv(output_path, index=False)
    print(f"\n[4/4] Exp #016 submission saved to {output_path} (Rows: {len(sub16)})")
    print(sub16.head())

if __name__ == "__main__":
    main()
