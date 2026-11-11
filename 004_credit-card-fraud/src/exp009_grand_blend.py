import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def main():
    sub4_path = "004_credit-card-fraud/data/submission_exp004.csv"
    sub8_path = "004_credit-card-fraud/data/submission_exp008.csv"
    sub6_path = "004_credit-card-fraud/data/submission_exp006_colab.csv"
    output_path = "004_credit-card-fraud/data/submission_exp009.csv"
    
    print("[1/3] Loading submission files...")
    sub4 = pd.read_csv(sub4_path)
    sub8 = pd.read_csv(sub8_path)
    sub6 = pd.read_csv(sub6_path)
    
    p4 = sub4['is_fraud'].values
    p8 = sub8['is_fraud'].values
    p6 = sub6['is_fraud'].values
    
    print(f"\n[2/3] Model Correlation Matrix:")
    corr_matrix = pd.DataFrame(
        np.corrcoef([p4, p8, p6]),
        index=['Exp004 (Pure Tree)', 'Exp008 (GLM Boosted)', 'Exp006 (3-GBDT)'],
        columns=['Exp004', 'Exp008', 'Exp006']
    )
    print(corr_matrix)
    
    # Grand Ensemble Weights
    w4 = 0.50  # LB Best (0.94808)
    w8 = 0.35  # CV Best (0.94766, GLM Boosted)
    w6 = 0.15  # Algorithmic Diversity (CatBoost + XGBoost + LGBM)
    
    prob_blend = w4 * p4 + w8 * p8 + w6 * p6
    
    print(f"\nPrediction Distribution Check:")
    print(f"  Exp004 (50%): Mean: {p4.mean():.5f}, Std: {p4.std():.5f}")
    print(f"  Exp008 (35%): Mean: {p8.mean():.5f}, Std: {p8.std():.5f}")
    print(f"  Exp006 (15%): Mean: {p6.mean():.5f}, Std: {p6.std():.5f}")
    print(f"  Grand Blend:  Mean: {prob_blend.mean():.5f}, Std: {prob_blend.std():.5f}, Min: {prob_blend.min():.5f}, Max: {prob_blend.max():.5f}")
    
    sub9 = pd.DataFrame({
        'id': sub4['id'],
        'is_fraud': prob_blend
    })
    sub9.to_csv(output_path, index=False)
    print(f"\n[3/3] Exp #009 Grand Ensemble saved to {output_path} (Rows: {len(sub9)})")
    print(sub9.head())

if __name__ == "__main__":
    main()
