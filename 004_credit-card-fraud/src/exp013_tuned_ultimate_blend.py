import os
import pandas as pd
import numpy as np

def main():
    sub4_path = "004_credit-card-fraud/data/submission_exp004.csv"
    sub10_path = "004_credit-card-fraud/data/submission_exp010.csv"
    sub12_path = "004_credit-card-fraud/data/submission_exp012_colab.csv"
    output_path = "004_credit-card-fraud/data/submission_exp013.csv"
    
    print("[1/3] Loading submission files...")
    sub4 = pd.read_csv(sub4_path)
    sub10 = pd.read_csv(sub10_path)
    sub12 = pd.read_csv(sub12_path)
    
    p4 = sub4['is_fraud'].values
    p10 = sub10['is_fraud'].values
    p12 = sub12['is_fraud'].values
    
    print(f"\n[2/3] Model Correlation Matrix:")
    corr_matrix = pd.DataFrame(
        np.corrcoef([p4, p10, p12]),
        index=['Exp004 (Pure Tree 0.94808)', 'Exp010 (Orig+GLM 0.94796)', 'Exp012 (Tuned 3-GBDT 0.94780)'],
        columns=['Exp004', 'Exp010', 'Exp012']
    )
    print(corr_matrix)
    
    # Tuned Ultimate Blend Weights
    w4 = 0.45   # Pure Tree LB Leader (0.94808)
    w10 = 0.40  # Original Data + GLM Boosting (0.94796)
    w12 = 0.15  # Tuned 2-Seed 3-GBDT on Colab L4 GPU (0.94780)
    
    p_blend = w4 * p4 + w10 * p10 + w12 * p12
    
    print(f"\nPrediction Distribution Check:")
    print(f"  Exp004 (45%): Mean: {p4.mean():.5f}, Std: {p4.std():.5f}")
    print(f"  Exp010 (40%): Mean: {p10.mean():.5f}, Std: {p10.std():.5f}")
    print(f"  Exp012 (15%): Mean: {p12.mean():.5f}, Std: {p12.std():.5f}")
    print(f"  Exp013 Blend: Mean: {p_blend.mean():.5f}, Std: {p_blend.std():.5f}, Min: {p_blend.min():.5f}, Max: {p_blend.max():.5f}")
    
    sub13 = pd.DataFrame({
        'id': sub4['id'],
        'is_fraud': p_blend
    })
    sub13.to_csv(output_path, index=False)
    print(f"\n[3/3] Exp #013 Tuned Ultimate Blend saved to {output_path} (Rows: {len(sub13)})")
    print(sub13.head())

if __name__ == "__main__":
    main()
