import os
import pandas as pd
import numpy as np

def main():
    sub4_path = "004_credit-card-fraud/data/submission_exp004.csv"
    sub6_path = "004_credit-card-fraud/data/submission_exp006_colab.csv"
    output_path = "004_credit-card-fraud/data/submission_exp007.csv"
    
    print("[1/3] Loading submission files...")
    sub4 = pd.read_csv(sub4_path)
    sub6 = pd.read_csv(sub6_path)
    
    p4 = sub4['is_fraud'].values
    p6 = sub6['is_fraud'].values
    
    # Check correlation
    corr = np.corrcoef(p4, p6)[0, 1]
    print(f"Correlation between Exp004 (LB 0.94808) and Exp006 (Colab 3-GBDT 0.94769): {corr:.5f}")
    
    # Golden Blend: 80% Exp004 + 20% Exp006
    w4 = 0.80
    w6 = 0.20
    p_blend = w4 * p4 + w6 * p6
    
    print(f"\n[2/3] Prediction Distribution:")
    print(f"  Exp004 Mean: {p4.mean():.5f}, Std: {p4.std():.5f}, Min: {p4.min():.5f}, Max: {p4.max():.5f}")
    print(f"  Exp006 Mean: {p6.mean():.5f}, Std: {p6.std():.5f}, Min: {p6.min():.5f}, Max: {p6.max():.5f}")
    print(f"  Exp007 Blend Mean: {p_blend.mean():.5f}, Std: {p_blend.std():.5f}, Min: {p_blend.min():.5f}, Max: {p_blend.max():.5f}")
    
    sub7 = pd.DataFrame({
        'id': sub4['id'],
        'is_fraud': p_blend
    })
    sub7.to_csv(output_path, index=False)
    print(f"\n[3/3] Exp #007 submission saved to {output_path} (Rows: {len(sub7)})")
    print(sub7.head())

if __name__ == "__main__":
    main()
