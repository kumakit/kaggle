import os
import pandas as pd
import numpy as np

def main():
    sub4_path = "004_credit-card-fraud/data/submission_exp004.csv"
    sub10_path = "004_credit-card-fraud/data/submission_exp010.csv"
    sub6_path = "004_credit-card-fraud/data/submission_exp006_colab.csv"
    output_path = "004_credit-card-fraud/data/submission_exp011.csv"
    
    print("[1/3] Loading submission files...")
    sub4 = pd.read_csv(sub4_path)
    sub10 = pd.read_csv(sub10_path)
    sub6 = pd.read_csv(sub6_path)
    
    p4 = sub4['is_fraud'].values
    p10 = sub10['is_fraud'].values
    p6 = sub6['is_fraud'].values
    
    print(f"\n[2/3] Model Correlation Matrix:")
    corr_matrix = pd.DataFrame(
        np.corrcoef([p4, p10, p6]),
        index=['Exp004 (Pure Tree 0.94808)', 'Exp010 (Orig+GLM 0.94796)', 'Exp006 (Colab 3-GBDT 0.94769)'],
        columns=['Exp004', 'Exp010', 'Exp006']
    )
    print(corr_matrix)
    
    # Ultimate Blend Weights
    w4 = 0.45   # Pure Tree LB Leader (0.94808)
    w10 = 0.40  # Original Data + GLM Boosting (0.94796)
    w6 = 0.15   # Algorithmic Diversity (CatBoost GPU + XGBoost GPU + LightGBM)
    
    p_blend = w4 * p4 + w10 * p10 + w6 * p6
    
    print(f"\nPrediction Distribution Check:")
    print(f"  Exp004 (45%): Mean: {p4.mean():.5f}, Std: {p4.std():.5f}")
    print(f"  Exp010 (40%): Mean: {p10.mean():.5f}, Std: {p10.std():.5f}")
    print(f"  Exp006 (15%): Mean: {p6.mean():.5f}, Std: {p6.std():.5f}")
    print(f"  Exp011 Ultimate Blend: Mean: {p_blend.mean():.5f}, Std: {p_blend.std():.5f}, Min: {p_blend.min():.5f}, Max: {p_blend.max():.5f}")
    
    sub11 = pd.DataFrame({
        'id': sub4['id'],
        'is_fraud': p_blend
    })
    sub11.to_csv(output_path, index=False)
    print(f"\n[3/3] Exp #011 Ultimate Blend saved to {output_path} (Rows: {len(sub11)})")
    print(sub11.head())

if __name__ == "__main__":
    main()
