import pandas as pd
import numpy as np
from scipy.stats import rankdata

def main():
    sub2_path = "004_credit-card-fraud/data/submission_exp002.csv"
    sub3_path = "004_credit-card-fraud/data/submission_exp003.csv"
    output_path = "004_credit-card-fraud/data/submission_exp004.csv"
    
    print("[1/3] Loading submission files...")
    sub2 = pd.read_csv(sub2_path)
    sub3 = pd.read_csv(sub3_path)
    
    p2 = sub2['is_fraud'].values
    p3 = sub3['is_fraud'].values
    
    # 1. Probability Blend (Weighted Average)
    # Exp003 has higher LB (0.94798 vs 0.94768), so give slightly more weight to Exp003
    w3 = 0.60
    w2 = 0.40
    prob_blend = w2 * p2 + w3 * p3
    
    # 2. Rank Blend (Percentile Rank)
    r2 = rankdata(p2) / len(p2)
    r3 = rankdata(p3) / len(p3)
    rank_blend = w2 * r2 + w3 * r3
    
    print(f"\n[2/3] Blend Statistics:")
    print(f"  Exp002 Mean: {p2.mean():.5f}, Std: {p2.std():.5f}")
    print(f"  Exp003 Mean: {p3.mean():.5f}, Std: {p3.std():.5f}")
    print(f"  Weighted Probability Blend (40% Exp002 + 60% Exp003): Mean: {prob_blend.mean():.5f}, Std: {prob_blend.std():.5f}")
    print(f"  Rank Blend: Mean: {rank_blend.mean():.5f}, Std: {rank_blend.std():.5f}")
    
    # We use Probability Blend because it perfectly preserves the calibrated probability scale (mean ~ 0.0166)
    sub4 = pd.DataFrame({
        'id': sub2['id'],
        'is_fraud': prob_blend
    })
    
    sub4.to_csv(output_path, index=False)
    print(f"\n[3/3] Exp #004 submission saved to {output_path} (Rows: {len(sub4)})")
    print(sub4.head())

if __name__ == "__main__":
    main()
