"""
Exp #031: Apex Sweetspot Fine-Tuned Blend (Standardized Rank-Logit Hybrid)
- コンペ名: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
- 目的: 本日のラスト提出（残り 1 枠）！
        Exp029 (Apex 95% -> LB 0.95336 / Rank 14) と Exp030 (Apex 94% -> LB 0.95335)
        の知見から導き出されたスイートスポット比率:
        - 96.0% Exp028_Apex (0.95333)
        -  2.5% Exp019_PureXGB (0.95098 / 相関 0.97196 の超直交アンカー)
        -  1.5% Exp022_BestBlend (0.95144 / 自力最高ブレンド)
        Apex の極めて高い基本順序関係を 96% 温存しつつ、外挿ノイズのみを外科手術的に
        シェイブして Top 10 (0.95338+ / 10位以上) への突入を狙う。
"""

import os
import pandas as pd
import numpy as np
from scipy.special import logit, expit

DATA_DIR = "004_credit-card-fraud/data"

def to_std_logit(p: np.ndarray) -> np.ndarray:
    n = len(p)
    r = pd.Series(p).rank(method='average').values
    u = (r - 0.5) / n
    u = np.clip(u, 1e-7, 1 - 1e-7)
    z = logit(u)
    return (z - z.mean()) / z.std()

def main():
    print("=== Exp #031: Apex Sweetspot Fine-Tuned Blend ===")
    
    # 1. 予測ファイルの読み込み
    path_apex = os.path.join(DATA_DIR, "submission_exp028_apex.csv")
    path_exp019 = os.path.join(DATA_DIR, "submission_exp019_colab.csv")
    path_exp022 = os.path.join(DATA_DIR, "submission_exp022.csv")
    path_exp029 = os.path.join(DATA_DIR, "submission_exp029.csv")
    
    df_apex = pd.read_csv(path_apex)
    df_exp019 = pd.read_csv(path_exp019)
    df_exp022 = pd.read_csv(path_exp022)
    df_exp029 = pd.read_csv(path_exp029)
    
    assert len(df_apex) == 299984, f"Unexpected len: {len(df_apex)}"
    assert (df_apex["id"].values == df_exp019["id"].values).all(), "ID mismatch exp019"
    assert (df_apex["id"].values == df_exp022["id"].values).all(), "ID mismatch exp022"
    
    p_apex = df_apex["is_fraud"].values
    p_exp019 = df_exp019["is_fraud"].values
    p_exp022 = df_exp022["is_fraud"].values
    p_exp029 = df_exp029["is_fraud"].values
    
    print(f"Loaded Apex (N={len(p_apex)}), Exp019, Exp022 successfully.")
    
    # 2. 標準化ロジット変換
    z_apex = to_std_logit(p_apex)
    z_exp019 = to_std_logit(p_exp019)
    z_exp022 = to_std_logit(p_exp022)
    
    # 3. 加重結合 (96.0% Apex + 2.5% Exp019 + 1.5% Exp022)
    w_apex = 0.960
    w_exp019 = 0.025
    w_exp022 = 0.015
    
    z_blend = w_apex * z_apex + w_exp019 * z_exp019 + w_exp022 * z_exp022
    
    # 4. 逆変換
    p_blend = expit(z_blend)
    
    # 相関分析
    spearman_apex = pd.Series(p_blend).corr(pd.Series(p_apex), method='spearman')
    spearman_exp029 = pd.Series(p_blend).corr(pd.Series(p_exp029), method='spearman')
    spearman_exp019 = pd.Series(p_blend).corr(pd.Series(p_exp019), method='spearman')
    spearman_exp022 = pd.Series(p_blend).corr(pd.Series(p_exp022), method='spearman')
    
    print("\n--- Blend Correlation Analysis ---")
    print(f"Spearman vs Apex (Exp028):      {spearman_apex:.6f}")
    print(f"Spearman vs Exp029 (Rank 14):   {spearman_exp029:.6f}")
    print(f"Spearman vs PureXGB (Exp019):   {spearman_exp019:.6f}")
    print(f"Spearman vs BestBlend (Exp022): {spearman_exp022:.6f}")
    
    # 5. 提出ファイルの作成
    sub = pd.DataFrame({
        "id": df_apex["id"],
        "is_fraud": p_blend
    })
    
    output_path = os.path.join(DATA_DIR, "submission_exp031.csv")
    sub.to_csv(output_path, index=False)
    
    print(f"\nSaved submission to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(sub.head(5))
    print("\nSummary stats:")
    print(sub["is_fraud"].describe())
    
    # NaN/Nullチェック
    assert sub["is_fraud"].isnull().sum() == 0, "Contains NaN!"
    assert len(sub) == 299984, "Wrong row count!"
    print("\nAll sanity checks passed successfully!")

if __name__ == "__main__":
    main()
