"""
Exp #038: Golden 4% Foundation + CatBoost GPU Micro-Blend
- コンペ名: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
- 目的: 自己ベスト 0.95349 (公式14位/Top 0.94%) から、Top 10 / 全体8位以内 (0.95350+ / 一桁順位！) への到達を目指す。
        Exp033/034 で実証された「4.0% 黄金基盤（PureXGB Exp019 2.5% + 自力ブレンド Exp022 1.5%）」を完全固定。
        そこに最新ラダーとの相関が 0.963999（約3.6%の巨大直交多様性）を持つ「CatBoost GPU 対称木 (Exp036)」を 0.7% 微量配合。
        - 95.3% Ladder v17 (Exp032 / 0.95347)
        -  2.5% Exp019_PureXGB (0.95098 / 超直交純粋木アンカー)
        -  1.5% Exp022_BestBlend (0.95144 / 自力最高3-GBDT+GLMブレンド)
        -  0.7% Exp036_CatBoost (OOF 0.94942 / 対称木Oblivious Treesアンカー)
- 手法: Standardized Rank-Logit Hybrid
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
    print("=== Exp #038: Golden Foundation + CatBoost GPU Micro-Blend ===")
    
    # 1. 予測ファイルの読み込み
    path_ladder = os.path.join(DATA_DIR, "submission_exp032_ladder.csv")
    path_exp019 = os.path.join(DATA_DIR, "submission_exp019_colab.csv")
    path_exp022 = os.path.join(DATA_DIR, "submission_exp022.csv")
    path_exp036 = os.path.join(DATA_DIR, "submission_exp036_catboost.csv")
    path_exp033 = os.path.join(DATA_DIR, "submission_exp033.csv")
    path_exp034 = os.path.join(DATA_DIR, "submission_exp034.csv")
    
    df_ladder = pd.read_csv(path_ladder)
    df_exp019 = pd.read_csv(path_exp019)
    df_exp022 = pd.read_csv(path_exp022)
    df_exp036 = pd.read_csv(path_exp036)
    df_exp033 = pd.read_csv(path_exp033)
    df_exp034 = pd.read_csv(path_exp034)
    
    assert len(df_ladder) == 299984, f"Unexpected len: {len(df_ladder)}"
    assert (df_ladder["id"].values == df_exp019["id"].values).all(), "ID mismatch exp019"
    assert (df_ladder["id"].values == df_exp022["id"].values).all(), "ID mismatch exp022"
    assert (df_ladder["id"].values == df_exp036["id"].values).all(), "ID mismatch exp036"
    assert (df_ladder["id"].values == df_exp033["id"].values).all(), "ID mismatch exp033"
    
    p_ladder = df_ladder["is_fraud"].values
    p_exp019 = df_exp019["is_fraud"].values
    p_exp022 = df_exp022["is_fraud"].values
    p_exp036 = df_exp036["is_fraud"].values
    p_exp033 = df_exp033["is_fraud"].values
    p_exp034 = df_exp034["is_fraud"].values
    
    print(f"Loaded Ladder, Exp019, Exp022, Exp036 (CatBoost) successfully.")
    
    # 2. 標準化ロジット変換
    z_ladder = to_std_logit(p_ladder)
    z_exp019 = to_std_logit(p_exp019)
    z_exp022 = to_std_logit(p_exp022)
    z_exp036 = to_std_logit(p_exp036)
    
    # 3. 加重結合 (95.3% Ladder + 2.5% Exp019 + 1.5% Exp022 + 0.7% CatBoost)
    w_ladder = 0.953
    w_exp019 = 0.025
    w_exp022 = 0.015
    w_exp036 = 0.007
    
    assert np.isclose(w_ladder + w_exp019 + w_exp022 + w_exp036, 1.0), "Weights do not sum to 1.0"
    
    z_blend = (w_ladder * z_ladder + 
               w_exp019 * z_exp019 + 
               w_exp022 * z_exp022 + 
               w_exp036 * z_exp036)
    
    # 4. 逆変換
    p_blend = expit(z_blend)
    
    # 相関分析
    spearman_ladder = pd.Series(p_blend).corr(pd.Series(p_ladder), method='spearman')
    spearman_exp033 = pd.Series(p_blend).corr(pd.Series(p_exp033), method='spearman')
    spearman_exp034 = pd.Series(p_blend).corr(pd.Series(p_exp034), method='spearman')
    spearman_exp019 = pd.Series(p_blend).corr(pd.Series(p_exp019), method='spearman')
    spearman_exp036 = pd.Series(p_blend).corr(pd.Series(p_exp036), method='spearman')
    spearman_exp022 = pd.Series(p_blend).corr(pd.Series(p_exp022), method='spearman')
    
    print("\n--- Blend Correlation Analysis ---")
    print(f"Spearman vs Ladder v17 (Exp032):   {spearman_ladder:.6f}")
    print(f"Spearman vs Exp033 (0.95349):      {spearman_exp033:.6f}")
    print(f"Spearman vs Exp034 (0.95349):      {spearman_exp034:.6f}")
    print(f"Spearman vs PureXGB (Exp019):      {spearman_exp019:.6f}")
    print(f"Spearman vs CatBoost (Exp036):     {spearman_exp036:.6f}")
    print(f"Spearman vs BestBlend (Exp022):    {spearman_exp022:.6f}")
    
    # 5. 提出ファイルの作成
    sub = pd.DataFrame({
        "id": df_ladder["id"],
        "is_fraud": p_blend
    })
    
    output_path = os.path.join(DATA_DIR, "submission_exp038.csv")
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
