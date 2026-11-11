"""
Exp #034: Ladder v17 Quad-Anchor Breakthrough Blend (Standardized Rank-Logit Hybrid)
- コンペ名: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
- 目的: 現在 13 位 (0.95349 / Top 0.88%) から全体一桁順位 (Top 8 以内: 0.95350+ / 8位以内！) への突入を目指す。
        最新頂点ラダー v17 (95.0%) に対し、我々の独自3大直交系統:
        - 2.5% Exp019_PureXGB (0.95098 / 相関 0.97396 の超直交純粋木アンカー)
        - 1.5% Exp022_BestBlend (0.95144 / 自力最高3-GBDT+GLMブレンド)
        - 1.0% Exp023_GLMXGB (OOF 0.95116 / 手元単体最高GLM-XGBoostアンカー)
        を Standardized Rank-Logit 空間で完全融合。
- 手法:
    1. 各モデルの予測確率をパーセンタイルランク u = (rank(p) - 0.5) / N に正規化
    2. 端点クリップ [1e-7, 1 - 1e-7] 後に logit(u) に変換
    3. 各ロジットを平均0・分散1に標準化 (Z-score)
    4. 加重線形結合: z_blend = 0.950 * z_ladder + 0.025 * z_exp019 + 0.015 * z_exp022 + 0.010 * z_exp023
    5. シグモイド逆変換: p_blend = sigmoid(z_blend)
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
    print("=== Exp #034: Ladder v17 Quad-Anchor Breakthrough Blend ===")
    
    # 1. 予測ファイルの読み込み
    path_ladder = os.path.join(DATA_DIR, "submission_exp032_ladder.csv")
    path_exp019 = os.path.join(DATA_DIR, "submission_exp019_colab.csv")
    path_exp022 = os.path.join(DATA_DIR, "submission_exp022.csv")
    path_exp023 = os.path.join(DATA_DIR, "submission_exp023_colab.csv")
    path_exp033 = os.path.join(DATA_DIR, "submission_exp033.csv")
    
    df_ladder = pd.read_csv(path_ladder)
    df_exp019 = pd.read_csv(path_exp019)
    df_exp022 = pd.read_csv(path_exp022)
    df_exp023 = pd.read_csv(path_exp023)
    df_exp033 = pd.read_csv(path_exp033)
    
    assert len(df_ladder) == 299984, f"Unexpected len: {len(df_ladder)}"
    assert (df_ladder["id"].values == df_exp019["id"].values).all(), "ID mismatch exp019"
    assert (df_ladder["id"].values == df_exp022["id"].values).all(), "ID mismatch exp022"
    assert (df_ladder["id"].values == df_exp023["id"].values).all(), "ID mismatch exp023"
    
    p_ladder = df_ladder["is_fraud"].values
    p_exp019 = df_exp019["is_fraud"].values
    p_exp022 = df_exp022["is_fraud"].values
    p_exp023 = df_exp023["is_fraud"].values
    p_exp033 = df_exp033["is_fraud"].values
    
    print(f"Loaded Ladder v17 (N={len(p_ladder)}), Exp019, Exp022, Exp023 successfully.")
    
    # 2. 標準化ロジット変換
    z_ladder = to_std_logit(p_ladder)
    z_exp019 = to_std_logit(p_exp019)
    z_exp022 = to_std_logit(p_exp022)
    z_exp023 = to_std_logit(p_exp023)
    
    # 3. 加重結合 (95.0% Ladder + 2.5% Exp019 + 1.5% Exp022 + 1.0% Exp023)
    w_ladder = 0.950
    w_exp019 = 0.025
    w_exp022 = 0.015
    w_exp023 = 0.010
    
    z_blend = (w_ladder * z_ladder + 
               w_exp019 * z_exp019 + 
               w_exp022 * z_exp022 + 
               w_exp023 * z_exp023)
    
    # 4. 逆変換
    p_blend = expit(z_blend)
    
    # 相関分析
    spearman_ladder = pd.Series(p_blend).corr(pd.Series(p_ladder), method='spearman')
    spearman_exp033 = pd.Series(p_blend).corr(pd.Series(p_exp033), method='spearman')
    spearman_exp019 = pd.Series(p_blend).corr(pd.Series(p_exp019), method='spearman')
    spearman_exp022 = pd.Series(p_blend).corr(pd.Series(p_exp022), method='spearman')
    spearman_exp023 = pd.Series(p_blend).corr(pd.Series(p_exp023), method='spearman')
    
    print("\n--- Blend Correlation Analysis ---")
    print(f"Spearman vs Ladder v17 (Exp032): {spearman_ladder:.6f}")
    print(f"Spearman vs Exp033 (Rank 13):    {spearman_exp033:.6f}")
    print(f"Spearman vs PureXGB (Exp019):    {spearman_exp019:.6f}")
    print(f"Spearman vs BestBlend (Exp022):  {spearman_exp022:.6f}")
    print(f"Spearman vs GLM-XGB (Exp023):    {spearman_exp023:.6f}")
    
    # 5. 提出ファイルの作成
    sub = pd.DataFrame({
        "id": df_ladder["id"],
        "is_fraud": p_blend
    })
    
    output_path = os.path.join(DATA_DIR, "submission_exp034.csv")
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
