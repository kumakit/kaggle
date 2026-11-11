"""
Exp #029: Apex Benchmark Diversity Blend (Standardized Rank-Logit Hybrid)
- コンペ名: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
- 目的: 現在 23 位 (0.95333) の頂点モデル (Apex) に対し、
        相関 0.97196 の自力純粋決定木 (Exp019) と自力最高ブレンド (Exp022) の直交シグナルを注入。
- 重み比率:
    - 95% Exp028_Apex (0.95333)
    -  3% Exp019_PureXGB (0.95098 / 相関 0.97196 の超直交アンカー)
    -  2% Exp022_BestBlend (0.95144 / 自力最高ブレンド)
- 手法: Standardized Rank-Logit Hybrid
    1. 各モデルの予測確率を平均順位パーセンタイル rank(p) / N に正規化
    2. 端点クリップ [1e-7, 1 - 1e-7] 後に logit(r) に変換
    3. 各ロジットを平均0・分散1に標準化 (Z-score)
    4. 加重線形結合: z_blend = 0.95 * z_apex + 0.03 * z_exp019 + 0.02 * z_exp022
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
    print("=== Exp #029: Apex Benchmark Diversity Blend ===")
    
    # 1. 予測ファイルの読み込み
    path_apex = os.path.join(DATA_DIR, "submission_exp028_apex.csv")
    path_exp019 = os.path.join(DATA_DIR, "submission_exp019_colab.csv")
    path_exp022 = os.path.join(DATA_DIR, "submission_exp022.csv")
    
    df_apex = pd.read_csv(path_apex)
    df_exp019 = pd.read_csv(path_exp019)
    df_exp022 = pd.read_csv(path_exp022)
    
    assert len(df_apex) == 299984, f"Unexpected len: {len(df_apex)}"
    assert (df_apex["id"].values == df_exp019["id"].values).all(), "ID mismatch exp019"
    assert (df_apex["id"].values == df_exp022["id"].values).all(), "ID mismatch exp022"
    
    p_apex = df_apex["is_fraud"].values
    p_exp019 = df_exp019["is_fraud"].values
    p_exp022 = df_exp022["is_fraud"].values
    
    print(f"Loaded Apex (N={len(p_apex)}), Exp019 (N={len(p_exp019)}), Exp022 (N={len(p_exp022)})")
    
    # 2. 標準化ロジット変換
    z_apex = to_std_logit(p_apex)
    z_exp019 = to_std_logit(p_exp019)
    z_exp022 = to_std_logit(p_exp022)
    
    # 3. 加重結合 (95% Apex + 3% Exp019 + 2% Exp022)
    w_apex = 0.95
    w_exp019 = 0.03
    w_exp022 = 0.02
    
    z_blend = w_apex * z_apex + w_exp019 * z_exp019 + w_exp022 * z_exp022
    
    # 4. 逆変換
    p_blend = expit(z_blend)
    
    # 相関分析
    spearman_apex = pd.Series(p_blend).corr(pd.Series(p_apex), method='spearman')
    spearman_exp019 = pd.Series(p_blend).corr(pd.Series(p_exp019), method='spearman')
    spearman_exp022 = pd.Series(p_blend).corr(pd.Series(p_exp022), method='spearman')
    
    print("\n--- Blend Correlation Analysis ---")
    print(f"Spearman vs Apex (Exp028):    {spearman_apex:.6f}")
    print(f"Spearman vs PureXGB (Exp019): {spearman_exp019:.6f}")
    print(f"Spearman vs BestBlend (Exp022): {spearman_exp022:.6f}")
    
    # 5. 提出ファイルの作成
    sub = pd.DataFrame({
        "id": df_apex["id"],
        "is_fraud": p_blend
    })
    
    output_path = os.path.join(DATA_DIR, "submission_exp029.csv")
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
