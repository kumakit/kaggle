"""
Exp #030: Top 10 Breakthrough Quad-Anchor Blend (Standardized Rank-Logit Hybrid)
- コンペ名: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
- 目的: 現在 14 位 (0.95336) から Top 10 (0.95338+ / 10位以上) への突入を目指す。
        23位頂点モデル Apex (94%) に対し、我々の独自3大系統:
        1. 純粋決定木 Exp019 (3.0%, 相関 0.97196)
        2. 自力最高ブレンド Exp022 (1.5%, 相関 0.98323)
        3. 自力単体最高 OOF Exp023 GLM-XGBoost (1.5%, 相関 0.98338)
        を Standardized Rank-Logit 空間で完全融合。
- 手法:
    1. 各モデルの予測確率をパーセンタイルランク u = (rank(p) - 0.5) / N に正規化
    2. 端点クリップ [1e-7, 1 - 1e-7] 後に logit(u) に変換
    3. 各ロジットを平均0・分散1に標準化 (Z-score)
    4. 加重線形結合: z_blend = 0.94 * z_apex + 0.030 * z_exp019 + 0.015 * z_exp022 + 0.015 * z_exp023
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
    print("=== Exp #030: Top 10 Breakthrough Quad-Anchor Blend ===")
    
    # 1. 予測ファイルの読み込み
    path_apex = os.path.join(DATA_DIR, "submission_exp028_apex.csv")
    path_exp019 = os.path.join(DATA_DIR, "submission_exp019_colab.csv")
    path_exp022 = os.path.join(DATA_DIR, "submission_exp022.csv")
    path_exp023 = os.path.join(DATA_DIR, "submission_exp023_colab.csv")
    path_exp029 = os.path.join(DATA_DIR, "submission_exp029.csv")
    
    df_apex = pd.read_csv(path_apex)
    df_exp019 = pd.read_csv(path_exp019)
    df_exp022 = pd.read_csv(path_exp022)
    df_exp023 = pd.read_csv(path_exp023)
    df_exp029 = pd.read_csv(path_exp029)
    
    assert len(df_apex) == 299984, f"Unexpected len: {len(df_apex)}"
    assert (df_apex["id"].values == df_exp019["id"].values).all(), "ID mismatch exp019"
    assert (df_apex["id"].values == df_exp022["id"].values).all(), "ID mismatch exp022"
    assert (df_apex["id"].values == df_exp023["id"].values).all(), "ID mismatch exp023"
    
    p_apex = df_apex["is_fraud"].values
    p_exp019 = df_exp019["is_fraud"].values
    p_exp022 = df_exp022["is_fraud"].values
    p_exp023 = df_exp023["is_fraud"].values
    p_exp029 = df_exp029["is_fraud"].values
    
    print(f"Loaded Apex (N={len(p_apex)}), Exp019, Exp022, Exp023 successfully.")
    
    # 2. 標準化ロジット変換
    z_apex = to_std_logit(p_apex)
    z_exp019 = to_std_logit(p_exp019)
    z_exp022 = to_std_logit(p_exp022)
    z_exp023 = to_std_logit(p_exp023)
    
    # 3. 加重結合 (94% Apex + 3% Exp019 + 1.5% Exp022 + 1.5% Exp023)
    w_apex = 0.940
    w_exp019 = 0.030
    w_exp022 = 0.015
    w_exp023 = 0.015
    
    z_blend = (w_apex * z_apex + 
               w_exp019 * z_exp019 + 
               w_exp022 * z_exp022 + 
               w_exp023 * z_exp023)
    
    # 4. 逆変換
    p_blend = expit(z_blend)
    
    # 相関分析
    spearman_apex = pd.Series(p_blend).corr(pd.Series(p_apex), method='spearman')
    spearman_exp029 = pd.Series(p_blend).corr(pd.Series(p_exp029), method='spearman')
    spearman_exp019 = pd.Series(p_blend).corr(pd.Series(p_exp019), method='spearman')
    spearman_exp022 = pd.Series(p_blend).corr(pd.Series(p_exp022), method='spearman')
    spearman_exp023 = pd.Series(p_blend).corr(pd.Series(p_exp023), method='spearman')
    
    print("\n--- Blend Correlation Analysis ---")
    print(f"Spearman vs Apex (Exp028):      {spearman_apex:.6f}")
    print(f"Spearman vs Exp029 (Rank 14):   {spearman_exp029:.6f}")
    print(f"Spearman vs PureXGB (Exp019):   {spearman_exp019:.6f}")
    print(f"Spearman vs BestBlend (Exp022): {spearman_exp022:.6f}")
    print(f"Spearman vs GLM-XGB (Exp023):   {spearman_exp023:.6f}")
    
    # 5. 提出ファイルの作成
    sub = pd.DataFrame({
        "id": df_apex["id"],
        "is_fraud": p_blend
    })
    
    output_path = os.path.join(DATA_DIR, "submission_exp030.csv")
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
