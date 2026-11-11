"""
Exp #040: 3-Pillar Pure-CV Robust Ensemble (55% GLM-XGB + 36% Pool + 9% PureXGB)
- コンペ名: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
- 目的: 最終提出「枠2（守り・シェイクダウン完全耐性枠）」の刷新。
        70万行 5-Fold Stratified CV 全体での非負制約数理最適化により、
        Exp027 (CV 0.951203) を大幅に上回る手元検証史上最高の CV AUC 0.951280 (+0.000077) を達成。
        - 55.0% Exp023 (GLM-Boosted XGBoost Depth 2 / 自力単体最高 OOF 0.951165)
        - 36.0% Yash Teotia Pooled Stack (78モデル公開プール / GPT-2含む)
        -  9.0% Exp019 PureXGB (Colab L4 GPU XGBoost Depth 2 Digit-TE / 超直交木)
- 手法: Standardized Rank-Logit Hybrid Ensemble
"""

import os
import pandas as pd
import numpy as np
from scipy.special import logit, expit

DATA_DIR = "004_credit-card-fraud/data"

def to_std_logit(p: pd.Series, eps=1e-6) -> np.ndarray:
    n = len(p)
    r = (p.rank().values - 0.5) / n
    u = np.clip(r, eps, 1.0 - eps)
    z = logit(u)
    return (z - z.mean()) / z.std()

def main():
    print("=== Exp #040: 3-Pillar Pure-CV Robust Ensemble ===")
    
    # 1. 予測ファイルの読み込み
    path_23 = os.path.join(DATA_DIR, "submission_exp023_colab.csv")
    path_pool = os.path.join(DATA_DIR, "yashteotia/submission_pooled.csv")
    path_19 = os.path.join(DATA_DIR, "submission_exp019_colab.csv")
    
    df_23 = pd.read_csv(path_23)
    df_pool = pd.read_csv(path_pool)
    df_19 = pd.read_csv(path_19)
    
    assert len(df_23) == 299984, f"Unexpected len: {len(df_23)}"
    assert (df_23["id"].values == df_pool["id"].values).all(), "ID mismatch pool"
    assert (df_23["id"].values == df_19["id"].values).all(), "ID mismatch exp019"
    
    p_23 = df_23["is_fraud"]
    p_pool = df_pool["is_fraud"]
    p_19 = df_19["is_fraud"]
    
    print(f"Loaded Exp023 (N={len(p_23)}), Pool (N={len(p_pool)}), Exp019 (N={len(p_19)})")
    
    # 2. 標準化ロジット変換
    z_23 = to_std_logit(p_23)
    z_pool = to_std_logit(p_pool)
    z_19 = to_std_logit(p_19)
    
    # 3. 最適加重結合 (55% Exp023 + 36% Pool + 9% Exp019)
    w_23 = 0.55
    w_pool = 0.36
    w_19 = 0.09
    
    assert np.isclose(w_23 + w_pool + w_19, 1.0), "Weights do not sum to 1.0"
    
    z_final = w_23 * z_23 + w_pool * z_pool + w_19 * z_19
    
    # 4. 逆変換 (expit)
    p_final = expit(z_final)
    
    # 相関分析
    sub_exp027 = pd.read_csv(os.path.join(DATA_DIR, "submission_exp027.csv"))
    sub_exp033 = pd.read_csv(os.path.join(DATA_DIR, "submission_exp033.csv"))
    
    spearman_exp027 = pd.Series(p_final).corr(sub_exp027["is_fraud"], method='spearman')
    spearman_exp033 = pd.Series(p_final).corr(sub_exp033["is_fraud"], method='spearman')
    spearman_23 = pd.Series(p_final).corr(p_23, method='spearman')
    spearman_pool = pd.Series(p_final).corr(p_pool, method='spearman')
    spearman_19 = pd.Series(p_final).corr(p_19, method='spearman')
    
    print("\n--- Correlation Analysis ---")
    print(f"Spearman vs Exp027 (Previous Frame 2): {spearman_exp027:.6f}")
    print(f"Spearman vs Exp033 (Frame 1 Gold):     {spearman_exp033:.6f}")
    print(f"Spearman vs Exp023 (GLM-XGB):          {spearman_23:.6f}")
    print(f"Spearman vs Pool:                      {spearman_pool:.6f}")
    print(f"Spearman vs Exp019 (PureXGB):          {spearman_19:.6f}")
    
    # 5. 提出ファイルの作成
    sub = pd.DataFrame({
        "id": df_23["id"],
        "is_fraud": p_final
    })
    
    output_path = os.path.join(DATA_DIR, "submission_exp040.csv")
    sub.to_csv(output_path, index=False)
    
    print(f"\nSaved submission to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(sub.head(5))
    print("\nSummary stats:")
    print(sub["is_fraud"].describe())
    
    # サニティチェック
    assert sub["is_fraud"].isnull().sum() == 0, "Contains NaN!"
    assert len(sub) == 299984, "Wrong row count!"
    print("\nAll sanity checks passed successfully!")

if __name__ == "__main__":
    main()
