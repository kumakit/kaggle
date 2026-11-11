"""
004_credit-card-fraud: Exp #024 - Dual GLM-Boost Mega Logit Blend
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Constituents:
  1. Exp023 (60%): Colab L4 GPU GLM-Boosted XGBoost Depth 2 (2 Seeds)
     - Single OOF: 0.95116 (史上最高単体スコア！)
  2. Exp021 (30%): GLM-Boosted 4-leaf LightGBM + 13 Pairs
     - Single OOF: 0.95085 | Single LB: 0.95113
  3. Exp015 (10%): Colab L4 GPU 3-GBDT (LightGBM + CatBoost GPU + XGBoost GPU) with Original Data
     - Single OOF: 0.94756 | Single LB: 0.94811

OOF ROC-AUC: 0.951206 (手元OOF史上最高新記録！)
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

def main():
    print("=" * 70)
    print("  Exp #024: Dual GLM-Boost Mega Logit Blend")
    print("=" * 70)

    # 1. Load train target
    train = pd.read_csv("004_credit-card-fraud/data/train.csv", usecols=["id", "is_fraud"])
    y = train["is_fraud"].values

    # 2. Load OOF predictions
    oof_23 = pd.read_csv("004_credit-card-fraud/data/oof_exp023_colab.csv")["oof_glm_xgb_d2"].values
    oof_21 = np.load("004_credit-card-fraud/data/oof_exp021.npy")
    oof_15 = pd.read_csv("004_credit-card-fraud/data/oof_exp015_colab.csv")["oof_blend"].values

    print(f"  Exp023 (GLM + XGB Depth 2) OOF AUC: {roc_auc_score(y, oof_23):.5f}")
    print(f"  Exp021 (GLM + LGB 4-leaf)  OOF AUC: {roc_auc_score(y, oof_21):.5f}")
    print(f"  Exp015 (Colab 3-GBDT)      OOF AUC: {roc_auc_score(y, oof_15):.5f}")

    # 3. Logit Transform and Blend
    eps = 1e-7
    L23 = logit(np.clip(oof_23, eps, 1 - eps))
    L21 = logit(np.clip(oof_21, eps, 1 - eps))
    L15 = logit(np.clip(oof_15, eps, 1 - eps))

    w23, w21, w15 = 0.60, 0.30, 0.10
    final_oof = expit(w23 * L23 + w21 * L21 + w15 * L15)
    blend_auc = roc_auc_score(y, final_oof)

    print("\n" + "=" * 60)
    print(f"Exp #024 Dual GLM-Boost Blend OOF ROC-AUC: {blend_auc:.6f}")
    print(f"  (Single Exp023 XGB: 0.95116 | Gain: {blend_auc - 0.95116:+.5f})")
    print(f"  (Single Exp021 LGB: 0.95085 | Gain: {blend_auc - 0.95085:+.5f})")
    print(f"  (Exp022 Previous:   0.95108 | Gain: {blend_auc - 0.95108:+.5f})")
    print("=" * 60)

    np.save("004_credit-card-fraud/data/oof_exp024.npy", final_oof)
    print("Saved OOF to 004_credit-card-fraud/data/oof_exp024.npy")

    # 4. Blend Test Submissions
    sub_23 = pd.read_csv("004_credit-card-fraud/data/submission_exp023_colab.csv")
    sub_21 = pd.read_csv("004_credit-card-fraud/data/submission_exp021.csv")
    sub_15 = pd.read_csv("004_credit-card-fraud/data/submission_exp015_colab.csv")

    p23 = sub_23["is_fraud"].values
    p21 = sub_21["is_fraud"].values
    p15 = sub_15["is_fraud"].values

    tL23 = logit(np.clip(p23, eps, 1 - eps))
    tL21 = logit(np.clip(p21, eps, 1 - eps))
    tL15 = logit(np.clip(p15, eps, 1 - eps))

    blend_test = expit(w23 * tL23 + w21 * tL21 + w15 * tL15)

    sub_df = pd.DataFrame({
        "id": sub_23["id"],
        "is_fraud": blend_test
    })

    out_file = "004_credit-card-fraud/data/submission_exp024.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\nGenerated submission saved to: {out_file}")
    print(f"Rows: {len(sub_df)}, Nulls: {sub_df.isnull().sum().to_dict()}")
    print("Head:\n", sub_df.head())


if __name__ == "__main__":
    main()
