"""
004_credit-card-fraud: Exp #022 - Final Grand Mega Logit Blend
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Constituents:
  1. Exp021 (65%): Boosting from a GLM (GLM init_score + 4-leaf LightGBM + 13 Pairs)
     - Single OOF: 0.95085 | Single LB: 0.95113
  2. Exp019 (30%): Colab L4 GPU XGBoost Depth 2 (2 Seeds) + 42 Digit-TE
     - Single OOF: 0.94994 | Single LB: 0.95098
  3. Exp015 (5%):  Colab L4 GPU 3-GBDT (LightGBM + CatBoost GPU + XGBoost GPU) with Original Data
     - Single OOF: 0.94756 | Single LB: 0.94811

OOF ROC-AUC: 0.951077 (史上初の 0.9510+ 超え！)
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

def main():
    print("=" * 70)
    print("  Exp #022: Final Grand Mega Logit Blend")
    print("=" * 70)

    # 1. Load train target
    train = pd.read_csv("004_credit-card-fraud/data/train.csv", usecols=["id", "is_fraud"])
    y = train["is_fraud"].values

    # 2. Load OOF predictions
    oof_021 = np.load("004_credit-card-fraud/data/oof_exp021.npy")
    oof_019 = pd.read_csv("004_credit-card-fraud/data/oof_exp019_colab.csv")["oof_xgb_d2_te"].values
    oof_015 = pd.read_csv("004_credit-card-fraud/data/oof_exp015_colab.csv")["oof_blend"].values

    print(f"  Exp021 (GLM + LGB 4-leaf) OOF AUC: {roc_auc_score(y, oof_021):.5f}")
    print(f"  Exp019 (XGB Depth 2)     OOF AUC: {roc_auc_score(y, oof_019):.5f}")
    print(f"  Exp015 (Colab 3-GBDT)     OOF AUC: {roc_auc_score(y, oof_015):.5f}")

    # 3. Logit Transform and Blend
    eps = 1e-7
    L21 = logit(np.clip(oof_021, eps, 1 - eps))
    L19 = logit(np.clip(oof_019, eps, 1 - eps))
    L15 = logit(np.clip(oof_015, eps, 1 - eps))

    w21, w19, w15 = 0.65, 0.30, 0.05
    final_oof = expit(w21 * L21 + w19 * L19 + w15 * L15)
    blend_auc = roc_auc_score(y, final_oof)

    print("\n" + "=" * 60)
    print(f"Exp #022 Final Blend OOF ROC-AUC: {blend_auc:.6f}")
    print(f"  (Single Exp021 LGB: 0.95085 | Gain: {blend_auc - 0.95085:+.5f})")
    print(f"  (Single Exp019 XGB: 0.94994 | Gain: {blend_auc - 0.94994:+.5f})")
    print("=" * 60)

    np.save("004_credit-card-fraud/data/oof_exp022.npy", final_oof)
    print("Saved OOF to 004_credit-card-fraud/data/oof_exp022.npy")

    # 4. Blend Test Submissions
    sub_021 = pd.read_csv("004_credit-card-fraud/data/submission_exp021.csv")
    sub_019 = pd.read_csv("004_credit-card-fraud/data/submission_exp019_colab.csv")
    sub_015 = pd.read_csv("004_credit-card-fraud/data/submission_exp015_colab.csv")

    p21 = sub_021["is_fraud"].values
    p19 = sub_019["is_fraud"].values
    p15 = sub_015["is_fraud"].values

    tL21 = logit(np.clip(p21, eps, 1 - eps))
    tL19 = logit(np.clip(p19, eps, 1 - eps))
    tL15 = logit(np.clip(p15, eps, 1 - eps))

    blend_test = expit(w21 * tL21 + w19 * tL19 + w15 * tL15)

    sub_df = pd.DataFrame({
        "id": sub_021["id"],
        "is_fraud": blend_test
    })

    out_file = "004_credit-card-fraud/data/submission_exp022.csv"
    sub_df.to_csv(out_file, index=False)
    print(f"\nGenerated submission saved to: {out_file}")
    print(f"Rows: {len(sub_df)}, Nulls: {sub_df.isnull().sum().to_dict()}")
    print("Head:\n", sub_df.head())


if __name__ == "__main__":
    main()
