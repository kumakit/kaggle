"""
004_credit-card-fraud: Exp #020 - Quad-Pillar Mega Logit Blend
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Pillars:
  1. Exp019 (76%): Colab L4 GPU XGBoost Depth 2 (2 Seeds) with 42 Digit-TE (OOF 0.94994, LB 0.95098)
  2. Exp015 (15%): Colab L4 GPU 3-GBDT (LightGBM + CatBoost + XGBoost) with Orig-Data (OOF 0.94756, LB 0.94811)
  3. Exp017 (5%):  LightGBM 4-leaf with 42 Digit-TE (OOF 0.94930)
  4. Exp010 (4%):  Original Data + GLM Boosting (OOF 0.94713, LB 0.94796)

Optimization:
  Logit Space: logit(p) = log(p / (1 - p))
  Weights: 0.76 * L19 + 0.15 * L15 + 0.05 * L17 + 0.04 * L10
  手元検証 OOF ROC-AUC: 0.950025 (史上初 0.950 超え！)
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

def main():
    print("=" * 70)
    print("  Exp #020: Quad-Pillar Mega Logit Blend")
    print("=" * 70)

    # 1. Load train target
    train = pd.read_csv("004_credit-card-fraud/data/train.csv", usecols=["id", "is_fraud"])
    y = train["is_fraud"].values
    print(f"Loaded train: {len(y)} rows, positive rate: {y.mean():.4%}")

    # 2. Load OOF predictions
    oof_019 = pd.read_csv("004_credit-card-fraud/data/oof_exp019_colab.csv")["oof_xgb_d2_te"].values
    oof_015 = pd.read_csv("004_credit-card-fraud/data/oof_exp015_colab.csv")["oof_blend"].values
    oof_017 = np.load("004_credit-card-fraud/data/oof_exp017.npy")
    oof_010 = np.load("004_credit-card-fraud/data/oof_exp010.npy")

    print(f"  Exp019 (XGB Digit-TE) OOF AUC: {roc_auc_score(y, oof_019):.5f}")
    print(f"  Exp015 (Colab 3-GBDT)  OOF AUC: {roc_auc_score(y, oof_015):.5f}")
    print(f"  Exp017 (LGB Digit-TE) OOF AUC: {roc_auc_score(y, oof_017):.5f}")
    print(f"  Exp010 (Orig GLM)      OOF AUC: {roc_auc_score(y, oof_010):.5f}")

    # 3. Transform to Logit Space
    eps = 1e-7
    L19 = logit(np.clip(oof_019, eps, 1 - eps))
    L15 = logit(np.clip(oof_015, eps, 1 - eps))
    L17 = logit(np.clip(oof_017, eps, 1 - eps))
    L10 = logit(np.clip(oof_010, eps, 1 - eps))

    w19, w15, w17, w10 = 0.76, 0.15, 0.05, 0.04
    blend_oof = expit(w19 * L19 + w15 * L15 + w17 * L17 + w10 * L10)
    blend_auc = roc_auc_score(y, blend_oof)

    print("\n" + "=" * 60)
    print(f"Exp #020 Mega Blend OOF ROC-AUC: {blend_auc:.6f}")
    print(f"  (Exp #019 Single XGBoost: 0.94994 | Gain: {blend_auc - 0.94994:+.5f})")
    print(f"  (Exp #018 3-Pillar Blend: 0.94959 | Gain: {blend_auc - 0.94959:+.5f})")
    print("=" * 60)

    np.save("004_credit-card-fraud/data/oof_exp020.npy", blend_oof)
    print("Saved OOF to 004_credit-card-fraud/data/oof_exp020.npy")

    # 4. Blend Test Submissions
    sub_019 = pd.read_csv("004_credit-card-fraud/data/submission_exp019_colab.csv")
    sub_015 = pd.read_csv("004_credit-card-fraud/data/submission_exp015_colab.csv")
    sub_017 = pd.read_csv("004_credit-card-fraud/data/submission_exp017.csv")
    sub_010 = pd.read_csv("004_credit-card-fraud/data/submission_exp010.csv")

    p19 = sub_019["is_fraud"].values
    p15 = sub_015["is_fraud"].values
    p17 = sub_017["is_fraud"].values
    p10 = sub_010["is_fraud"].values

    tL19 = logit(np.clip(p19, eps, 1 - eps))
    tL15 = logit(np.clip(p15, eps, 1 - eps))
    tL17 = logit(np.clip(p17, eps, 1 - eps))
    tL10 = logit(np.clip(p10, eps, 1 - eps))

    blend_test = expit(w19 * tL19 + w15 * tL15 + w17 * tL17 + w10 * tL10)

    sub_df = pd.DataFrame({
        "id": sub_019["id"],
        "is_fraud": blend_test
    })

    out_path = "004_credit-card-fraud/data/submission_exp020.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"\nGenerated submission saved to: {out_path}")
    print(f"Rows: {len(sub_df)}, Nulls: {sub_df.isnull().sum().to_dict()}")
    print("Head:\n", sub_df.head())

if __name__ == "__main__":
    main()
