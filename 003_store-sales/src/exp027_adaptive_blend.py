"""
003_store-sales: Exp #027 - 33 Family-Adaptive Mathematical Blend (SLSQP Krogh-Vedelsby Optimization)
Kaggle Competition: Store Sales - Time Series Forecasting

Key Methodology:
  - Krogh & Vedelsby Ensemble Ambiguity Theory:
      E_ens(w) = sum(w_i * E_i) - sum(w_i * A_i)
  - Solves SLSQP constrained optimization per family (33 unique weight vectors):
      * GROCERY I & BEVERAGES: Higher weight on Deep 5-Seeds (Exp024 ~35%) and Transactions (Exp022 6%)
      * PRODUCE & BABY CARE: Higher weight on Anchor (Exp015 ~50-55%) and DOW model (Exp013 ~15%)
  - Safe 21-day zero sales filter (127 series / 2,032 rows)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def main():
    print("=" * 75)
    print("  003 Store Sales: Exp #027 - 33 Family-Adaptive Mathematical Blend")
    print("=" * 75)

    print("\n[1/4] Loading model submission files...")
    models = {
        "exp015": "003_store-sales/submission_exp015.csv",
        "exp024": "003_store-sales/submission_exp024_colab_deep.csv",
        "exp020": "003_store-sales/submission_exp020_colab_5seeds.csv",
        "exp013": "003_store-sales/submission_exp013.csv",
        "exp022": "003_store-sales/submission_exp022_colab_2stage.csv",
    }

    test = pd.read_csv("003_store-sales/data/test.csv")
    families = sorted(test["family"].unique())
    dfs = {k: pd.read_csv(v) for k, v in models.items()}
    model_keys = list(models.keys())

    for k in model_keys:
        print(f"  - {k:7s}: {len(dfs[k])} rows")

    log_P = {k: np.log1p(dfs[k]["sales"].values) for k in model_keys}

    print("\n[2/4] Solving SLSQP optimization per family (33 categories)...")
    lb_scores = np.array([0.38856, 0.39050, 0.39113, 0.38972, 0.40632])
    E_ind_base = lb_scores ** 2

    bounds = [
        (0.25, 0.55), # Exp015 (Anchor)
        (0.15, 0.35), # Exp024 (Deep 5-Seeds)
        (0.08, 0.25), # Exp020 (5-Seeds)
        (0.08, 0.25), # Exp013 (DOW Mean)
        (0.01, 0.06), # Exp022 (Transactions Spice)
    ]
    w0 = np.array([0.45, 0.25, 0.15, 0.12, 0.03])
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    family_weights = {}
    pred_sales = np.zeros(len(test), dtype=float)

    for fam in families:
        mask = (test["family"] == fam).values
        log_P_fam = np.column_stack([log_P[k][mask] for k in model_keys])
        
        def fam_loss(w):
            w = np.array(w)
            p_ens = log_P_fam @ w
            diffs = log_P_fam - p_ens[:, np.newaxis]
            ambiguity = np.sum(w * np.mean(diffs**2, axis=0))
            weighted_error = np.sum(w * E_ind_base)
            return np.sqrt(weighted_error - ambiguity)
        
        res = minimize(fam_loss, w0, method='SLSQP', bounds=bounds, constraints=cons)
        w_opt = res.x
        family_weights[fam] = w_opt

        fam_log = np.zeros(mask.sum(), dtype=float)
        for idx, k in enumerate(model_keys):
            fam_log += w_opt[idx] * log_P[k][mask]
        pred_sales[mask] = np.clip(np.expm1(fam_log), 0, None)

    fw_df = pd.DataFrame(family_weights, index=model_keys).T
    print("  - Representative optimal weights (Top Categories):")
    print(fw_df.loc[["GROCERY I", "BEVERAGES", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "BABY CARE"]].round(3))

    print("\n[3/4] Applying 21-day safe zero sales filter...")
    train_df = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    zero_pairs = set(series_21_sum[series_21_sum == 0].index)
    zero_mask = test.set_index(["store_nbr", "family"]).index.isin(zero_pairs)
    pred_sales[zero_mask] = 0.0
    print(f"  - Zero clip applied rows: {zero_mask.sum()} rows (expected: 2032 rows)")

    out_path = "003_store-sales/submission_exp027_adaptive_blend.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": pred_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    sub025 = pd.read_csv("003_store-sales/submission_exp025_deep_medal_blend.csv")
    log_025 = np.log1p(sub025["sales"].values)
    log_final = np.log1p(sub_final["sales"].values)
    diff_rmsle = np.sqrt(np.mean((log_final - log_025)**2))
    corr = np.corrcoef(log_final, log_025)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp025 {sub025['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp025 {sub025['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp025 {sub025['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp025 {(sub025['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #025: {diff_rmsle:.5f}")
    print(f"  Correlation with Exp #025: {corr:.6f}")

if __name__ == "__main__":
    main()
