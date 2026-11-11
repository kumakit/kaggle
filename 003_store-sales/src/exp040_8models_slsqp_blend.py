"""
003_store-sales: Exp #040 - 8-Model 33 Family-Adaptive SLSQP Mathematical Blend
Kaggle Competition: Store Sales - Time Series Forecasting

8 Top Models Integrated:
  1. Exp #036: Current Personal Best (PB 0.38788, Rank 65)
  2. Exp #038: Colab L4 GPU Regional School Disentangled 5-Seeds (Val 0.39116, Record Best Single!)
  3. Exp #015: Anchor Blend (PB 0.38856)
  4. Exp #028: Colab L4 GPU School Season & Calendar 5-Seeds (Val 0.39754)
  5. Exp #024: Colab L4 GPU Deeper Trees 5-Seeds (Val 0.39729)
  6. Exp #020: Colab L4 GPU Clean 5-Seeds (LB 0.39113)
  7. Exp #013: Day-of-Week Group Mean Single Best (LB 0.38972)
  8. Exp #022: Transactions 2-Stage Diversity Spice Model (Corr 0.99850)

Mathematical Formulation:
  - Per-family SLSQP minimizing ensemble generalized error E = \bar{E} - A
  - Produces continuous weight vector across all 33 families.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #040 - 8-Model 33 Family-Adaptive SLSQP Blend")
    print("=" * 80)

    print("\n[1/3] Loading 8 model submission files...")
    models = {
        "exp036": "003_store-sales/submission_exp036_golden_spice.csv",
        "exp038": "003_store-sales/submission_exp038_colab_regional.csv",
        "exp015": "003_store-sales/submission_exp015.csv",
        "exp028": "003_store-sales/submission_exp028_colab_calendar.csv",
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

    log_P_all = {k: np.log1p(dfs[k]["sales"].values) for k in model_keys}

    print("\n[2/3] Solving SLSQP optimization across 33 families...")
    lb_scores = np.array([0.38788, 0.38880, 0.38856, 0.39010, 0.39050, 0.39113, 0.38972, 0.40632])
    E_ind_base = lb_scores ** 2

    bounds = [
        (0.35, 0.65), # Exp036 (PB Anchor)
        (0.10, 0.35), # Exp038 (Regional School L4, Val 0.39116)
        (0.05, 0.20), # Exp015 (Anchor Blend)
        (0.03, 0.15), # Exp028 (Calendar Deep 5S)
        (0.03, 0.15), # Exp024 (Deep 5S)
        (0.02, 0.10), # Exp020 (Clean 5S)
        (0.03, 0.12), # Exp013 (DOW Mean)
        (0.01, 0.03), # Exp022 (Transactions Spice)
    ]
    w0 = np.array([0.48, 0.20, 0.10, 0.06, 0.06, 0.04, 0.05, 0.01])
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    family_weights = {}
    pred_sales = np.zeros(len(test), dtype=float)

    for fam in families:
        mask = (test["family"] == fam).values
        log_P_fam = np.column_stack([log_P_all[k][mask] for k in model_keys])
        
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
            fam_log += w_opt[idx] * log_P_all[k][mask]
        pred_sales[mask] = np.clip(np.expm1(fam_log), 0, None)

    fw_df = pd.DataFrame(family_weights, index=model_keys).T
    print("  - Representative optimal weights (Top Categories):")
    print(fw_df.loc[["GROCERY I", "BEVERAGES", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "CLEANING", "MEATS"]].round(3))
    print("\n  - Mean weights across all 33 families:")
    print(fw_df.mean().round(3))

    out_path = "003_store-sales/submission_exp040_8models_slsqp_blend.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": pred_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[3/3] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    sub036 = pd.read_csv("003_store-sales/submission_exp036_golden_spice.csv")
    log_036 = np.log1p(sub036["sales"].values)
    log_final = np.log1p(sub_final["sales"].values)
    diff_rmsle = np.sqrt(np.mean((log_final - log_036)**2))
    corr = np.corrcoef(log_final, log_036)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp036 PB {sub036['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp036 PB {sub036['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp036 PB {sub036['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp036 PB {(sub036['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #036 (PB 0.38788): {diff_rmsle:.6f}")
    print(f"  Correlation with Exp #036 (PB 0.38788): {corr:.7f}")


if __name__ == "__main__":
    main()
