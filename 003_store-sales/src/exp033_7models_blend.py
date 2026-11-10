"""
003_store-sales: Exp #033 - 7-Model 33 Family-Adaptive Mathematical Blend (SLSQP Krogh-Vedelsby Optimization)
Kaggle Competition: Store Sales - Time Series Forecasting

7 Top Models Integrated:
  1. Exp #015: Anchor Blend (PB 0.38856)
  2. Exp #032: Colab L4 GPU Earthquake Decay & Holiday Granular 5-Seeds 1000-Trees (Val 0.39835, NEW)
  3. Exp #028: Colab L4 GPU School Season & Calendar Enhanced 5-Seeds 1000-Trees (Val 0.39754)
  4. Exp #024: Colab L4 GPU Deeper Trees 5-Seeds 1000-Trees (Val 0.39729)
  5. Exp #020: Colab L4 GPU Clean 5-Seeds 1000-Trees (LB 0.39113)
  6. Exp #013: Day-of-Week Group Mean Single Best (LB 0.38972)
  7. Exp #022: Transactions 2-Stage Diversity Spice Model (Corr 0.99850)

Mathematical Formulation:
  - Per-family SLSQP minimizing ensemble generalized error E = \bar{E} - A
  - \bar{E} = \sum w_i E_i (weighted individual generalization error)
  - A = \sum w_i (p_i - p_ens)^2 (ensemble ambiguity / diversity)
  - 21-day safe zero sales filter (127 series / 2,032 rows).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def main():
    print("=" * 80)
    print("  003 Store Sales: Exp #033 - 7-Model 33 Family-Adaptive Mathematical Blend")
    print("=" * 80)

    print("\n[1/4] Loading 7 model submission files...")
    models = {
        "exp015": "003_store-sales/submission_exp015.csv",
        "exp032": "003_store-sales/submission_exp032_colab_earthquake.csv",
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

    print("\n[2/4] Solving SLSQP optimization across 33 families...")
    lb_scores = np.array([0.38856, 0.38980, 0.39010, 0.39050, 0.39113, 0.38972, 0.40632])
    E_ind_base = lb_scores ** 2

    bounds = [
        (0.18, 0.40), # Exp015 (Anchor)
        (0.08, 0.25), # Exp032 (Earthquake & Holiday 5S, NEW)
        (0.08, 0.25), # Exp028 (Calendar Deep 5S)
        (0.08, 0.25), # Exp024 (Deep 5S)
        (0.04, 0.15), # Exp020 (Clean 5S)
        (0.05, 0.18), # Exp013 (DOW Mean)
        (0.01, 0.04), # Exp022 (Transactions Spice)
    ]
    w0 = np.array([0.32, 0.16, 0.15, 0.15, 0.09, 0.10, 0.03])
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

    print("\n[3/4] Applying 21-day safe zero sales filter...")
    train_df = pd.read_csv("003_store-sales/data/train.csv", parse_dates=["date"])
    last_21_days = train_df[train_df["date"] >= "2017-07-26"]
    series_21_sum = last_21_days.groupby(["store_nbr", "family"])["sales"].sum()
    zero_pairs = set(series_21_sum[series_21_sum == 0].index)
    zero_mask = test.set_index(["store_nbr", "family"]).index.isin(zero_pairs)
    pred_sales[zero_mask] = 0.0
    print(f"  - Zero clip applied rows: {zero_mask.sum()} rows (expected: 2032 rows)")

    out_path = "003_store-sales/submission_exp033_7models_blend.csv"
    sub_final = pd.DataFrame({
        "id": test["id"],
        "sales": pred_sales
    })
    sub_final.to_csv(out_path, index=False)
    print(f"\n[4/4] Output saved to: {out_path} ({len(sub_final)} rows)")

    print("\n=== Top 10 Predictions ===")
    print(sub_final.head(10))

    sub030 = pd.read_csv("003_store-sales/submission_exp030_dual_optimal_blend.csv")
    log_030 = np.log1p(sub030["sales"].values)
    log_final = np.log1p(sub_final["sales"].values)
    diff_rmsle = np.sqrt(np.mean((log_final - log_030)**2))
    corr = np.corrcoef(log_final, log_030)[0, 1]

    print("\n=== Prediction Statistics ===")
    print(f"  Mean sales:   {sub_final['sales'].mean():.4f} (vs Exp030 PB {sub030['sales'].mean():.4f})")
    print(f"  Median sales: {sub_final['sales'].median():.4f} (vs Exp030 PB {sub030['sales'].median():.4f})")
    print(f"  Max sales:    {sub_final['sales'].max():.4f} (vs Exp030 PB {sub030['sales'].max():.4f})")
    print(f"  Zero count:   {(sub_final['sales'] == 0).sum()} (vs Exp030 PB {(sub030['sales'] == 0).sum()})")
    print(f"\n  Shift RMSLE from Exp #030 (PB 0.38793): {diff_rmsle:.5f}")
    print(f"  Correlation with Exp #030 (PB 0.38793): {corr:.6f}")


if __name__ == "__main__":
    main()
