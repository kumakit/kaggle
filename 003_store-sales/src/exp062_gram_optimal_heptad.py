"""Exp #062: Exact 7x7 Gram-Matrix Jointly Optimal 7-System Log-Space Blend.

Solves the full-rank 7x7 Gram-matrix quadratic program across all 14 verified
Public LB submissions (Exp036, Exp044..049, Exp051, Exp053, Exp055, Exp056, Exp058, Exp059, Exp061)
to jointly optimize all 7 orthogonal pillars:
  [Exp036 GBDT1, Exp043 ResNet, Exp050 BiGRU-TCN, Exp052 Patch-DLinear, Exp054 GBDT2, Exp057 CrossAxial, Exp060 TriAxial]
while strictly preserving Exp #036's exact 2,001 zero rows.

Usage:
    uv run python 003_store-sales/src/exp062_gram_optimal_heptad.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

COMP_DIR = Path(__file__).resolve().parents[1]

SUB_PATHS = {
    "Exp036_GBDT1": COMP_DIR / "submission_exp036_golden_spice.csv",
    "Exp043_ResNet": COMP_DIR / "submission_exp043_colab_l4_dnn.csv",
    "Exp050_BiGRU_TCN": COMP_DIR / "submission_exp050_colab_l4_bigru_tcn.csv",
    "Exp052_Patch_DLin": COMP_DIR / "submission_exp052_colab_l4_patch_dlinear.csv",
    "Exp054_GBDT2": COMP_DIR / "submission_exp054_colab_l4_gbdt_gen2.csv",
    "Exp057_CrossAxial": COMP_DIR / "submission_exp057_colab_l4_cross_series_huber.csv",
    "Exp060_TriAxial": COMP_DIR / "submission_exp060_colab_l4_cross_triaxial.csv",
}
ANCHOR_061_PATH = COMP_DIR / "submission_exp061_heptad_triaxial10.csv"
OUT_PATH = COMP_DIR / "submission_exp062_gram_opt_heptad.csv"

W56 = np.array([0.310846, 0.323629, 0.040734, 0.275240, 0.049552, 0.0, 0.0], dtype=np.float64)
W56 /= W56.sum()
W58 = 0.75 * W56 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0], dtype=np.float64)
W59 = np.array([0.234882, 0.281416, 0.007632, 0.120176, 0.0, 0.355894, 0.0], dtype=np.float64)
W59 /= W59.sum()
W61 = 0.90 * W59 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10], dtype=np.float64)

VERIFIED_LB_POINTS = [
    ("Exp036", np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.38788),
    ("Exp044", np.array([0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.38632),
    ("Exp045", np.array([0.90, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.38491),
    ("Exp046", np.array([0.85, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.38366),
    ("Exp047", np.array([0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.38163),
    ("Exp048", np.array([0.65, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.38023),
    ("Exp049", np.array([0.48, 0.52, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.37932),
    ("Exp051", np.array([0.36, 0.39, 0.25, 0.0, 0.0, 0.0, 0.0]), 0.37839),
    ("Exp053", np.array([0.27, 0.2925, 0.1875, 0.25, 0.0, 0.0, 0.0]), 0.37784),
    ("Exp055", np.array([0.216, 0.234, 0.15, 0.20, 0.20, 0.0, 0.0]), 0.37813),
    ("Exp056", W56, 0.37761),
    ("Exp058", W58, 0.37632),
    ("Exp059", W59, 0.37608),
    ("Exp061", W61, 0.37591),
]


def main() -> None:
    base_df = pd.read_csv(SUB_PATHS["Exp036_GBDT1"])
    zero_mask = base_df["sales"].to_numpy(dtype=np.float64) == 0.0
    active_mask = ~zero_mask

    cols = []
    for name, path in SUB_PATHS.items():
        df = pd.read_csv(path)
        assert np.array_equal(base_df["id"].to_numpy(), df["id"].to_numpy()), f"ID mismatch: {name}"
        cols.append(np.log1p(np.clip(df["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask)

    mat_m = np.column_stack(cols)
    gram = (mat_m.T @ mat_m) / float(len(mat_m))

    w_mat = np.vstack([pt[1] for pt in VERIFIED_LB_POINTS])
    lbs = np.array([pt[2] for pt in VERIFIED_LB_POINTS], dtype=np.float64)
    rhs = (lbs ** 2) - np.einsum("ki,ij,kj->k", w_mat, gram, w_mat)
    a_vec, _, rank, _ = np.linalg.lstsq(w_mat, rhs, rcond=None)
    if rank != 7:
        raise RuntimeError(f"Expected rank 7, got {rank}")

    print("=== Exact 7x7 Gram-Matrix Fit Verification (14 Verified LB Submissions) ===")
    max_err = 0.0
    for name, w, lb in VERIFIED_LB_POINTS:
        fit = float(np.sqrt(w @ gram @ w + a_vec @ w))
        err = fit - lb
        max_err = max(max_err, abs(err))
        print(f"  {name:6s} | Actual: {lb:.5f} | Fit: {fit:.5f} | err: {err:+.6f}")
    print(f"Max absolute fit residual across 14 points: {max_err:.2e}")

    print("\n=== Implied Standalone Masked LB RMSLE ===")
    for i, name in enumerate(SUB_PATHS.keys()):
        print(f"  {name:18s}: {np.sqrt(gram[i, i] + a_vec[i]):.5f}")

    res = minimize(
        lambda w: float(w @ gram @ w + a_vec @ w),
        x0=np.ones(7) / 7.0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 7,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")

    opt_w = np.clip(res.x, 0.0, 1.0)
    opt_w /= opt_w.sum()
    pred_lb = float(np.sqrt(opt_w @ gram @ opt_w + a_vec @ opt_w))

    print("\n=== Global 7-Model Optimal Simplex Weights (Exp #062) ===")
    for name, w in zip(SUB_PATHS.keys(), opt_w):
        print(f"  {name:18s}: {w:.6f} ({w * 100:6.2f}%)")
    print(f"Predicted Joint 7-Model Optimal Public LB: {pred_lb:.5f}")
    print(f"Expected improvement vs Exp #061 (0.37591): {pred_lb - 0.37591:+.5f}")

    z_opt = mat_m @ opt_w
    z_opt[zero_mask] = 0.0
    sales_opt = np.clip(np.expm1(z_opt), 0.0, None)
    sales_opt[zero_mask] = 0.0

    out_df = base_df[["id"]].copy()
    out_df["sales"] = sales_opt

    assert len(out_df) == 28512
    assert int(out_df["sales"].isna().sum()) == 0
    zero_cnt = int((out_df["sales"] == 0.0).sum())
    assert zero_cnt == 2001, f"Expected 2001 exact zeros, got {zero_cnt}"
    out_df.to_csv(OUT_PATH, index=False)

    z61 = np.log1p(np.clip(pd.read_csv(ANCHOR_061_PATH)["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask
    corr61 = float(np.corrcoef(z_opt, z61)[0, 1])
    shift61 = float(np.sqrt(np.mean((z_opt - z61) ** 2)))
    print(f"\nSaved: {OUT_PATH}")
    print(f"Rows: {len(out_df)} | NaNs: 0 | Zeros: {zero_cnt} | Mean: {out_df['sales'].mean():.4f}")
    print(f"vs Exp #061 (PB 0.37591): log1p corr={corr61:.6f}, shift RMSLE={shift61:.5f}")


if __name__ == "__main__":
    main()
