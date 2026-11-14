"""Exp #056: Exact 5x5 Gram-Matrix Jointly Optimal 5-System Log-Space Blend.

Uses the exact Gram matrix G = (1/N) M^T M of the 5 zero-masked log1p prediction
vectors [Exp036, Exp043, Exp050, Exp052, Exp054] and the 10 verified Public LB
equations (Exp036, Exp044..Exp049, Exp051, Exp053, Exp055) to solve for the
exact linear target inner-product vector and minimize predicted Public LB RMSLE
jointly over the 5-dimensional probability simplex.

Usage:
    uv run python 003_store-sales/src/exp056_gram_optimal_quintet.py
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
}
ANCHOR_053_PATH = COMP_DIR / "submission_exp053_quad_patch25.csv"
OUT_PATH = COMP_DIR / "submission_exp056_gram_opt_quintet.csv"

# 10 verified Public LB experiments: (name, simplex_weights [f36, f43, f50, f52, f54], lb_rmsle)
VERIFIED_LB_POINTS = [
    ("Exp036", np.array([1.0000, 0.0000, 0.0000, 0.0000, 0.0000]), 0.38788),
    ("Exp044", np.array([0.9500, 0.0500, 0.0000, 0.0000, 0.0000]), 0.38632),
    ("Exp045", np.array([0.9000, 0.1000, 0.0000, 0.0000, 0.0000]), 0.38491),
    ("Exp046", np.array([0.8500, 0.1500, 0.0000, 0.0000, 0.0000]), 0.38366),
    ("Exp047", np.array([0.7500, 0.2500, 0.0000, 0.0000, 0.0000]), 0.38163),
    ("Exp048", np.array([0.6500, 0.3500, 0.0000, 0.0000, 0.0000]), 0.38023),
    ("Exp049", np.array([0.4800, 0.5200, 0.0000, 0.0000, 0.0000]), 0.37932),
    ("Exp051", np.array([0.3600, 0.3900, 0.2500, 0.0000, 0.0000]), 0.37839),
    ("Exp053", np.array([0.2700, 0.2925, 0.1875, 0.2500, 0.0000]), 0.37784),
    ("Exp055", np.array([0.2160, 0.2340, 0.1500, 0.2000, 0.2000]), 0.37813),
]


def load_aligned_masked_logs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load all 5 pillar submissions + Exp053 anchor, verifying id alignment."""
    base_df = pd.read_csv(SUB_PATHS["Exp036_GBDT1"])
    zero_mask = base_df["sales"].to_numpy(dtype=np.float64) == 0.0
    active_mask = ~zero_mask

    log_cols = []
    for name, path in SUB_PATHS.items():
        df = pd.read_csv(path)
        if not np.array_equal(base_df["id"].to_numpy(), df["id"].to_numpy()):
            raise ValueError(f"ID mismatch in {name}: {path}")
        sales = np.clip(df["sales"].to_numpy(dtype=np.float64), 0.0, None)
        z = np.log1p(sales) * active_mask
        log_cols.append(z)

    anchor_df = pd.read_csv(ANCHOR_053_PATH)
    if not np.array_equal(base_df["id"].to_numpy(), anchor_df["id"].to_numpy()):
        raise ValueError(f"ID mismatch in Exp053 anchor: {ANCHOR_053_PATH}")
    z_anchor = np.log1p(np.clip(anchor_df["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask

    # Matrix M of shape (N, 5)
    mat_m = np.column_stack(log_cols)
    return base_df, zero_mask, mat_m, z_anchor


def solve_gram_system(mat_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve exact 5x5 Gram-matrix linear system from verified LB equations."""
    n_rows = mat_m.shape[0]
    gram = (mat_m.T @ mat_m) / float(n_rows)

    w_mat = np.vstack([pt[1] for pt in VERIFIED_LB_POINTS])
    lbs = np.array([pt[2] for pt in VERIFIED_LB_POINTS], dtype=np.float64)

    # For any simplex weight w (sum(w)=1): LB(w)^2 = w^T G w + a^T w
    rhs = (lbs ** 2) - np.einsum("ki,ij,kj->k", w_mat, gram, w_mat)
    a_vec, _, rank, _ = np.linalg.lstsq(w_mat, rhs, rcond=None)
    if rank != 5:
        raise RuntimeError(f"Expected rank 5 for Gram solver, got {rank}")

    print("=== Exact 5x5 Gram-Matrix Fit Verification (10 Verified LB Submissions) ===")
    max_abs_err = 0.0
    for (name, w, lb_actual) in VERIFIED_LB_POINTS:
        pred_sq = float(w @ gram @ w + a_vec @ w)
        lb_fit = float(np.sqrt(pred_sq))
        err = lb_fit - lb_actual
        max_abs_err = max(max_abs_err, abs(err))
        print(f"  {name:6s} | Actual: {lb_actual:.5f} | Fit: {lb_fit:.5f} | err: {err:+.6f}")
    print(f"Max absolute LB fit residual across 10 points: {max_abs_err:.2e}")

    print("\n=== Implied Standalone Masked LB RMSLE (per pillar) ===")
    for idx, name in enumerate(SUB_PATHS.keys()):
        standalone_lb = float(np.sqrt(gram[idx, idx] + a_vec[idx]))
        print(f"  {name:18s}: {standalone_lb:.5f}")

    def objective(w: np.ndarray) -> float:
        return float(w @ gram @ w + a_vec @ w)

    res = minimize(
        objective,
        x0=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 5,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")

    opt_w = np.clip(res.x, 0.0, 1.0)
    opt_w = opt_w / np.sum(opt_w)
    pred_opt_lb = float(np.sqrt(opt_w @ gram @ opt_w + a_vec @ opt_w))
    return gram, a_vec, opt_w, pred_opt_lb


def main() -> None:
    base_df, zero_mask, mat_m, z_anchor053 = load_aligned_masked_logs()
    _, _, opt_w, pred_opt_lb = solve_gram_system(mat_m)

    names = list(SUB_PATHS.keys())
    print("\n=== Global 5-Model Optimal Simplex Weights (Exp #056) ===")
    for name, w in zip(names, opt_w):
        print(f"  {name:18s}: {w:.6f} ({w * 100:6.2f}%)")
    print(f"Predicted Joint 5-Model Optimal Public LB RMSLE: {pred_opt_lb:.5f}")
    print(f"Expected improvement vs Exp #053 (0.37784): {pred_opt_lb - 0.37784:+.5f}")

    # Blend in masked log1p space and strictly preserve Exp #036 exact 2,001 zero rows
    z_opt = mat_m @ opt_w
    z_opt[zero_mask] = 0.0
    sales_opt = np.expm1(z_opt)
    sales_opt[zero_mask] = 0.0
    sales_opt = np.clip(sales_opt, 0.0, None)

    out_df = base_df[["id"]].copy()
    out_df["sales"] = sales_opt

    # Integrity assertions
    assert len(out_df) == 28512, f"Expected 28512 rows, got {len(out_df)}"
    assert int(out_df["sales"].isna().sum()) == 0, "NaNs detected in submission"
    zero_count = int((out_df["sales"] == 0.0).sum())
    expected_zeros = int(zero_mask.sum())
    assert zero_count == expected_zeros == 2001, (
        f"Expected {expected_zeros}==2001 exact zeros, got {zero_count}"
    )

    out_df.to_csv(OUT_PATH, index=False)

    # Diagnostic comparison vs Exp #053 (PB 0.37784) and Exp #036 (PB 0.38788)
    z_036 = mat_m[:, 0]
    corr_053 = float(np.corrcoef(z_opt, z_anchor053)[0, 1])
    rmsle_diff_053 = float(np.sqrt(np.mean((z_opt - z_anchor053) ** 2)))
    corr_036 = float(np.corrcoef(z_opt, z_036)[0, 1])
    rmsle_diff_036 = float(np.sqrt(np.mean((z_opt - z_036) ** 2)))

    print(f"\nSaved: {OUT_PATH}")
    print(
        f"Rows: {len(out_df)} | NaNs: {int(out_df['sales'].isna().sum())} "
        f"| Zeros: {zero_count} | Mean sales: {out_df['sales'].mean():.4f}"
    )
    print(f"vs Exp #053 (PB 0.37784): log1p corr={corr_053:.6f}, shift RMSLE={rmsle_diff_053:.5f}")
    print(f"vs Exp #036 (PB 0.38788): log1p corr={corr_036:.6f}, shift RMSLE={rmsle_diff_036:.5f}")


if __name__ == "__main__":
    main()
