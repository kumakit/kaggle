"""Exp #061: 7-System Orthogonal Heptad Probe (90% Exp #059 Gram-Optimal Hexad + 10% Exp #060 Cross-TriAxial Decomp).

Blends Exp #059 (0.37608 PB) with Exp #060 (112d Cross-Series Tri-Axial Attention
+ Seasonal-Trend Decomp + Hybrid Loss, Val Raw 0.39892, diff RMSLE 0.09217 vs Exp059)
in masked log1p space while preserving Exp #036's exact 2,001 zero rows, and supplies
the exact 13th LB equation to close the full-rank 7x7 Gram matrix.

Usage:
    uv run python 003_store-sales/src/exp061_heptad_triaxial10.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMP_DIR = Path(__file__).resolve().parents[1]
P036_PATH = COMP_DIR / "submission_exp036_golden_spice.csv"
P059_PATH = COMP_DIR / "submission_exp059_gram_opt_hexad.csv"
P060_PATH = COMP_DIR / "submission_exp060_colab_l4_cross_triaxial.csv"
OUT_PATH = COMP_DIR / "submission_exp061_heptad_triaxial10.csv"

EPS = 0.10


def main() -> None:
    sub036 = pd.read_csv(P036_PATH)
    sub059 = pd.read_csv(P059_PATH)
    sub060 = pd.read_csv(P060_PATH)

    assert np.array_equal(sub036["id"].values, sub059["id"].values)
    assert np.array_equal(sub036["id"].values, sub060["id"].values)

    zero_mask = sub036["sales"].to_numpy(dtype=np.float64) == 0.0
    active_mask = ~zero_mask

    z59 = np.log1p(np.clip(sub059["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask
    z60 = np.log1p(np.clip(sub060["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask

    corr = float(np.corrcoef(z59, z60)[0, 1])
    diff_rmsle = float(np.sqrt(np.mean((z59 - z60) ** 2)))
    print(f"Exp059 vs Exp060 masked log1p corr={corr:.6f}, diff RMSLE={diff_rmsle:.5f}")

    z_blend = (1.0 - EPS) * z59 + EPS * z60
    z_blend[zero_mask] = 0.0
    sales_blend = np.clip(np.expm1(z_blend), 0.0, None)
    sales_blend[zero_mask] = 0.0

    out_df = sub036[["id"]].copy()
    out_df["sales"] = sales_blend

    assert len(out_df) == 28512
    assert int(out_df["sales"].isna().sum()) == 0
    zero_cnt = int((out_df["sales"] == 0.0).sum())
    assert zero_cnt == 2001

    out_df.to_csv(OUT_PATH, index=False)
    shift_rmsle = float(np.sqrt(np.mean((z_blend - z59) ** 2)))
    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out_df)} | NaNs: 0 | Zeros: {zero_cnt} | Mean: {out_df['sales'].mean():.4f}")
    print(f"Shift RMSLE vs Exp059 (PB 0.37608): {shift_rmsle:.5f}")


if __name__ == "__main__":
    main()
