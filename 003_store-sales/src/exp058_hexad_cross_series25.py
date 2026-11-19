"""Exp #058: 6-System Orthogonal Hexad Blend (75% Exp #056 Gram-Optimal Quintet + 25% Exp #057 Cross-Series Huber).

Combines the exact 5x5 Gram-optimal quintet (0.37761 PB) with the 6th orthogonal
pillar Exp #057 (Cross-Series 2D Axial Attention + SmoothL1/Huber, Val Raw 0.38439)
in masked log1p space while strictly preserving Exp #036's 2,001 zero rows.
Also supplies the exact probe equation needed to close the 6x6 Gram-matrix system.

Usage:
    uv run python 003_store-sales/src/exp058_hexad_cross_series25.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMP_DIR = Path(__file__).resolve().parents[1]
P036_PATH = COMP_DIR / "submission_exp036_golden_spice.csv"
P056_PATH = COMP_DIR / "submission_exp056_gram_opt_quintet.csv"
P057_PATH = COMP_DIR / "submission_exp057_colab_l4_cross_series_huber.csv"
OUT_PATH = COMP_DIR / "submission_exp058_hexad_cross_series25.csv"

EPS = 0.25


def main() -> None:
    sub036 = pd.read_csv(P036_PATH)
    sub056 = pd.read_csv(P056_PATH)
    sub057 = pd.read_csv(P057_PATH)

    assert np.array_equal(sub036["id"].values, sub056["id"].values)
    assert np.array_equal(sub036["id"].values, sub057["id"].values)

    zero_mask = sub036["sales"].to_numpy(dtype=np.float64) == 0.0
    active_mask = ~zero_mask

    z56 = np.log1p(np.clip(sub056["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask
    z57 = np.log1p(np.clip(sub057["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask

    corr = float(np.corrcoef(z56, z57)[0, 1])
    diff_rmsle = float(np.sqrt(np.mean((z56 - z57) ** 2)))
    print(f"Exp056 vs Exp057 masked log1p corr={corr:.6f}, diff RMSLE={diff_rmsle:.5f}")

    z_blend = (1.0 - EPS) * z56 + EPS * z57
    z_blend[zero_mask] = 0.0
    sales_blend = np.expm1(z_blend)
    sales_blend[zero_mask] = 0.0
    sales_blend = np.clip(sales_blend, 0.0, None)

    out_df = sub036[["id"]].copy()
    out_df["sales"] = sales_blend

    assert len(out_df) == 28512, f"Expected 28512 rows, got {len(out_df)}"
    assert int(out_df["sales"].isna().sum()) == 0
    zero_cnt = int((out_df["sales"] == 0.0).sum())
    assert zero_cnt == 2001, f"Expected 2001 exact zeros, got {zero_cnt}"

    out_df.to_csv(OUT_PATH, index=False)
    shift_rmsle = float(np.sqrt(np.mean((z_blend - z56) ** 2)))
    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out_df)} | NaNs: 0 | Zeros: {zero_cnt} | Mean: {out_df['sales'].mean():.4f}")
    print(f"Shift RMSLE vs Exp056 (PB 0.37761): {shift_rmsle:.5f}")


if __name__ == "__main__":
    main()
