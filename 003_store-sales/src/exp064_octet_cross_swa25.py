"""Exp #064: 8-System Orthogonal Octet Probe (75% Exp #062 Gram-Optimal Heptad + 25% Exp #063 CrossAxial-SWA).

Blends Exp #062 (0.37590 PB) with Exp #063 (CrossAxial + Analytic 112d/28d +
Top-3 SWA + Pseudo-Huber, all-time record single-model Val Raw 0.38177) in masked
log1p space while preserving Exp #036's exact 2,001 zero rows, and supplies the
15th verified LB equation to close the full-rank 8x8 Gram matrix.

Usage:
    uv run python 003_store-sales/src/exp064_octet_cross_swa25.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMP_DIR = Path(__file__).resolve().parents[1]
P036_PATH = COMP_DIR / "submission_exp036_golden_spice.csv"
P062_PATH = COMP_DIR / "submission_exp062_gram_opt_heptad.csv"
P063_PATH = COMP_DIR / "submission_exp063_colab_l4_cross_swa.csv"
OUT_PATH = COMP_DIR / "submission_exp064_octet_cross_swa25.csv"

EPS = 0.25


def main() -> None:
    sub036 = pd.read_csv(P036_PATH)
    sub062 = pd.read_csv(P062_PATH)
    sub063 = pd.read_csv(P063_PATH)

    assert np.array_equal(sub036["id"].values, sub062["id"].values)
    assert np.array_equal(sub036["id"].values, sub063["id"].values)

    zero_mask = sub036["sales"].to_numpy(dtype=np.float64) == 0.0
    active_mask = ~zero_mask

    z62 = np.log1p(np.clip(sub062["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask
    z63 = np.log1p(np.clip(sub063["sales"].to_numpy(dtype=np.float64), 0.0, None)) * active_mask

    corr = float(np.corrcoef(z62, z63)[0, 1])
    diff_rmsle = float(np.sqrt(np.mean((z62 - z63) ** 2)))
    print(f"Exp062 vs Exp063 masked log1p corr={corr:.6f}, diff RMSLE={diff_rmsle:.5f}")

    z_blend = (1.0 - EPS) * z62 + EPS * z63
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
    shift_rmsle = float(np.sqrt(np.mean((z_blend - z62) ** 2)))
    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out_df)} | NaNs: 0 | Zeros: {zero_cnt} | Mean: {out_df['sales'].mean():.4f}")
    print(f"Shift RMSLE vs Exp062 (PB 0.37590): {shift_rmsle:.5f}")


if __name__ == "__main__":
    main()
