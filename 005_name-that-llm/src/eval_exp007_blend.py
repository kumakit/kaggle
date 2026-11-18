"""
005_name-that-llm: Exp #007 OOF Decoding, Correlation, and 3-Model Ensemble Optimization
Combines:
  - Exp #007: OpenPool (fidelity >= 0.45) + 5-Fold Qwen3-4B-Base LoRA
  - Exp #006: OpenPool + 5-Fold Qwen3-1.7B-Base LoRA
  - Exp #005: OpenPool + 5-Fold ModernBERT-base
"""

import os
import re
import sys
import base64
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold


def decode_b64_from_log(log_path, out_oof_path):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    m = re.search(r"===BASE64_OOF_START===\s*([A-Za-z0-9+/=\s]+?)\s*===BASE64_OOF_END===", text)
    if not m:
        raise ValueError(f"Could not find BASE64_OOF block in {log_path}")
    raw = base64.b64decode("".join(m.group(1).split()))
    with open(out_oof_path, "wb") as f:
        f.write(raw)
    print(f"[DECODE] Saved OOF CSV ({len(raw):,} bytes) -> {out_oof_path}")


def main():
    data_dir = os.path.join("005_name-that-llm", "data")
    oof7_path = os.path.join(data_dir, "oof_exp007_qwen3_4b_lora.csv")

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        decode_b64_from_log(sys.argv[1], oof7_path)

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    classes = sorted(train["model"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    y = train["model"].map(label2idx).values
    groups = train["prompt_id"].values

    p5 = pd.read_csv(os.path.join(data_dir, "oof_exp005_openpool_modernbert.csv"))[classes].values
    p6 = pd.read_csv(os.path.join(data_dir, "oof_exp006_qwen3_lora.csv"))[classes].values
    o7 = pd.read_csv(oof7_path)
    assert (o7["id"].values == train["id"].values).all(), "ID mismatch in Exp007 OOF"
    p7 = o7[classes].values

    print("\n=== Single Model OOF Balanced Accuracy ===")
    for name, p in [
        ("Exp005 OpenPool ModernBERT-base", p5),
        ("Exp006 OpenPool Qwen3-1.7B", p6),
        ("Exp007 OpenPool Qwen3-4B", p7),
    ]:
        print(f"  {name:32s}: {balanced_accuracy_score(y, np.argmax(p, axis=1)):.5f}")

    print("\n=== Correlation with Exp007 Qwen3-4B ===")
    for name, p in [("Exp006 Qwen3-1.7B", p6), ("Exp005 ModernBERT", p5)]:
        corr = np.corrcoef(p7.ravel(), p.ravel())[0, 1]
        agree = np.mean(np.argmax(p7, axis=1) == np.argmax(p, axis=1))
        print(f"  vs {name:20s}: prob_corr={corr:.4f}, argmax_agree={agree:.4f}")

    eps = 1e-8
    for mode_name, arrs in [
        ("Linear Prob Blend", (p7, p6, p5)),
        ("Log-Prob (Geometric) Blend", (np.log(p7 + eps), np.log(p6 + eps), np.log(p5 + eps))),
    ]:
        a7, a6, a5 = arrs
        best_s = -1.0
        best_w = None
        grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
        for w7 in grid:
            for w6 in grid:
                if w7 + w6 > 1.0 + 1e-9:
                    continue
                w5 = round(1.0 - w7 - w6, 2)
                b = w7 * a7 + w6 * a6 + w5 * a5
                s = balanced_accuracy_score(y, np.argmax(b, axis=1))
                if s > best_s:
                    best_s = s
                    best_w = (w7, w6, w5)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        nested_preds = np.zeros(len(train), dtype=int)
        fold_ws = []
        for tr_idx, va_idx in sgkf.split(train, y, groups=groups):
            f_best = -1.0
            f_w = None
            for w7 in np.round(np.arange(0.25, 0.90, 0.05), 2):
                for w6 in np.round(np.arange(0.0, 1.01 - w7 + 1e-9, 0.05), 2):
                    w5 = round(1.0 - w7 - w6, 2)
                    if w5 < 0:
                        continue
                    b_tr = w7 * a7[tr_idx] + w6 * a6[tr_idx] + w5 * a5[tr_idx]
                    sc = balanced_accuracy_score(y[tr_idx], np.argmax(b_tr, axis=1))
                    if sc > f_best:
                        f_best = sc
                        f_w = (w7, w6, w5)
            fold_ws.append(f_w)
            b_va = f_w[0] * a7[va_idx] + f_w[1] * a6[va_idx] + f_w[2] * a5[va_idx]
            nested_preds[va_idx] = np.argmax(b_va, axis=1)

        print(f"\n=== {mode_name} ===")
        print(f"  Global Best (Qwen3-4B, Qwen3-1.7B, ModernBERT) = {best_w} -> OOF BalAcc: {best_s:.5f}")
        print(f"  Nested-CV Out-of-Fold BalAcc                   = {balanced_accuracy_score(y, nested_preds):.5f} | folds={fold_ws}")


if __name__ == "__main__":
    main()
