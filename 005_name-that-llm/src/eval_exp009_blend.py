"""
005_name-that-llm: Exp #009 OOF Decoding, Correlation, and Multi-Architecture Ensemble Optimization
Combines:
  - Exp #009: OpenPool (fidelity >= 0.45) + 5-Fold Qwen3-8B-Base Multi-Layer LoRA
  - Exp #007: OpenPool (fidelity >= 0.45) + 5-Fold Qwen3-4B-Base LoRA
  - Exp #008: OpenPool (fidelity >= 0.45) + 5-Fold ModernBERT-large
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


def search_3model(arrs, y, groups, sgkf, names):
    a1, a2, a3 = arrs
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
    best_s, best_w = -1.0, None
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0 + 1e-9:
                continue
            w3 = round(1.0 - w1 - w2, 2)
            s = balanced_accuracy_score(y, np.argmax(w1 * a1 + w2 * a2 + w3 * a3, axis=1))
            if s > best_s:
                best_s, best_w = s, (w1, w2, w3)

    nested = np.zeros(len(y), dtype=int)
    fold_ws = []
    for tr_idx, va_idx in sgkf.split(np.zeros(len(y)), y, groups=groups):
        fb, fw = -1.0, None
        for w1 in grid:
            for w2 in grid:
                if w1 + w2 > 1.0 + 1e-9:
                    continue
                w3 = round(1.0 - w1 - w2, 2)
                sc = balanced_accuracy_score(y[tr_idx], np.argmax(w1 * a1[tr_idx] + w2 * a2[tr_idx] + w3 * a3[tr_idx], axis=1))
                if sc > fb:
                    fb, fw = sc, (w1, w2, w3)
        fold_ws.append(fw)
        w1, w2, w3 = fw
        nested[va_idx] = np.argmax(w1 * a1[va_idx] + w2 * a2[va_idx] + w3 * a3[va_idx], axis=1)
    ns = balanced_accuracy_score(y, nested)
    print(f"  3-Model {names}: Global={best_s:.5f} @ {best_w} | Nested-CV={ns:.5f} | folds={fold_ws}")


def search_4model(arrs, y, groups, sgkf, names):
    a1, a2, a3, a4 = arrs
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)

    def best_on_subset(idx=None):
        sub_y = y if idx is None else y[idx]
        s1 = a1 if idx is None else a1[idx]
        s2 = a2 if idx is None else a2[idx]
        s3 = a3 if idx is None else a3[idx]
        s4 = a4 if idx is None else a4[idx]
        fb, fw = -1.0, None
        for w1 in grid:
            if w1 < 0.25:
                continue
            for w2 in grid:
                if w1 + w2 > 1.0 + 1e-9:
                    continue
                for w3 in grid:
                    if w1 + w2 + w3 > 1.0 + 1e-9:
                        continue
                    w4 = round(1.0 - w1 - w2 - w3, 2)
                    sc = balanced_accuracy_score(sub_y, np.argmax(w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4, axis=1))
                    if sc > fb:
                        fb, fw = sc, (w1, w2, w3, w4)
        return fb, fw

    gs, gw = best_on_subset()
    nested = np.zeros(len(y), dtype=int)
    fold_ws = []
    for tr_idx, va_idx in sgkf.split(np.zeros(len(y)), y, groups=groups):
        _, fw = best_on_subset(tr_idx)
        fold_ws.append(fw)
        w1, w2, w3, w4 = fw
        nested[va_idx] = np.argmax(w1 * a1[va_idx] + w2 * a2[va_idx] + w3 * a3[va_idx] + w4 * a4[va_idx], axis=1)
    ns = balanced_accuracy_score(y, nested)
    print(f"  4-Model {names}: Global={gs:.5f} @ {gw} | Nested-CV={ns:.5f} | folds={fold_ws}")


def main():
    data_dir = os.path.join("005_name-that-llm", "data")
    oof9_path = os.path.join(data_dir, "oof_exp009_qwen3_8b_lora.csv")

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        decode_b64_from_log(sys.argv[1], oof9_path)

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    classes = sorted(train["model"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    y = train["model"].map(label2idx).values
    groups = train["prompt_id"].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    p5 = pd.read_csv(os.path.join(data_dir, "oof_exp005_openpool_modernbert.csv"))[classes].values
    p6 = pd.read_csv(os.path.join(data_dir, "oof_exp006_qwen3_lora.csv"))[classes].values
    p7 = pd.read_csv(os.path.join(data_dir, "oof_exp007_qwen3_4b_lora.csv"))[classes].values
    p8 = pd.read_csv(os.path.join(data_dir, "oof_exp008_modernbert_large.csv"))[classes].values
    o9 = pd.read_csv(oof9_path)
    assert (o9["id"].values == train["id"].values).all(), "ID mismatch in Exp009 OOF"
    p9 = o9[classes].values

    print("\n=== Single Model OOF Balanced Accuracy ===")
    for name, p in [
        ("Exp005 OpenPool ModernBERT-base", p5),
        ("Exp006 OpenPool Qwen3-1.7B", p6),
        ("Exp008 OpenPool ModernBERT-large", p8),
        ("Exp007 OpenPool Qwen3-4B", p7),
        ("Exp009 OpenPool Qwen3-8B MultiLayer", p9),
    ]:
        print(f"  {name:36s}: {balanced_accuracy_score(y, np.argmax(p, axis=1)):.5f}")

    print("\n=== Correlation with Exp009 Qwen3-8B MultiLayer ===")
    for name, p in [("Exp007 Qwen3-4B", p7), ("Exp008 ModernBERT-large", p8), ("Exp006 Qwen3-1.7B", p6)]:
        corr = np.corrcoef(p9.ravel(), p.ravel())[0, 1]
        agree = np.mean(np.argmax(p9, axis=1) == np.argmax(p, axis=1))
        print(f"  vs {name:24s}: prob_corr={corr:.4f}, argmax_agree={agree:.4f}")

    eps = 1e-8
    for mode_name, trans in [
        ("Linear Prob Blend", lambda x: x),
        ("Log-Prob (Geometric) Blend", lambda x: np.log(x + eps)),
    ]:
        print(f"\n=== {mode_name} ===")
        search_3model((trans(p9), trans(p7), trans(p8)), y, groups, sgkf, "(Qwen3-8B, Qwen3-4B, MB-large)")
        search_4model((trans(p9), trans(p7), trans(p8), trans(p6)), y, groups, sgkf, "(8B, 4B, MB-L, 1.7B)")


if __name__ == "__main__":
    main()
