"""
005_name-that-llm: Exp #005 OOF Decoding, Diagnostics, and Multi-Pillar Ensemble Evaluation
"""

import os
import re
import sys
import base64
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
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
    oof5_path = os.path.join(data_dir, "oof_exp005_openpool_modernbert.csv")

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        decode_b64_from_log(sys.argv[1], oof5_path)

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    classes = sorted(train["model"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    y = train["model"].map(label2idx).values
    groups = train["prompt_id"].values

    o2 = pd.read_csv(os.path.join(data_dir, "oof_exp002_c2.csv"))
    o3 = pd.read_csv(os.path.join(data_dir, "oof_exp003_lgb.csv"))
    o4 = pd.read_csv(os.path.join(data_dir, "oof_exp004_modernbert.csv"))
    o5 = pd.read_csv(oof5_path)

    for name, df in [("LR_Exp002", o2), ("LGBM_Exp003", o3), ("BERT_Exp004", o4), ("OpenPool_Exp005", o5)]:
        assert (df["id"].values == train["id"].values).all(), f"ID mismatch in {name}"

    p_lr = o2[classes].values
    p_lgb = o3[classes].values
    p_b4 = o4[classes].values
    p_b5 = o5[classes].values

    print("\n=== Single Model OOF Balanced Accuracy ===")
    for name, p in [("LR (Exp002)", p_lr), ("LGBM (Exp003)", p_lgb), ("ModernBERT (Exp004)", p_b4), ("OpenPool+ModernBERT (Exp005)", p_b5)]:
        score = balanced_accuracy_score(y, np.argmax(p, axis=1))
        print(f"  {name:30s}: {score:.5f}")

    print("\n=== Per-Class Recall (Exp005 vs Exp004 vs LR) ===")
    pred5 = np.argmax(p_b5, axis=1)
    pred4 = np.argmax(p_b4, axis=1)
    pred2 = np.argmax(p_lr, axis=1)
    print(f"{'Class':24s} | {'Exp002 LR':>9s} | {'Exp004 MB':>9s} | {'Exp005 OP':>9s} | {'Delta 5-4':>9s}")
    print("-" * 70)
    for i, cls in enumerate(classes):
        mask = (y == i)
        r2 = np.mean(pred2[mask] == i)
        r4 = np.mean(pred4[mask] == i)
        r5 = np.mean(pred5[mask] == i)
        print(f"{cls:24s} | {r2:9.4f} | {r4:9.4f} | {r5:9.4f} | {r5 - r4:+9.4f}")

    print("\n=== Mean Diagonal Probability & Prediction Correlation with Exp005 ===")
    for name, p in [("LR (Exp002)", p_lr), ("LGBM (Exp003)", p_lgb), ("ModernBERT (Exp004)", p_b4)]:
        corr = np.corrcoef(p_b5.ravel(), p.ravel())[0, 1]
        agree = np.mean(np.argmax(p_b5, axis=1) == np.argmax(p, axis=1))
        print(f"  vs {name:20s}: prob_corr={corr:.4f}, argmax_agree={agree:.4f}")

    # Simplex grid search on (w_b5, w_lr, w_lgb, w_b4)
    best_score = -1.0
    best_w = None
    step = 0.05
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    for w5 in grid:
        for w_lr in grid:
            if w5 + w_lr > 1.0 + 1e-9:
                continue
            for w_lgb in np.round(np.arange(0.0, 0.30 + 1e-9, step), 2):
                if w5 + w_lr + w_lgb > 1.0 + 1e-9:
                    continue
                w4 = round(1.0 - (w5 + w_lr + w_lgb), 2)
                if w4 < 0 or w4 > 0.35:
                    continue
                blend = w5 * p_b5 + w_lr * p_lr + w_lgb * p_lgb + w4 * p_b4
                s = balanced_accuracy_score(y, np.argmax(blend, axis=1))
                if s > best_score:
                    best_score = s
                    best_w = (w5, w_lr, w_lgb, w4)

    print(f"\n=== Best Grid Blend (step={step}) ===")
    print(f"  Weights (Exp005, LR, LGBM, Exp004) = {best_w} -> OOF BalAcc: {best_score:.5f}")

    # Fine local refinement around best_w
    best_fine_score = best_score
    best_fine_w = best_w
    for d5 in np.linspace(-0.06, 0.06, 7):
        for dl in np.linspace(-0.06, 0.06, 7):
            for dg in np.linspace(-0.04, 0.04, 5):
                w5 = max(0.0, best_w[0] + d5)
                w_lr = max(0.0, best_w[1] + dl)
                w_lgb = max(0.0, best_w[2] + dg)
                w4 = max(0.0, 1.0 - (w5 + w_lr + w_lgb))
                tot = w5 + w_lr + w_lgb + w4
                if tot <= 0:
                    continue
                w5, w_lr, w_lgb, w4 = w5 / tot, w_lr / tot, w_lgb / tot, w4 / tot
                blend = w5 * p_b5 + w_lr * p_lr + w_lgb * p_lgb + w4 * p_b4
                s = balanced_accuracy_score(y, np.argmax(blend, axis=1))
                if s > best_fine_score:
                    best_fine_score = s
                    best_fine_w = (round(w5, 3), round(w_lr, 3), round(w_lgb, 3), round(w4, 3))

    print(f"  Refined Weights (Exp005, LR, LGBM, Exp004) = {best_fine_w} -> OOF BalAcc: {best_fine_score:.5f}")

    # Nested 5-Fold CV check of simple 2-way (Exp005 + LR) and 3-way (Exp005 + LR + LGBM) weights
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    nested_preds = np.zeros(len(train), dtype=int)
    fold_ws = []
    for tr_idx, va_idx in sgkf.split(train, y, groups=groups):
        f_best = -1.0
        f_w = None
        for w5 in np.round(np.arange(0.50, 1.01, 0.05), 2):
            for w_lr in np.round(np.arange(0.0, 1.01 - w5 + 1e-9, 0.05), 2):
                w_lgb = round(1.0 - w5 - w_lr, 2)
                if w_lgb < 0 or w_lgb > 0.25:
                    continue
                b_tr = w5 * p_b5[tr_idx] + w_lr * p_lr[tr_idx] + w_lgb * p_lgb[tr_idx]
                sc = balanced_accuracy_score(y[tr_idx], np.argmax(b_tr, axis=1))
                if sc > f_best:
                    f_best = sc
                    f_w = (w5, w_lr, w_lgb)
        fold_ws.append(f_w)
        b_va = f_w[0] * p_b5[va_idx] + f_w[1] * p_lr[va_idx] + f_w[2] * p_lgb[va_idx]
        nested_preds[va_idx] = np.argmax(b_va, axis=1)

    nested_score = balanced_accuracy_score(y, nested_preds)
    print(f"  Nested-CV 3-Pillar (Exp005, LR, LGBM) OOF BalAcc: {nested_score:.5f} | Fold weights: {fold_ws}")


if __name__ == "__main__":
    main()
