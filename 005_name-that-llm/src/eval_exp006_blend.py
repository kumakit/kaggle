"""
005_name-that-llm: Exp #006 OOF Decoding, Correlation, and Multi-Architecture Ensemble Evaluation
Combines:
  - Exp #006: OpenPool + 5-Fold Qwen3-1.7B-Base LoRA (Causal Decoder)
  - Exp #005: OpenPool + 5-Fold ModernBERT-base (Bidirectional Encoder)
  - Exp #002: Stylometry + Word/Char TF-IDF LogisticRegression (C=2.0)
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
    oof6_path = os.path.join(data_dir, "oof_exp006_qwen3_lora.csv")

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        decode_b64_from_log(sys.argv[1], oof6_path)

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    classes = sorted(train["model"].unique().tolist())
    label2idx = {c: i for i, c in enumerate(classes)}
    y = train["model"].map(label2idx).values
    groups = train["prompt_id"].values

    p_lr = pd.read_csv(os.path.join(data_dir, "oof_exp002_c2.csv"))[classes].values
    p_b5 = pd.read_csv(os.path.join(data_dir, "oof_exp005_openpool_modernbert.csv"))[classes].values
    o6 = pd.read_csv(oof6_path)
    assert (o6["id"].values == train["id"].values).all(), "ID mismatch in Exp006 OOF"
    p_q6 = o6[classes].values

    print("\n=== Single Model OOF Balanced Accuracy ===")
    for name, p in [
        ("Exp002 Stylometry+TFIDF LR", p_lr),
        ("Exp005 OpenPool ModernBERT", p_b5),
        ("Exp006 OpenPool Qwen3-1.7B", p_q6),
    ]:
        print(f"  {name:30s}: {balanced_accuracy_score(y, np.argmax(p, axis=1)):.5f}")

    print("\n=== Architectural Diversity (Exp006 Qwen3 vs Exp005 ModernBERT / Exp002 LR) ===")
    for name, p in [("Exp005 ModernBERT", p_b5), ("Exp002 LR", p_lr)]:
        corr = np.corrcoef(p_q6.ravel(), p.ravel())[0, 1]
        agree = np.mean(np.argmax(p_q6, axis=1) == np.argmax(p, axis=1))
        print(f"  vs {name:20s}: prob_corr={corr:.4f}, argmax_agree={agree:.4f}")

    print("\n=== Per-Class Recall (Exp006 Qwen3 vs Exp005 ModernBERT) ===")
    pred5 = np.argmax(p_b5, axis=1)
    pred6 = np.argmax(p_q6, axis=1)
    for i, cls in enumerate(classes):
        mask = (y == i)
        r5 = np.mean(pred5[mask] == i)
        r6 = np.mean(pred6[mask] == i)
        print(f"  {cls:24s} | Exp005={r5:.4f} | Exp006={r6:.4f} | Delta={r6 - r5:+.4f}")

    # Evaluate linear vs geometric (log-prob) blends over (w_q6, w_b5, w_lr)
    eps = 1e-8
    lp_q6, lp_b5, lp_lr = np.log(p_q6 + eps), np.log(p_b5 + eps), np.log(p_lr + eps)

    for mode_name, arrs in [("Linear Prob Blend", (p_q6, p_b5, p_lr)), ("Log-Prob (Geometric) Blend", (lp_q6, lp_b5, lp_lr))]:
        a6, a5, alr = arrs
        best_s = -1.0
        best_w = None
        grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
        for w6 in grid:
            for w5 in grid:
                if w6 + w5 > 1.0 + 1e-9:
                    continue
                wlr = round(1.0 - w6 - w5, 2)
                if wlr < 0 or wlr > 0.30:
                    continue
                b = w6 * a6 + w5 * a5 + wlr * alr
                s = balanced_accuracy_score(y, np.argmax(b, axis=1))
                if s > best_s:
                    best_s = s
                    best_w = (w6, w5, wlr)

        # Nested 5-Fold CV check
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        nested_preds = np.zeros(len(train), dtype=int)
        fold_ws = []
        for tr_idx, va_idx in sgkf.split(train, y, groups=groups):
            f_best = -1.0
            f_w = None
            for w6 in np.round(np.arange(0.30, 0.96, 0.05), 2):
                for w5 in np.round(np.arange(0.0, 1.01 - w6 + 1e-9, 0.05), 2):
                    wlr = round(1.0 - w6 - w5, 2)
                    if wlr < 0 or wlr > 0.25:
                        continue
                    b_tr = w6 * a6[tr_idx] + w5 * a5[tr_idx] + wlr * alr[tr_idx]
                    sc = balanced_accuracy_score(y[tr_idx], np.argmax(b_tr, axis=1))
                    if sc > f_best:
                        f_best = sc
                        f_w = (w6, w5, wlr)
            fold_ws.append(f_w)
            b_va = f_w[0] * a6[va_idx] + f_w[1] * a5[va_idx] + f_w[2] * alr[va_idx]
            nested_preds[va_idx] = np.argmax(b_va, axis=1)

        print(f"\n=== {mode_name} ===")
        print(f"  Global Grid Best (Qwen3, ModernBERT, LR) = {best_w} -> OOF BalAcc: {best_s:.5f}")
        print(f"  Nested-CV Out-of-Fold BalAcc             = {balanced_accuracy_score(y, nested_preds):.5f} | folds={fold_ws}")


if __name__ == "__main__":
    main()
