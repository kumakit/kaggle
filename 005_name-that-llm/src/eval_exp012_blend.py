"""
005_name-that-llm: Exp #012 Full 5-Fold CV & Nested-CV Evaluator
Evaluates 5 Neural Backbones (Qwen3-4B, ModernBERT-large, DeBERTa-v3-large, Qwen3-1.7B, ModernBERT-base)
+ Exp #011 StyleGram + capped_agglo Cluster Sinkhorn + Global Sinkhorn + Exp #010 GLM-5.3/Flash.
"""

import os
import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
DATA_DIR = "005_name-that-llm/data"


def capped_agglo(S, cap=18, thr=0.08):
    n = len(S)
    A = S.astype(np.float32).copy()
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32)
    alive = np.ones(n, bool)
    lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new
        A[:, i] = new
        A[i, i] = -np.inf
        A[j] = -np.inf
        A[:, j] = -np.inf
        size[i] += size[j]
        alive[j] = False
        lab[lab == j] = i
    return lab


def cluster_sinkhorn(logp, groups_c, n_iters=3, alpha=0.75, lo=6, hi=18):
    out = logp.copy()
    _, inv, cnt = np.unique(groups_c, return_inverse=True, return_counts=True)
    for g in np.flatnonzero((cnt >= lo) & (cnt <= hi)):
        idx = np.flatnonzero(inv == g)
        z = out[idx].copy()
        for _ in range(n_iters):
            z = log_softmax(z, axis=1)
            z = z - alpha * z.mean(axis=0, keepdims=True)
        out[idx] = log_softmax(z, axis=1)
    return out


def global_sinkhorn(logp, mix=0.35):
    P = np.exp(logp - logp.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    Q = P.copy()
    target = np.full(P.shape[1], 1.0 / P.shape[1])
    for _ in range(100):
        Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return log_softmax((1 - mix) * np.log(P + 1e-12) + mix * np.log(Q + 1e-12), axis=1)


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    classes = sorted(df["model"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = df["model"].map(c2i).values
    groups = df["prompt_id"].values
    eps = 1e-7

    o5 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp005_openpool_modernbert.csv"))[classes].values
    o6 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp006_qwen3_lora.csv"))[classes].values
    o7 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp007_qwen3_4b_lora.csv"))[classes].values
    o8 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp008_modernbert_large.csv"))[classes].values
    o12 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp012_deberta_large.csv"))[classes].values
    o11 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp011_stylegram.csv"))[classes].values
    o10 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp010_glm_blend.csv"))[classes].values
    l5, l6, l7, l8, l12, l11, ll = [np.log(np.clip(x, eps, 1.0)) for x in (o5, o6, o7, o8, o12, o11, o10)]

    v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
    Xc = v.fit_transform(df["text"].fillna(""))
    G = capped_agglo((Xc @ Xc.T).toarray(), 18, 0.08)

    # Nested-CV on Neural weights + w_sg + Sinkhorn params
    weight_cands = [
        (0.30, 0.30, 0.20, 0.15, 0.05),
        (0.30, 0.30, 0.20, 0.10, 0.10),
        (0.25, 0.30, 0.25, 0.15, 0.05),
        (0.25, 0.25, 0.25, 0.20, 0.05),
        (0.30, 0.25, 0.25, 0.15, 0.05),
    ]
    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]

    def full_pipeline(idx_mask, w_neu, w_sg, c_sk, g_sk):
        w7, w8, w12, w6, w5 = w_neu
        bl = log_softmax(w7 * l7[idx_mask] + w8 * l8[idx_mask] + w12 * l12[idx_mask] + w6 * l6[idx_mask] + w5 * l5[idx_mask] + w_sg * l11[idx_mask], 1)
        if c_sk > 0:
            bl = cluster_sinkhorn(bl, G[idx_mask], 3, c_sk, 6, 18)
        if g_sk > 0:
            bl = global_sinkhorn(bl, g_sk)
        glm_mass = np.exp(bl)[:, glm_idxs].sum(1)
        sel = np.sort(np.argsort(-glm_mass, kind="stable")[: int(round(0.30 * len(idx_mask)))])
        before = bl.argmax(1)
        after = before.copy()
        after[sel] = (bl[sel] + 1.15 * ll[idx_mask][sel]).argmax(1)
        glm_rel = np.isin(before, glm_idxs) | np.isin(after, glm_idxs)
        after[(before != after) & ~glm_rel] = before[(before != after) & ~glm_rel]
        return after

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    nested_preds = np.zeros_like(y)
    chosen = []
    for tr_idx, va_idx in splitter.split(df, y, groups=groups):
        best_tr, best_cfg = -1.0, None
        for w_neu in weight_cands:
            for w_sg in [0.25, 0.35]:
                for c_sk in [0.75, 1.00]:
                    p_tr = full_pipeline(tr_idx, w_neu, w_sg, c_sk, 0.35)
                    sc = balanced_accuracy_score(y[tr_idx], p_tr)
                    if sc > best_tr:
                        best_tr, best_cfg = sc, (w_neu, w_sg, c_sk)
        chosen.append(best_cfg)
        nested_preds[va_idx] = full_pipeline(va_idx, best_cfg[0], best_cfg[1], best_cfg[2], 0.35)

    print(f"Nested-CV Full Pipeline BalAcc: {balanced_accuracy_score(y, nested_preds):.5f}")
    for i, c in enumerate(chosen):
        print(f"  Fold {i + 1} selected: w_neu={c[0]}, w_sg={c[1]}, c_sk={c[2]}")

    all_idx = np.arange(len(y))
    for w_neu in [(0.30, 0.30, 0.20, 0.15, 0.05), (0.30, 0.30, 0.20, 0.10, 0.10)]:
        for c_sk in [0.75, 1.00]:
            p = full_pipeline(all_idx, w_neu, 0.35, c_sk, 0.35)
            sc = balanced_accuracy_score(y, p)
            rec = recall_score(y, p, average=None)
            print(f"Global w_neu={w_neu} sg=0.35 c_sk={c_sk} g_sk=0.35 -> BalAcc={sc:.5f} | min_rec={rec.min():.3f} ({classes[rec.argmin()]})")


if __name__ == "__main__":
    main()
