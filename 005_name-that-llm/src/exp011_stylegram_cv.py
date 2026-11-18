"""
005_name-that-llm: Exp #011 Local 5-Fold CV Experiment
Multi-View StyleGram (char + word + shape + function) + style_groups Centering
+ Cluster Log-Prob Centering / Sinkhorn check on Exp #008 & Exp #010 (GLM-5.3/Flash)!
Zero test leakage: style_groups() is computed independently inside each val fold without prompt_id.
"""

import os
import re
import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags, hstack
from scipy.spatial.distance import squareform
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

SEED = 42
DATA_DIR = "005_name-that-llm/data"

FUNCTION_WORDS = frozenset(
    "a an the and or but if while although because as so than that which who whom whose "
    "this these those it its they them their theirs he him his she her hers we us our ours "
    "you your yours i me my mine in on at to from with without for of by over under between "
    "through into about is are was were be been being have has had do does did can could will "
    "would shall should may might must not no nor all any some each every both either neither "
    "other another such only also just very too more most less least here there when where why "
    "how what whether then now still yet even much many few little".split()
)

VIEWS = [
    ("char", "char_wb", (3, 5), 180000, 1.0),
    ("word", "word", (1, 3), 120000, 0.7),
    ("shape", "char", (2, 6), 50000, 0.5),
    ("function", "word", (1, 3), 40000, 0.5),
]


def shape_view(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_view(text):
    return " ".join(
        t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
        for t in re.findall(r"\w+|[^\w\s]", text)
    )


def prepare_texts(texts, view):
    if view == "shape":
        return [shape_view(t) for t in texts]
    if view == "function":
        return [function_view(t) for t in texts]
    return texts


def style_groups(texts, thresh=0.90):
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=100000, dtype=np.float32)
    X = vec.fit_transform(texts)
    D = np.clip(1.0 - (X @ X.T).toarray(), 0.0, 1.0)
    med = max(float(np.median(D[np.triu_indices(len(texts), 1)])), 1e-6)
    D = np.asarray(D / med, dtype=np.float64).copy()
    np.fill_diagonal(D, 0.0)
    return fcluster(linkage(squareform(D, checks=False), method="average"), thresh, criterion="distance")


def sparse_group_centre(X, groups, lo=2, hi=18):
    X = csr_matrix(X)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep):
        return X.copy()
    col = np.full(len(cnt), -1)
    col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows)), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)))
    means = diags(1.0 / cnt[keep]) @ (M.T @ X)
    return (X - M @ means).tocsr()


def apply_cluster_sinkhorn(logp, groups, n_iters=3, alpha=0.5, lo=6, hi=18):
    out = logp.copy()
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    for g_id in np.flatnonzero((cnt >= lo) & (cnt <= hi)):
        idx = np.flatnonzero(inv == g_id)
        z = out[idx].copy()
        for _ in range(n_iters):
            z = log_softmax(z, axis=1)
            col_mean = z.mean(axis=0, keepdims=True)
            z = z - alpha * col_mean
        out[idx] = log_softmax(z, axis=1)
    return out


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    texts = train_df["text"].fillna("").astype(str).tolist()
    classes = sorted(train_df["model"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y = train_df["model"].map(c2i).values
    groups = train_df["prompt_id"].values

    # Precompute prepared text views once
    prepared = {v[0]: prepare_texts(texts, v[0]) for v in VIEWS}

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_raw = np.zeros((len(y), len(classes)), dtype=np.float64)
    oof_ctr = np.zeros((len(y), len(classes)), dtype=np.float64)
    oof_svc_ctr = np.zeros((len(y), len(classes)), dtype=np.float64)
    val_style_clusters = np.zeros(len(y), dtype=np.int64)

    cluster_offset = 0
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(texts, y, groups=groups)):
        t_f = time.time()
        g_tr_true = groups[tr_idx]
        g_va_style = style_groups([texts[i] for i in va_idx], thresh=0.90)
        val_style_clusters[va_idx] = g_va_style + cluster_offset
        cluster_offset += int(g_va_style.max()) + 1

        # Check purity / size of discovered style_groups on val fold
        _, cnt_s = np.unique(g_va_style, return_counts=True)
        in_range = int(((cnt_s >= 2) & (cnt_s <= 18)).sum())

        X_tr_parts, X_va_parts = [], []
        for vname, analyzer, ngram_range, max_feat, weight in VIEWS:
            kw = dict(
                analyzer=analyzer,
                ngram_range=ngram_range,
                max_features=max_feat,
                sublinear_tf=True,
                dtype=np.float32,
            )
            if vname == "shape":
                kw["lowercase"] = False
            vec = TfidfVectorizer(**kw)
            X_tr_v = vec.fit_transform([prepared[vname][i] for i in tr_idx]) * weight
            X_va_v = vec.transform([prepared[vname][i] for i in va_idx]) * weight
            X_tr_parts.append(X_tr_v)
            X_va_parts.append(X_va_v)

        X_tr = hstack(X_tr_parts, format="csr")
        X_va = hstack(X_va_parts, format="csr")

        # 1) Uncentered LogReg baseline
        clf_raw = LogisticRegression(C=3.0, max_iter=1000, random_state=SEED)
        clf_raw.fit(X_tr, y[tr_idx])
        oof_raw[va_idx] = clf_raw.predict_log_proba(X_va)

        # 2) Group-Centered LogReg (Train uses prompt_id or style_groups; Val uses strictly text-only style_groups!)
        X_tr_c = sparse_group_centre(X_tr, g_tr_true, lo=2, hi=18)
        X_va_c = sparse_group_centre(X_va, g_va_style, lo=2, hi=18)
        clf_ctr = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
        clf_ctr.fit(X_tr_c, y[tr_idx])
        oof_ctr[va_idx] = clf_ctr.predict_log_proba(X_va_c)

        # 3) Group-Centered LinearSVC
        clf_svc = LinearSVC(C=0.4, random_state=SEED)
        clf_svc.fit(X_tr_c, y[tr_idx])
        oof_svc_ctr[va_idx] = log_softmax(clf_svc.decision_function(X_va_c), axis=1)

        print(
            f"[Fold {fold + 1}/5] ({time.time() - t_f:.1f}s) style_clusters_in_2_18={in_range}/{len(cnt_s)} | "
            f"raw={balanced_accuracy_score(y[va_idx], oof_raw[va_idx].argmax(1)):.5f} | "
            f"ctr_LR={balanced_accuracy_score(y[va_idx], oof_ctr[va_idx].argmax(1)):.5f} | "
            f"ctr_SVC={balanced_accuracy_score(y[va_idx], oof_svc_ctr[va_idx].argmax(1)):.5f}",
            flush=True,
        )

    sc_raw = balanced_accuracy_score(y, oof_raw.argmax(1))
    sc_ctr = balanced_accuracy_score(y, oof_ctr.argmax(1))
    sc_svc = balanced_accuracy_score(y, oof_svc_ctr.argmax(1))
    oof_sg = log_softmax(0.6 * oof_ctr + 0.4 * oof_svc_ctr, axis=1)
    sc_sg = balanced_accuracy_score(y, oof_sg.argmax(1))
    print(f"\n[StyleGram OOF Summary] raw={sc_raw:.5f} | centered_LR={sc_ctr:.5f} | centered_SVC={sc_svc:.5f} | LR+SVC={sc_sg:.5f}")

    # Load Exp #008 base & Exp #010 GLM LL OOF
    eps = 1e-7
    o5 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp005_openpool_modernbert.csv"))[classes].values
    o6 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp006_qwen3_lora.csv"))[classes].values
    o7 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp007_qwen3_4b_lora.csv"))[classes].values
    o8 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp008_modernbert_large.csv"))[classes].values
    base_lp = log_softmax(
        0.25 * np.log(o7 + eps) + 0.40 * np.log(o8 + eps) + 0.30 * np.log(o6 + eps) + 0.05 * np.log(o5 + eps),
        axis=1,
    )
    o10 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp010_glm_blend.csv"))
    ll_lp = np.log(np.clip(o10[classes].values, eps, 1.0))

    print("\n=== Correlation with Exp #008 Base ===")
    corr = np.corrcoef(np.exp(base_lp).ravel(), np.exp(oof_sg).ravel())[0, 1]
    print(f"StyleGram LR+SVC vs Exp008 prob_corr={corr:.4f}")

    print("\n=== Pre-GLM Blend (Exp008 Base + StyleGram) ===")
    for w_sg in [0.0, 0.15, 0.25, 0.35, 0.50, 0.65]:
        bl = log_softmax(base_lp + w_sg * oof_sg, axis=1)
        print(f"  w_sg={w_sg:.2f} -> Pre-GLM BalAcc={balanced_accuracy_score(y, bl.argmax(1)):.5f}")

    print("\n=== Cluster Sinkhorn on Pre-GLM Blend (style_groups, zero leakage) ===")
    for alpha in [0.0, 0.25, 0.50, 0.75]:
        bl = log_softmax(base_lp + 0.35 * oof_sg, axis=1)
        bl_sh = apply_cluster_sinkhorn(bl, val_style_clusters, n_iters=3, alpha=alpha, lo=6, hi=18)
        print(f"  w_sg=0.35, sinkhorn alpha={alpha:.2f} -> Pre-GLM BalAcc={balanced_accuracy_score(y, bl_sh.argmax(1)):.5f}")

    print("\n=== Post-GLM (+Exp010 GLM stage cov=0.30, w=1.15, restrict=True) ===")
    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]
    for w_sg in [0.0, 0.25, 0.35, 0.50]:
        for alpha in [0.0, 0.35, 0.50]:
            bl = log_softmax(base_lp + w_sg * oof_sg, axis=1)
            if alpha > 0:
                bl = apply_cluster_sinkhorn(bl, val_style_clusters, n_iters=3, alpha=alpha, lo=6, hi=18)
            glm_mass = np.exp(bl)[:, glm_idxs].sum(1)
            sel = np.sort(np.argsort(-glm_mass, kind="stable")[: int(round(0.30 * len(y)))])
            before = bl.argmax(1)
            after = before.copy()
            after[sel] = (bl[sel] + 1.15 * ll_lp[sel]).argmax(1)
            glm_rel = np.isin(before, glm_idxs) | np.isin(after, glm_idxs)
            after[(before != after) & ~glm_rel] = before[(before != after) & ~glm_rel]
            print(f"  w_sg={w_sg:.2f}, sinkhorn={alpha:.2f} -> Exp011+GLM BalAcc={balanced_accuracy_score(y, after):.5f}")

    oof_df = pd.DataFrame(np.exp(oof_sg), columns=classes)
    oof_df["id"] = train_df["id"]
    oof_df.to_csv(os.path.join(DATA_DIR, "oof_exp011_stylegram.csv"), index=False)
    print(f"\n[DONE] Saved oof_exp011_stylegram.csv in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
