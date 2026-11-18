"""
005_name-that-llm: Exp #013 StyleGram v2 + Family Gates Bundle Builder
Builds & uploads Kaggle Private Dataset: nobuyukikumagai/name-that-llm-exp013-stylegram-v2
Contains:
  - vectorizers.joblib: 4-view TF-IDF (char + word + shape + function) fit cleanly on 44.7k external pool
  - fold_0..4/models.npz: svc_plain, svc_centred, and 4 family gate LinearSVC weights
  - bundle_meta.json: canonical class order and metadata
"""

import json
import os
import subprocess
import sys
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, "005_name-that-llm/src")
from exp011_stylegram_cv import prepare_texts, sparse_group_centre, style_groups
from exp013_stylegram_v2_cv import EXTRA_SHARE, VIEWS_V2, compute_sample_weights, load_clean_extra

SEED = 42
DATA_DIR = "005_name-that-llm/data"
BUNDLE_DIR = "005_name-that-llm/data/exp013_stylegram_v2_bundle"
DATASET_SLUG = "nobuyukikumagai/name-that-llm-exp013-stylegram-v2"
DATASET_TITLE = "Exp013 StyleGram v2 and Family Gates"
assert 6 <= len(DATASET_TITLE) <= 50


def main():
    t0 = time.time()
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    classes = sorted(train_df["model"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    y = train_df["model"].map(c2i).values
    groups = train_df["prompt_id"].values
    tr_texts = train_df["text"].fillna("").astype(str).tolist()

    extra_df = load_clean_extra(train_df, classes)
    ex_texts = extra_df["text"].tolist()
    ex_y = extra_df["model"].map(c2i).values
    ex_groups = extra_df["prompt_id"].values
    print(f"[DATA] train={len(tr_texts):,} | extra={len(ex_texts):,}", flush=True)

    t_vec = time.time()
    vecs = []
    X_tr_parts, X_ex_parts = [], []
    for vname, analyzer, ngram_range, max_feat, weight in VIEWS_V2:
        prep_ex_v = prepare_texts(ex_texts, vname)
        prep_tr_v = prepare_texts(tr_texts, vname)
        kw = dict(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_feat,
            sublinear_tf=True,
            min_df=3,
            lowercase=False,
            dtype=np.float32,
        )
        if analyzer == "word":
            kw["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**kw)
        X_ex_v = vec.fit_transform(prep_ex_v) * weight
        X_tr_v = vec.transform(prep_tr_v) * weight
        X_ex_parts.append(X_ex_v)
        X_tr_parts.append(X_tr_v)
        vecs.append({"view": vname, "weight": weight, "vec": vec})

    joblib.dump(vecs, os.path.join(BUNDLE_DIR, "vectorizers.joblib"), compress=3)
    X_tr_all = hstack(X_tr_parts, format="csr")
    X_ex_all = hstack(X_ex_parts, format="csr")
    X_ex_c = sparse_group_centre(X_ex_all, ex_groups, lo=2, hi=10**6)
    print(f"[VECTORIZERS] Saved vectorizers.joblib ({time.time() - t_vec:.1f}s), dim={X_tr_all.shape[1]:,}", flush=True)

    families = {f: np.array([i for i, c in enumerate(classes) if c.startswith(f)]) for f in ("claude", "gpt", "gemini", "glm")}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_hyb = np.zeros((len(y), k), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(tr_texts, y, groups=groups)):
        t_f = time.time()
        f_dir = os.path.join(BUNDLE_DIR, f"fold_{fold}")
        os.makedirs(f_dir, exist_ok=True)

        g_va = style_groups([tr_texts[i] for i in va_idx], thresh=0.90)
        _, inv_va, cnt_va = np.unique(g_va, return_inverse=True, return_counts=True)
        in_clust = (cnt_va[inv_va] >= 2) & (cnt_va[inv_va] <= 18)

        X_tr = X_tr_all[tr_idx]
        X_va = X_tr_all[va_idx]
        X_va_c = sparse_group_centre(X_va, g_va, lo=2, hi=18)

        sw = compute_sample_weights(y[tr_idx], ex_y, k, share=EXTRA_SHARE)
        y_comb = np.concatenate([y[tr_idx], ex_y])

        X_comb_p = vstack([X_tr, X_ex_all]).tocsr()
        svc_p = LinearSVC(C=0.5, tol=1e-2, random_state=SEED, max_iter=2000).fit(X_comb_p, y_comb, sample_weight=sw)

        X_tr_c = sparse_group_centre(X_tr, groups[tr_idx], lo=2, hi=18)
        X_comb_c = vstack([X_tr_c, X_ex_c]).tocsr()
        svc_c = LinearSVC(C=0.5, tol=1e-2, random_state=SEED, max_iter=2000).fit(X_comb_c, y_comb, sample_weight=sw)

        lp_p = log_softmax(svc_p.decision_function(X_va), axis=1)
        lp_c = log_softmax(svc_c.decision_function(X_va_c), axis=1)
        oof_hyb[va_idx] = np.where(in_clust[:, None], lp_c, lp_p)

        save_dict = {
            "svc_p_coef": svc_p.coef_.astype(np.float32),
            "svc_p_intercept": svc_p.intercept_.astype(np.float32),
            "svc_c_coef": svc_c.coef_.astype(np.float32),
            "svc_c_intercept": svc_c.intercept_.astype(np.float32),
        }

        for fam, idxs in families.items():
            m_tr = np.isin(y[tr_idx], idxs)
            m_ex = np.isin(ex_y, idxs)
            X_f = vstack([X_tr[m_tr], X_ex_all[m_ex]]).tocsr()
            y_f = np.concatenate([y[tr_idx][m_tr], ex_y[m_ex]])
            w_f = np.concatenate([np.ones(int(m_tr.sum())), np.full(int(m_ex.sum()), 0.25)])
            C_f = 1.0 if fam in ("gpt", "gemini") else 0.5
            g_svc = LinearSVC(C=C_f, class_weight="balanced", tol=1e-2, random_state=SEED, max_iter=2000).fit(X_f, y_f, sample_weight=w_f)
            save_dict[f"gate_{fam}_coef"] = g_svc.coef_.astype(np.float32)
            save_dict[f"gate_{fam}_intercept"] = np.atleast_1d(g_svc.intercept_).astype(np.float32)

        np.savez_compressed(os.path.join(f_dir, "models.npz"), **save_dict)
        print(
            f"  [FOLD {fold + 1}/5] ({time.time() - t_f:.1f}s) Hyb BalAcc={balanced_accuracy_score(y[va_idx], oof_hyb[va_idx].argmax(1)):.5f}",
            flush=True,
        )

    oof_sc = balanced_accuracy_score(y, oof_hyb.argmax(1))
    print(f"[OOF] Overall StyleGram v2 Bundle BalAcc={oof_sc:.5f}", flush=True)

    meta = {
        "classes": classes,
        "n_folds": 5,
        "families": {f: idxs.tolist() for f, idxs in families.items()},
        "oof_bal_acc": float(oof_sc),
    }
    with open(os.path.join(BUNDLE_DIR, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": DATASET_TITLE, "id": DATASET_SLUG, "licenses": [{"name": "CC-BY-SA-4.0"}]}, f, indent=2)

    check = subprocess.run(["uv", "run", "kaggle", "datasets", "status", DATASET_SLUG], capture_output=True, text=True)
    if check.returncode == 0:
        cmd = ["uv", "run", "kaggle", "datasets", "version", "-p", ".", "-m", f"Exp013 SGv2 OOF {oof_sc:.5f}", "-r", "zip"]
    else:
        cmd = ["uv", "run", "kaggle", "datasets", "create", "-p", ".", "-r", "zip"]
    res = subprocess.run(cmd, cwd=BUNDLE_DIR, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.returncode != 0:
        print("[ERR]", res.stderr, flush=True)
        raise RuntimeError("Kaggle dataset upload failed")
    print(f"[DONE] StyleGram v2 bundle uploaded in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
