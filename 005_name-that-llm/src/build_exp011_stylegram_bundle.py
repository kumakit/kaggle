"""
005_name-that-llm: Exp #011 StyleGram Bundle Builder
Exports the 5-Fold StratifiedGroupKFold (seed=42) Multi-View TF-IDF (char + word + shape + function)
+ group-centered LogisticRegression & LinearSVC weights to Kaggle Private Dataset.
"""

import json
import os
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

import sys
sys.path.insert(0, "005_name-that-llm/src")
from exp011_stylegram_cv import VIEWS, prepare_texts, sparse_group_centre, style_groups

SEED = 42
DATA_DIR = "005_name-that-llm/data"
BUNDLE_DIR = "005_name-that-llm/data/exp011_stylegram_bundle"
DATASET_SLUG = "nobuyukikumagai/name-that-llm-exp011-stylegram"
DATASET_TITLE = "Exp011 MultiView StyleGram Bundle"
assert 6 <= len(DATASET_TITLE) <= 50


def main():
    t0 = time.time()
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    texts = train_df["text"].fillna("").astype(str).tolist()
    classes = sorted(train_df["model"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y = train_df["model"].map(c2i).values
    groups = train_df["prompt_id"].values

    prepared = {v[0]: prepare_texts(texts, v[0]) for v in VIEWS}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_sg = np.zeros((len(y), len(classes)), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(texts, y, groups=groups)):
        f_dir = os.path.join(BUNDLE_DIR, f"fold_{fold}")
        os.makedirs(f_dir, exist_ok=True)

        vecs = []
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
            vecs.append({"view": vname, "weight": weight, "vec": vec})

        joblib.dump(vecs, os.path.join(f_dir, "vectorizers.joblib"), compress=3)
        X_tr = hstack(X_tr_parts, format="csr")
        X_va = hstack(X_va_parts, format="csr")

        X_tr_c = sparse_group_centre(X_tr, groups[tr_idx], lo=2, hi=18)
        g_va = style_groups([texts[i] for i in va_idx], thresh=0.90)
        X_va_c = sparse_group_centre(X_va, g_va, lo=2, hi=18)

        clf_lr = LogisticRegression(C=5.0, max_iter=1000, random_state=SEED)
        clf_lr.fit(X_tr_c, y[tr_idx])
        clf_svc = LinearSVC(C=0.4, random_state=SEED)
        clf_svc.fit(X_tr_c, y[tr_idx])

        np.savez_compressed(
            os.path.join(f_dir, "models.npz"),
            lr_coef=clf_lr.coef_.astype(np.float32),
            lr_intercept=clf_lr.intercept_.astype(np.float32),
            svc_coef=clf_svc.coef_.astype(np.float32),
            svc_intercept=clf_svc.intercept_.astype(np.float32),
        )

        z_lr = log_softmax(X_va_c @ clf_lr.coef_.T + clf_lr.intercept_, axis=1)
        z_svc = log_softmax(X_va_c @ clf_svc.coef_.T + clf_svc.intercept_, axis=1)
        oof_sg[va_idx] = log_softmax(0.6 * z_lr + 0.4 * z_svc, axis=1)
        print(f"  [FOLD {fold + 1}/5] Fold BalAcc={balanced_accuracy_score(y[va_idx], oof_sg[va_idx].argmax(1)):.5f}", flush=True)

    print(f"[OOF] Overall StyleGram BalAcc={balanced_accuracy_score(y, oof_sg.argmax(1)):.5f}", flush=True)

    with open(os.path.join(BUNDLE_DIR, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "n_folds": 5, "lr_weight": 0.6, "svc_weight": 0.4}, f, indent=2)
    with open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": DATASET_TITLE, "id": DATASET_SLUG, "licenses": [{"name": "CC-BY-SA-4.0"}]}, f, indent=2)

    check = subprocess.run(["uv", "run", "kaggle", "datasets", "status", DATASET_SLUG], capture_output=True, text=True)
    if check.returncode == 0:
        cmd = ["uv", "run", "kaggle", "datasets", "version", "-p", ".", "-m", "Exp011 StyleGram 5Fold bundle", "-r", "zip"]
    else:
        cmd = ["uv", "run", "kaggle", "datasets", "create", "-p", ".", "-r", "zip"]
    res = subprocess.run(cmd, cwd=BUNDLE_DIR, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[ERR]", res.stderr)
        raise RuntimeError("Kaggle dataset upload failed")
    print(f"[DONE] StyleGram bundle uploaded in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
