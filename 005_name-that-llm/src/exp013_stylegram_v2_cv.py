"""
005_name-that-llm: Exp #013 StyleGram v2 + Family Gate 5-Fold Local CV (Fast Vectorized Edition)
Fit TF-IDF once purely on clean external pool (~44.7k rows, strictly disjoint from train.csv)
to guarantee ZERO validation leakage and 5x faster 5-Fold CV & lightweight Kaggle bundle.
"""

import os
import re
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from scipy.special import log_softmax, logsumexp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

import sys
sys.path.insert(0, "005_name-that-llm/src")
from exp011_stylegram_cv import prepare_texts, sparse_group_centre, style_groups
from eval_exp012_blend import capped_agglo, cluster_sinkhorn, global_sinkhorn

SEED = 42
DATA_DIR = "005_name-that-llm/data"
EXTRA_SHARE = 0.50
EXTRA_MIN_LABELS = 12

VIEWS_V2 = [
    ("char", "char_wb", (3, 5), 180000, 1.0),
    ("word", "word", (1, 3), 120000, 0.7),
    ("shape", "char", (2, 6), 50000, 0.5),
    ("function", "word", (1, 3), 40000, 0.5),
]


def load_clean_extra(train_df, classes):
    tr_texts = set(train_df["text"].fillna("").astype(str).str.strip().tolist())

    op = pd.read_parquet(
        os.path.join(DATA_DIR, "open_pool/ntl_open_pool.parquet"),
        columns=["prompt_group", "model", "text", "flag_near_train_prompt", "flag_train_prompt_regen", "flag_kb_proxy_after_ruling"],
    )
    op = op[
        op["model"].isin(classes)
        & ~op["flag_near_train_prompt"].fillna(False)
        & ~op["flag_train_prompt_regen"].fillna(False)
        & ~op["flag_kb_proxy_after_ruling"].fillna(False)
    ].copy()
    op = op.rename(columns={"prompt_group": "prompt_id"})[["prompt_id", "model", "text"]]
    op["prompt_id"] = "op_" + op["prompt_id"].astype(str)

    pv = pd.read_parquet(os.path.join(DATA_DIR, "prvsiyan/pool_v1.parquet"), columns=["prompt_id", "model", "text"])
    pv["prompt_id"] = "pv_" + pv["prompt_id"].astype(str)

    cl = pd.read_csv(os.path.join(DATA_DIR, "extra_passages/ntl_extra_passages_claude.csv"))[["prompt_id", "model", "text"]]
    cl["prompt_id"] = "cl_" + cl["prompt_id"].astype(str)

    gp = pd.read_csv(os.path.join(DATA_DIR, "extra_passages/ntl_gpt_passages.csv"))[["prompt_id", "model", "text"]]
    gp["prompt_id"] = "gp_" + gp["prompt_id"].astype(str)

    x = pd.concat([pv, cl, gp, op], ignore_index=True)
    x = x[x["model"].isin(classes)].copy()
    x["text"] = x["text"].fillna("").astype(str).map(lambda t: re.sub(r"\s+", " ", t.strip()))
    x = x[~x["text"].isin(tr_texts)]
    x = x[x["text"].str.split().str.len().between(30, 350)].drop_duplicates(subset=["model", "text"])
    x = x.drop_duplicates(subset=["prompt_id", "model"])
    x = x[x.groupby("prompt_id")["model"].transform("nunique") >= EXTRA_MIN_LABELS].reset_index(drop=True)
    return x


def compute_sample_weights(y_tr, y_ex, k, share=EXTRA_SHARE):
    n_tr = np.bincount(y_tr, minlength=k).astype(np.float64)
    n_ex = np.bincount(y_ex, minlength=k).astype(np.float64)
    w_tr = np.where(n_ex[y_tr] > 0, 1.0 - share, 1.0)
    w_ex = share * n_tr[y_ex] / np.maximum(n_ex[y_ex], 1.0)
    return np.concatenate([w_tr, w_ex])


def main():
    t0 = time.time()
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
    print(f"[DATA] train={len(tr_texts):,} | clean extra={len(ex_texts):,} ({extra_df['prompt_id'].nunique()} prompts)", flush=True)

    t_vec = time.time()
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
        print(f"  [TF-IDF] {vname:8s} vocab={X_ex_v.shape[1]:,} ({time.time() - t_vec:.1f}s)", flush=True)

    X_tr_all = hstack(X_tr_parts, format="csr")
    X_ex_all = hstack(X_ex_parts, format="csr")
    X_ex_c = sparse_group_centre(X_ex_all, ex_groups, lo=2, hi=10**6)
    print(f"[TF-IDF DONE] X_tr={X_tr_all.shape} X_ex={X_ex_all.shape} in {time.time() - t_vec:.1f}s", flush=True)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_svc_plain = np.zeros((len(y), k), dtype=np.float64)
    oof_svc_hyb = np.zeros((len(y), k), dtype=np.float64)
    oof_sgd_hyb = np.zeros((len(y), k), dtype=np.float64)

    families = {f: np.array([i for i, c in enumerate(classes) if c.startswith(f)]) for f in ("claude", "gpt", "gemini", "glm")}
    oof_gates = {f: np.zeros((len(y), len(idxs)), dtype=np.float64) for f, idxs in families.items()}

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(tr_texts, y, groups=groups)):
        t_f = time.time()
        g_va = style_groups([tr_texts[i] for i in va_idx], thresh=0.90)
        _, inv_va, cnt_va = np.unique(g_va, return_inverse=True, return_counts=True)
        in_clust = (cnt_va[inv_va] >= 2) & (cnt_va[inv_va] <= 18)

        X_tr = X_tr_all[tr_idx]
        X_va = X_tr_all[va_idx]
        X_va_c = sparse_group_centre(X_va, g_va, lo=2, hi=18)

        sw = compute_sample_weights(y[tr_idx], ex_y, k, share=EXTRA_SHARE)
        y_comb = np.concatenate([y[tr_idx], ex_y])

        X_comb_plain = vstack([X_tr, X_ex_all]).tocsr()
        svc_p = LinearSVC(C=0.5, tol=1e-3, random_state=SEED, max_iter=2000).fit(X_comb_plain, y_comb, sample_weight=sw)
        sgd_p = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=35, tol=1e-3, n_jobs=-1, random_state=SEED).fit(X_comb_plain, y_comb, sample_weight=sw)

        X_tr_c = sparse_group_centre(X_tr, groups[tr_idx], lo=2, hi=18)
        X_comb_c = vstack([X_tr_c, X_ex_c]).tocsr()
        svc_c = LinearSVC(C=0.5, tol=1e-3, random_state=SEED, max_iter=2000).fit(X_comb_c, y_comb, sample_weight=sw)
        sgd_c = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=35, tol=1e-3, n_jobs=-1, random_state=SEED).fit(X_comb_c, y_comb, sample_weight=sw)

        lp_svc_p = log_softmax(svc_p.decision_function(X_va), axis=1)
        lp_svc_c = log_softmax(svc_c.decision_function(X_va_c), axis=1)
        lp_sgd_p = sgd_p.predict_log_proba(X_va)
        lp_sgd_c = sgd_c.predict_log_proba(X_va_c)

        oof_svc_plain[va_idx] = lp_svc_p
        oof_svc_hyb[va_idx] = np.where(in_clust[:, None], lp_svc_c, lp_svc_p)
        oof_sgd_hyb[va_idx] = np.where(in_clust[:, None], lp_sgd_c, lp_sgd_p)

        for fam, idxs in families.items():
            m_tr = np.isin(y[tr_idx], idxs)
            m_ex = np.isin(ex_y, idxs)
            X_f = vstack([X_tr[m_tr], X_ex_all[m_ex]]).tocsr()
            y_f = np.concatenate([y[tr_idx][m_tr], ex_y[m_ex]])
            w_f = np.concatenate([np.ones(int(m_tr.sum())), np.full(int(m_ex.sum()), 0.25)])
            C_f = 1.0 if fam in ("gpt", "gemini") else 0.5
            g_svc = LinearSVC(C=C_f, class_weight="balanced", tol=1e-3, random_state=SEED, max_iter=2000).fit(X_f, y_f, sample_weight=w_f)
            d = g_svc.decision_function(X_va)
            if len(idxs) == 2:
                d = np.column_stack([-d, d])
            d = d - d.mean(axis=1, keepdims=True)
            sc = np.sqrt(np.mean(d * d, axis=1, keepdims=True))
            oof_gates[fam][va_idx] = log_softmax(d / np.maximum(sc, 0.25), axis=1)

        fold_hyb = log_softmax(0.75 * oof_svc_hyb[va_idx] + 0.25 * oof_sgd_hyb[va_idx], axis=1)
        print(
            f"[Fold {fold + 1}/5] ({time.time() - t_f:.1f}s) in_cluster={int(in_clust.sum())}/{len(va_idx)} | "
            f"svc_plain={balanced_accuracy_score(y[va_idx], oof_svc_plain[va_idx].argmax(1)):.5f} | "
            f"svc_hyb={balanced_accuracy_score(y[va_idx], oof_svc_hyb[va_idx].argmax(1)):.5f} | "
            f"hyb_blend={balanced_accuracy_score(y[va_idx], fold_hyb.argmax(1)):.5f}",
            flush=True,
        )

    oof_v2_svc = oof_svc_hyb
    oof_v2_blend = log_softmax(0.75 * oof_svc_hyb + 0.25 * oof_sgd_hyb, axis=1)
    print("\n=== StyleGram v2 Single OOF Summary ===", flush=True)
    print(f"  svc_plain       : {balanced_accuracy_score(y, oof_svc_plain.argmax(1)):.5f}", flush=True)
    print(f"  svc_hyb         : {balanced_accuracy_score(y, oof_svc_hyb.argmax(1)):.5f}", flush=True)
    print(f"  sgd_hyb         : {balanced_accuracy_score(y, oof_sgd_hyb.argmax(1)):.5f}", flush=True)
    print(f"  0.75SVC+0.25SGD : {balanced_accuracy_score(y, oof_v2_blend.argmax(1)):.5f}", flush=True)

    oof_v2 = oof_v2_svc if balanced_accuracy_score(y, oof_v2_svc.argmax(1)) >= balanced_accuracy_score(y, oof_v2_blend.argmax(1)) else oof_v2_blend
    oof_df = pd.DataFrame(np.exp(oof_v2), columns=classes)
    oof_df["id"] = train_df["id"]
    oof_df.to_csv(os.path.join(DATA_DIR, "oof_exp013_stylegram_v2.csv"), index=False)
    np.savez_compressed(os.path.join(DATA_DIR, "tmp_exp012/oof_exp013_gates.npz"), **oof_gates)

    g_cache = os.path.join(DATA_DIR, "tmp_exp012/capped_agglo_G.npy")
    G = np.load(g_cache)

    eps = 1e-7
    l5, l6, l7, l8, l9, l12, l11, ll = [
        np.log(np.clip(pd.read_csv(os.path.join(DATA_DIR, fn))[classes].values, eps, 1.0))
        for fn in [
            "oof_exp005_openpool_modernbert.csv", "oof_exp006_qwen3_lora.csv",
            "oof_exp007_qwen3_4b_lora.csv", "oof_exp008_modernbert_large.csv",
            "oof_exp009_qwen3_8b_lora.csv", "oof_exp012_deberta_large.csv",
            "oof_exp011_stylegram.csv", "oof_exp010_glm_blend.csv",
        ]
    ]
    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]

    def apply_gates(logp, lam=0.35):
        out = logp.copy()
        for fam, idxs in families.items():
            base = out[:, idxs]
            base_lp = log_softmax(base, axis=1)
            base_p = np.exp(base_lp)
            sp = np.sort(base_p, axis=1)
            margin = sp[:, -1] - sp[:, -2]
            conf_gate = np.clip((0.45 - margin) / 0.45, 0.0, 1.0)
            strength = min(0.45, lam / (1.0 + lam)) * conf_gate
            mixed_lp = log_softmax((1.0 - strength[:, None]) * base_lp + strength[:, None] * oof_gates[fam], axis=1)
            fam_mass = logsumexp(base, axis=1)
            out[:, idxs] = fam_mass[:, None] + mixed_lp
        return out

    def eval_pipe(neu_lp, sg_lp, w_sg, c_sk=0.65, g_sk=0.35, use_gate=False, lam=0.35, cov=0.30):
        bl = log_softmax(neu_lp + w_sg * sg_lp, axis=1)
        if use_gate:
            bl = apply_gates(bl, lam=lam)
        if c_sk > 0:
            bl = cluster_sinkhorn(bl, G, 3, c_sk, 6, 18)
        if g_sk > 0:
            bl = global_sinkhorn(bl, g_sk)
        glm_mass = np.exp(bl)[:, glm_idxs].sum(1)
        sel = np.sort(np.argsort(-glm_mass, kind="stable")[: int(round(cov * len(bl)))])
        before = bl.argmax(1)
        after = before.copy()
        after[sel] = (bl[sel] + 1.15 * ll[sel]).argmax(1)
        glm_rel = np.isin(before, glm_idxs) | np.isin(after, glm_idxs)
        after[(before != after) & ~glm_rel] = before[(before != after) & ~glm_rel]
        return balanced_accuracy_score(y, after), recall_score(y, after, average=None)

    neu_12 = 0.30 * l7 + 0.30 * l8 + 0.20 * l12 + 0.15 * l6 + 0.05 * l5
    neu_13 = 0.20 * l9 + 0.24 * l7 + 0.24 * l8 + 0.16 * l12 + 0.12 * l6 + 0.04 * l5

    print("\n=== Ensemble Impact (Without vs With StyleGram v2, Family Gate, Qwen3-8B) ===", flush=True)
    for name, neu, sg, w_sg, c_sk, gate in [
        ("Exp012 baseline (SGv1 w=0.35, c_sk=0.85)", neu_12, l11, 0.35, 0.85, False),
        ("Exp012 + SGv2 w=0.35, c_sk=0.65", neu_12, oof_v2, 0.35, 0.65, False),
        ("Exp012 + SGv2 w=0.50, c_sk=0.65", neu_12, oof_v2, 0.50, 0.65, False),
        ("Exp012 + SGv2 w=0.65, c_sk=0.65", neu_12, oof_v2, 0.65, 0.65, False),
        ("Exp012 + SGv2 w=0.80, c_sk=0.65", neu_12, oof_v2, 0.80, 0.65, False),
        ("Exp012 + SGv2 w=0.50 + Gates, c_sk=0.65", neu_12, oof_v2, 0.50, 0.65, True),
        ("Exp013 (+Qwen3-8B + SGv2 w=0.50, c_sk=0.65)", neu_13, oof_v2, 0.50, 0.65, False),
        ("Exp013 (+Qwen3-8B + SGv2 w=0.65, c_sk=0.65)", neu_13, oof_v2, 0.65, 0.65, False),
        ("Exp013 (+Qwen3-8B + SGv2 w=0.65 + Gates, c_sk=0.65)", neu_13, oof_v2, 0.65, 0.65, True),
        ("Exp013 (+Qwen3-8B + SGv2 w=0.65 + Gates, c_sk=0.00)", neu_13, oof_v2, 0.65, 0.00, True),
    ]:
        sc, rec = eval_pipe(neu, sg, w_sg, c_sk=c_sk, use_gate=gate)
        print(f"  {name:55s} -> BalAcc={sc:.5f} | min_rec={rec.min():.3f} ({classes[rec.argmin()]})", flush=True)

    print(f"\n[DONE] Finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
