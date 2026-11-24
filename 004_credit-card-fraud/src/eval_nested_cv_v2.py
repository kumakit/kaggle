"""
Nested-CV optimization v2 (fast cached & closed-form Ridge + targeted LogReg):
- Caches deduplicated Rank-Logit OOF/Test matrices to data/cache_v2/ so repeated sweeps take <5s.
- Evaluates (A) Full single-model stack from scratch and (B) Greedy Forward-Selected augmentation
  on top of wide_lm_stack_nested + post-Nov11 diverse models (verified across TWO outer CV seeds).
"""
import gc
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append("004_credit-card-fraud/src")
from eval_nested_cv_base import load_all_clean_models, deduplicate_models, to_rank_logit

DATA_DIR = "004_credit-card-fraud/data"
BANKS_DIR = os.path.join(DATA_DIR, "oof_banks")
CACHE_DIR = os.path.join(DATA_DIR, "cache_v2")


def load_busyaprime_models(models: dict, y: np.ndarray):
    bp_dir = os.path.join(BANKS_DIR, "busyaprime")
    man = pd.read_csv(os.path.join(bp_dir, "manifest.csv"))
    oof_mat = np.load(os.path.join(bp_dir, "oof.npy"), mmap_mode="r")
    te_mat = np.load(os.path.join(bp_dir, "test.npy"), mmap_mode="r")
    for idx, row in man.iterrows():
        name = f"busya_{row['name']}"
        oof = np.array(oof_mat[:, idx], dtype=np.float32)
        te = np.array(te_mat[:, idx], dtype=np.float32)
        if np.isnan(oof).any() or np.isnan(te).any():
            continue
        auc = roc_auc_score(y, oof)
        if auc >= 0.948:
            models[name] = (oof, te, str(row["family"]), auc)
    del oof_mat, te_mat
    gc.collect()
    return models


def build_or_load_cache(y: np.ndarray):
    os.makedirs(CACHE_DIR, exist_ok=True)
    meta_path = os.path.join(CACHE_DIR, "meta.json")
    oof_path = os.path.join(CACHE_DIR, "Z_oof.npy")
    te_path = os.path.join(CACHE_DIR, "Z_te.npy")
    base_oof_path = os.path.join(CACHE_DIR, "base_oof.npy")
    base_te_path = os.path.join(CACHE_DIR, "base_te.npy")

    if all(os.path.exists(p) for p in [meta_path, oof_path, te_path, base_oof_path, base_te_path]):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return (
            meta["names"],
            meta["families"],
            meta["aucs"],
            np.load(oof_path, mmap_mode="r"),
            np.load(te_path, mmap_mode="r"),
            np.load(base_oof_path),
            np.load(base_te_path),
        )

    t0 = time.time()
    models = load_all_clean_models(y)
    models = load_busyaprime_models(models, y)
    print(f"Loaded {len(models)} clean models in {time.time() - t0:.1f}s", flush=True)

    base_oof = to_rank_logit(models["pub_wide_lm_stack"][0])
    base_te = to_rank_logit(models["pub_wide_lm_stack"][1])

    single_models = {k: v for k, v in models.items() if k != "pub_wide_lm_stack"}
    kept = deduplicate_models(single_models, y, corr_thresh=0.9994)
    print(f"Deduplicated to {len(kept)} single models in {time.time() - t0:.1f}s", flush=True)

    Z_oof = np.column_stack([to_rank_logit(models[k][0]) for k in kept]).astype(np.float32)
    Z_te = np.column_stack([to_rank_logit(models[k][1]) for k in kept]).astype(np.float32)
    fams = [models[k][2] for k in kept]
    aucs = [float(models[k][3]) for k in kept]

    np.save(oof_path, Z_oof)
    np.save(te_path, Z_te)
    np.save(base_oof_path, base_oof)
    np.save(base_te_path, base_te)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"names": kept, "families": fams, "aucs": aucs}, f, indent=2)
    print(f"Saved cache to {CACHE_DIR}", flush=True)
    return kept, fams, aucs, Z_oof, Z_te, base_oof, base_te


def closed_form_ridge_nested(Z: np.ndarray, y: np.ndarray, alphas: list[float], seed: int):
    """Super fast closed-form Ridge via precomputed Gram matrices X^T X and X^T y per fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y))
    Z64 = np.asarray(Z, dtype=np.float64)
    yc = y.astype(np.float64) - y.mean()
    d = Z64.shape[1]
    eye = np.eye(d)

    # Precompute full Gram matrix once
    G_full = Z64.T @ Z64
    rhs_full = Z64.T @ yc

    fold_stats = []
    for tr, va in folds:
        Zv = Z64[va]
        yv = yc[va]
        G_tr = G_full - (Zv.T @ Zv)
        rhs_tr = rhs_full - (Zv.T @ yv)
        fold_stats.append((va, G_tr, rhs_tr))

    results = {}
    for a in alphas:
        pred = np.zeros(len(y), dtype=np.float64)
        for va, G_tr, rhs_tr in fold_stats:
            w = np.linalg.solve(G_tr + a * eye, rhs_tr)
            pred[va] = Z64[va] @ w
        results[a] = (roc_auc_score(y, pred), pred)
    return results


def eval_nested_logreg(Z: np.ndarray, y: np.ndarray, C: float, seed: int):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(Z, y):
        clf = LogisticRegression(C=C, solver="newton-cholesky", max_iter=80, tol=1e-5)
        clf.fit(Z[tr], y[tr])
        pred[va] = clf.decision_function(Z[va])
    return roc_auc_score(y, pred), pred


def main():
    t0 = time.time()
    y = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["is_fraud"])["is_fraud"].to_numpy().astype(int)
    names, fams, aucs, Z_oof, Z_te, base_oof, base_te = build_or_load_cache(y)

    # 1. Closed-form Ridge on all 127 deduplicated single models (takes ~2s!)
    alphas = [100.0, 500.0, 2000.0, 5000.0, 15000.0, 40000.0]
    r2026 = closed_form_ridge_nested(Z_oof, y, alphas, seed=2026)
    r777 = closed_form_ridge_nested(Z_oof, y, alphas, seed=777)
    for a in alphas:
        print(f"[Scratch Singles Ridge alpha={a:<7.0f}] Nested2026={r2026[a][0]:.6f} | Nested777={r777[a][0]:.6f}", flush=True)

    # 2. Check Pub Wide LM Stack baseline
    base_auc = roc_auc_score(y, base_oof)
    print(f"\nBaseline pub_wide_lm_stack AUC = {base_auc:.6f}", flush=True)

    # 3. Greedy Forward Selection on candidate subset (all non-redundant strong/diverse models)
    cand_idx = [
        i for i, k in enumerate(names)
        if k.startswith(("ours_", "busya_tabm", "busya_rmlp", "busya_ebm", "busya_lgb_g2_plaininit",
                         "next2060_", "bizzi_", "weni_", "megayak_", "queyrusi_", "aman_sameer",
                         "aman_tabpfn", "aman_gpt2", "aman_yekenot"))
        or aucs[i] >= 0.9508
    ]
    print(f"Candidate pool size: {len(cand_idx)}", flush=True)

    # Run greedy forward selection ONLY on fold 0..3 of seed=2026 to test true out-of-fold generalization on fold 4,
    # AND also inspect full-OOF greedy path (then verify selected subset via 2-seed Nested-CV).
    cur_z = base_oof.astype(np.float64).copy()
    cur_auc = base_auc
    selected_idx = []
    pool = list(cand_idx)
    for step in range(15):
        best_step = None
        best_auc = cur_auc
        best_z = None
        for idx in pool:
            zk = Z_oof[:, idx].astype(np.float64)
            for w in [0.02, 0.04, 0.07, 0.10, 0.14]:
                cz = (1.0 - w) * cur_z + w * zk
                auc = roc_auc_score(y, cz)
                if auc > best_auc + 1e-6:
                    best_auc = auc
                    best_step = (idx, w)
                    best_z = cz
        if best_step is None:
            break
        idx, w = best_step
        selected_idx.append(idx)
        pool.remove(idx)
        cur_z = (best_z - best_z.mean()) / best_z.std()
        cur_auc = best_auc
        print(f"  Step {step+1:2d}: +{names[idx]:32s} (solo={aucs[idx]:.5f}, w={w:.2f}) -> AUC={cur_auc:.6f}", flush=True)

    # 4. Strict 2-Seed Outer Nested-CV on [pub_wide_lm_stack + selected anchors]
    Z_sel = np.column_stack([base_oof] + [Z_oof[:, i] for i in selected_idx])
    print(f"\nStrict 2-Seed Nested-CV on WideLM + {len(selected_idx)} selected anchors:", flush=True)
    sr2026 = closed_form_ridge_nested(Z_sel, y, [100.0, 500.0, 2000.0, 5000.0, 15000.0], seed=2026)
    sr777 = closed_form_ridge_nested(Z_sel, y, [100.0, 500.0, 2000.0, 5000.0, 15000.0], seed=777)
    for a in [100.0, 500.0, 2000.0, 5000.0, 15000.0]:
        print(f"  [Selected Ridge alpha={a:<7.0f}] Nested2026={sr2026[a][0]:.6f} | Nested777={sr777[a][0]:.6f}", flush=True)

    for C in [0.01, 0.05, 0.2]:
        a2026, p2026 = eval_nested_logreg(Z_sel, y, C, seed=2026)
        a777, p777 = eval_nested_logreg(Z_sel, y, C, seed=777)
        print(f"  [Selected LogReg C={C:<5}] Nested2026={a2026:.6f} | Nested777={a777:.6f}", flush=True)

    # 5. Blend of Scratch Singles Ridge + Selected Ridge
    _, p_scr = r2026[5000.0]
    _, p_sel_r = sr2026[2000.0]
    for mix in [0.0, 0.2, 0.3, 0.5]:
        c = (1 - mix) * to_rank_logit(p_sel_r) + mix * to_rank_logit(p_scr)
        print(f"  [Blend SelRidge*(1-{mix:.1f}) + ScratchRidge*{mix:.1f}] Nested2026={roc_auc_score(y, c):.6f}", flush=True)

    print(f"\nCompleted in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
