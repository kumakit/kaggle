"""
005_name-that-llm: Exp #016 Local 5-Fold CV Evaluator for Two-Way Gaussian Head & Qwen3-Embedding Clustering
Strictly fits gauss_head inside each StratifiedGroupKFold training split (4 folds) and evaluates on the held-out fold,
comparing true prompt_id vs Qwen3-Embedding unsupervised clustering vs TF-IDF clustering,
both standalone and combined with the 7-backbone Exp #015 pipeline (with/without Cluster Sinkhorn).
"""

import gc
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "005_name-that-llm/src")
import gauss_head
from eval_exp012_blend import cluster_sinkhorn, global_sinkhorn

SEED = 42
DATA_DIR = "005_name-that-llm/data"
ASSETS_DIR = os.path.join(DATA_DIR, "exp016_assets")


def capped_agglo_sim(sim, cap=18, thr=0.70):
    n = len(sim)
    a = sim.astype(np.float32).copy()
    np.fill_diagonal(a, -np.inf)
    size = np.ones(n, dtype=np.int32)
    alive = np.ones(n, dtype=bool)
    lab = np.arange(n, dtype=np.int32)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        m = np.where(ok, a, -np.inf)
        k = int(np.argmax(m))
        i, j = divmod(k, n)
        if not np.isfinite(m[i, j]) or m[i, j] < thr:
            break
        new = (size[i] * a[i] + size[j] * a[j]) / (size[i] + size[j])
        a[i] = new
        a[:, i] = new
        a[i, i] = -np.inf
        a[j] = -np.inf
        a[:, j] = -np.inf
        size[i] += size[j]
        alive[j] = False
        lab[lab == j] = i
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def eval_gauss_oof(z_path, y, prompt_ids, g_eval, folds, d=128, temp=3.0):
    z = np.load(z_path)
    n, n_cls = len(y), gauss_head.K_CLASSES
    oof_lp = np.zeros((n, n_cls), dtype=np.float64)
    cfg = dict(gauss_head.DEFAULT_CFG, d=d)
    for tr_idx, va_idx in folds:
        x_tr = np.asarray(z[tr_idx], dtype=np.float64)
        prm = gauss_head.fit(x_tr, y[tr_idx], prompt_ids[tr_idx], cfg=cfg)
        prm["T1"] = prm["Tg"] = float(temp)
        del x_tr
        x_va = np.asarray(z[va_idx], dtype=np.float64)
        _, va_lab = np.unique(g_eval[va_idx], return_inverse=True)
        oof_lp[va_idx] = gauss_head.scores(x_va, va_lab, prm)
        del x_va
    del z
    gc.collect()
    return oof_lp


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    classes = sorted(df["model"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = df["model"].map(c2i).values
    prompt_ids = df["prompt_id"].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df, y, groups=prompt_ids))

    g_tfidf = np.load(os.path.join(DATA_DIR, "tmp_exp012/capped_agglo_G.npy"))
    e_emb = np.load(os.path.join(ASSETS_DIR, "E_qwen_emb8b.npy")).astype(np.float32)
    sim = e_emb @ e_emb.T
    del e_emb

    print("=== Clustering Quality Comparison (vs true prompt_id) ===")
    print(f"  TF-IDF capped_agglo(0.08): ARI={adjusted_rand_score(prompt_ids, g_tfidf):.4f}, clusters={len(np.unique(g_tfidf))}")
    g_emb_map = {}
    for thr in [0.60, 0.65, 0.70, 0.75]:
        g_e = capped_agglo_sim(sim, cap=18, thr=thr)
        sz = np.bincount(g_e)[g_e]
        ari = adjusted_rand_score(prompt_ids, g_e)
        frac_grp = ((sz >= 2) & (sz <= 18)).mean()
        print(f"  Qwen-Emb capped(thr={thr:.2f}): ARI={ari:.4f}, clusters={len(np.unique(g_e))}, grp2..18={frac_grp:.3f}, max_sz={sz.max()}")
        g_emb_map[thr] = g_e
    del sim
    gc.collect()

    g_emb = g_emb_map[0.70]
    candidates = [
        ("Z_llama_s1", 128, 3.0),
        ("Z_llama_s1", 256, 3.0),
        ("Z_llama_oof", 128, 3.0),
        ("Z_qwen8b_base", 128, 3.0),
        ("Z_qwen8b_oof", 128, 3.0),
        ("Z_qwen8b_oof", 256, 3.0),
        ("E_qwen_emb8b", 128, 3.0),
    ]

    print("\n=== Standalone Gauss Head 5-Fold OOF Balanced Accuracy ===")
    gauss_oofs = {}
    for name, d, temp in candidates:
        p = os.path.join(ASSETS_DIR, f"{name}.npy")
        if not os.path.exists(p):
            continue
        lp_g = eval_gauss_oof(p, y, prompt_ids, g_emb, folds, d=d, temp=temp)
        sc_g = balanced_accuracy_score(y, lp_g.argmax(1))
        lp_t = eval_gauss_oof(p, y, prompt_ids, g_tfidf, folds, d=d, temp=temp)
        sc_t = balanced_accuracy_score(y, lp_t.argmax(1))
        lp_o = eval_gauss_oof(p, y, prompt_ids, prompt_ids, folds, d=d, temp=temp)
        sc_o = balanced_accuracy_score(y, lp_o.argmax(1))
        print(f"  {name:14s} (d={d:3d}, T={temp:.1f}): QwenEmb={sc_g:.5f} | TFIDF={sc_t:.5f} | Oracle={sc_o:.5f}")
        gauss_oofs[(name, d)] = lp_g
        np.save(os.path.join(ASSETS_DIR, f"oof_gauss_{name}_d{d}.npy"), lp_g.astype(np.float32))

    # Load Exp #015 baseline OOF terms
    eps = 1e-7
    l5, l6, l7, l8, l12, l13q8, l15ll, l11sg, l13sg, ll = [
        np.log(np.clip(pd.read_csv(os.path.join(DATA_DIR, fn))[classes].values, eps, 1.0))
        for fn in [
            "oof_exp005_openpool_modernbert.csv",
            "oof_exp006_qwen3_lora.csv",
            "oof_exp007_qwen3_4b_lora.csv",
            "oof_exp008_modernbert_large.csv",
            "oof_exp012_deberta_large.csv",
            "oof_exp013_qwen3_8b_lora.csv",
            "oof_exp015_llama3_8b_lora.csv",
            "oof_exp011_stylegram.csv",
            "oof_exp013_stylegram_v2.csv",
            "oof_exp010_glm_blend.csv",
        ]
    ]
    gates = np.load(os.path.join(DATA_DIR, "tmp_exp012/oof_exp013_gates.npz"))
    families = {f: np.array([i for i, c in enumerate(classes) if c.startswith(f)]) for f in ("claude", "gpt", "gemini", "glm")}
    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]
    base6 = 0.25 * l13q8 + 0.225 * l7 + 0.225 * l8 + 0.15 * l12 + 0.1125 * l6 + 0.0375 * l5
    neu15 = 0.90 * base6 + 0.10 * l15ll

    def run_pipe(gauss_term, w_g=0.0, w_sg1=0.35, w_sg2=0.35, lam=0.15, c_sk=0.0, g_sk=0.30, g_clust=g_emb):
        bl = log_softmax(neu15 + w_sg1 * l11sg + w_sg2 * l13sg + w_g * gauss_term, axis=1)
        if lam > 0:
            out = bl.copy()
            for fam, idxs in families.items():
                base = out[:, idxs]
                base_lp = log_softmax(base, axis=1)
                sp = np.sort(np.exp(base_lp), axis=1)
                st = min(0.45, lam / (1.0 + lam)) * np.clip((0.45 - (sp[:, -1] - sp[:, -2])) / 0.45, 0.0, 1.0)
                mlp = log_softmax((1.0 - st[:, None]) * base_lp + st[:, None] * gates[fam], axis=1)
                out[:, idxs] = logsumexp(base, axis=1)[:, None] + mlp
            bl = out
        if c_sk > 0:
            bl = cluster_sinkhorn(bl, g_clust, 3, c_sk, 6, 18)
        bl = global_sinkhorn(bl, g_sk)
        gm = np.exp(bl)[:, glm_idxs].sum(1)
        sel = np.sort(np.argsort(-gm, kind="stable")[: int(round(0.30 * len(bl)))])
        bef = bl.argmax(1)
        aft = bef.copy()
        aft[sel] = (bl[sel] + 1.15 * ll[sel]).argmax(1)
        grel = np.isin(bef, glm_idxs) | np.isin(aft, glm_idxs)
        aft[(bef != aft) & ~grel] = bef[(bef != aft) & ~grel]
        return aft

    print("\n=== Full Blend Integration (WITHOUT Cluster Sinkhorn, Global Sinkhorn=0.30) ===")
    sc0 = balanced_accuracy_score(y, run_pipe(0.0, w_g=0.0, c_sk=0.0, g_sk=0.30))
    print(f"  Baseline Exp015 (c_sk=0.00, g_sk=0.30, no gauss): {sc0:.5f}")
    for (name, d), lp_g in gauss_oofs.items():
        for w_g in [0.25, 0.45, 0.65, 0.85, 1.05]:
            sc = balanced_accuracy_score(y, run_pipe(lp_g, w_g=w_g, c_sk=0.0, g_sk=0.30))
            print(f"  + {name:14s} (d={d}, w_g={w_g:.2f}, c_sk=0.00): {sc:.5f} ({sc - sc0:+.5f})")

    if ("Z_llama_s1", 128) in gauss_oofs and ("Z_qwen8b_oof", 128) in gauss_oofs:
        g_comb = 0.5 * gauss_oofs[("Z_llama_s1", 128)] + 0.5 * gauss_oofs[("Z_qwen8b_oof", 128)]
        print("\n=== Combined Multi-Family Gauss (Llama_S1 + Qwen8B) ===")
        for w_g in [0.40, 0.70, 1.00, 1.30]:
            for c_sk in [0.00, 0.30, 0.85]:
                sc = balanced_accuracy_score(y, run_pipe(g_comb, w_g=w_g, c_sk=c_sk, g_sk=0.30))
                print(f"  w_g={w_g:.2f}, c_sk={c_sk:.2f}, g_sk=0.30 -> BalAcc={sc:.5f}")


if __name__ == "__main__":
    main()
