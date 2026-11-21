"""
005_name-that-llm: Exp #017a Scale-Calibrated Gauss Head (d=128) & Cluster Sinkhorn Isolation Study
Computes and caches d=128 In-Fold Gauss OOFs (both clustered via Qwen3-Emb thr=0.65 and singleton-only),
checks logit std / GLM change rate, and evaluates safe W_GAUSS (10-15% effective weight) x c_sk via 5-Fold Nested-CV.
"""

import gc
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import log_softmax, logsumexp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "005_name-that-llm/src")
import gauss_head
from eval_exp012_blend import cluster_sinkhorn, global_sinkhorn
from eval_exp016_gauss_cv import capped_agglo_sim

SEED = 42
DATA_DIR = "005_name-that-llm/data"
ASSETS_DIR = os.path.join(DATA_DIR, "exp016_assets")


def get_or_compute_gauss(name, y, pids, g65, folds, d=128, temp=3.0, singleton=False):
    tag = "single" if singleton else "qemb65"
    cache = os.path.join(ASSETS_DIR, f"oof_gauss_{name}_d{d}_{tag}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    z = np.load(os.path.join(ASSETS_DIR, f"{name}.npy"))
    oof = np.zeros((len(y), 18), dtype=np.float64)
    cfg = dict(gauss_head.DEFAULT_CFG, d=d)
    for tr, va in folds:
        prm = gauss_head.fit(np.asarray(z[tr], np.float64), y[tr], pids[tr], cfg=cfg)
        prm["T1"] = prm["Tg"] = float(temp)
        vl = np.arange(len(va)) if singleton else np.unique(g65[va], return_inverse=True)[1]
        oof[va] = gauss_head.scores(np.asarray(z[va], np.float64), vl, prm)
    del z
    gc.collect()
    np.save(cache, oof.astype(np.float32))
    return oof


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    classes = sorted(df["model"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    y = df["model"].map(c2i).values
    pids = df["prompt_id"].values
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, groups=pids))

    e_emb = np.load(os.path.join(ASSETS_DIR, "E_qwen_emb8b.npy")).astype(np.float32)
    sim = e_emb @ e_emb.T
    del e_emb
    g65 = capped_agglo_sim(sim, 18, 0.65)
    del sim
    gc.collect()

    print("=== d=128 Gauss Head Standalone (Grouped QwenEmb vs Singleton) ===")
    g_grp, g_sng = {}, {}
    for name in ("Z_llama_s1", "Z_qwen8b_oof", "Z_qwen8b_base"):
        gg = get_or_compute_gauss(name, y, pids, g65, folds, d=128, temp=3.0, singleton=False)
        gs = get_or_compute_gauss(name, y, pids, g65, folds, d=128, temp=3.0, singleton=True)
        g_grp[name] = gg
        g_sng[name] = gs
        print(
            f"  {name:14s}: Grouped={balanced_accuracy_score(y, gg.argmax(1)):.5f} (std={gg.std():.2f}) | "
            f"Singleton={balanced_accuracy_score(y, gs.argmax(1)):.5f} (std={gs.std():.2f})"
        )

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

    def eval_va(idx, g_mat, w_g, c_sk, g_sk=0.35):
        bl = log_softmax(neu15[idx] + 0.35 * l11sg[idx] + 0.35 * l13sg[idx] + w_g * g_mat[idx], axis=1)
        out = bl.copy()
        for fam, idxs in families.items():
            base = out[:, idxs]
            base_lp = log_softmax(base, axis=1)
            sp = np.sort(np.exp(base_lp), axis=1)
            st = min(0.45, 0.15 / 1.15) * np.clip((0.45 - (sp[:, -1] - sp[:, -2])) / 0.45, 0.0, 1.0)
            mlp = log_softmax((1.0 - st[:, None]) * base_lp + st[:, None] * gates[fam][idx], axis=1)
            out[:, idxs] = logsumexp(base, axis=1)[:, None] + mlp
        if c_sk > 0:
            out = cluster_sinkhorn(out, g65[idx], 3, c_sk, 6, 18)
        bl = global_sinkhorn(out, g_sk)
        gm = np.exp(bl)[:, glm_idxs].sum(1)
        sel = np.sort(np.argsort(-gm, kind="stable")[: int(round(0.30 * len(bl)))])
        bef = bl.argmax(1)
        aft = bef.copy()
        aft[sel] = (bl[sel] + 1.15 * ll[idx][sel]).argmax(1)
        grel = np.isin(bef, glm_idxs) | np.isin(aft, glm_idxs)
        aft[(bef != aft) & ~grel] = bef[(bef != aft) & ~grel]
        return aft, float((bef != aft).mean()), float(bl.std())

    def fold_cv(g_mat, w_g, c_sk, g_sk=0.35):
        p = np.zeros_like(y)
        chgs, stds = [], []
        for _, va in folds:
            p[va], chg, st = eval_va(va, g_mat, w_g, c_sk, g_sk)
            chgs.append(chg)
            stds.append(st)
        return balanced_accuracy_score(y, p), float(np.mean(chgs)), float(np.mean(stds))

    # Compare: Llama_S1 only vs Llama_S1 + Qwen8B_OOF vs Llama_S1 + Qwen8B_Base (d=128)
    pairs = [
        ("LlamaS1_only", g_grp["Z_llama_s1"]),
        ("0.5*LlamaS1+0.5*Q8OOF", 0.5 * g_grp["Z_llama_s1"] + 0.5 * g_grp["Z_qwen8b_oof"]),
        ("0.5*LlamaS1+0.5*Q8Base", 0.5 * g_grp["Z_llama_s1"] + 0.5 * g_grp["Z_qwen8b_base"]),
    ]

    print("\n=== Baseline (w_g=0.0, Exp015 + Qwen3-Emb clusters) ===")
    for c_sk in [0.00, 0.35, 0.65, 0.85]:
        sc, chg, st = fold_cv(g_grp["Z_llama_s1"], 0.0, c_sk, 0.30 if c_sk == 0 else 0.35)
        print(f"  w_g=0.00, c_sk={c_sk:.2f} -> Fold-OOF={sc:.5f} | bl_std={st:.2f} | glm_chg={chg:.3f}")

    for pname, gmat in pairs:
        print(f"\n=== {pname} (d=128, Safe Scaled W_GAUSS vs c_sk) ===")
        for w_g in [0.10, 0.15, 0.22, 0.30, 0.45]:
            for c_sk in [0.00, 0.35, 0.65]:
                g_sk = 0.30 if c_sk == 0 else 0.35
                sc, chg, st = fold_cv(gmat, w_g, c_sk, g_sk)
                eff_pct = (w_g * gmat.std()) / (2.097 + w_g * gmat.std()) * 100.0
                print(
                    f"  w_g={w_g:.2f} (gauss~{eff_pct:4.1f}%, std={st:.2f}), c_sk={c_sk:.2f} -> "
                    f"Fold-OOF={sc:.5f} | glm_chg={chg:.3f}"
                )


if __name__ == "__main__":
    main()
