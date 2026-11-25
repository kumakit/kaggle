"""
Exp #048 (v2): Direct ROC-AUC Coordinate Ascent Convex Ensemble on 2-Seed Outer Nested-CV.
Optimizes non-negative weights directly for ROC-AUC (not LogLoss!) inside each outer training fold,
integrating WideLM Stack + 10F GLM-Boost (Sameer + Exp042 + Exp046) + RealMLP Bag (Megayak + Ours 3-Seed + Weni BPE).
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"
BANKS_DIR = os.path.join(DATA_DIR, "oof_banks")


def to_rank_logit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    z = logit(np.clip(r, eps, 1.0 - eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float64)


def optimize_auc_coord_ascent(Z_tr: np.ndarray, y_tr: np.ndarray, init_w: np.ndarray, max_sub: int = 200000):
    """Fast non-negative coordinate ascent directly maximizing ROC AUC on stratified subsample of train fold."""
    if len(y_tr) > max_sub:
        rs = np.random.RandomState(42)
        pos_idx = np.where(y_tr == 1)[0]
        neg_idx = np.where(y_tr == 0)[0]
        n_pos = int(max_sub * (len(pos_idx) / len(y_tr)))
        n_neg = max_sub - n_pos
        sub_idx = np.concatenate([
            rs.choice(pos_idx, size=n_pos, replace=False),
            rs.choice(neg_idx, size=n_neg, replace=False),
        ])
        Z_opt, y_opt = Z_tr[sub_idx], y_tr[sub_idx]
    else:
        Z_opt, y_opt = Z_tr, y_tr

    w = init_w.copy()
    w = w / w.sum()
    best_auc = roc_auc_score(y_opt, Z_opt @ w)

    for step_size in [0.06, 0.03, 0.015]:
        improved = True
        while improved:
            improved = False
            for j in range(Z_opt.shape[1]):
                for direction in [step_size, -step_size]:
                    if w[j] + direction < 0.0:
                        continue
                    w_cand = w.copy()
                    w_cand[j] += direction
                    if w_cand.sum() <= 0:
                        continue
                    w_cand /= w_cand.sum()
                    auc = roc_auc_score(y_opt, Z_opt @ w_cand)
                    if auc > best_auc + 1e-7:
                        best_auc = auc
                        w = w_cand
                        improved = True
                        break
    return w


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=["id"])
    y = train["is_fraud"].to_numpy().astype(int)

    z_am = np.load(f"{BANKS_DIR}/amanatar/rank1_q17_streams.npz", allow_pickle=False)

    oof41 = pd.read_csv(f"{DATA_DIR}/exp041_out/oof_exp041_realmlp.csv")["oof_realmlp_exp041"].to_numpy()
    te41 = pd.read_csv(f"{DATA_DIR}/exp041_out/submission_exp041_refit.csv")["is_fraud"].to_numpy()
    oof47 = pd.read_csv(f"{DATA_DIR}/exp047_out/oof_exp047_realmlp.csv")["oof_realmlp_exp041"].to_numpy()
    te47 = pd.read_csv(f"{DATA_DIR}/exp047_out/submission_exp047_refit.csv")["is_fraud"].to_numpy()
    oof_rmlp3 = (rankdata(oof41) / len(y) + 2.0 * rankdata(oof47) / len(y)) / 3.0
    te_rmlp3 = (rankdata(te41) / len(test) + 2.0 * rankdata(te47) / len(test)) / 3.0

    oof42 = pd.read_csv(f"{DATA_DIR}/exp042_out/oof_exp042_gb10.csv")["oof_exp042_gb10"].to_numpy()
    te42 = pd.read_csv(f"{DATA_DIR}/exp042_out/submission_exp042_refit.csv")["is_fraud"].to_numpy()
    oof46 = pd.read_csv(f"{DATA_DIR}/exp046_out/oof_exp046_xgb10.csv")["oof_exp046_xgb10"].to_numpy()
    te46 = pd.read_csv(f"{DATA_DIR}/exp046_out/submission_exp046_refit.csv")["is_fraud"].to_numpy()

    # Combine RealMLP family & 10F GB family cleanly + individual members
    oof_megayak_rmlp = np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_oof.npy")
    te_megayak_rmlp = np.load(f"{BANKS_DIR}/megayak_realmlp/mlp_test.npy")
    oof_weni_rmlp = np.load(f"{BANKS_DIR}/wenimohamed/oof_weni_realmlp_bpe_v2.npy")
    te_weni_rmlp = np.load(f"{BANKS_DIR}/wenimohamed/test_weni_realmlp_bpe_v2.npy")

    items = [
        ("pub_wide_lm_stack", np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_nested.npy"),
         np.load(f"{BANKS_DIR}/goodpjw_artifacts/wide_lm_stack_test.npy")),
        ("aman_sameer_glm10f", z_am["oof_sameer_glm10f"], z_am["test_sameer_glm10f"]),
        ("ours_exp042_lgb10_refit", oof42, te42),
        ("ours_exp046_xgb10_refit", oof46, te46),
        ("megayak_realmlp_v2", oof_megayak_rmlp, te_megayak_rmlp),
        ("ours_realmlp_3seed_refit", oof_rmlp3, te_rmlp3),
        ("weni_realmlp_bpe_v2", oof_weni_rmlp, te_weni_rmlp),
    ]
    names = [x[0] for x in items]
    Z_oof = np.column_stack([to_rank_logit(x[1]) for x in items])
    Z_te = np.column_stack([to_rank_logit(x[2]) for x in items])

    init_w = np.array([0.80, 0.08, 0.02, 0.02, 0.04, 0.02, 0.02], dtype=np.float64)

    oof_bag = np.zeros(len(y), dtype=np.float64)
    te_bag = np.zeros(len(test), dtype=np.float64)
    ws = []

    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_s = np.zeros(len(y), dtype=np.float64)
        te_s = np.zeros(len(test), dtype=np.float64)
        for f, (tr, va) in enumerate(skf.split(Z_oof, y)):
            w = optimize_auc_coord_ascent(Z_oof[tr], y[tr], init_w)
            oof_s[va] = Z_oof[va] @ w
            te_s += (Z_te @ w) / 5.0
            ws.append(w)
        auc_s = roc_auc_score(y, oof_s)
        print(f"[Outer Nested-CV seed={seed}] Direct-AUC Convex Blend = {auc_s:.6f}", flush=True)
        oof_bag += to_rank_logit(oof_s) / 2.0
        te_bag += to_rank_logit(te_s) / 2.0

    final_auc = roc_auc_score(y, oof_bag)
    print(f"\n=== Final Exp #048 Direct-AUC Convex Fortress ({(time.time()-t0):.1f}s) ===", flush=True)
    print(f"2-Seed Bagged Outer Nested-CV AUC = {final_auc:.6f} (vs WideLM {roc_auc_score(y, Z_oof[:,0]):.6f}, Gain={(final_auc-roc_auc_score(y, Z_oof[:,0]))*1e4:+.3f} pts)", flush=True)
    mean_w = np.mean(ws, axis=0)
    for n, w in sorted(zip(names, mean_w), key=lambda x: -x[1]):
        print(f"  {n:28s}: {w:.4f}", flush=True)

    p_final = expit(te_bag)
    sub = pd.DataFrame({"id": test["id"], "is_fraud": p_final})
    out_sub = os.path.join(DATA_DIR, "submission_exp048.csv")
    out_oof = os.path.join(DATA_DIR, "oof_exp048.npy")
    sub.to_csv(out_sub, index=False)
    np.save(out_oof, expit(oof_bag).astype(np.float32))
    assert sub["is_fraud"].isnull().sum() == 0
    assert len(sub) == 299984
    print(f"Saved: {out_sub} & {out_oof}", flush=True)

    # Also evaluate Meta-Stabilized Blend with Exp #043 (L2 LogReg)
    oof43 = np.load(os.path.join(DATA_DIR, "oof_exp043.npy"))
    te43 = pd.read_csv(os.path.join(DATA_DIR, "submission_exp043.csv"))["is_fraud"].to_numpy()
    z_oof43 = to_rank_logit(oof43)
    z_te43 = to_rank_logit(te43)
    z_oof48 = oof_bag
    z_te48 = te_bag
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        blend_oof = (1.0 - alpha) * z_oof48 + alpha * z_oof43
        print(f"  MetaBlend (Exp048*{1-alpha:.1f} + Exp043*{alpha:.1f}): OOF AUC = {roc_auc_score(y, blend_oof):.6f}", flush=True)
    # Save best meta-stabilized blend as submission_exp048_meta.csv
    best_a = max([0.0, 0.3, 0.5, 0.7, 1.0], key=lambda a: roc_auc_score(y, (1.0 - a) * z_oof48 + a * z_oof43))
    print(f"  Best MetaBlend alpha={best_a:.1f} -> OOF AUC = {roc_auc_score(y, (1.0 - best_a) * z_oof48 + best_a * z_oof43):.6f}", flush=True)
    p_meta = expit((1.0 - best_a) * z_te48 + best_a * z_te43)
    pd.DataFrame({"id": test["id"], "is_fraud": p_meta}).to_csv(os.path.join(DATA_DIR, "submission_exp048_meta.csv"), index=False)
    np.save(os.path.join(DATA_DIR, "oof_exp048_meta.npy"), expit((1.0 - best_a) * z_oof48 + best_a * z_oof43).astype(np.float32))


if __name__ == "__main__":
    main()
