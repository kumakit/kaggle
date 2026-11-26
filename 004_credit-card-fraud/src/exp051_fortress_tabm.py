"""
Exp #051: Pure Nested-CV Fortress + TabM (Exp #049 Full-Data Refit)
Verifies on 2-seed Outer Nested-CV (seed=2026, 777) that adding 1.5% of our self-trained TabM
(corr=0.9609 vs Exp048) genuinely improves out-of-fold generalization, and builds submission_exp051.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "004_credit-card-fraud/data"


def to_rank_logit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (rankdata(arr) - 0.5) / len(arr)
    z = logit(np.clip(r, eps, 1.0 - eps))
    return ((z - z.mean()) / (z.std() + 1e-12)).astype(np.float64)


def main():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=["id", "is_fraud"])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=["id"])
    y = train["is_fraud"].to_numpy().astype(int)

    oof48 = np.load(f"{DATA_DIR}/oof_exp048.npy")
    te48 = pd.read_csv(f"{DATA_DIR}/submission_exp048.csv")["is_fraud"].to_numpy()
    oof49 = pd.read_csv(f"{DATA_DIR}/exp049_out/oof_exp049_tabm.csv")["oof_exp049_tabm"].to_numpy()
    te49 = pd.read_csv(f"{DATA_DIR}/exp049_out/submission_exp049_refit.csv")["is_fraud"].to_numpy()

    z_o48 = to_rank_logit(oof48)
    z_o49 = to_rank_logit(oof49)
    z_t48 = to_rank_logit(te48)
    z_t49 = to_rank_logit(te49)

    print(f"Base Exp #048 OOF AUC = {roc_auc_score(y, z_o48):.6f}")
    print(f"TabM Exp #049 OOF AUC = {roc_auc_score(y, z_o49):.6f}")

    # Strict 2-seed Outer Nested-CV: select w in [0.005..0.030] inside tr, score on va
    for seed in [2026, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        p_nested = np.zeros(len(y))
        ws = []
        for tr, va in skf.split(z_o48, y):
            best_w, best_tr = 0.0, roc_auc_score(y[tr], z_o48[tr])
            for w in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
                auc = roc_auc_score(y[tr], (1.0 - w) * z_o48[tr] + w * z_o49[tr])
                if auc > best_tr:
                    best_tr, best_w = auc, w
            ws.append(best_w)
            p_nested[va] = (1.0 - best_w) * z_o48[va] + best_w * z_o49[va]
        print(f"  [Nested-CV seed={seed}] selected ws={ws} -> Nested AUC = {roc_auc_score(y, p_nested):.6f} (vs Base {roc_auc_score(y, z_o48):.6f})")

    w_final = 0.015
    z_oof51 = (1.0 - w_final) * z_o48 + w_final * z_o49
    z_te51 = (1.0 - w_final) * z_t48 + w_final * z_t49
    auc51 = roc_auc_score(y, z_oof51)
    print(f"\nExp #051 (98.5% Exp048 + 1.5% Exp049 TabM Refit) OOF AUC = {auc51:.6f}")

    p_final = expit(z_te51)
    out_sub = os.path.join(DATA_DIR, "submission_exp051.csv")
    out_oof = os.path.join(DATA_DIR, "oof_exp051.npy")
    pd.DataFrame({"id": test["id"], "is_fraud": p_final}).to_csv(out_sub, index=False)
    np.save(out_oof, expit(z_oof51).astype(np.float32))
    print(f"Saved {out_sub} & {out_oof}")


if __name__ == "__main__":
    main()
