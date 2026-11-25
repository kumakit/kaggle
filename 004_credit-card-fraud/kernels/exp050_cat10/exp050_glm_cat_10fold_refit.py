"""
004_credit-card-fraud: Exp #050 - Kaggle GPU 10-Fold GLM-Boosted CatBoost (Symmetric Oblivious Trees) + Full-Data Refit
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
"""

import gc
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 10
TARGET = "is_fraud"
BAG = [(222, 333), (444, 555)]

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
NUMS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
FLOATS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]
FEATS = NUMS + FLAGS + CATS

JOINT_PAIRS = [
    ("merchant_category", "merchant_risk_score"),
    ("is_foreign_transaction", "ip_country_mismatch"),
    ("merchant_category", "card_type"),
    ("channel", "device_type"),
    ("auth_method", "is_new_merchant"),
    ("used_vpn", "ip_country_mismatch"),
    ("is_ai_generated_scam_attempt", "channel"),
]


def find_file(name: str) -> str:
    for root in ["/kaggle/input", "004_credit-card-fraud/data", "/content/data", "data"]:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def te_rate(k, t, ncat, prior, smooth):
    s = np.bincount(k, weights=t, minlength=ncat)
    n = np.bincount(k, minlength=ncat)
    return (s + prior * smooth) / (n + smooth)


def build_key_matrix(full: pd.DataFrame):
    key_cols, key_names, cents_list = [], [], []
    for c in FEATS:
        codes, _ = pd.factorize(full[c].astype(str))
        key_cols.append(codes.astype(np.int32))
        key_names.append(f"raw_{c}")

    for c in FLOATS:
        units = np.rint(full[c].to_numpy() * 100).astype(np.int64)
        integer = units // 100
        s = pd.Series(np.abs(integer)).astype(str)
        cents = (units % 100).astype(np.int32)
        cents_list.append(cents)
        digs = {
            f"{c}__int": integer,
            f"{c}__int10": units // 1000,
            f"{c}__tenths": units // 10,
            f"{c}__cents": cents,
            f"{c}__lead1": (s.str.len() * 10 + s.str[0].astype(int)).to_numpy(),
            f"{c}__lead2": (s.str.len() * 100 + s.str[:2].astype(int)).to_numpy(),
        }
        for name, arr in digs.items():
            _, inv = np.unique(arr, return_inverse=True)
            key_cols.append(inv.astype(np.int32))
            key_names.append(name)

    for c1, c2 in JOINT_PAIRS:
        comb = full[c1].astype(str) + "_" + full[c2].astype(str)
        codes, _ = pd.factorize(comb)
        key_cols.append(codes.astype(np.int32))
        key_names.append(f"pair_{c1}__{c2}")

    kmat = np.column_stack(key_cols)
    ncat = [int(kmat[:, j].max()) + 1 for j in range(kmat.shape[1])]
    kcnt = np.empty_like(kmat, dtype=np.int32)
    for j in range(kmat.shape[1]):
        cnts = np.bincount(kmat[:, j], minlength=ncat[j])
        kcnt[:, j] = cnts[kmat[:, j]]
    return kmat, kcnt, ncat, key_names, np.column_stack(cents_list)


def main():
    t0 = time.time()
    tr = pd.read_csv(find_file("train.csv"))
    te = pd.read_csv(find_file("test.csv"))
    for d in (tr, te):
        d[FLAGS] = d[FLAGS].astype("int8")

    y = tr[TARGET].astype(int).to_numpy()
    ntr, nte = len(tr), len(te)
    print(f"[DATA] train={ntr}, test={nte}, fraud_rate={y.mean():.5f}", flush=True)

    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    kmat, kcnt, ncat, knames, cents_mat = build_key_matrix(full)

    num_df = full[NUMS].astype(float).copy()
    for c in ("amount_usd", "account_balance_usd", "hours_since_last_txn"):
        num_df[c] = np.log1p(np.maximum(num_df[c], 0.0))
    z_num = StandardScaler().fit_transform(num_df).astype(np.float32)
    oh = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(full[CATS].astype(str))
    base_glm = np.hstack([z_num, full[FLAGS].to_numpy(dtype=np.float32), oh])
    del num_df, z_num, oh
    gc.collect()

    raw_tree = full[FEATS].copy()
    for c in CATS:
        raw_tree[c] = raw_tree[c].astype(str)
    del full
    gc.collect()

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(ntr), y))

    def encode(a, b, smooth=20.0, inner=5, teseed=0):
        a = np.arange(ntr) if a is None else a
        ya = y[a].astype(np.float64)
        ka, kt = kmat[a], kmat[ntr:]
        kb = kmat[b] if b is not None else None
        ea = np.empty((len(a), kmat.shape[1]), dtype=np.float32)
        et = np.empty((nte, kmat.shape[1]), dtype=np.float32)
        eb = np.empty((len(b), kmat.shape[1]), dtype=np.float32) if b is not None else None

        inner_f = list(StratifiedKFold(inner, shuffle=True, random_state=teseed).split(np.zeros(len(a)), ya))
        for j in range(kmat.shape[1]):
            for i, o in inner_f:
                ea[o, j] = te_rate(ka[i, j], ya[i], ncat[j], ya[i].mean(), smooth)[ka[o, j]]
            r = te_rate(ka[:, j], ya, ncat[j], ya.mean(), smooth)
            et[:, j] = r[kt[:, j]]
            if b is not None:
                eb[:, j] = r[kb[:, j]]
        return ea, eb, et

    def glm_logits(a, ea, b, eb, et):
        a = np.arange(ntr) if a is None else a
        eps = 1e-6
        la = np.log(np.clip(ea, eps, 1 - eps) / (1 - np.clip(ea, eps, 1 - eps)))
        lt = np.log(np.clip(et, eps, 1 - eps) / (1 - np.clip(et, eps, 1 - eps)))
        sc = StandardScaler().fit(la)
        xa = np.hstack([sc.transform(la), base_glm[a]])
        m = LogisticRegression(C=0.3, max_iter=300, tol=1e-7, solver="newton-cholesky").fit(xa, y[a])
        ga = m.decision_function(xa)
        gt = m.decision_function(np.hstack([sc.transform(lt), base_glm[ntr:]]))
        gb = None
        if b is not None:
            lb = np.log(np.clip(eb, eps, 1 - eps) / (1 - np.clip(eb, eps, 1 - eps)))
            gb = m.decision_function(np.hstack([sc.transform(lb), base_glm[b]]))
        return ga, gb, gt

    def tree_frame(idx, e_mat):
        d = raw_tree.iloc[idx].reset_index(drop=True).copy()
        for j, c in enumerate(FLOATS):
            d[f"cents_{c}"] = cents_mat[idx, j]
        for j, k in enumerate(knames):
            d[f"te_{k}"] = e_mat[:, j]
            d[f"cnt_{k}"] = kcnt[idx, j]
        return d

    oof_total = np.zeros(ntr, dtype=np.float64)
    foldavg_total = np.zeros(nte, dtype=np.float64)
    refit_total = np.zeros(nte, dtype=np.float64)

    for teseed, mseed in BAG:
        oof_s = np.zeros(ntr, dtype=np.float64)
        foldavg_s = np.zeros(nte, dtype=np.float64)
        best_iters = []

        for f, (a, b) in enumerate(folds):
            tf0 = time.time()
            ea, eb, et = encode(a, b, teseed=teseed)
            ga, gb, gt = glm_logits(a, ea, b, eb, et)
            xa, xb, xt = tree_frame(a, ea), tree_frame(b, eb), tree_frame(np.arange(ntr, ntr + nte), et)

            p_tr = Pool(xa, label=y[a], cat_features=CATS, baseline=ga)
            p_va = Pool(xb, label=y[b], cat_features=CATS, baseline=gb)
            p_te = Pool(xt, cat_features=CATS, baseline=gt)

            cb = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                iterations=8000,
                learning_rate=0.03,
                depth=3,
                l2_leaf_reg=5.0,
                random_seed=mseed + f,
                task_type="GPU",
                verbose=False,
            )
            cb.fit(p_tr, eval_set=p_va, early_stopping_rounds=300, verbose=False)
            pb = cb.predict_proba(p_va)[:, 1]
            pt = cb.predict_proba(p_te)[:, 1]

            oof_s[b] = pb
            foldavg_s += pt / N_FOLDS
            best_it = cb.get_best_iteration() or 2000
            best_iters.append(best_it)
            print(
                f"[Seed {teseed}/{mseed} Fold {f}] GLM={roc_auc_score(y[b], gb):.5f} -> "
                f"Cat3={roc_auc_score(y[b], pb):.5f} (trees={best_it}) | {(time.time() - tf0):.0f}s",
                flush=True,
            )
            del ea, eb, et, xa, xb, xt, p_tr, p_va, p_te, cb
            gc.collect()

        print(f"[SEED {teseed}] OOF AUC = {roc_auc_score(y, oof_s):.6f}", flush=True)

        n_refit = int(round(np.mean(best_iters) * 1.11))
        tf0 = time.time()
        ea, _, et = encode(None, None, teseed=teseed)
        ga, _, gt = glm_logits(None, ea, None, None, et)
        xa, xt = tree_frame(np.arange(ntr), ea), tree_frame(np.arange(ntr, ntr + nte), et)
        p_tr = Pool(xa, label=y, cat_features=CATS, baseline=ga)
        p_te = Pool(xt, cat_features=CATS, baseline=gt)
        cb_ref = CatBoostClassifier(
            loss_function="Logloss",
            iterations=n_refit,
            learning_rate=0.03,
            depth=3,
            l2_leaf_reg=5.0,
            random_seed=mseed + 99,
            task_type="GPU",
            verbose=False,
        )
        cb_ref.fit(p_tr, verbose=False)
        pt_ref = cb_ref.predict_proba(p_te)[:, 1]
        print(f"[REFIT {teseed}] {n_refit} trees done in {(time.time() - tf0):.0f}s", flush=True)

        del ea, et, xa, xt, p_tr, p_te, cb_ref
        gc.collect()

        oof_total += rankdata(oof_s) / (ntr * len(BAG))
        foldavg_total += rankdata(foldavg_s) / (nte * len(BAG))
        refit_total += rankdata(pt_ref) / (nte * len(BAG))

    overall_auc = roc_auc_score(y, oof_total)
    print("=" * 70, flush=True)
    print(f"[DONE] Exp #050 10-Fold GLM-CatBoost OOF AUC = {overall_auc:.6f} | Total {(time.time() - t0)/60:.1f} min", flush=True)
    print("=" * 70, flush=True)

    pd.DataFrame({"id": tr["id"], "oof_exp050_cat10": oof_total.astype(np.float32)}).to_csv("oof_exp050_cat10.csv", index=False)
    pd.DataFrame({"id": te["id"], "is_fraud": foldavg_total.astype(np.float32)}).to_csv("submission_exp050_foldavg.csv", index=False)
    pd.DataFrame({"id": te["id"], "is_fraud": refit_total.astype(np.float32)}).to_csv("submission_exp050_refit.csv", index=False)


if __name__ == "__main__":
    main()
