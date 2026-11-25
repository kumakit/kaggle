"""
004_credit-card-fraud: Exp #049 - Kaggle GPU Digit-TE Enhanced TabM (BatchEnsemble MLP) + Full-Data Refit
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)
"""

import argparse
import base64
import gc
import glob
import inspect
import io
import json
import os
import subprocess
import sys
import threading
import time
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

FLOAT_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]


def start_heartbeat(stop_event: threading.Event, interval_sec: int = 20):
    t0 = time.time()

    def _beat():
        while not stop_event.wait(interval_sec):
            print(f"[HEARTBEAT] Running... elapsed={(time.time() - t0) / 60.0:.1f} min", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return th


def setup_environment():
    print("[INIT] Checking packages and datasets...", flush=True)
    try:
        import torch
        import pytabkit
        print(f"  - PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit>=1.8.0", "scikit-learn"], check=True)
        import torch
        import pytabkit
        print(f"  - Installed PyTabKit. PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    for cand in ["/content/data", "004_credit-card-fraud/data", "data"]:
        if os.path.exists(os.path.join(cand, "train.csv")):
            return cand

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if username and key:
            with open(kaggle_json_path, "w") as f:
                json.dump({"username": username, "key": key}, f)
            os.chmod(kaggle_json_path, 0o600)
        else:
            raise FileNotFoundError("train.csv and Kaggle credentials not found")
    data_dir = "/content/data"
    os.makedirs(data_dir, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", "playground-series-s6e11", "-p", data_dir], check=True)
    for zf in glob.glob(os.path.join(data_dir, "*.zip")):
        subprocess.run(["unzip", "-q", "-o", zf, "-d", data_dir], check=True)
    return data_dir


def coded(a: pd.Series, b: pd.Series):
    codes, uniq = pd.factorize(a)
    look = {u: i for i, u in enumerate(uniq)}
    return codes.astype("int32"), b.map(look).fillna(-1).astype("int32")


def extract_digit_key_codes(tr_df: pd.DataFrame, te_df: pd.DataFrame):
    tr_keys, te_keys = {}, {}
    for c in FLOAT_COLS:
        for df, store in [(tr_df, tr_keys), (te_df, te_keys)]:
            units = np.rint(df[c].to_numpy() * 100).astype(np.int64)
            integer = units // 100
            s = pd.Series(np.abs(integer)).astype(str)
            store[f"{c}__int"] = integer
            store[f"{c}__int10"] = units // 1000
            store[f"{c}__tenths"] = units // 10
            store[f"{c}__cents"] = units % 100
            store[f"{c}__lead1"] = (s.str.len() * 10 + s.str[0].astype(int)).to_numpy()
            store[f"{c}__lead2"] = (s.str.len() * 100 + s.str[:2].astype(int)).to_numpy()

    codes_dict, levels_dict = {}, {}
    n_tr = len(tr_df)
    for k in tr_keys:
        comb = np.concatenate([tr_keys[k], te_keys[k]])
        _, inv = np.unique(comb, return_inverse=True)
        inv = inv.astype(np.int32)
        codes_dict[k] = (inv[:n_tr], inv[n_tr:])
        levels_dict[k] = int(inv.max()) + 1
    return codes_dict, levels_dict


def te_table(codes: np.ndarray, target: np.ndarray, n_levels: int, m: float = 20.0):
    frauds = np.bincount(codes, weights=target, minlength=n_levels)
    rows = np.bincount(codes, minlength=n_levels)
    return (frauds + m * target.mean()) / (rows + m)


def compute_digit_te_logits(codes_dict, levels_dict, tr_idx, va_idx, y_all, seed, m=20.0, inner_cv=5):
    keys = list(codes_dict.keys())
    y_tr = y_all[tr_idx].astype(np.float64)
    inner = list(KFold(inner_cv, shuffle=True, random_state=seed).split(tr_idx))
    n_te = len(codes_dict[keys[0]][1])
    E_tr = np.empty((len(tr_idx), len(keys)), dtype=np.float32)
    E_va = np.empty((len(va_idx), len(keys)), dtype=np.float32) if va_idx is not None else None
    E_te = np.empty((n_te, len(keys)), dtype=np.float32)

    eps = 1e-5
    for j, k in enumerate(keys):
        c_tr_all, c_te = codes_dict[k]
        k_tr = c_tr_all[tr_idx]
        n_lev = levels_dict[k]
        for fit_i, enc_i in inner:
            tab = te_table(k_tr[fit_i], y_tr[fit_i], n_lev, m=m)
            E_tr[enc_i, j] = logit(np.clip(tab[k_tr[enc_i]], eps, 1.0 - eps))
        full_tab = te_table(k_tr, y_tr, n_lev, m=m)
        if va_idx is not None:
            E_va[:, j] = logit(np.clip(full_tab[c_tr_all[va_idx]], eps, 1.0 - eps))
        E_te[:, j] = logit(np.clip(full_tab[c_te], eps, 1.0 - eps))

    col_names = [f"dte_{k}" for k in keys]
    return (
        pd.DataFrame(E_tr, columns=col_names),
        pd.DataFrame(E_va, columns=col_names) if va_idx is not None else None,
        pd.DataFrame(E_te, columns=col_names),
    )


def build_base_features(tr: pd.DataFrame, te: pd.DataFrame):
    cat = [c for c in tr.columns if tr[c].dtype == bool or not pd.api.types.is_numeric_dtype(tr[c])]
    num = [c for c in tr.columns if c not in cat]
    dec = [c for c in num if (tr[c] % 1 != 0).any()]

    new_cats = []
    for c in num:
        col = f"{c}_int"
        tr[col], te[col] = coded(np.floor(tr[c]), np.floor(te[c]))
        new_cats.append(col)
    for c in dec:
        col = f"{c}_cents"
        tr[col], te[col] = coded((tr[c] * 100).round() % 100, (te[c] * 100).round() % 100)
        new_cats.append(col)
    for c, nb in [("merchant_risk_score", 7), ("hours_since_last_txn", 500)]:
        kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile", subsample=None)
        col = f"{c}_q{nb}"
        tr[col] = kb.fit_transform(tr[[c]]).ravel().astype("int32")
        te[col] = kb.transform(te[[c]]).ravel().astype("int32")
        new_cats.append(col)
    crosses = []
    for cols in [("merchant_category", "card_type", "txn_count_last_24h"), ("merchant_category", "customer_age")]:
        col = "x_" + "_".join(cols)
        j = lambda d: d[list(cols)].astype(str).agg("_".join, axis=1)
        tr[col], te[col] = coded(j(tr), j(te))
        new_cats.append(col)
        crosses.append(col)

    for col in cat:
        tr[col], te[col] = coded(tr[col].astype(str), te[col].astype(str))
        tr[col] = tr[col].astype("int32")
        te[col] = te[col].astype("int32")
    for col in new_cats:
        tr[col] = tr[col].astype("int32")
        te[col] = te[col].astype("int32")

    all_cats = cat + new_cats
    te_src = crosses + ["hours_since_last_txn"]
    te_names = [f"{c}_te" for c in te_src]
    return tr, te, all_cats, te_src, te_names


def main():
    stop_event = threading.Event()
    start_heartbeat(stop_event, interval_sec=20)
    try:
        t0 = time.time()
        data_dir = setup_environment()
        import torch
        from pytabkit import TabM_D_Classifier

        tr_raw = pd.read_csv(os.path.join(data_dir, "train.csv"))
        te_raw = pd.read_csv(os.path.join(data_dir, "test.csv"))
        train_id = tr_raw["id"].to_numpy()
        test_id = te_raw["id"].to_numpy()
        y = tr_raw["is_fraud"].astype(int).to_numpy()
        tr_feat = tr_raw.drop(columns=["id", "is_fraud"])
        te_feat = te_raw.drop(columns=["id"])
        del tr_raw, te_raw
        gc.collect()

        n_tr, n_te = len(tr_feat), len(te_feat)
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(n_tr), y))
        codes_dict, levels_dict = extract_digit_key_codes(tr_feat, te_feat)
        tr_feat, te_feat, all_cats, te_src, te_names = build_base_features(tr_feat, te_feat)
        print(f"[DATA] train={n_tr}, test={n_te}, cols={tr_feat.shape[1]}, cats={len(all_cats)}", flush=True)

        # Safely configure TabM_D_Classifier parameters
        sig = inspect.signature(TabM_D_Classifier.__init__).parameters
        cand_params = {
            "verbosity": 0,
            "val_metric_name": "1-auc_ovr",
            "n_epochs": 15,
            "batch_size": 512,
            "lr": 0.002,
            "tabm_k": 32,
            "num_emb_type": "pwl",
            "d_embedding": 8,
            "patience": 4,
            "device": "cuda",
        }
        base_params = {k: v for k, v in cand_params.items() if k in sig}
        print(f"[TABM] Using params: {base_params}", flush=True)

        def run_one(tr_idx, va_idx, seed):
            Xa = tr_feat.iloc[tr_idx].reset_index(drop=True)
            Xt = te_feat.reset_index(drop=True)
            Xb = tr_feat.iloc[va_idx].reset_index(drop=True) if va_idx is not None else None

            enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=seed)
            Xa[te_names] = enc.fit_transform(Xa[te_src], y[tr_idx]).astype(np.float32)
            Xt[te_names] = enc.transform(Xt[te_src]).astype(np.float32)
            if Xb is not None:
                Xb[te_names] = enc.transform(Xb[te_src]).astype(np.float32)

            dte_tr, dte_va, dte_te = compute_digit_te_logits(
                codes_dict, levels_dict, tr_idx, va_idx, y, seed=seed, m=20.0, inner_cv=5
            )
            Xa = pd.concat([Xa, dte_tr], axis=1)
            Xt = pd.concat([Xt, dte_te], axis=1)
            if Xb is not None:
                Xb = pd.concat([Xb, dte_va], axis=1)

            if Xb is None:
                rs = np.random.RandomState(seed)
                m = rs.rand(len(tr_idx)) < 0.03
                Xf, yf = Xa.loc[~m], y[tr_idx][~m]
                Xv, yv = Xa.loc[m], y[tr_idx][m]
            else:
                Xf, yf = Xa, y[tr_idx]
                Xv, yv = Xb, y[va_idx]

            p_kw = {**base_params}
            if "random_state" in sig:
                p_kw["random_state"] = seed
            model = TabM_D_Classifier(**p_kw)
            with redirect_stdout(io.StringIO()):
                model.fit(Xf, yf, Xv, yv, cat_col_names=all_cats)

            pb = None if Xb is None else model.predict_proba(Xb)[:, 1].astype(np.float32)
            pt = model.predict_proba(Xt)[:, 1].astype(np.float32)
            del model, Xa, Xt, Xb, Xf, Xv, dte_tr, dte_va, dte_te
            gc.collect()
            torch.cuda.empty_cache()
            return pb, pt

        seed = 42
        oof = np.zeros(n_tr, dtype=np.float64)
        foldavg = np.zeros(n_te, dtype=np.float64)
        for f, (tr_idx, va_idx) in enumerate(folds):
            tf0 = time.time()
            pb, pt = run_one(tr_idx, va_idx, seed + f)
            oof[va_idx] = pb
            foldavg += pt / 5.0
            print(f"[FOLD {f}] TabM AUC={roc_auc_score(y[va_idx], pb):.6f} | {(time.time()-tf0)/60:.2f} min", flush=True)

        oof_auc = roc_auc_score(y, oof)
        print(f"[OOF] Exp #049 TabM OOF AUC = {oof_auc:.6f}", flush=True)

        tf0 = time.time()
        _, refit = run_one(np.arange(n_tr), None, seed + 100)
        print(f"[REFIT] done | {(time.time()-tf0)/60:.2f} min | Total {(time.time()-t0)/60:.2f} min", flush=True)

        pd.DataFrame({"id": train_id, "oof_exp049_tabm": oof.astype(np.float32)}).to_csv("oof_exp049_tabm.csv", index=False)
        pd.DataFrame({"id": test_id, "is_fraud": foldavg.astype(np.float32)}).to_csv("submission_exp049_foldavg.csv", index=False)
        pd.DataFrame({"id": test_id, "is_fraud": refit.astype(np.float32)}).to_csv("submission_exp049_refit.csv", index=False)
        if os.path.exists("/content/drive/MyDrive"):
            pd.DataFrame({"id": train_id, "oof_exp049_tabm": oof.astype(np.float32)}).to_csv("/content/drive/MyDrive/oof_exp049_tabm.csv", index=False)
            pd.DataFrame({"id": test_id, "is_fraud": refit.astype(np.float32)}).to_csv("/content/drive/MyDrive/submission_exp049_refit.csv", index=False)
        if not os.path.exists("/kaggle/working"):
            np.savez_compressed("exp049_outputs.npz", oof=oof.astype(np.float32), foldavg=foldavg.astype(np.float32), refit=refit.astype(np.float32))
            with open("exp049_outputs.npz", "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            print("===BASE64_EXP049_NPZ_START===", flush=True)
            for i in range(0, len(b64), 4000):
                print(b64[i : i + 4000], flush=True)
            print("===BASE64_EXP049_NPZ_END===", flush=True)
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
