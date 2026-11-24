"""
004_credit-card-fraud: Exp #041 - Colab L4 GPU Digit-TE Enhanced RealMLP + Full-Data Refit
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Key Features:
  1. PyTabKit RealMLP_TD_Classifier (Neural Network with PLR embeddings & smooth-clip scaling).
  2. Digit-Key & Cross features: integer floor categories, decimal cents categories, quantile bins,
     interaction crosses, sklearn TargetEncoder + m-estimate in-fold Digit-Key TE logits.
  3. Exact community 5-Fold StratifiedKFold(n_splits=5, shuffle=True, random_state=42) alignment.
  4. Produces both 5-Fold Mean test predictions and Full-Data Refit (100% training data) test predictions.
  5. Supports --smoke-test, --resume, Heartbeat logging, Google Drive sync, and Base64 stdout recovery.
"""

import argparse
import base64
import gc
import glob
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
            elapsed = (time.time() - t0) / 60.0
            print(f"[HEARTBEAT] Running... elapsed={elapsed:.1f} min", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return th


def setup_environment():
    print("[INIT] Checking packages and Kaggle credentials...", flush=True)
    try:
        import torch
        import pytabkit
        print(f"  - PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, PyTabKit ready", flush=True)
    except ImportError:
        print("  - Installing pytabkit...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit>=1.8.0", "scikit-learn", "kaggle"], check=True)
        import torch
        import pytabkit
        print(f"  - Installed PyTabKit. PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    kaggle_inputs = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if kaggle_inputs:
        data_dir = os.path.dirname(kaggle_inputs[0])
        print(f"  - Detected Kaggle input dataset at: {data_dir}", flush=True)
        return data_dir

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
            print(f"  - Configured kaggle.json from env for {username}", flush=True)
        else:
            raise RuntimeError("Kaggle credentials not found in ~/.kaggle/kaggle.json or env (KAGGLE_USERNAME/KAGGLE_KEY).")

    data_dir = "/content/data"
    for cand in ["/content/data", "data", "004_credit-card-fraud/data"]:
        if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(os.path.join(cand, "test.csv")):
            data_dir = cand
            break

    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print("  - Downloading playground-series-s6e11 from Kaggle...", flush=True)
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(["kaggle", "competitions", "download", "-c", "playground-series-s6e11", "-p", data_dir], check=True)
        for zf in glob.glob(os.path.join(data_dir, "*.zip")):
            subprocess.run(["unzip", "-q", "-o", zf, "-d", data_dir], check=True)

    print(f"  - Dataset ready at: {data_dir}", flush=True)
    return data_dir


def coded(a: pd.Series, b: pd.Series):
    codes, uniq = pd.factorize(a)
    look = {u: i for i, u in enumerate(uniq)}
    return codes.astype("int32"), b.map(look).fillna(-1).astype("int32")


def extract_digit_key_codes(tr_df: pd.DataFrame, te_df: pd.DataFrame):
    """Extract digit keys from the 6 float columns and factorize jointly for fast m-estimate TE."""
    tr_keys = {}
    te_keys = {}
    for c in FLOAT_COLS:
        for prefix, df, store in [("tr", tr_df, tr_keys), ("te", te_df, te_keys)]:
            units = np.rint(df[c].to_numpy() * 100).astype(np.int64)
            integer = units // 100
            s = pd.Series(np.abs(integer)).astype(str)
            store[f"{c}__int"] = integer
            store[f"{c}__int10"] = units // 1000
            store[f"{c}__tenths"] = units // 10
            store[f"{c}__cents"] = units % 100
            store[f"{c}__lead1"] = (s.str.len() * 10 + s.str[0].astype(int)).to_numpy()
            store[f"{c}__lead2"] = (s.str.len() * 100 + s.str[:2].astype(int)).to_numpy()

    codes_dict = {}
    levels_dict = {}
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


def compute_digit_te_logits(
    codes_dict: dict,
    levels_dict: dict,
    tr_idx: np.ndarray,
    va_idx: np.ndarray | None,
    y_all: np.ndarray,
    seed: int,
    m: float = 20.0,
    inner_cv: int = 5,
):
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
    df_tr = pd.DataFrame(E_tr, columns=col_names)
    df_va = pd.DataFrame(E_va, columns=col_names) if va_idx is not None else None
    df_te = pd.DataFrame(E_te, columns=col_names)
    return df_tr, df_va, df_te


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
    for c, nb in [("merchant_risk_score", 7), ("hours_since_last_txn", 500), ("hours_since_last_txn", 1000)]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Fast smoke test on 50k rows")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--n-ens", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints if available")
    parser.add_argument("--no-refit", action="store_true", help="Skip full-data refit")
    args = parser.parse_args()

    stop_event = threading.Event()
    start_heartbeat(stop_event, interval_sec=20)

    try:
        t0 = time.time()
        data_dir = setup_environment()
        import torch
        from pytabkit import RealMLP_TD_Classifier

        tr_raw = pd.read_csv(os.path.join(data_dir, "train.csv"))
        te_raw = pd.read_csv(os.path.join(data_dir, "test.csv"))

        if args.smoke_test:
            print("[SMOKE-TEST] Subsampling to 50k train and 10k test...", flush=True)
            skf_sub = StratifiedKFold(n_splits=14, shuffle=True, random_state=42)
            _, sub_idx = next(skf_sub.split(tr_raw, tr_raw["is_fraud"]))
            tr_raw = tr_raw.iloc[sub_idx].reset_index(drop=True)
            te_raw = te_raw.iloc[:10000].reset_index(drop=True)
            args.n_ens = 2
            args.n_epochs = 1

        train_id = tr_raw["id"].to_numpy()
        test_id = te_raw["id"].to_numpy()
        y = tr_raw["is_fraud"].astype(int).to_numpy()
        tr_feat = tr_raw.drop(columns=["id", "is_fraud"])
        te_feat = te_raw.drop(columns=["id"])
        del tr_raw, te_raw
        gc.collect()

        n_tr, n_te = len(tr_feat), len(te_feat)
        n_splits = 2 if args.smoke_test else 5
        folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(np.zeros(n_tr), y))
        print(f"[DATA] train={n_tr}, test={n_te}, fraud_rate={y.mean():.5f}, folds={n_splits}", flush=True)

        print("[FEAT] Extracting digit-key codes & building base features...", flush=True)
        codes_dict, levels_dict = extract_digit_key_codes(tr_feat, te_feat)
        tr_feat, te_feat, all_cats, te_src, te_names = build_base_features(tr_feat, te_feat)
        print(f"[FEAT] Base cols={tr_feat.shape[1]}, cats={len(all_cats)}, digit_te_keys={len(codes_dict)}", flush=True)

        params = dict(
            verbosity=0,
            val_metric_name="1-auc_ovr",
            n_ens=args.n_ens,
            n_epochs=args.n_epochs,
            stop_epoch=args.n_epochs,
            batch_size=288,
            use_early_stopping=False,
            lr=0.053,
            wd=0.018,
            sq_mom=0.988,
            lr_sched="flat_anneal",
            wd_sched="cos_log_15",
            first_layer_lr_factor=0.25,
            embedding_size=5,
            max_one_hot_cat_size=16,
            hidden_sizes=[512, 256, 128],
            act="silu",
            p_drop=0.05,
            p_drop_sched="pow5",
            plr_hidden_1=16,
            plr_hidden_2=8,
            plr_act_name="gelu",
            plr_lr_factor=0.1151,
            plr_sigma=0.75,
            ls_eps=0.01,
            ls_eps_sched="sqrt_cos",
            add_front_scale=False,
            bias_init_mode="neg-uniform-dynamic-2",
            tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
        )

        if os.path.exists("/content/drive/MyDrive"):
            ckpt_dir = "/content/drive/MyDrive/exp041_ckpt"
        elif os.path.exists("/kaggle/working"):
            ckpt_dir = "/tmp/exp041_ckpt"
        else:
            ckpt_dir = "/content/exp041_ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)

        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

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
                m = rs.rand(len(tr_idx)) < 0.02
                Xf, yf = Xa.loc[~m], y[tr_idx][~m]
                Xv, yv = Xa.loc[m], y[tr_idx][m]
            else:
                Xf, yf = Xa, y[tr_idx]
                Xv, yv = Xb, y[va_idx]

            model = RealMLP_TD_Classifier(**{**params, "random_state": seed})
            with redirect_stdout(io.StringIO()):
                model.fit(Xf, yf, Xv, yv, cat_col_names=all_cats)

            pb = None if Xb is None else model.predict_proba(Xb)[:, 1].astype(np.float32)
            pt = model.predict_proba(Xt)[:, 1].astype(np.float32)
            del model, Xa, Xt, Xb, Xf, Xv, dte_tr, dte_va, dte_te
            gc.collect()
            torch.cuda.empty_cache()
            return pb, pt

        oof_all = np.zeros(n_tr, dtype=np.float64)
        foldavg_all = np.zeros(n_te, dtype=np.float64)
        refit_all = np.zeros(n_te, dtype=np.float64)

        for seed in seeds:
            oof_s = np.zeros(n_tr, dtype=np.float64)
            foldavg_s = np.zeros(n_te, dtype=np.float64)
            for f, (tr_idx, va_idx) in enumerate(folds):
                ft_oof = os.path.join(ckpt_dir, f"oof_s{seed}_f{f}.npy")
                ft_te = os.path.join(ckpt_dir, f"te_s{seed}_f{f}.npy")
                if args.resume and not args.smoke_test and os.path.exists(ft_oof) and os.path.exists(ft_te):
                    oof_s[va_idx] = np.load(ft_oof)
                    foldavg_s += np.load(ft_te) / n_splits
                    print(f"[RESUME] Loaded seed={seed} fold={f} AUC={roc_auc_score(y[va_idx], oof_s[va_idx]):.6f}", flush=True)
                    continue

                tf0 = time.time()
                pb, pt = run_one(tr_idx, va_idx, seed + f)
                oof_s[va_idx] = pb
                foldavg_s += pt / n_splits
                if not args.smoke_test:
                    np.save(ft_oof, pb)
                    np.save(ft_te, pt)
                print(
                    f"[FOLD] seed={seed} fold={f} AUC={roc_auc_score(y[va_idx], pb):.6f} "
                    f"| {(time.time() - tf0) / 60:.2f} min",
                    flush=True,
                )

            seed_auc = roc_auc_score(y, oof_s)
            print(f"[SEED {seed}] OOF AUC={seed_auc:.6f}", flush=True)
            oof_all += oof_s / len(seeds)
            foldavg_all += foldavg_s / len(seeds)

            if not args.no_refit:
                ft_ref = os.path.join(ckpt_dir, f"refit_s{seed}.npy")
                if args.resume and not args.smoke_test and os.path.exists(ft_ref):
                    pt_ref = np.load(ft_ref)
                    print(f"[RESUME] Loaded refit seed={seed}", flush=True)
                else:
                    tf0 = time.time()
                    _, pt_ref = run_one(np.arange(n_tr), None, seed + 100)
                    if not args.smoke_test:
                        np.save(ft_ref, pt_ref)
                    print(f"[REFIT] seed={seed} done | {(time.time() - tf0) / 60:.2f} min", flush=True)
                refit_all += pt_ref / len(seeds)

        total_auc = roc_auc_score(y, oof_all)
        print("=" * 70, flush=True)
        print(f"[DONE] Exp #041 RealMLP Overall OOF AUC = {total_auc:.6f} | Total {(time.time() - t0) / 60:.2f} min", flush=True)
        print("=" * 70, flush=True)

        oof_df = pd.DataFrame({"id": train_id, "oof_realmlp_exp041": oof_all.astype(np.float32)})
        sub_fold_df = pd.DataFrame({"id": test_id, "is_fraud": foldavg_all.astype(np.float32)})
        sub_refit_df = pd.DataFrame({"id": test_id, "is_fraud": refit_all.astype(np.float32)})

        oof_df.to_csv("oof_exp041_realmlp.csv", index=False)
        sub_fold_df.to_csv("submission_exp041_foldavg.csv", index=False)
        if not args.no_refit:
            sub_refit_df.to_csv("submission_exp041_refit.csv", index=False)

        if os.path.exists("/content/drive/MyDrive"):
            oof_df.to_csv("/content/drive/MyDrive/oof_exp041_realmlp.csv", index=False)
            sub_fold_df.to_csv("/content/drive/MyDrive/submission_exp041_foldavg.csv", index=False)
            if not args.no_refit:
                sub_refit_df.to_csv("/content/drive/MyDrive/submission_exp041_refit.csv", index=False)
            print("[DRIVE] Synced outputs to /content/drive/MyDrive/", flush=True)

        if args.smoke_test:
            print("[SMOKE-TEST] Sanity check passed! Skipping Kaggle upload/Base64 dump.", flush=True)
            return

        # Save outputs in compressed npz as well
        np.savez_compressed(
            "exp041_outputs.npz",
            oof=oof_all.astype(np.float32),
            foldavg=foldavg_all.astype(np.float32),
            refit=refit_all.astype(np.float32),
        )
        if not os.path.exists("/kaggle/working"):
            with open("exp041_outputs.npz", "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            print("===BASE64_EXP041_NPZ_START===", flush=True)
            for i in range(0, len(b64), 4000):
                print(b64[i : i + 4000], flush=True)
            print("===BASE64_EXP041_NPZ_END===", flush=True)

    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
