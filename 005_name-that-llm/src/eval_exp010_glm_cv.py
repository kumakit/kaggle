"""
005_name-that-llm: Exp #010 Local 5-Fold CV Evaluator & Meta-Model Builder
Evaluates real GLM-5.3-Flash + GLM-5.3 likelihood features (L160 & L320) strictly on
StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) grouped by prompt_id.
Compares:
  1. Standalone GLM-5.3/Flash likelihood classifiers (LogisticRegression & LightGBM)
  2. Blending into Exp #008 base ensemble across coverage, weight W, and GLM-restriction
  3. Full meta-stacker combining Exp #005-#008 log-probs + GLM features (Nested-CV verified)
Exports fitted numpy-portable JSON weights for offline Kaggle T4x2 inference.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_DIR = "005_name-that-llm/data"
GLM_OUT_DIR = "005_name-that-llm/data/exp010_glm_train/out"

DIFF_COLS = [
    "lp_mean", "lp_q50", "lp_q25", "lp_q75", "ent_mean",
    "top1", "top5", "logrank_mean", "fdg", "surprise_mean", "z_mean", "bpc",
]
DIFF_COLS_V1 = ["lp_mean", "lp_q50", "ent_mean", "top1", "logrank_mean", "fdg"]


def entropy(z):
    p = np.exp(z)
    return float(-(p * z).sum(1).mean())


def match_temperature(z, h_ref):
    lo, hi = 1e-3, 1e3
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if entropy(log_softmax(z * mid, 1)) > h_ref:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def build_feature_frame(flash_csv, glm53_csv, train_ids, extra_diffs=False):
    f_fl = pd.read_csv(flash_csv, dtype={"id": str}).set_index("id").loc[train_ids]
    f_53 = pd.read_csv(glm53_csv, dtype={"id": str}).set_index("id").loc[train_ids]
    df_feat = pd.concat([f_fl.add_prefix("flash:"), f_53.add_prefix("glm53:")], axis=1)
    d_cols = DIFF_COLS if extra_diffs else DIFF_COLS_V1
    for c in d_cols:
        df_feat[f"d:{c}"] = df_feat[f"flash:{c}"] - df_feat[f"glm53:{c}"]
    if extra_diffs:
        df_feat["flash:drift"] = df_feat["flash:lp_mean_second"] - df_feat["flash:lp_mean_first"]
        df_feat["glm53:drift"] = df_feat["glm53:lp_mean_second"] - df_feat["glm53:lp_mean_first"]
        df_feat["d:drift"] = df_feat["flash:drift"] - df_feat["glm53:drift"]
    return df_feat


def get_base_ensemble_logp(train_df, classes):
    eps = 1e-7
    o5 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp005_openpool_modernbert.csv"))
    o6 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp006_qwen3_lora.csv"))
    o7 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp007_qwen3_4b_lora.csv"))
    o8 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp008_modernbert_large.csv"))
    for o in (o5, o6, o7, o8):
        assert o["id"].astype(str).tolist() == train_df["id"].astype(str).tolist()
    logp = (
        0.25 * np.log(o7[classes].values + eps)
        + 0.40 * np.log(o8[classes].values + eps)
        + 0.30 * np.log(o6[classes].values + eps)
        + 0.05 * np.log(o5[classes].values + eps)
    )
    return log_softmax(logp, axis=1), (o5[classes].values, o6[classes].values, o7[classes].values, o8[classes].values)


def run_cv_logreg(X, y, groups, C=0.05):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_logp = np.zeros((len(y), len(np.unique(y))), dtype=np.float64)
    for tr_idx, va_idx in splitter.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y[tr_idx])
        oof_logp[va_idx] = clf.predict_log_proba(X_va)
    return oof_logp


def run_cv_lgb(X, y, groups):
    import lightgbm as lgb

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(y), len(np.unique(y))), dtype=np.float64)
    for tr_idx, va_idx in splitter.split(X, y, groups=groups):
        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        oof_probs[va_idx] = clf.predict_proba(X[va_idx])
    return np.log(np.clip(oof_probs, 1e-7, 1.0))


def evaluate_blend(base_logp, ll_logp, y, classes, cover=0.45, weight=1.0, h_ref=None, restrict_glm=True):
    probs_base = np.exp(base_logp)
    glm_idxs = [classes.index("glm-5.3"), classes.index("glm-5.3-flash")]
    glm_mass = probs_base[:, glm_idxs].sum(1)
    n_sel = int(round(cover * len(y)))
    sel = np.sort(np.argsort(-glm_mass, kind="stable")[:n_sel])

    if h_ref is not None:
        tp = match_temperature(base_logp, h_ref)
        w = weight / tp
    else:
        w = weight

    before = base_logp.argmax(1)
    after = before.copy()
    after[sel] = (base_logp[sel] + w * ll_logp[sel]).argmax(1)

    if restrict_glm:
        glm_arr = np.asarray(glm_idxs)
        glm_rel = np.isin(before, glm_arr) | np.isin(after, glm_arr)
        blocked = (before != after) & ~glm_rel
        after[blocked] = before[blocked]

    bal = balanced_accuracy_score(y, after)
    chg = float((before != after).mean())
    rec = recall_score(y, after, labels=list(range(len(classes))), average=None)
    return bal, chg, rec[glm_idxs[0]], rec[glm_idxs[1]], after


def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_ids = train_df["id"].astype(str).tolist()
    classes = sorted(train_df["model"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y = train_df["model"].map(c2i).values
    groups = train_df["prompt_id"].values

    base_logp, _ = get_base_ensemble_logp(train_df, classes)
    base_pred = base_logp.argmax(1)
    base_bal = balanced_accuracy_score(y, base_pred)
    base_rec = recall_score(y, base_pred, average=None)
    print(
        f"Base Exp008 Log-Prob OOF: {base_bal:.5f} "
        f"(glm-5.3={base_rec[c2i['glm-5.3']]:.3f}, glm-5.3-flash={base_rec[c2i['glm-5.3-flash']]:.3f})"
    )

    for tag, fl_file, g53_file in [
        ("L160", "flash_features_L160.csv", "glm53_features_L160.csv"),
        ("L320", "flash_features.csv", "glm53_features.csv"),
    ]:
        fl_path = os.path.join(GLM_OUT_DIR, fl_file)
        g53_path = os.path.join(GLM_OUT_DIR, g53_file)
        if not (os.path.exists(fl_path) and os.path.exists(g53_path)):
            print(f"[SKIP {tag}] Missing {fl_path} or {g53_path}")
            continue

        print(f"\n================ {tag} Evaluation ================")
        feat_54 = build_feature_frame(fl_path, g53_path, train_ids, extra_diffs=False)
        feat_ext = build_feature_frame(fl_path, g53_path, train_ids, extra_diffs=True)

        for C in [0.02, 0.05, 0.1, 0.25, 0.5]:
            lp = run_cv_logreg(feat_54.values, y, groups, C=C)
            sc = balanced_accuracy_score(y, lp.argmax(1))
            r = recall_score(y, lp.argmax(1), average=None)
            print(
                f"  LogReg-54 C={C:<4} single BalAcc: {sc:.5f} "
                f"(glm-5.3={r[c2i['glm-5.3']]:.3f}, flash={r[c2i['glm-5.3-flash']]:.3f})"
            )

        lp_ext = run_cv_logreg(feat_ext.values, y, groups, C=0.05)
        sc_ext = balanced_accuracy_score(y, lp_ext.argmax(1))
        print(f"  LogReg-Ext ({feat_ext.shape[1]}f) C=0.05 single BalAcc: {sc_ext:.5f}")

        lp_lgb = run_cv_lgb(feat_ext.values, y, groups)
        sc_lgb = balanced_accuracy_score(y, lp_lgb.argmax(1))
        print(f"  LightGBM-Ext single BalAcc: {sc_lgb:.5f}")

        lp_v1_best = run_cv_logreg(feat_54.values, y, groups, C=0.05)
        for name, ll_lp in [("LogReg-54(C=0.05)", lp_v1_best), ("LogReg-Ext", lp_ext), ("LGBM-Ext", lp_lgb)]:
            print(f"\n  --- Blend with {name} ---")
            for restrict in [True, False]:
                for cov in [0.30, 0.45, 0.60, 1.00]:
                    for w in [0.75, 1.00, 1.25, 1.50]:
                        bal, chg, r53, rfl, _ = evaluate_blend(
                            base_logp, ll_lp, y, classes, cover=cov, weight=w, h_ref=None, restrict_glm=restrict
                        )
                        print(
                            f"    restrict={str(restrict):5s} cov={cov:.2f} w={w:.2f} -> "
                            f"BalAcc={bal:.5f} (diff={bal - base_bal:+.5f}, chg={chg:.3f}, "
                            f"glm53={r53:.3f}, flash={rfl:.3f})"
                        )

        # Nested CV for choosing (restrict, cov, w) on LogReg-Ext
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        nested_preds = np.zeros_like(y)
        chosen = []
        for tr_idx, va_idx in splitter.split(feat_ext.values, y, groups=groups):
            best_tr = -1.0
            best_cfg = None
            for restrict in [True, False]:
                for cov in [0.30, 0.40, 0.50, 0.65, 1.00]:
                    for w in [0.50, 0.75, 1.00, 1.25, 1.50]:
                        bal_tr, _, _, _, _ = evaluate_blend(
                            base_logp[tr_idx], lp_ext[tr_idx], y[tr_idx], classes,
                            cover=cov, weight=w, h_ref=None, restrict_glm=restrict,
                        )
                        if bal_tr > best_tr:
                            best_tr = bal_tr
                            best_cfg = (restrict, cov, w)
            chosen.append(best_cfg)
            _, _, _, _, pred_va = evaluate_blend(
                base_logp[va_idx], lp_ext[va_idx], y[va_idx], classes,
                cover=best_cfg[1], weight=best_cfg[2], h_ref=None, restrict_glm=best_cfg[0],
            )
            nested_preds[va_idx] = pred_va
        print(f"  [Nested-CV {tag} LogReg-Ext] BalAcc={balanced_accuracy_score(y, nested_preds):.5f} | folds={chosen}")


if __name__ == "__main__":
    main()
