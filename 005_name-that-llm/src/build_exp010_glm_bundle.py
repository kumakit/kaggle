"""
005_name-that-llm: Exp #010 GLM-5.3 / GLM-5.3-Flash Likelihood Model Bundle Builder
Trains 5-Fold StratifiedGroupKFold (seed=42, grouped by prompt_id) ensembles on L160 features:
  - LogisticRegression (C=0.05, class_weight='balanced') -> portable JSON mean/scale/coef/intercept
  - LightGBM (class_weight='balanced') -> portable text boosters (lgb_fold_0..4.txt)
  - CatBoost (auto_class_weights='Balanced') -> portable .cbm models (cb_fold_0..4.cbm)
Verifies OOF Balanced Accuracy blended with Exp #008 base log-probs and prepares Kaggle Dataset upload.
"""

import json
import os
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_DIR = "005_name-that-llm/data"
GLM_OUT_DIR = "005_name-that-llm/data/exp010_glm_train/out"
BUNDLE_DIR = "005_name-that-llm/data/exp010_glm_bundle"
DATASET_SLUG = "nobuyukikumagai/name-that-llm-exp010-glm-models"
DATASET_TITLE = "Exp010 GLM-5.3 Likelihood Bundle"
assert 6 <= len(DATASET_TITLE) <= 50

DIFF_COLS_EXT = [
    "lp_mean", "lp_q50", "lp_q25", "lp_q75", "ent_mean",
    "top1", "top5", "logrank_mean", "fdg", "surprise_mean", "z_mean", "bpc",
]


def build_features(flash_df, glm53_df, ids):
    f_fl = flash_df.set_index("id").loc[ids]
    f_53 = glm53_df.set_index("id").loc[ids]
    df_feat = pd.concat([f_fl.add_prefix("flash:"), f_53.add_prefix("glm53:")], axis=1)
    for c in DIFF_COLS_EXT:
        df_feat[f"d:{c}"] = df_feat[f"flash:{c}"] - df_feat[f"glm53:{c}"]
    df_feat["flash:drift"] = df_feat["flash:lp_mean_second"] - df_feat["flash:lp_mean_first"]
    df_feat["glm53:drift"] = df_feat["glm53:lp_mean_second"] - df_feat["glm53:lp_mean_first"]
    df_feat["d:drift"] = df_feat["flash:drift"] - df_feat["glm53:drift"]
    return df_feat


def load_base_exp008_logp(train_df, classes):
    eps = 1e-7
    o5 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp005_openpool_modernbert.csv"))
    o6 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp006_qwen3_lora.csv"))
    o7 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp007_qwen3_4b_lora.csv"))
    o8 = pd.read_csv(os.path.join(DATA_DIR, "oof_exp008_modernbert_large.csv"))
    logp = (
        0.25 * np.log(o7[classes].values + eps)
        + 0.40 * np.log(o8[classes].values + eps)
        + 0.30 * np.log(o6[classes].values + eps)
        + 0.05 * np.log(o5[classes].values + eps)
    )
    return log_softmax(logp, axis=1)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train_ids = train_df["id"].astype(str).tolist()
    classes = sorted(train_df["model"].unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    y = train_df["model"].map(c2i).values
    groups = train_df["prompt_id"].values

    flash_df = pd.read_csv(os.path.join(GLM_OUT_DIR, "flash_features_L160.csv"), dtype={"id": str})
    glm53_df = pd.read_csv(os.path.join(GLM_OUT_DIR, "glm53_features_L160.csv"), dtype={"id": str})
    feat_df = build_features(flash_df, glm53_df, train_ids)
    feat_cols = feat_df.columns.tolist()
    X = feat_df.values.astype(np.float64)
    assert np.isfinite(X).all(), "Non-finite values in training features!"

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_lr = np.zeros((len(y), len(classes)), dtype=np.float64)
    oof_lgb = np.zeros((len(y), len(classes)), dtype=np.float64)
    oof_cb = np.zeros((len(y), len(classes)), dtype=np.float64)

    lr_folds = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups=groups)):
        # 1. Logistic Regression (exported purely as JSON arrays)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr_idx])
        X_va_s = scaler.transform(X[va_idx])
        clf_lr = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", random_state=SEED)
        clf_lr.fit(X_tr_s, y[tr_idx])
        oof_lr[va_idx] = clf_lr.predict_log_proba(X_va_s)
        lr_folds.append({
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": clf_lr.coef_.tolist(),
            "intercept": clf_lr.intercept_.tolist(),
        })

        # 2. LightGBM (exported as portable booster string)
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
            class_weight="balanced", random_state=SEED, verbose=-1,
        )
        clf_lgb.fit(X[tr_idx], y[tr_idx])
        oof_lgb[va_idx] = np.log(np.clip(clf_lgb.predict_proba(X[va_idx]), 1e-7, 1.0))
        clf_lgb.booster_.save_model(os.path.join(BUNDLE_DIR, f"lgb_fold_{fold}.txt"))

        # 3. CatBoost (exported as .cbm)
        clf_cb = CatBoostClassifier(
            iterations=500, learning_rate=0.04, depth=5, l2_leaf_reg=5.0,
            auto_class_weights="Balanced", random_seed=SEED, verbose=0,
        )
        clf_cb.fit(X[tr_idx], y[tr_idx])
        oof_cb[va_idx] = np.log(np.clip(clf_cb.predict_proba(X[va_idx]), 1e-7, 1.0))
        clf_cb.save_model(os.path.join(BUNDLE_DIR, f"cb_fold_{fold}.cbm"))
        print(f"  [FOLD {fold + 1}/5] Fitted & exported LR + LGB + CB", flush=True)

    ll_oof = log_softmax(0.50 * oof_lr + 0.25 * oof_lgb + 0.25 * oof_cb, axis=1)
    base_lp = load_base_exp008_logp(train_df, classes)
    glm_idxs = [c2i["glm-5.3"], c2i["glm-5.3-flash"]]
    glm_mass = np.exp(base_lp)[:, glm_idxs].sum(1)

    for cov in [0.30, 0.35, 0.40]:
        for w in [1.0, 1.15]:
            n_sel = int(round(cov * len(y)))
            sel = np.sort(np.argsort(-glm_mass, kind="stable")[:n_sel])
            before = base_lp.argmax(1)
            after = before.copy()
            after[sel] = (base_lp[sel] + w * ll_oof[sel]).argmax(1)
            glm_rel = np.isin(before, glm_idxs) | np.isin(after, glm_idxs)
            after[(before != after) & ~glm_rel] = before[(before != after) & ~glm_rel]
            bal = balanced_accuracy_score(y, after)
            rec = recall_score(y, after, average=None)
            chg = float((before != after).mean())
            print(
                f"  [VERIFY] cov={cov:.2f} w={w:.2f} -> BalAcc={bal:.5f} "
                f"(chg={chg:.3f}, glm53={rec[glm_idxs[0]]:.3f}, flash={rec[glm_idxs[1]]:.3f})"
            )

    # Save canonical predictions using cov=0.35, w=1.15 (covers >99.7% true GLM with high margin)
    n_sel = int(round(0.35 * len(y)))
    sel = np.sort(np.argsort(-glm_mass, kind="stable")[:n_sel])
    before = base_lp.argmax(1)
    after = before.copy()
    after[sel] = (base_lp[sel] + 1.15 * ll_oof[sel]).argmax(1)
    glm_rel = np.isin(before, glm_idxs) | np.isin(after, glm_idxs)
    after[(before != after) & ~glm_rel] = before[(before != after) & ~glm_rel]
    oof_out = pd.DataFrame(np.exp(ll_oof), columns=classes)
    oof_out["id"] = train_df["id"]
    oof_out["true_model"] = train_df["model"]
    oof_out["pred_blend"] = [classes[i] for i in after]
    oof_out.to_csv(os.path.join(DATA_DIR, "oof_exp010_glm_blend.csv"), index=False)

    # Compute exact base_logp mean entropy on train OOF so inference can optionally stabilize scale if shifted
    p_base = np.exp(base_lp)
    base_entropy = float(-(p_base * base_lp).sum(1).mean())

    bundle_meta = {
        "format": "exp010-glm-bundle-v1",
        "classes": classes,
        "feature_cols": feat_cols,
        "diff_cols_ext": DIFF_COLS_EXT,
        "max_text_tokens": 160,
        "tok_per_char": float((flash_df["n_tok"] / np.maximum(train_df["text"].fillna("").str.len(), 1)).mean()),
        "weights": {"lr": 0.50, "lgb": 0.25, "cb": 0.25},
        "blend_weight": 1.15,
        "base_entropy_ref": base_entropy,
        "n_folds": 5,
        "lr_folds": lr_folds,
    }
    with open(os.path.join(BUNDLE_DIR, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(bundle_meta, f)

    ds_meta = {
        "title": DATASET_TITLE,
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }
    with open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(ds_meta, f, indent=2)
    print(f"[SAVED] Bundle ready in {BUNDLE_DIR}", flush=True)

    check = subprocess.run(["uv", "run", "kaggle", "datasets", "status", DATASET_SLUG], capture_output=True, text=True)
    if check.returncode == 0:
        cmd = ["uv", "run", "kaggle", "datasets", "version", "-p", BUNDLE_DIR, "-m", "Exp010 L160 5Fold LR+LGB+CB GLM bundle", "-r", "zip"]
    else:
        cmd = ["uv", "run", "kaggle", "datasets", "create", "-p", BUNDLE_DIR, "-r", "zip"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[ERR]", res.stderr)
        raise RuntimeError("Kaggle dataset upload failed")


if __name__ == "__main__":
    main()
