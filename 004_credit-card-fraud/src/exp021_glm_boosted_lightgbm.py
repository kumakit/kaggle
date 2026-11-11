"""
004_credit-card-fraud: Exp #021 - Boosting from a GLM (GLM Initial Logit + 4-leaf LightGBM)
Competition: Playground Series Season 6 Episode 11 (Credit Card Fraud Detection)

Architecture:
  1. 36 Digit Keys (from 6 float columns) + 13 Interaction Pairs + 24 Base Features.
  2. High-speed In-Fold Target Encoding with np.bincount (smoothing m=50).
  3. Additive GLM (LogisticRegression solver='newton-cholesky', C=0.3) fitted on logit of encodings.
     -> Captures the underlying logistic formula generating the labels.
  4. 4-leaf LightGBM starting from GLM's logit (init_score=z).
     -> Trees fit strictly the non-linear interactions & residual errors.
  5. 5-Fold Stratified CV (seed=42).
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

FLOAT_COLS = ['amount_usd', 'hours_since_last_txn', 'distance_from_home_km', 'account_balance_usd', 'velocity_score', 'merchant_risk_score']

PAIRS = [
    ("merchant_category", "merchant_risk_score"),
    ("is_foreign_transaction", "ip_country_mismatch"),
    ("merchant_category", "q:amount_usd"),
    ("txn_count_last_24h", "q:velocity_score"),
    ("merchant_category", "customer_age"),
    ("q:amount_usd", "q:merchant_risk_score"),
    ("q:hours_since_last_txn", "q:velocity_score"),
    ("customer_age", "q:merchant_risk_score"),
    ("is_new_merchant", "day_of_week"),
    ("is_ai_generated_scam_attempt", "q:velocity_score"),
    ("customer_age", "card_age_months"),
    ("time_of_day_hour", "day_of_week"),
    ("txn_count_last_24h", "customer_age")
]


def extract_digit_keys(df):
    keys = {}
    for c in FLOAT_COLS:
        units = np.rint(df[c].to_numpy() * 100).astype(np.int64)
        integer = units // 100
        s = pd.Series(integer).astype(str)
        keys[f"{c}__int"] = integer
        keys[f"{c}__int10"] = units // 1000
        keys[f"{c}__tenths"] = units // 10
        keys[f"{c}__cents"] = units % 100
        keys[f"{c}__lead1"] = (s.str.len() * 10 + s.str[0].astype(int)).to_numpy()
        keys[f"{c}__lead2"] = (s.str.len() * 100 + s.str[:2].astype(int)).to_numpy()
    return pd.DataFrame(keys, index=df.index)


def factorize_shared(frame_tr, frame_te, cols):
    codes, n_levels = {}, {}
    for c in cols:
        combined = pd.concat([frame_tr[c], frame_te[c]], ignore_index=True)
        k, uniques = pd.factorize(combined)
        k = k.astype(np.int32)
        codes[c] = (k[:len(frame_tr)], k[len(frame_tr):])
        n_levels[c] = len(uniques)
    return codes, n_levels


def te_table(codes, target, n_levels, m):
    frauds = np.bincount(codes, weights=target, minlength=n_levels)
    rows = np.bincount(codes, minlength=n_levels)
    return (frauds + m * target.mean()) / (rows + m)


def target_encode_fold(cols, codes_dict, levels_dict, tr, va, y_all, seed, m=50, inner_cv=5):
    y_tr = y_all[tr].astype(float)
    inner = list(KFold(inner_cv, shuffle=True, random_state=seed).split(tr))
    
    E_tr = np.empty((len(tr), len(cols)), dtype=np.float32)
    E_va = np.empty((len(va), len(cols)), dtype=np.float32)
    n_test = len(codes_dict[cols[0]][1])
    E_te = np.empty((n_test, len(cols)), dtype=np.float32)
    
    for j, c in enumerate(cols):
        c_train, c_test = codes_dict[c]
        k_tr = c_train[tr]
        n_lev = levels_dict[c]
        
        # Cross-fitted for training rows
        for fit_idx, enc_idx in inner:
            tab = te_table(k_tr[fit_idx], y_tr[fit_idx], n_lev, m)
            E_tr[enc_idx, j] = tab[k_tr[enc_idx]]
            
        # Full training fit for valid & test
        full_tab = te_table(k_tr, y_tr, n_lev, m)
        E_va[:, j] = full_tab[c_train[va]]
        E_te[:, j] = full_tab[c_test]
        
    names = [f"te_{c}" for c in cols]
    return (pd.DataFrame(E_tr, columns=names),
            pd.DataFrame(E_va, columns=names),
            pd.DataFrame(E_te, columns=names))


def main():
    print("=" * 80)
    print("  004 Credit Card Fraud: Exp #021 - Boosting from a GLM (GLM + 4-leaf LightGBM)")
    print("=" * 80)

    # 1. Load data
    train = pd.read_csv("004_credit-card-fraud/data/train.csv")
    test = pd.read_csv("004_credit-card-fraud/data/test.csv")
    print(f"Train: {train.shape}, Test: {test.shape}")
    y = train["is_fraud"].values

    FEATURES = [c for c in test.columns if c != "id"]
    BOOL = [c for c in FEATURES if train[c].dtype == 'bool']
    NUM = [c for c in FEATURES if train[c].dtype in ['int64', 'float64'] and c not in BOOL]
    CAT = [c for c in FEATURES if c not in BOOL and c not in NUM]
    print(f"{len(NUM)} numeric · {len(BOOL)} boolean · {len(CAT)} categorical")

    # 2. Extract digit keys
    print("\n[1/5] Extracting 36 digit keys...")
    keys_tr = extract_digit_keys(train)
    keys_te = extract_digit_keys(test)
    KEYS = list(keys_tr.columns)
    print(f"Extracted {len(KEYS)} digit keys.")

    # 3. Build 13 Interaction Pairs
    print("\n[2/5] Constructing 13 Interaction Pairs...")
    both = pd.concat([train[FEATURES], test[FEATURES]], ignore_index=True)
    def pair_part(col):
        return pd.qcut(both[col[2:]], 10, labels=False, duplicates="drop") if col.startswith("q:") else both[col]
    
    pairs_df = pd.DataFrame({
        f"{a}*{b}".replace("q:", ""): pair_part(a).astype(str) + "|" + pair_part(b).astype(str)
        for a, b in PAIRS
    })
    PAIR_KEYS = list(pairs_df.columns)
    pairs_tr = pairs_df.iloc[:len(train)].reset_index(drop=True)
    pairs_te = pairs_df.iloc[len(train):].reset_index(drop=True)
    del both, pairs_df
    gc.collect()
    print(f"Created {len(PAIR_KEYS)} pair columns.")

    # 4. Factorize all keys (base + digits + pairs)
    print("\n[3/5] Factorizing categorical and integer levels for Target Encoding...")
    ALL_KEYS = FEATURES + KEYS
    GLM_KEYS = ALL_KEYS + PAIR_KEYS

    # Merge into unified frames for factorizing
    frame_tr = pd.concat([train[FEATURES], keys_tr, pairs_tr], axis=1)
    frame_te = pd.concat([test[FEATURES], keys_te, pairs_te], axis=1)

    CODES, N_LEVELS = factorize_shared(frame_tr, frame_te, GLM_KEYS)
    print(f"Total encoded features for GLM: {len(GLM_KEYS)}")

    # Prepare raw base frames for LightGBM
    CATEGORIES = {c: sorted(pd.concat([train[c], test[c]]).unique()) for c in CAT}
    def model_frame(df):
        out = df[FEATURES].copy()
        for c in CAT:
            out[c] = pd.Categorical(out[c], categories=CATEGORIES[c])
        out[BOOL] = out[BOOL].astype(int)
        return out

    X_RAW = model_frame(train)
    T_RAW = model_frame(test)

    # 5. Cross-Validation
    print("\n[4/5] Running 5-Fold Stratified CV with GLM Boosting...")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_glm = np.zeros(len(train))
    test_glm = np.zeros(len(test))

    oof_boost = np.zeros(len(train))
    test_boost = np.zeros(len(test))

    to_logit = lambda E: logit(np.clip(E.to_numpy(np.float64), 1e-5, 1 - 1e-5))
    log_num = lambda df: np.log1p(df[NUM].to_numpy(float))

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 4,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42
    }

    for fold, (tr, va) in enumerate(skf.split(train, y), 1):
        print(f"\n--- FOLD {fold}/{n_splits} ---")

        # A. Target Encoding (m=50) for GLM
        E_tr, E_va, E_te = target_encode_fold(GLM_KEYS, CODES, N_LEVELS, tr, va, y, seed=42 + fold, m=50)

        # B. Design matrix for GLM
        sc_e = StandardScaler().fit(to_logit(E_tr))
        sc_n = StandardScaler().fit(log_num(train.iloc[tr]))
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[CAT].iloc[tr])

        def build_glm_design(E, df):
            return np.hstack([
                sc_e.transform(to_logit(E)),
                oh.transform(df[CAT]),
                sc_n.transform(log_num(df)),
                df[BOOL].to_numpy(float)
            ])

        X_glm_tr = build_glm_design(E_tr, train.iloc[tr])
        X_glm_va = build_glm_design(E_va, train.iloc[va])
        X_glm_te = build_glm_design(E_te, test)

        glm = LogisticRegression(C=0.3, solver="newton-cholesky", tol=1e-8, max_iter=200)
        glm.fit(X_glm_tr, y[tr])

        z_tr = glm.decision_function(X_glm_tr)
        z_va = glm.decision_function(X_glm_va)
        z_te = glm.decision_function(X_glm_te)

        p_glm_va = expit(z_va)
        oof_glm[va] = p_glm_va
        test_glm += expit(z_te) / n_splits
        glm_auc = roc_auc_score(y[va], p_glm_va)
        print(f"  [Fold {fold}] Pure GLM AUC: {glm_auc:.5f} (converged in {glm.n_iter_[0]} iter)")

        # C. 4-leaf LightGBM on top of GLM's logit
        # Extract encodings for ALL_KEYS (base + digits) for tree features
        lgb_E_tr = E_tr[[f"te_{c}" for c in ALL_KEYS]]
        lgb_E_va = E_va[[f"te_{c}" for c in ALL_KEYS]]
        lgb_E_te = E_te[[f"te_{c}" for c in ALL_KEYS]]

        X_lgb_tr = pd.concat([X_RAW.iloc[tr].reset_index(drop=True), lgb_E_tr], axis=1)
        X_lgb_va = pd.concat([X_RAW.iloc[va].reset_index(drop=True), lgb_E_va], axis=1)
        X_lgb_te = pd.concat([T_RAW.reset_index(drop=True), lgb_E_te], axis=1)

        d_tr = lgb.Dataset(X_lgb_tr, y[tr], init_score=z_tr)
        d_va = lgb.Dataset(X_lgb_va, y[va], init_score=z_va, reference=d_tr)

        booster = lgb.train(
            lgb_params,
            d_tr,
            num_boost_round=30000,
            valid_sets=[d_va],
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )

        delta_va = booster.predict(X_lgb_va, num_iteration=booster.best_iteration, raw_score=True)
        delta_te = booster.predict(X_lgb_te, num_iteration=booster.best_iteration, raw_score=True)

        p_boost_va = expit(z_va + delta_va)
        p_boost_te = expit(z_te + delta_te)

        oof_boost[va] = p_boost_va
        test_boost += p_boost_te / n_splits

        boost_auc = roc_auc_score(y[va], p_boost_va)
        print(f"  [Fold {fold}] GLM + LightGBM AUC: {boost_auc:.5f} (gain: {boost_auc - glm_auc:+.5f}, trees: {booster.best_iteration})")

        del E_tr, E_va, E_te, X_glm_tr, X_glm_va, X_glm_te, X_lgb_tr, X_lgb_va, X_lgb_te, d_tr, d_va
        gc.collect()

    # 6. Evaluation
    total_glm_auc = roc_auc_score(y, oof_glm)
    total_boost_auc = roc_auc_score(y, oof_boost)

    print("\n" + "=" * 70)
    print(f"Exp #021 Pure GLM OOF ROC-AUC:              {total_glm_auc:.5f}")
    print(f"Exp #021 GLM + LightGBM OOF ROC-AUC:       {total_boost_auc:.5f} (Tree gain: {total_boost_auc - total_glm_auc:+.5f})")
    print(f"  (Reference Exp #019 Single XGBoost:       0.94994 | Gain: {total_boost_auc - 0.94994:+.5f})")
    print(f"  (Reference Exp #020 Mega Blend:           0.95002 | Gain: {total_boost_auc - 0.95002:+.5f})")
    print("=" * 70)

    # 7. Save Submissions & OOFs
    np.save("004_credit-card-fraud/data/oof_exp021.npy", oof_boost)
    np.save("004_credit-card-fraud/data/oof_exp021_pure_glm.npy", oof_glm)

    sub_boost = pd.DataFrame({"id": test["id"], "is_fraud": test_boost})
    sub_boost.to_csv("004_credit-card-fraud/data/submission_exp021.csv", index=False)
    print(f"Saved submission_exp021.csv (Rows: {len(sub_boost)})")

    sub_glm = pd.DataFrame({"id": test["id"], "is_fraud": test_glm})
    sub_glm.to_csv("004_credit-card-fraud/data/submission_exp021_pure_glm.csv", index=False)
    print(f"Saved submission_exp021_pure_glm.csv (Rows: {len(sub_glm)})")


if __name__ == "__main__":
    main()
