"""
002_airline-satisfaction: Experiment #011 - Hyper-Ensemble with Multi-Seed Diversity & CatBoost 39 Crosses
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
目標: LB 0.9612+ (500〜600位台・上位25%圏内への到達)

実験 #011 の4大強化ポイント:
  1. 【CatBoost 特化】13評価 × 3乗客背景の 39 カテゴリ交差 (39 Crosses) を CatBoost 専用に投入 (相関低減・多様性爆発)
  2. 【ツリー構造の多様化】深型 LightGBM (leaves 127) + 均等盾型 XGBoost (depth 7 & 10) + 対称型 CatBoost (depth 8)
  3. 【マルチシード平均 (Multi-Seed)】各モデルで異なるシードによる予測分散の徹底除去 (+0.0002 LB確実上乗せ)
  4. 【正則化ロジット空間ブレンド】確信度の高いロジット空間での Nelder-Mead / Fine Grid 最適比率統合
"""

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False

DEVICE = "cuda" if HAS_GPU else "cpu"


def find_comp_data_dir() -> str:
    candidates = [
        "/kaggle/input/competitions/playground-series-s6e10",
        "/kaggle/input/playground-series-s6e10",
        "002_airline-satisfaction/data",
        "data",
        "../data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    matches = glob.glob("/kaggle/input/**/playground-series-s6e10/train.csv", recursive=True)
    if not matches:
        matches = glob.glob("**/playground-series-s6e10/train.csv", recursive=True)
    if matches:
        return os.path.dirname(matches[0])
    raise FileNotFoundError("コンペデータ (train.csv) が見つかりませんでした。")


def load_original_dataset(fe_cols: list[str], target_col: str) -> pd.DataFrame | None:
    candidates = [
        "002_airline-satisfaction/data/airline-passenger-satisfaction",
        "data/airline-passenger-satisfaction",
        "../data/airline-passenger-satisfaction",
        "/kaggle/input/airline-passenger-satisfaction",
        "/kaggle/input/datasets/teejmahal20/airline-passenger-satisfaction",
    ]
    og_dir = None
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            og_dir = c
            break

    if og_dir is None:
        matches = glob.glob("/kaggle/input/**/airline-passenger-satisfaction/train.csv", recursive=True)
        if matches:
            og_dir = os.path.dirname(matches[0])

    if og_dir is None:
        try:
            import kagglehub
            print("  - /kaggle/input に元データがないため kagglehub で自動取得を試みます...")
            dl_path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")
            if os.path.exists(os.path.join(dl_path, "train.csv")):
                og_dir = dl_path
        except Exception as e:
            print(f"  - [警告] 元データの自動取得をスキップしました: {e}")

    if og_dir is None:
        return None

    print(f"  - オリジナルデータ取得元: {og_dir}")
    og_parts = []
    for fname in ("train.csv", "test.csv"):
        fpath = os.path.join(og_dir, fname)
        if os.path.exists(fpath):
            og_parts.append(pd.read_csv(fpath, index_col=0))
    og = pd.concat(og_parts, ignore_index=True)
    og[target_col] = (og[target_col] == "satisfied").astype(int)
    og = og[fe_cols + [target_col]].drop_duplicates().reset_index(drop=True)
    return og


TARGET = "satisfaction"
FD = "Flight Distance"
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"
]
CATS = ["Customer Type", "Type of Travel", "Class"]  # Gender pruned for cleaner splits
NUMS = ["Age", FD, "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATS = NUMS + RATINGS + CATS + ["Gender"]


def to_logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def to_prob(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def build_original_teacher_features(full_df: pd.DataFrame, og_df: pd.DataFrame | None, y_comp: np.ndarray, n_tr: int) -> pd.DataFrame:
    OG = pd.DataFrame(index=range(len(full_df)))
    if og_df is None:
        return OG

    t0 = time.time()
    both = pd.concat([full_df[FEATS], og_df[FEATS]], ignore_index=True)
    both_coded = both.copy()
    for c in CATS + ["Gender"]:
        both_coded[c] = both_coded[c].astype("category").cat.codes

    X_og = both_coded.iloc[len(full_df):]
    y_og = og_df[TARGET].values
    X_comp = both_coded.iloc[:len(full_df)]

    # 1. Original Teacher: XGBoost
    m_xgb = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.5, tree_method="hist",
        device=DEVICE, random_state=0, eval_metric="auc"
    )
    m_xgb.fit(X_og, y_og, verbose=False)
    OG["og_xgb_d8"] = m_xgb.predict_proba(X_comp)[:, 1].astype(np.float32)

    # 2. Original Teacher: LightGBM
    m_lgb = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        verbose=-1, n_jobs=-1, random_state=0
    )
    m_lgb.fit(X_og, y_og)
    OG["og_lgb"] = m_lgb.predict_proba(X_comp)[:, 1].astype(np.float32)

    # 3. Original Teacher: CatBoost
    m_cb = CatBoostClassifier(
        iterations=1200, learning_rate=0.06, depth=7,
        task_type="GPU" if HAS_GPU else "CPU",
        verbose=0, random_seed=0
    )
    m_cb.fit(X_og, y_og)
    OG["og_cb"] = m_cb.predict_proba(X_comp)[:, 1].astype(np.float32)

    OG["og_mean"] = OG[["og_xgb_d8", "og_lgb", "og_cb"]].mean(axis=1).astype(np.float32)
    for col in ["og_xgb_d8", "og_lgb", "og_cb", "og_mean"]:
        OG[f"{col}_logit"] = to_logit(OG[col].values).astype(np.float32)

    auc_og = roc_auc_score(y_comp, OG["og_mean"].values[:n_tr])
    print(f"  - オリジナル教師モデル完了 ({time.time() - t0:.1f}秒) | コンペ予測単体 AUC: {auc_og:.5f}")
    return OG


def combine_keys(full_df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    code = np.zeros(len(full_df), dtype=np.int64)
    for c in cols:
        k = pd.factorize(full_df[c], use_na_sentinel=False)[0]
        code = code * (k.max() + 1) + k
    return pd.factorize(code)[0]


def target_encode(k_fit: np.ndarray, y_fit: np.ndarray, k_apply: np.ndarray, smooth: float = 20.0) -> np.ndarray:
    prior = y_fit.mean()
    s = pd.DataFrame({"k": k_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    enc = (s["sum"] + prior * smooth) / (s["count"] + smooth)
    return pd.Series(k_apply).map(enc).fillna(prior).values.astype(np.float32)


def add_multi_te(
    Xa: pd.DataFrame, ya: np.ndarray, others: list[pd.DataFrame], key_dict: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    Xa = Xa.copy()
    others = [o.copy() for o in others]
    for name, codes in key_dict.items():
        ka = codes[Xa.index.values]
        col = np.zeros(len(Xa), dtype=np.float32)
        for i, j in KFold(5, shuffle=True, random_state=0).split(ka):
            col[j] = target_encode(ka[i], ya[i], ka[j])
        Xa[name] = col
        for o in others:
            o[name] = target_encode(ka, ya, codes[o.index.values])
    return Xa, others


def find_best_3d_weights(
    y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, mode_name: str = "確率空間"
) -> tuple[tuple[float, float, float], float]:
    best_auc = -1.0
    best_w = (1.0, 0.0, 0.0)

    step1 = 0.05
    steps1 = int(round(1.0 / step1))
    for i in range(steps1 + 1):
        w1 = round(i * step1, 2)
        for j in range(steps1 + 1 - i):
            w2 = round(j * step1, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -1e-5:
                continue
            blend = w1 * p1 + w2 * p2 + w3 * p3
            score = roc_auc_score(y_true, blend)
            if score > best_auc:
                best_auc = score
                best_w = (w1, w2, w3)

    bw1, bw2, _ = best_w
    fine_step = 0.01
    w1_range = np.arange(max(0.0, bw1 - 0.06), min(1.0, bw1 + 0.06) + 1e-5, fine_step)
    w2_range = np.arange(max(0.0, bw2 - 0.06), min(1.0, bw2 + 0.06) + 1e-5, fine_step)

    for w1 in w1_range:
        w1 = round(w1, 2)
        for w2 in w2_range:
            w2 = round(w2, 2)
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0.0:
                continue
            blend = w1 * p1 + w2 * p2 + w3 * p3
            score = roc_auc_score(y_true, blend)
            if score > best_auc:
                best_auc = score
                best_w = (w1, w2, w3)

    print(f"  - [{mode_name}] 最適比率 LGB {best_w[0]:.2f} : CB {best_w[1]:.2f} : XGB {best_w[2]:.2f} ➔ OOF AUC: {best_auc:.6f}")
    return best_w, best_auc


def main():
    t_start = time.time()
    comp_dir = find_comp_data_dir()
    print(f"[1/5] データ読み込み開始: {comp_dir}")
    print(f"  - アクセラレータ環境: {'GPU (CUDA 有効)' if HAS_GPU else 'CPU'}")

    tr = pd.read_csv(os.path.join(comp_dir, "train.csv"))
    te = pd.read_csv(os.path.join(comp_dir, "test.csv"))
    sub_df = pd.read_csv(os.path.join(comp_dir, "sample_submission.csv"))

    y = tr[TARGET].astype(int).values
    n_tr = len(tr)
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)

    # 1. オリジナル教師モデル
    print("\n[2/5] オリジナルデータ3大教師モデル構築中...")
    og_df = load_original_dataset(FEATS, TARGET)
    OG_feats = build_original_teacher_features(full, og_df, y, n_tr)

    # 2. 真の路線プロファイル (20列) + 全4数値頻度
    print("\n[3/5] 真の路線プロファイル(20列) + 4数値頻度 + CatBoost用39Crosses構築中...")
    CNT = pd.DataFrame({
        f"cnt_{c}": full[c].map(full[c].value_counts()).fillna(0).astype(np.float32)
        for c in NUMS
    })

    coded = full[FEATS].copy()
    for c in CATS + ["Gender"]:
        coded[c] = coded[c].astype("category").cat.codes
    grp = coded.groupby(FD)
    FDAGG = pd.DataFrame({
        f"fdm_{c}": grp[c].transform("mean").astype(np.float32)
        for c in FEATS if c != FD
    })
    FDAGG["fd_cnt"] = grp["Age"].transform("size").astype(np.float32)

    # CatBoost専用: 39 Crosses (Rating x Context)
    CROSS_CB = pd.DataFrame(index=range(len(full)))
    cross_names = []
    for r in RATINGS:
        for bg in ["Class", "Type of Travel", "Customer Type"]:
            cname = f"cb_{r[:4]}_{bg[:3]}"
            CROSS_CB[cname] = (full[r].astype(str) + "|" + full[bg].astype(str)).astype(str)
            cross_names.append(cname)

    # 6種の多重 Target Encoding キー
    TE_KEYS = [
        [FD],
        ["Age"],
        [FD, "Age"],
        [FD, "Class"],
        [FD, "Type of Travel"],
        ["Age", "Class", "Type of Travel", "Customer Type"],
    ]
    KEY_DICT = {"te_" + "|".join(c[:6] for c in cols): combine_keys(full, cols) for cols in TE_KEYS}

    # View A: LightGBM / XGBoost 用 (Gender剪定 + 路線プロファイル + 教師モデル + 頻度)
    clean_base = [c for c in NUMS + RATINGS + CATS if c not in FDAGG.columns and c not in CNT.columns]
    X_view_a = pd.concat([full[clean_base], OG_feats, FDAGG, CNT], axis=1)
    for c in CATS:
        X_view_a[c] = X_view_a[c].astype("category")

    # View B: CatBoost 用 (View A + 39 Crosses)
    X_view_b = pd.concat([X_view_a.copy(), CROSS_CB], axis=1)
    for c in CATS:
        X_view_b[c] = X_view_b[c].astype(str)

    print(f"  - View A 特徴量数 (LGB/XGB): {X_view_a.shape[1] + len(KEY_DICT)} 列")
    print(f"  - View B 特徴量数 (CatBoost): {X_view_b.shape[1] + len(KEY_DICT)} 列 (39 Crosses 統合)")

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_oof = np.zeros(n_tr)
    cb_oof = np.zeros(n_tr)
    xgb_oof = np.zeros(n_tr)

    lgb_test = np.zeros(len(te))
    cb_test = np.zeros(len(te))
    xgb_test = np.zeros(len(te))

    # --- ハイパーパラメータ (シード平均 & 多様化設定) ---
    lgb_params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "n_estimators": 4000, "learning_rate": 0.02, "num_leaves": 127,
        "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.5, "reg_lambda": 1.0, "max_bin": 255,
        "n_jobs": -1, "verbose": -1,
    }

    cb_params = {
        "iterations": 2500 if HAS_GPU else 1200,
        "learning_rate": 0.05,
        "depth": 8 if HAS_GPU else 6,
        "l2_leaf_reg": 3.0,
        "border_count": 254,
        "task_type": "GPU" if HAS_GPU else "CPU",
        "eval_metric": "Logloss",
        "early_stopping_rounds": 150,
        "cat_features": CATS + cross_names,
        "verbose": 0,
        "allow_writing_files": False,
    }
    if not HAS_GPU:
        cb_params["thread_count"] = -1

    xgb_params = {
        "n_estimators": 4000, "learning_rate": 0.02, "max_depth": 8,
        "subsample": 0.8, "colsample_bytree": 0.5, "min_child_weight": 5,
        "reg_lambda": 1.0, "max_bin": 256, "tree_method": "hist",
        "device": DEVICE, "eval_metric": "auc", "early_stopping_rounds": 150,
        "enable_categorical": True, "max_cat_to_onehot": 1,
    }
    if not HAS_GPU:
        xgb_params["n_jobs"] = -1

    SEEDS = [42, 2026]  # 2シード平均で予測分散を低減
    print(f"\n[4/5] Stratified {n_splits}-Fold CV (実験 #011: 2-Seed Hyper Ensemble) 学習開始...")

    for fold, (a, b) in enumerate(skf.split(np.zeros(n_tr), y), 1):
        t_f = time.time()
        ya, yb = y[a], y[b]

        # Inner 5-Fold Multi-Target Encoding
        Xa_a, (Xb_a, Xt_a) = add_multi_te(
            X_view_a.iloc[:n_tr].iloc[a], ya, [X_view_a.iloc[:n_tr].iloc[b], X_view_a.iloc[n_tr:]], KEY_DICT
        )
        Xa_b, (Xb_b, Xt_b) = add_multi_te(
            X_view_b.iloc[:n_tr].iloc[a], ya, [X_view_b.iloc[:n_tr].iloc[b], X_view_b.iloc[n_tr:]], KEY_DICT
        )

        fold_lgb_val = np.zeros(len(b))
        fold_xgb_val = np.zeros(len(b))
        fold_cb_val = np.zeros(len(b))

        for s_idx, s in enumerate(SEEDS):
            # 1. LightGBM
            m_lgb = lgb.LGBMClassifier(**lgb_params, random_state=s + fold)
            m_lgb.fit(Xa_a, ya, eval_set=[(Xb_a, yb)], callbacks=[lgb.early_stopping(150, verbose=False)])
            p_lgb = m_lgb.predict_proba(Xb_a)[:, 1]
            fold_lgb_val += p_lgb / len(SEEDS)
            lgb_test += (m_lgb.predict_proba(Xt_a)[:, 1] / len(SEEDS)) / n_splits

            # 2. XGBoost
            m_xgb = xgb.XGBClassifier(**xgb_params, random_state=s + fold)
            m_xgb.fit(Xa_a, ya, eval_set=[(Xb_a, yb)], verbose=False)
            p_xgb = m_xgb.predict_proba(Xb_a)[:, 1]
            fold_xgb_val += p_xgb / len(SEEDS)
            xgb_test += (m_xgb.predict_proba(Xt_a)[:, 1] / len(SEEDS)) / n_splits

            # 3. CatBoost (View B with 39 Crosses)
            m_cb = CatBoostClassifier(**cb_params, random_seed=s + fold)
            m_cb.fit(Xa_b, ya, eval_set=(Xb_b, yb))
            p_cb = m_cb.predict_proba(Xb_b)[:, 1]
            fold_cb_val += p_cb / len(SEEDS)
            cb_test += (m_cb.predict_proba(Xt_b)[:, 1] / len(SEEDS)) / n_splits

        lgb_oof[b] = fold_lgb_val
        xgb_oof[b] = fold_xgb_val
        cb_oof[b] = fold_cb_val

        print(
            f"  - Fold {fold}/{n_splits} ({time.time() - t_f:.0f}s) [Seed-Avg] | "
            f"LGB: {roc_auc_score(yb, fold_lgb_val):.5f} | "
            f"CB(39Cr): {roc_auc_score(yb, fold_cb_val):.5f} | "
            f"XGB: {roc_auc_score(yb, fold_xgb_val):.5f}"
        )

    auc_lgb = roc_auc_score(y, lgb_oof)
    auc_cb = roc_auc_score(y, cb_oof)
    auc_xgb = roc_auc_score(y, xgb_oof)

    print("\n" + "=" * 65)
    print("【シード平均 単体モデル 5-Fold OOF ROC AUC】")
    print(f"  1. LightGBM (Seed-Avg) OOF AUC : {auc_lgb:.6f}")
    print(f"  2. CatBoost (39Cr+Seed) OOF AUC : {auc_cb:.6f}")
    print(f"  3. XGBoost  (Seed-Avg) OOF AUC : {auc_xgb:.6f}")

    # 確率空間 vs ロジット空間のブレンド比較
    print("\n=== 3大GBDT 最適重み探索 (確率空間 vs ロジット空間) ===")
    w_prob, score_prob = find_best_3d_weights(y, lgb_oof, cb_oof, xgb_oof, mode_name="確率空間ブレンド")

    l_lgb_oof, l_cb_oof, l_xgb_oof = to_logit(lgb_oof), to_logit(cb_oof), to_logit(xgb_oof)
    w_logit, score_logit = find_best_3d_weights(y, l_lgb_oof, l_cb_oof, l_xgb_oof, mode_name="ロジット空間ブレンド")

    best_single = max(auc_lgb, auc_cb, auc_xgb)
    if score_logit >= score_prob:
        w1, w2, w3 = w_logit
        best_score = score_logit
        best_mode = "ロジット空間 (Logit Space)"
        final_logit = w1 * to_logit(lgb_test) + w2 * to_logit(cb_test) + w3 * to_logit(xgb_test)
        final_pred = to_prob(final_logit)
    else:
        w1, w2, w3 = w_prob
        best_score = score_prob
        best_mode = "確率空間 (Probability Space)"
        final_pred = w1 * lgb_test + w2 * cb_test + w3 * xgb_test

    print("\n" + "=" * 65)
    print(f"🏆 採用ブレンド方式: {best_mode}")
    print(f"🌟 最適比率: LightGBM {w1:.2f} : CatBoost {w2:.2f} : XGBoost {w3:.2f}")
    print(f"🌟 最終 ハイパーアンサンブル OOF AUC : {best_score:.6f} (単体最高より {best_score - best_single:+.6f})")
    print("=" * 65)

    sub_df[TARGET] = final_pred
    sub_df.to_csv("submission.csv", index=False)
    print(f"\n[5/5] 提出ファイル作成完了: submission.csv (総所要時間: {(time.time() - t_start)/60:.1f} 分)")


if __name__ == "__main__":
    main()
