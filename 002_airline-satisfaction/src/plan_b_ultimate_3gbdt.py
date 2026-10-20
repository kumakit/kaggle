"""
002_airline-satisfaction: Plan B Ultimate - Original Data Teachers + Route Profile (20 cols) + 6 Multi-TE + 3-GBDT Logit Ensemble
Kaggle Playground Series Season 6 Episode 10: Airline Passenger Satisfaction

タスク: 航空会社顧客満足度予測（2値分類）
評価指標: ROC AUC (Area Under the ROC Curve)
実験 #010 (プランB・上限突破版):
  1. Original Dataset (13万件) による3大教師モデル予測 (og_xgb, og_lgb, og_cat + ロジット変換)
  2. 真の路線プロファイル (Flight Distance ごとの他20特徴量の平均値 fdm_* をそのまま付与)
  3. 6種の多重交差 In-Fold Target Encoding (Inner 5-Fold による完全リークフリー設計)
  4. 4つの全数値列の出現頻度カウント (cnt_*) + 従来のドメイン統計特徴量
  5. 3大GBDT (LightGBM + CatBoost + XGBoost) の確率空間 & ロジット空間 3次元最適ブレンド
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

# GPU 利用可能性の自動検知
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
    """
    オリジナルデータセット (teejmahal20/airline-passenger-satisfaction) を検索・取得する。
    1. ローカルまたは /kaggle/input 内を検索
    2. 見つからない場合、kagglehub が使えれば自動ダウンロードを試みる
    """
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
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMS = ["Age", FD, "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATS = NUMS + RATINGS + CATS


def to_logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def to_prob(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def build_original_teacher_features(full_df: pd.DataFrame, og_df: pd.DataFrame | None, y_comp: np.ndarray, n_tr: int) -> pd.DataFrame:
    """オリジナルデータのみで3つの教師モデルを学習し、コンペデータの予測値とロジットを特徴量化する（リーク完全ゼロ）"""
    OG = pd.DataFrame(index=range(len(full_df)))
    if og_df is None:
        print("  - [注意] オリジナルデータが見つからないため教師特徴量をスキップします。")
        return OG

    t0 = time.time()
    both = pd.concat([full_df[FEATS], og_df[FEATS]], ignore_index=True)
    both_coded = both.copy()
    for c in CATS:
        both_coded[c] = both_coded[c].astype("category").cat.codes

    X_og = both_coded.iloc[len(full_df):]
    y_og = og_df[TARGET].values
    X_comp = both_coded.iloc[:len(full_df)]

    # 1. Original Teacher: XGBoost (depth 8)
    m_xgb = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.5, tree_method="hist",
        device=DEVICE, random_state=0, eval_metric="auc"
    )
    m_xgb.fit(X_og, y_og, verbose=False)
    OG["og_xgb_d8"] = m_xgb.predict_proba(X_comp)[:, 1].astype(np.float32)

    # 2. Original Teacher: LightGBM (leaves 63)
    m_lgb = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.03, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        verbose=-1, n_jobs=-1, random_state=0
    )
    m_lgb.fit(X_og, y_og)
    OG["og_lgb"] = m_lgb.predict_proba(X_comp)[:, 1].astype(np.float32)

    # 3. Original Teacher: CatBoost (depth 7)
    m_cb = CatBoostClassifier(
        iterations=1200, learning_rate=0.06, depth=7,
        task_type="GPU" if HAS_GPU else "CPU",
        verbose=0, random_seed=0
    )
    m_cb.fit(X_og, y_og)
    OG["og_cb"] = m_cb.predict_proba(X_comp)[:, 1].astype(np.float32)

    # 3教師の平均 & ロジット特徴量
    OG["og_mean"] = OG[["og_xgb_d8", "og_lgb", "og_cb"]].mean(axis=1).astype(np.float32)
    for col in ["og_xgb_d8", "og_lgb", "og_cb", "og_mean"]:
        OG[f"{col}_logit"] = to_logit(OG[col].values).astype(np.float32)

    auc_og = roc_auc_score(y_comp, OG["og_mean"].values[:n_tr])
    print(f"  - オリジナル3教師モデル学習完了 ({time.time() - t0:.1f}秒) | 元データ単体でのコンペ予測 AUC: {auc_og:.5f}")
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
    """
    学習Fold内部をさらに5分割(Inner 5-Fold)してOut-of-Foldで安全にTarget Encodingを計算し、
    検証・テストデータには学習Fold全体を使ってマッピングする。
    """
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

    # オリジナルデータの読み込みと教師モデル構築
    print("\n[2/5] オリジナルデータ教師モデル (Original-Data Teachers) 構築中...")
    og_df = load_original_dataset(FEATS, TARGET)
    OG_feats = build_original_teacher_features(full, og_df, y, n_tr)

    # 真の路線プロファイル (20列) + 4数値頻度カウント + ドメイン特徴量
    print("\n[3/5] 真の路線プロファイル(20列) + 数値頻度カウント + 6種多重TEキー作成中...")
    CNT = pd.DataFrame({
        f"cnt_{c}": full[c].map(full[c].value_counts()).fillna(0).astype(np.float32)
        for c in NUMS
    })

    coded = full.copy()
    for c in CATS:
        coded[c] = coded[c].astype("category").cat.codes
    grp = coded.groupby(FD)
    FDAGG = pd.DataFrame({
        f"fdm_{c}": grp[c].transform("mean").astype(np.float32)
        for c in FEATS if c != FD
    })
    FDAGG["fd_cnt"] = grp["Age"].transform("size").astype(np.float32)

    # ドメイン要約特徴量（遅延差分・サービス基本統計）
    DOM = pd.DataFrame(index=range(len(full)))
    dep_d = full["Departure Delay in Minutes"].fillna(0)
    arr_d = full["Arrival Delay in Minutes"].fillna(0)
    DOM["Delay_Diff"] = (arr_d - dep_d).astype(np.float32)
    DOM["Service_Mean"] = full[RATINGS].mean(axis=1).astype(np.float32)
    DOM["Service_Std"] = full[RATINGS].std(axis=1).astype(np.float32)

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

    X_all = pd.concat([full, OG_feats, FDAGG, CNT, DOM], axis=1)
    print(f"  - ベース特徴量構築完了: {X_all.shape[1]} 列 (+ Fold内で6列の多重TEを追加 ➔ 計 {X_all.shape[1] + len(KEY_DICT)} 列)")

    # 各モデル用の型変換
    X_lgb = X_all.copy()
    for c in CATS:
        X_lgb[c] = X_lgb[c].astype("category")

    X_cb = X_all.copy()
    for c in CATS:
        X_cb[c] = X_cb[c].astype(str)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_oof = np.zeros(n_tr)
    cb_oof = np.zeros(n_tr)
    xgb_oof = np.zeros(n_tr)

    lgb_test = np.zeros(len(te))
    cb_test = np.zeros(len(te))
    xgb_test = np.zeros(len(te))

    # --- モデルハイパーパラメータ (上位検証済み設定) ---
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
        "cat_features": CATS,
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

    print(f"\n[4/5] Stratified {n_splits}-Fold CV (上限突破 3大GBDT) 学習開始...")
    for fold, (a, b) in enumerate(skf.split(np.zeros(n_tr), y), 1):
        t_f = time.time()
        ya, yb = y[a], y[b]

        # Inner 5-Fold Multi-Target Encoding (LGB / XGB 用)
        Xa_l, (Xb_l, Xt_l) = add_multi_te(
            X_lgb.iloc[:n_tr].iloc[a], ya, [X_lgb.iloc[:n_tr].iloc[b], X_lgb.iloc[n_tr:]], KEY_DICT
        )
        # CatBoost 用 (TE列のみコピー)
        Xa_c = X_cb.iloc[:n_tr].iloc[a].copy()
        Xb_c = X_cb.iloc[:n_tr].iloc[b].copy()
        Xt_c = X_cb.iloc[n_tr:].copy()
        for k_name in KEY_DICT.keys():
            Xa_c[k_name] = Xa_l[k_name]
            Xb_c[k_name] = Xb_l[k_name]
            Xt_c[k_name] = Xt_l[k_name]

        # 1. LightGBM
        m_lgb = lgb.LGBMClassifier(**lgb_params, random_state=42 + fold)
        m_lgb.fit(Xa_l, ya, eval_set=[(Xb_l, yb)], callbacks=[lgb.early_stopping(150, verbose=False)])
        p_lgb_val = m_lgb.predict_proba(Xb_l)[:, 1]
        lgb_oof[b] = p_lgb_val
        lgb_test += m_lgb.predict_proba(Xt_l)[:, 1] / n_splits

        # 2. CatBoost
        m_cb = CatBoostClassifier(**cb_params, random_seed=42 + fold)
        m_cb.fit(Xa_c, ya, eval_set=(Xb_c, yb))
        p_cb_val = m_cb.predict_proba(Xb_c)[:, 1]
        cb_oof[b] = p_cb_val
        cb_test += m_cb.predict_proba(Xt_c)[:, 1] / n_splits

        # 3. XGBoost
        m_xgb = xgb.XGBClassifier(**xgb_params, random_state=42 + fold)
        m_xgb.fit(Xa_l, ya, eval_set=[(Xb_l, yb)], verbose=False)
        p_xgb_val = m_xgb.predict_proba(Xb_l)[:, 1]
        xgb_oof[b] = p_xgb_val
        xgb_test += m_xgb.predict_proba(Xt_l)[:, 1] / n_splits

        print(
            f"  - Fold {fold}/{n_splits} ({time.time() - t_f:.0f}s) | "
            f"LGB: {roc_auc_score(yb, p_lgb_val):.5f} | "
            f"CB: {roc_auc_score(yb, p_cb_val):.5f} | "
            f"XGB: {roc_auc_score(yb, p_xgb_val):.5f}"
        )

    auc_lgb = roc_auc_score(y, lgb_oof)
    auc_cb = roc_auc_score(y, cb_oof)
    auc_xgb = roc_auc_score(y, xgb_oof)

    print("\n" + "=" * 65)
    print("【単体モデル 5-Fold OOF ROC AUC】")
    print(f"  1. LightGBM 単体 OOF AUC : {auc_lgb:.6f}")
    print(f"  2. CatBoost 単体 OOF AUC : {auc_cb:.6f}")
    print(f"  3. XGBoost  単体 OOF AUC : {auc_xgb:.6f}")

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
    print(f"🌟 最終 3大GBDT OOF AUC : {best_score:.6f} (単体最高より {best_score - best_single:+.6f})")
    print("=" * 65)

    sub_df[TARGET] = final_pred
    sub_df.to_csv("submission.csv", index=False)
    print(f"\n[5/5] 提出ファイル作成完了: submission.csv (総所要時間: {(time.time() - t_start)/60:.1f} 分)")


if __name__ == "__main__":
    main()
