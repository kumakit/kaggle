# 学習・実験ログ (Learning Log): Store Sales - Time Series Forecasting

- **コンペ名:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **関連Issue:** [kumakit/mission-control#407](https://github.com/kumakit/mission-control/issues/407)
- **評価指標:** **RMSLE (Root Mean Squared Logarithmic Error)**（※値が小さいほど高精度）

---

## 実験一覧サマリー

| 実験 # | 手法・モデル | 検証スコア (Val RMSLE) | Kaggle本番スコア (Public LB) | 順位推移 | 備考 |
|---|---|---|---|---|---|
| **#001** | **LightGBM カレンダー特徴量 (Time-based Split)** | 約 `0.53` 前後 | **`0.52338`** | **443 位** | **初提出完了！** カレンダー＋カテゴリのみで堅牢なベースライン確立 |

---

## 実験詳細

### 実験 #001: LightGBM による初回ベースライン提出
- **実施日:** 2026-10-22
- **担当・実施者:** 学習者本人（Kaggle Notebook経由で提出）

#### 1. 仮説 (Hypothesis)
- 時系列予測において、まずは「直近のトレンド（2016年以降）」に絞り、カレンダー特徴量（曜日、日、月、年、週末フラグ、dayofyear）と基本カテゴリ（店舗、商品ファミリー）を与えることで、安定したベースライン予測ができるはず。
- 目的変数 $y$ を $\ln(1+y)$（`log1p`）変換してRMSEで学習させることで、RMSLEの数理構造と完全に一致した最適化が行えるはず。
- テスト期間と同じ長さの「直近16日間（2017-07-31 〜 2017-08-15）」を手元検証データ（Validation）とすることで、未来リークを防ぎ、手元検証スコアと本番提出スコアが連動するはず。

#### 2. 実験内容 (Method)
- **データ:** `train.csv` (2016-01-01以降を使用), `test.csv` (16日間, 28,512行)
- **特徴量 (9列):** `store_nbr`, `family`, `onpromotion`, `year`, `month`, `day`, `dayofweek`, `is_weekend`, `dayofyear`
- **データ分割:** Time-based Split（Train: 2016-01-01〜2017-07-30、Val: 2017-07-31〜2017-08-15）
- **モデル設定 (LightGBM Regressor):**
  - `objective='regression'`, `metric='rmse'`, `learning_rate=0.05`, `num_leaves=63`, `n_estimators=1000`
  - `early_stopping_rounds=50`
- **目的変数処理:**
  - 学習時: `np.log1p(sales)`
  - 予測時: `np.clip(np.expm1(pred), 0, None)`
- **成果物:**
  - ノートブック: `003_store-sales/notebooks/01-baseline.ipynb`
  - スクリプト: `003_store-sales/src/baseline.py`

#### 3. 結果 (Result)
- **Kaggle本番スコア (Public Leaderboard):** **`0.52338`**
- **提出後の順位:** **`443 位`**（Leaderboard登載完了 🎉）

#### 4. 分かったこと・考察 (Insights)
1. **Time-based Split の健全性:**
   - 目的変数を `np.log1p` 変換し、手元で直近16日間の検証期間を設けた設計により、未来データのリークなく本番でも `0.52338` という安定したスコアが出た。
2. **カレンダーとカテゴリの基礎表現力:**
   - 店舗・商品カテゴリと日付属性（曜日や特売数 `onpromotion`）だけでも、曜日ごとの買い出しサイクルや商品の規模感をLightGBMが捉えられている。
3. **伸び代（まだ使っていない情報）:**
   - まだ「給与日（15日・月末）」や「祝日・イベント（`holidays_events.csv`）」、「原油価格（`oil.csv`）」などの外部要因を使っていない。
   - 何より、時系列モデリングの主役である「過去の売上実績（ラグ特徴量・ローリング平均）」を投入していないため、ここからの伸び代が非常に大きい。

#### 5. 次の疑問・ネクストアクション (Next Action)
- **Phase 2:** カレンダー・ドメイン特徴量の拡充（給与日フラグ、祝日・振替休日、地震影響）の導入でどこまでスコアが伸びるか？
- **Phase 3:** 過去売上のラグ特徴量（テスト16日間を考慮した Lag 16〜）とローリング移動平均の設計。
