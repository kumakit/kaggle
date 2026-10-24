# 学習・実験ログ (Learning Log): Store Sales - Time Series Forecasting

- **コンペ名:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **関連Issue:** [kumakit/mission-control#407](https://github.com/kumakit/mission-control/issues/407)
- **評価指標:** **RMSLE (Root Mean Squared Logarithmic Error)**（※値が小さいほど高精度）

---

## 実験一覧サマリー

| 実験 # | 手法・モデル | 検証スコア (Val RMSLE) | Kaggle本番スコア (Public LB) | 順位推移 | 備考 |
|---|---|---|---|---|---|
| **#001** | **LightGBM カレンダー特徴量 (Time-based Split)** | 約 `0.41440` | `0.52338` | 443 位 | 初回ベースライン（カレンダー＋基本カテゴリ） |
| **#002** | **ドメイン特徴量拡張 (給与日・祝日・店舗メタ・原油価格)** | `0.41426` | `0.52581` | 445 位 前後 | 手元微増も本番微減（マクロ原油・店舗冗長列の過学習を特定） |
| **#003** | **時系列ラグ（Lag 16〜）& ローリング移動平均モデル** | **`0.39628`** | `0.52383` (V4) | 443 位 前後 | 手元大幅改善も全商品一括＋ゼロ系列ペナルティで本番足踏み |
| **#004** | **33ファミリー別個別LightGBM ＋ ゼロ売上フィルター** | `0.40444` | **`0.45505`** | **277 位 (一撃166人抜き！🚀)** | **超特大ブレイクスルー！自己ベスト-0.06833急伸！** スケール格差解消とゼロクリップが完全的中 |
| **#005** | **3大GBDT多様性アンサンブル (LGB+CB+XGB with RTX 4060 GPU) ＋ 前年同期ラグ** | **`0.40163`** | **`0.43078`** (V6) | **`210 位` (さらに67人抜き！累計233人抜き！🎉)** | **大幅自己ベスト更新 (-0.02427)！** 3大GBDTの多様性と52週前同曜日ラグ(Lag 364)が猛威を振るい上位35%以内突入 |
| **#006** | **特売リード＆ラグ特徴量 ＋ 3大GBDT多様性アンサンブル (RTX 4060 GPU)** | **`0.40074`** | **`0.41064`** (V7) | **`149 位` (さらに61人抜き！累計294人抜き！🚀)** | **大幅自己ベスト更新 (-0.02014)！** 特売先行指標(Lead)と反動減(Lag)が炸裂し100位台突入！ |

---

## 実験詳細

### 実験 #006: 特売リード＆ラグ特徴量 ＋ 3大GBDT多様性アンサンブル (RTX 4060 GPU)
- **実施日:** 2026-10-24
- **担当・実施者:** 学習者本人（RTX 4060 Laptop GPU を活用して提出）

#### 1. 仮説 (Hypothesis)
- スーパーマーケットの需要予測において、「特売（`onpromotion`）」は単なる当日効果だけでなく、事前・事後に大きな需要変動を引き起こす。
  - **事後影響（反動減・買いだめ反動）:** 特売で買いだめ（Stockpiling）された直後は、家庭内に在庫があるため需要が急減する（`promo_lag_1`, `promo_lag_2`, `promo_lag_7`）。
  - **先行指標（買い控え）:** チラシや事前告知等で明日・明後日に特売が控えていると分かっている場合、消費者は買い控えを起こす（`promo_lead_1`, `promo_lead_2`）。
  - **初日バースト効果:** 特売が前日より急増した初日の勢い（`promo_diff_1 = onpromotion - promo_lag_1`）。
- テストデータ `test.csv` には 16 日間すべての `onpromotion` があらかじめ収録されているため、未来リード（shift(-1)）を使っても完全に合法であり未来リークは一切生じない。
- これら 7 つの特売特徴量を 3 大 GBDT アンサンブル（LightGBM 50% + CatBoost 30% + XGBoost 20%）に投入することで、手元検証および本番スコアをさらに向上させ、200位切り・100位台突入を狙えるはず。

#### 2. 実験内容 (Method)
- **データ:** `train.csv` + `test.csv` (全系列ラグ計算)
- **特徴量 (28列):**
  - カレンダー・給与日 (11列): `year`, `month`, `day`, `dayofweek`, `is_weekend`, `dayofyear`, `weekofyear`, `is_month_start`, `is_month_end`, `is_wage_day`, `is_wage_day_after`
  - カテゴリ (2列): `store_nbr`, `onpromotion`
  - 過去売上ラグ & 移動平均 (8列): `sales_lag_16`, `sales_lag_21`, `sales_lag_28`, `sales_lag_35`, `sales_lag_364`, `sales_rolling_mean_7`, `sales_rolling_mean_14`, `sales_rolling_mean_28`
  - **新規投入: 特売リード＆ラグ特徴量 (7列):**
    - `promo_lag_1`: 前日の特売数（反動減）
    - `promo_lag_2`: 2日前の特売数
    - `promo_lag_7`: 前週同曜日の特売数
    - `promo_lead_1`: 翌日（1日後）の特売数（事前買い控え）
    - `promo_lead_2`: 翌々日（2日後）の特売数
    - `promo_diff_1`: 特売の前日差分（初日急増バースト）
    - `promo_rolling_mean_7`: 直近7日間の特売平均水準
- **モデル:** 33商品ファミリーごとに 3 大モデルを学習しブレンド：
  - **LightGBM:** 50%
  - **CatBoost GPU:** 30% (`task_type='GPU'`)
  - **XGBoost GPU:** 20% (`tree_method='hist'`, `device='cuda'`)
- **後処理:** 直近21日間売上ゼロ系列の完全0クリップ
- **成果物:**
  - ノートブック: `003_store-sales/notebooks/06-promo-features.ipynb`
  - スクリプト: `003_store-sales/src/promo_features.py`

#### 3. 結果 (Result)
- **手元検証スコア (Val RMSLE):** **`0.40074`**（#005の `0.40163` から **さらに -0.00089 改善！ 0.400目前！** 📈）
- **Kaggle本番スコア (Public Leaderboard):** **`0.41064`**（前回 `0.43078` から **-0.02014 の超大幅自己ベスト更新！** 🏆）
- **提出後の順位:** **`149 位`**（210位から **さらに 61人抜き**！初日443位から **累計 294人抜き** で一気に100位台突入！Top 25%圏内！🎉）

#### 4. 勝因の分析（何がこの特大ジャンプを生んだのか）
1. **外生変数「特売（onpromotion）」の事前・事後効果の完全捕捉:**
   - 既存のモデルでは「当日特売があったかどうか」しか見ていなかったが、スーパーマーケットの実務では「昨日特売だった商品の反動減（買いだめ消費）」と「明日特売が控えている商品の買い控え」が極めて大きく作用する。
   - `promo_lag_1` / `promo_lag_2` / `promo_lag_7` による反動減補正と、`promo_lead_1` / `promo_lead_2` による買い控え補正が、テスト期間16日間の日次需要の波を驚くほど正確に再現した。
2. **合法的な未来情報（Known Future）の威力:**
   - 売上データは未来予測の対象だが、特売スケジュールは店舗側が事前に計画・確定している外生変数である。このドメイン特性を活かして `shift(-1)` / `shift(-2)` をリークフリーで組み込んだことがスコア急伸（0.430 ➔ 0.410）に直結した。
3. **3大GBDTアンサンブル（RTX 4060 GPU）との相乗効果:**
   - 28列に拡張された特徴量空間でも、Leaf-wise（LGB）、対称木（CB）、正則化木（XGB）の多様な決定木アンサンブルが過学習を起こさず、特売モメンタム（`promo_diff_1`）と頻度（`promo_rolling_mean_7`）の非線形な関係を綺麗に拾い上げた。

---

### 実験 #005: 3大GBDT多様性アンサンブル (LightGBM + CatBoost GPU + XGBoost GPU) ＋ 前年同期ラグ
- **実施日:** 2026-10-24
- **担当・実施者:** 学習者本人（RTX 4060 Laptop GPU を活用して提出）

#### 1. 仮説 (Hypothesis)
- 単一の決定木アルゴリズム（LightGBM）だけでは、特定のスパイクやノイズに過敏に反応するリスクがある。
- 異なる構造（Leaf-wiseのLightGBM、Symmetric TreesのCatBoost、Depth-wiseのXGBoost）を組み合わせることで、予測のバリアンス（ばらつき）を大幅に抑制できるはず。
- 8月後半の新学期特需（2016年8月の急増需要）を直接参照できるよう、**「前年同期（ちょうど52週前の同じ曜日＝364日前）のラグ (`sales_lag_364`)」** を加えることで、8月後半の特殊需要を正確に推計できるはず。
- RTX 4060 Laptop GPU（CUDA）を活用し、CatBoost GPU と XGBoost GPU を高速並列学習させる。

#### 2. 実験内容 (Method)
- **データ:** `train.csv` + `test.csv` (全系列ラグ計算)
- **特徴量 (21列):**
  - カレンダー・給与日: `year`, `month`, `day`, `dayofweek`, `is_weekend`, `dayofyear`, `weekofyear`, `is_month_start`, `is_month_end`, `is_wage_day`, `is_wage_day_after`
  - カテゴリ: `store_nbr`, `onpromotion`
  - 直近ラグ: `sales_lag_16`, `sales_lag_21`, `sales_lag_28`, `sales_lag_35`
  - **前年同期ラグ:** `sales_lag_364`（52週前同曜日売上）
  - ローリング移動平均: `sales_rolling_mean_7`, `sales_rolling_mean_14`, `sales_rolling_mean_28`
- **モデル:** 33商品ファミリーごとに以下の3大モデルを学習しブレンド：
  - **LightGBM:** 50%
  - **CatBoost GPU:** 30% (`task_type='GPU'`)
  - **XGBoost GPU:** 20% (`tree_method='hist'`, `device='cuda'`)
- **後処理:** 直近21日間売上ゼロ系列の完全0クリップ
- **成果物:**
  - ノートブック: `003_store-sales/notebooks/05-ensemble-gpu.ipynb`
  - スクリプト: `003_store-sales/src/ensemble_gpu.py`

#### 3. 結果 (Result)
- **手元検証スコア (Val RMSLE):** **`0.40163`**（#004の `0.40444` から **さらに -0.00281 改善！** 📈）
- **Kaggle本番スコア (Public Leaderboard):** **`0.43078`**（前回 `0.45505` から **-0.02427 の大幅自己ベスト更新！** 🏆）
- **提出後の順位:** **`210 位`**（277位から **さらに 67人抜き**！初日443位から **累計 233人抜き** で上位35%以内突入！🎉）

#### 4. 勝因の分析（何がこのスコア更新をもたらしたか）
1. **3大GBDT多様性アンサンブルのバリアンス抑制効果:**
   - LightGBM（葉単位分割）、CatBoost（対称木・GPU）、XGBoost（深さ単位分割・正則化ヒストグラム・GPU）というアルゴリズム・決定木構造の根本的な多様性により、単一モデル特有の過小・過大評価を相殺・平滑化できた。
2. **前年同期ラグ（52週前同曜日＝Lag 364）の決定打:**
   - 8月後半のテスト期間はエクアドルの新学期特需（`SCHOOL AND OFFICE SUPPLIES` 等の特需急増）があり、直近16日〜35日前（7月）のトレンドだけでは捉えきれなかった前年8月の特有の急伸パターンを決定木が正確に予測に織り込むことができた。
3. **RTX 4060 Laptop GPU による高速学習と安定実行:**
   - 33商品ファミリーごとのCatBoost GPU・XGBoost GPUの学習が数分で完了し、GPU上での安定したモデル構築が実現。
   - ※XGBoost実行時の `Falling back to prediction using DMatrix due to mismatched devices` 警告は、推論時にCPU上のDataFrameを入力した際の自動転送通知であり、予測結果の精度には全く影響しないことを確認。

---

## 実験詳細

### 実験 #004: 33商品ファミリー別個別モデル ＋ 直近0売上完全フィルター
- **実施日:** 2026-10-24
- **担当・実施者:** 学習者本人（Kaggle Notebook経由で提出）

#### 1. 仮説 (Hypothesis)
- 全商品を1つのモデルで学習させると、1日何千本も売れる「飲料水」と1日0〜1個の「書籍」のスケール格差により、小規模カテゴリの予測が狂ってしまう。33商品ファミリーごとに独立した専用LightGBMモデルを構築すれば、各カテゴリの周期性とトレンドを固有のスケールで正確に捉えられるはず。
- 直近21日間に売上が全くない124系列（約2,000行）を完全に `0.0` にクリップ（Zero Forecast Filter）することで、微小な予測値（0.1等）によるRMSLEペナルティを完全ゼロに排除できるはず。

#### 2. 実験内容 (Method)
- **データ:** `train.csv` + `test.csv` (全系列を安全にラグ計算)
- **特徴量 (20列):**
  - カレンダー・給与日: `year`, `month`, `day`, `dayofweek`, `is_weekend`, `dayofyear`, `weekofyear`, `is_month_start`, `is_month_end`, `is_wage_day`, `is_wage_day_after`
  - カテゴリ: `store_nbr`, `onpromotion`
  - ラグ特徴量: `sales_lag_16`, `sales_lag_21`, `sales_lag_28`, `sales_lag_35`
  - 移動平均: `sales_rolling_mean_7`, `sales_rolling_mean_14`, `sales_rolling_mean_28`
- **モデル設定:** 33商品ファミリーごとに個別に `LGBMRegressor`（33モデル）を順次学習
- **後処理:** 直近21日間売上ゼロの124ペアを `0.0` にクリップ
- **成果物:**
  - ノートブック: `003_store-sales/notebooks/04-family-models.ipynb`
  - スクリプト: `003_store-sales/src/family_models.py`

#### 3. 結果 (Result)
- **手元検証スコア (Val RMSLE):** `0.40444`
- **Kaggle本番スコア (Public Leaderboard):** **`0.45505`**（前回 `0.52338` から **-0.06833 の特大スコア更新！** 🏆）
- **順位推移:** 443 位 ➔ **`277 位`（一撃で 166人抜き達成！！🚀）**

#### 4. 勝因の分析（何がこの特大ジャンプを生んだのか）
1. **商品ファミリー別の独立モデリングによるスケール格差の完全解消:**
   - 巨大な飲料水や食料品のノイズに小規模カテゴリが潰されず、各カテゴリ特有の周期性や特売効果を決定木が精密に捉えられた。
2. **ゼロ売上フィルターによる無駄なペナルティの完全消滅:**
   - テスト期間約2,000行に及ぶ「売上ゼロ系列」への微小予測値ペナルティが綺麗さっぱり消えた。
3. **過去実績ラグ（Lag 16〜）と移動平均の相乗効果:**
   - 各商品×各店舗の「本来の需要水準」が正しく予測の土台となった。

---

## 実験詳細

### 実験 #003: 過去実績ラグ（Lag 16〜）＆ ローリング移動平均モデル
- **実施日:** 2026-10-23
- **担当・実施者:** 学習者本人（Kaggle Notebook経由で提出）

#### 1. 仮説 (Hypothesis)
- スーパーの売上需要予測で最も本質的な情報は「直近の売上実績（先週・先々週の同じ曜日にいくら売れたか）」。
- テスト期間（16日間）を考慮し、未来リークを完全にシャットアウトした **最小ラグ16日（$t - 16$ 以上）** の過去実績ラグ（Lag 16, 21, 28, 35）とローリング移動平均（7日, 14日, 28日）を店舗×商品ファミリーごとに構築すれば、大幅な精度向上が見込めるはず。
- ノイズとなった原油価格や冗長な店舗属性はアブレーション（完全除外）する。

#### 2. 実験内容 (Method)
- **データ:** `train.csv` + `test.csv` (時系列を結合して安全にシフト計算)
- **特徴量 (22列):**
  - カレンダー・給与日: `year`, `month`, `day`, `dayofweek`, `is_weekend`, `dayofyear`, `weekofyear`, `is_month_start`, `is_month_end`, `is_wage_day`, `is_wage_day_after`
  - カテゴリ: `store_nbr`, `family`, `onpromotion`
  - ラグ特徴量: `sales_lag_16`, `sales_lag_21` (3週前同曜日), `sales_lag_28` (4週前同曜日), `sales_lag_35` (5週前同曜日)
  - ローリング統計量: `sales_rolling_mean_7`, `sales_rolling_mean_14`, `sales_rolling_mean_28`, `sales_rolling_std_7`
- **モデル設定:** LightGBM Regressor (`learning_rate=0.05`, `num_leaves=63`, `n_estimators=1000`)
- **成果物:**
  - ノートブック: `003_store-sales/notebooks/03-lag-features.ipynb`
  - スクリプト: `003_store-sales/src/lag_features.py`

#### 3. 結果 (Result)
- **手元検証スコア (Val RMSLE):** **`0.39628`**（#001の `0.41440` から **-0.01812 の特大改善！念願の0.3点台突入！** 🚀）
- **特徴量重要度 Top 10:**
  1. `store_nbr` (11,780)
  2. `family` (7,975)
  3. `dayofyear` (7,882)
  4. **`sales_rolling_mean_28` (4,382)** 📈
  5. `day` (3,627)
  6. **`sales_rolling_mean_14` (3,102)** 📈
  7. **`sales_rolling_std_7` (2,609)** 📈
  8. **`sales_rolling_mean_7` (2,553)** 📈
  9. `dayofweek` (2,493)
  10. `weekofyear` (2,401)
- **Kaggle本番スコア (Public Leaderboard):** 
- **提出後の順位:** 

---

## 実験詳細

### 実験 #002: ドメイン特徴量拡張モデル（5テーブル統合）
- **実施日:** 2026-10-23
- **担当・実施者:** 学習者本人（Kaggle Notebook経由で提出）

#### 1. 仮説 (Hypothesis)
- 単なるカレンダー情報だけでなく、エクアドル特有の経済・商習慣を反映したドメイン特徴量（給与日15日/月末、祝日振替、店舗属性、原油価格）を追加することで、需要変動の波をより精密に捉えられるはず。
- 特にエクアドルは石油依存国であるため、原油価格（WTI）の推移や店舗の都市・クラスタ情報が売上予測の重要な手掛かりになるはず。

#### 2. 実験内容 (Method)
- **データ:** `train.csv`, `test.csv`, `stores.csv`, `holidays_events.csv`, `oil.csv`
- **特徴量 (22列に拡張):**
  - カレンダー: `year`, `month`, `day`, `dayofweek`, `is_weekend`, `dayofyear`, `weekofyear`, `is_month_start`, `is_month_end`
  - 給与日: `is_wage_day`（15日または月末）, `is_wage_day_after`（16日または月初）
  - 震災イベント: `is_earthquake`（2016-04-16から1ヶ月間）
  - 店舗メタ: `city`, `state`, `store_type`, `cluster`
  - 祝日: `is_national_holiday`（振替除外、全国公休日）
  - 原油: `dcoilwtico`（線形補間）, `oil_7d_mean`（7日移動平均）
- **モデル設定:** LightGBM Regressor (`learning_rate=0.05`, `num_leaves=63`, `n_estimators=1000`)
- **成果物:**
  - ノートブック: `003_store-sales/notebooks/02-domain-features.ipynb`
  - スクリプト: `003_store-sales/src/feature_engineering.py`

#### 3. 結果 (Result)
- **手元検証スコア (Val RMSLE):** **`0.41426`**（#001の `0.41440` から微増）
- **Kaggle本番スコア (Public Leaderboard):** **`0.52581`**（前回 `0.52338` より **+0.00243 微減**）

#### 4. 分かったこと・統計的考察 (Insights)
1. **原油価格（マクロ指標）の過学習:**
   - 特徴量重要度で原油価格（`dcoilwtico`, `oil_7d_mean`）が2,000回以上も分岐に使われていたが、日次（たった16日間）のミクロな売上変動に対して原油価格は動きが緩やか、あるいは無関係な短期ノイズを過剰学習（Overfitting）してしまった可能性が高い。
2. **店舗属性の冗長性（多重共線性・特徴量希釈）:**
   - すでに54店舗を示す `store_nbr` が入っているため、`city`, `state`, `store_type`, `cluster` は店舗番号と1対1で完全に決定論的な関係にある。追加したことで決定木の探索空間が無駄に広がり、本質的な `family`（商品カテゴリ）との主効果の学習を邪魔してしまった。
3. **何が本質的なのか？（次のブレイクスルーへの示唆）:**
   - スーパーの売上需要予測において最も強力なシグナルは、「原油価格や州名」ではなく、**「その店舗・その商品が直近（先週・先々週）どれだけ売れていたか（直近実績）」** である。
   - 外部データを闇雲に足すよりも、**「過去売上のラグ特徴量（Lag 16〜）とローリング移動平均」** を設計することが不可欠であると実証された。

#### 5. 次の疑問・ネクストアクション (Next Action)
- **Phase 3:** 外部のノイズ特徴量を削ぎ落とし（アブレーション）、テスト期間（16日間）を踏まえた **リークのないラグ特徴量（Lag 16, 21, 28等）とローリング平均** を導入して真のブレイクスルーを狙う！

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
