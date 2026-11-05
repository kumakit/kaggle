# Kaggle 実践ガイド: Store Sales - Time Series Forecasting（店舗売上の時系列需要予測）

本ガイドは、機械学習の実務で最も頻出する「需要予測・時系列回帰」を体系的に習得するための実践入門ドキュメントです。
GitHub Issue [#407](https://github.com/kumakit/mission-control/issues/407) を作業計画・進捗の正本として管理します。

> [!TIP]
> **本実践（003）の方針と狙い**
> - **時系列データ（Time Series）の基礎を完全制覇:** 画像（001: CNN）、テーブル分類（002: GBDT）に続き、未来を予測する「時系列モデリング」を習得します。
> - **未来リーク厳禁！時系列専用の検証設計:** 通常のランダムK-Fold交差検証は使えません。未来のデータで過去を予測してしまうデータ漏洩（Data Leakage）を防ぐ「Time-based Validation」を学びます。
> - **統計検定2級の知識が直結:** トレンド、季節変動（曜日・月・給与日）、自己相関、移動平均、ラグ特徴量、そして対数変換（$\log(1+y)$）とRMSLEの関係を数理的に体感します。

---

## 1. コンペティションの概要
- **コンペ名:** [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **カテゴリ:** Getting Started（常設コンペ・自分のペースでじっくり実験可能）
- **タスク:** エクアドルの大手スーパーチェーン「Corporación Favorita」における、各店舗・商品ファミリーごとの日次売上数量（`sales`）の予測（**時系列回帰タスク**）。
- **予測期間:** 訓練データの最終日直後からの **16日間**（約半月分）

---

## 2. 予測する対象と提出形式

### 予測対象 (Target)
- **列名:** `sales`
- **内容:** 特定店舗（`store_nbr`）における特定商品ファミリー（`family`）の特定日（`date`）の売上数量（小数点を含む正の実数値）。

### 提出形式 (Submission Format)
```csv
id,sales
3000888,0.0
3000889,15.2
3000890,128.5
...
```

---

## 3. データの構造（複数テーブルの関連）

本コンペは複数の補助テーブルを結合して特徴量を作成する、実務そのままの構造です。

| ファイル名 | 行数・サイズ | 主なカラム | データの意味と役割 |
|---|---|---|---|
| **`train.csv`** | 約300万行 (121MB) | `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` | 2013〜2017年の日次売上実績データ（学習用） |
| **`test.csv`** | 約2.8万行 (1MB) | `id`, `date`, `store_nbr`, `family`, `onpromotion` | 予測対象の16日間のデータ |
| **`stores.csv`** | 54行 | `store_nbr`, `city`, `state`, `type`, `cluster` | 店舗メタ情報（都市、州、店舗タイプ、クラスタ） |
| **`oil.csv`** | 約1200行 | `date`, `dcoilwtico` | 日次原油価格（石油輸出国エクアドルの景気動向） |
| **`holidays_events.csv`** | 約350行 | `date`, `type`, `locale`, `description`, `transferred` | 祝日・イベント情報（※振替休日フラグに注意） |
| **`transactions.csv`** | 約8.3万行 | `date`, `store_nbr`, `transactions` | 店舗ごとの日次レジ通過客数・取引件数 |

---

## 4. 評価指標: RMSLE（Root Mean Squared Logarithmic Error）

本コンペの評価指標は **RMSLE（対数平均二乗誤差の平方根）** です。

$$\text{RMSLE} = \sqrt{ \frac{1}{n} \sum_{i=1}^n \left( \ln(1 + \hat{y}_i) - \ln(1 + y_i) \right)^2 }$$

### 専門用語のやさしい解説: なぜ単なるRMSEではなく「対数」なのか？
- **売上の規模格差に対応するため:**
  - 1日1万個売れる「パン」の100個の誤差と、1日10個しか売れない「家電」の100個の誤差はビジネス的重要度が全く異なります。
  - 対数を取ると「比率（パーセンテージ）の誤差」を評価することになり、高価格・低頻度商品と定番消耗品を公平に評価できます。
- **モデリングの定石（鉄則）:**
  - 目的変数 $y$ を学習前に **$z = \ln(1 + y)$（`np.log1p(y)`）** に変換して通常のRMSEで学習し、予測時に **$\hat{y} = \exp(z) - 1$（`np.expm1(z)`）** で元に戻すアプローチが最も安定します。

---

## 5. 時系列モデリングの最重要鉄則

### ① 未来のカンニング厳禁（時系列分割バリデーション）
通常のK-Fold（ランダムシャッフル）を行うと、未来の売上データを使って過去の売上を予測してしまう致命的なリークが発生します。手元検証は**「訓練データの最後の16日間」を検証用（Validation）として切り出す Time-based Split** を採用します。

### ② ドメイン知識の反映（実務的アイデア）
- **給与日の影響:** エクアドルの公務員給与は「毎月15日」と「月末」に支給されます（支給日直後にスーパー売上が急増）。
- **震災イベント:** 2016年4月16日にエクアドル大地震（M7.8）が発生し、救援物資需要で数週間にわたり特定の水・缶詰などの売上が異常値を示します。
- **原油価格:** エクアドル経済は原油相場と連動（欠損値補間が必要）。
- **振替休日:** `holidays_events.csv` の `transferred == True` は公休日が移動しているため注意が必要。

---

## 6. 学習・挑戦ロードマップ

- [x] **Step 1: コンペ参加規約への同意**（完了！）
  - 公式ページで「Join Competition」を完了。
- [x] **Step 2: 最小ベースラインの作成と初回提出**（完了！スコア: **`0.52338`**、順位: **443位** 🎉）
  - [notebooks/01-baseline.ipynb](notebooks/01-baseline.ipynb)：LightGBM カレンダー特徴量ベースライン
  - [src/baseline.py](src/baseline.py)：ローカル再現スクリプト
- [x] **Step 3: カレンダー・ドメイン特徴量の導入**（完了！スコア: `0.52581`）
  - [notebooks/02-domain-features.ipynb](notebooks/02-domain-features.ipynb)：原油・店舗冗長列の過学習を特定（アブレーション）
- [x] **Step 4: 時系列の真骨頂「ラグ特徴量」＆「33ファミリー別モデル」**（完了！スコア: **`0.45505`**、順位: **`277位`（166人抜き！🚀）**）
  - [notebooks/04-family-models.ipynb](notebooks/04-family-models.ipynb)：33ファミリー個別LightGBM ＋ ゼロ売上フィルター
  - [src/family_models.py](src/family_models.py)：再現スクリプト
  - [learning-log.md](learning-log.md)：実験 #004 記録完了
- [x] **Step 5: GBDT多様性アンサンブル ＋ 前年同期ラグ（RTX 4060 GPU活用）**（完了！スコア: **`0.43078`**、順位: **`210位`（さらに67人抜き・累計233人抜き！🎉）**）
  - [notebooks/05-ensemble-gpu.ipynb](notebooks/05-ensemble-gpu.ipynb)：LightGBM + CatBoost GPU + XGBoost GPU ＋ 52週前同曜日ラグ (`sales_lag_364`)
  - [src/ensemble_gpu.py](src/ensemble_gpu.py)：GPU並列ローカルスクリプト
  - [learning-log.md](learning-log.md)：実験 #005 記録完了
- [x] **Step 6: 特売プロモーション先行・後行特徴量 ＆ 3大GBDTアンサンブル**（完了！スコア: **`0.41064`**、順位: **`149位`（さらに61人抜き・累計294人抜き！🚀）**）
  - [notebooks/06-promo-features.ipynb](notebooks/06-promo-features.ipynb)：特売リード（Lead 1/2）・ラグ（Lag 1/2/7）・差分・移動平均特徴量
  - [src/promo_features.py](src/promo_features.py)：再現スクリプト
  - [learning-log.md](learning-log.md)：実験 #006 記録完了
- [x] **Step 7: ファミリー別アンサンブル重み数理最適化**（完了！スコア: `0.41188`、順位: 155位）
  - [notebooks/07-optimize-weights.ipynb](notebooks/07-optimize-weights.ipynb)：SciPy SLSQP による各カテゴリの黄金比率探索
  - [src/optimize_weights.py](src/optimize_weights.py)：再現スクリプト
  - [learning-log.md](learning-log.md)：実験 #007 記録完了（手元0.3点台達成、固定重みの汎化力を立証）
- [x] **Step 8: 再帰的マルチステップ予測 ＆ 直近ラグ完全解禁**（完了！スコア: **`0.39084`**、順位: **`82位`（一撃73人抜き！累計361人抜き！🎉 念願の100位切り・2桁順位達成！🚀）**）
  - [notebooks/08-recursive-forecasting.ipynb](notebooks/08-recursive-forecasting.ipynb)：直近Lag 1〜15完全解禁 ＆ 1日ずつ順次予測する再帰推論
  - [src/recursive_forecasting.py](src/recursive_forecasting.py)：再現スクリプト
  - [learning-log.md](learning-log.md)：実験 #008 記録完了（-0.01980 特大ブレイクスルー！）
