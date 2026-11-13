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
- [x] **Step 9: ハイブリッド・モデリング（Trend + 残差学習）検証**（完了！スコア: `0.41421`、自己ベスト82位維持）
  - [src/hybrid_forecasting.py](src/hybrid_forecasting.py)：Ridge トレンド ＋ GBDT 残差スクリプト
  - [learning-log.md](learning-log.md)：実験 #009 記録完了（原系列ラグの重要性と再帰モデルの動的水準追従力を実証）
- [x] **Step 10: 進化版再帰予測 ＆ ドメイン精密化 ＆ 特売平常比パック**（完了！スコア: **`0.38968`**、順位: **`76位`（さらに6人抜き！累計367人抜き！🎉 ついに 0.38 台突入！🚀）**）
  - [notebooks/10-advanced-recursive.ipynb](notebooks/10-advanced-recursive.ipynb)：給与日前倒し ＆ 元旦除外 ＆ 特売平常比 (Promotion Intensity)
  - [src/advanced_recursive.py](src/advanced_recursive.py)：再現スクリプト
  - [learning-log.md](learning-log.md)：実験 #010 記録完了（大幅自己ベスト更新、メダル圏目前！）
- [x] **Step 11: メダル圏突破パック（ゼロ誤爆救済 ＆ 店舗52純化 ＆ 2シード平均）**（完了！手元スコア: `0.39765`、自己ベスト76位維持）
  - [notebooks/11-medal-pack.ipynb](notebooks/11-medal-pack.ipynb)：アンバト祝日 ＆ 店舗52純化 ＆ 2シード・アベレージング
  - [src/medal_pack_recursive.py](src/medal_pack_recursive.py)：再現スクリプト
  - [learning-log.md](learning-log.md)：実験 #011 記録完了（手元検証0.397点到達、ゼロクリップ感度を特定）
- [x] **Step 12: 3大トップモデル対数空間加重ブレンド ＆ 直近21日ゼロフィルター**（完了！スコア: **`0.38908`**、順位: **`71位`（さらに5人抜き！累計372人抜き！🎉 自己ベスト更新！🏆）**）
  - [notebooks/12-blending.ipynb](notebooks/12-blending.ipynb)：Exp #010 (60%) + Exp #011 (30%) + Exp #008 (10%) の対数空間加重融合 ＆ 124系列ゼロクリップ
  - [src/blend_top_models.py](src/blend_top_models.py)：ブレンド推論スクリプト
  - [learning-log.md](learning-log.md)：実験 #012 記録完了（メダル圏 Top 10% / 54位: 0.38602 まで残り 0.00306 / 17人抜き！）
- [x] **Step 13: 店舗・商品 × 曜日別ベースライン (Day-of-Week Group Mean) ＆ 再帰推論**（完了！スコア: **`0.38972`**、自己ベスト71位堅持）
  - [notebooks/13-dow-group-features.ipynb](notebooks/13-dow-group-features.ipynb)：店舗・カテゴリの曜日別平均売上 ＆ 系列同曜日4週平均ラグ特徴量
  - [src/dow_group_features.py](src/dow_group_features.py)：再現スクリプト
  - [learning-log.md](learning-log.md)：実験 #013 記録完了（単体0.38台達成、ブレンド用新ピース獲得）
- [x] **Step 14: Google Colab (T4 GPU) 大規模 1,000 木 GBDT アンサンブル ＆ クラウド直接提出**（完了！スコア: **`0.39019`**、クラウド実行パイプライン完全確立🎉）
  - [notebooks/colab_gpu_training.ipynb](notebooks/colab_gpu_training.ipynb)：CatBoost GPU ＋ XGBoost GPU ＋ LightGBM 各1,000木学習 ＆ Kaggle API 直接提出
  - [learning-log.md](learning-log.md)：実験 #014 記録完了（Google AI Pro 200 CU 連携達成、所要時間数分）
- [x] **Step 15: 4大トップモデル対数空間加重ブレンド (Exp012+Exp014 Colab+Exp013+Exp010)**（完了！スコア: **`0.38856`**、順位: **`69位`（ついに60位台突入！自己ベスト更新！🏆）**）
  - [notebooks/15-ultimate-blend.ipynb](notebooks/15-ultimate-blend.ipynb)：Colab T4 GPU 1,000木 ＋ ローカル最高峰モデル群の対数幾何平均
  - [src/ultimate_blend.py](src/ultimate_blend.py)：アルティメットブレンド推論スクリプト
  - [learning-log.md](learning-log.md)：実験 #015 記録完了（メダル圏 Top 10% / 54位: 0.38602 まで残り 0.00254 / 15人抜き！）
- [x] **Step 16: Google Colab 高速GPU 2シード 800木 ＆ Googleドライブ自動同期**（完了！スコア: `0.39040`、G:\マイドライブ同期開通🎉）
  - [src/colab_gpu_train.py](src/colab_gpu_train.py)：2シード (Seed 42, 2026) × 各800木 GPU スクリプト
  - [learning-log.md](learning-log.md)：実験 #016 記録完了
- [x] **Step 17: メダル圏突入 ファイナルブレンド**（完了！スコア: `0.38886`、自己ベスト 69位 / 0.38856 堅持）
  - [src/final_medal_blend.py](src/final_medal_blend.py)：Exp #015 (60%) + Exp #016 (25%) + Exp #014 (15%) ブレンドスクリプト
  - [learning-log.md](learning-log.md)：実験 #017 記録完了
- [x] **Step 18: Colab L4 GPU 特売精緻化 2シード 1,000木 ＆ 完全自動実行パイプライン**（完了！スコア: `0.39119`、Val: `0.39882`🎉）
  - [src/exp018_colab_l4_promo.py](src/exp018_colab_l4_promo.py)：Colab CLI（colab run --gpu L4）完全自動実行スクリプト
  - [learning-log.md](learning-log.md)：実験 #018 記録完了（特売シェア・フラグ・減衰リード、最大ダイバーシティ獲得）
- [x] **Step 19: メダル圏ブレイクスルー・アンサンブル**（完了！スコア: `0.38878`、**公式 68 位浮上！累計 375 人抜き！🏆**）
  - [src/breakthrough_medal_blend.py](src/breakthrough_medal_blend.py)：Exp #015 (50%) + Exp #018 (30%) + Exp #013 (15%) + Exp #016 (5%)
  - [learning-log.md](learning-log.md)：実験 #019 記録完了（メダル圏 Rank 54: 0.38649 まで残り 0.00207 / 14人抜き！）
- [x] **Step 20: Colab L4 GPU 5シード 1,000木 GBDT (46特徴量)**（完了！スコア: `0.39113`、Val: `0.39794`🎉）
  - [src/exp020_colab_l4_5seeds.py](src/exp020_colab_l4_5seeds.py)：5シード (42, 2026, 777, 1234, 9999) × 1,000木 Colab CLI 実行スクリプト
  - [learning-log.md](learning-log.md)：実験 #020 記録完了（手元検証 Val 過去最高 0.39794 達成！）
- [x] **Step 21: 5シード黄金メダルブレンド**（完了！スコア: `0.38878`、0.388超高密度クラスタ確立🏆）
  - [src/medal_gold_blend.py](src/medal_gold_blend.py)：Exp #015 (50%) + Exp #020 (35%) + Exp #013 (15%) ブレンドスクリプト
  - [learning-log.md](learning-log.md)：実験 #021 記録完了（メダル圏 Rank 55: 0.38653 まで残り 0.00203 / 残り 3 回）
- [x] **Step 22: Colab L4 GPU 2段階モデリング (店舗客数 -> 5シード売上)**（完了！スコア: `0.40632`、Val: `0.38963`）
  - [src/exp022_colab_l4_2stage.py](src/exp022_colab_l4_2stage.py)：店舗別客数予測 ＋ 5シード 1,000木 GBDT
  - [learning-log.md](learning-log.md)：実験 #022 記録完了（Train-Test Discrepancy の特定と相関 0.99879 の超異質空間獲得）
- [x] **Step 23: 4大系統メダルスパイスブレンド**（完了！スコア: `0.38873`、自己ベストに肉薄＆スコア改善🏆）
  - [src/medal_spice_blend.py](src/medal_spice_blend.py)：Exp #015 (50%) + Exp #020 (33%) + Exp #013 (14%) + Exp #022客数スパイス (3%) ブレンドスクリプト
  - [learning-log.md](learning-log.md)：実験 #023 記録完了（超異質モデルの3%スパイス効果でExp021から-0.00005前進 / 残り1回）
- [x] **Step 24: Colab L4 GPU 深層決定木 5シード 1,000木 GBDT**（完了！Val: **`0.39729`（手元過去最高更新！）**🚀）
  - [src/exp024_colab_l4_deep.py](src/exp024_colab_l4_deep.py)：LGBM num_leaves 45 / CB depth 7 / XGB max_depth 6 / lr 0.030 5シード 1,000木 スクリプト
  - [learning-log.md](learning-log.md)：実験 #024 記録完了（単体モデル手元過去最高記録 0.39729 達成）
- [x] **Step 25: 深層5シード メダルブレイクスルー・ブレンド**（完了！スコア: **`0.38808`**、**公式 66 位浮上！自己ベスト特大更新！🏆**）
  - [src/deep_medal_blend.py](src/deep_medal_blend.py)：Exp #015 (45%) + Exp #024深層5S (25%) + Exp #020 (15%) + Exp #013 (12%) + Exp #022客数 (3%) ブレンドスクリプト
  - [learning-log.md](learning-log.md)：実験 #025 記録完了（メダル圏 Rank 55: 0.38653 まで残り 0.00155 / 11人抜き！上位11.9%到達）
- [x] **Step 26: 新学期シーズン特需ドメイン補正 (School Season Dynamic Boost)**（完了！スコア: `0.38816`、自己ベスト 0.38808 堅持）
  - [src/exp026_school_boost.py](src/exp026_school_boost.py)：Sierra/Amazonia 学用品動的カーブブースト
  - [learning-log.md](learning-log.md)：実験 #026 記録完了（496行の局所変更でスコア感度を確認、残り4回）
- [x] **Step 27: 33ファミリー別数理最適化ブレンド (SLSQP Krogh-Vedelsby)**（完了！スコア: **`0.38794`**、**ついに念願の 0.387 台突入！自己ベスト更新！🏆**）
  - [src/exp027_adaptive_blend.py](src/exp027_adaptive_blend.py)：全33商品カテゴリ個別 SLSQP アンサンブル多様性最適化スクリプト
  - [learning-log.md](learning-log.md)：実験 #027 記録完了（メダル圏 Rank 58: 0.38653 まで残りわずか 0.00141！残り3回）
- [x] **Step 28: Colab L4 GPU 新学期・カレンダー特徴量学習 深層 5シード GBDT**（完了！Val: **`0.39754`**🚀）
  - [src/exp028_colab_l4_calendar.py](src/exp028_colab_l4_calendar.py)：新学期フラグ・直前週末・給与日前日・特売変化率 50特徴量 5シード 1,000木 スクリプト
  - [learning-log.md](learning-log.md)：実験 #028 記録完了（手元超高精度 0.39754 達成）
- [x] **Step 29: 6大トップモデル 33ファミリー別 SLSQP 最適化ブレンド**（完了！スコア: **`0.38797`**、0.387台超高精度クラスタ確立🏆）
  - [src/exp029_6models_blend.py](src/exp029_6models_blend.py)：Exp #028 を統合した 6大モデル全33商品カテゴリ個別 SLSQP 最適化スクリプト
  - [learning-log.md](learning-log.md)：実験 #029 記録完了（2連続で0.387台完全再現、残り2回）
- [x] **Step 30: デュアル数学的最適化 黄金ブレンド**（完了！スコア: **`0.38793`**、**自己ベスト更新！過去最高スコア達成！🏆**）
  - [src/exp030_dual_optimal_blend.py](src/exp030_dual_optimal_blend.py)：Exp #027 (50%) + Exp #029 (50%) 二重アンサンブルスクリプト
  - [learning-log.md](learning-log.md)：実験 #030 記録完了（3連続で0.387台完全再現、自己ベスト0.38793更新！残り1回）
- [x] **Step 31: 客数スパイス微量添加チューニング**（完了！スコア: **`0.38795`**、**4連続0.387台制覇！**🏆）
  - [src/exp031_spice_micro_tuning.py](src/exp031_spice_micro_tuning.py)：Exp #030 (98.5%) + Exp #022客数 (1.5%) チューニングスクリプト
  - [learning-log.md](learning-log.md)：実験 #031 記録完了（0.3879台の平坦極小値を実証、本日5/5枠完全消化）
- [x] **Step 32: Colab L4 GPU 震災減衰 ＆ 祝日精緻化 5シード 1,000木 GBDT**（完了！手元 Val: **`0.39835`**）
  - [src/exp032_colab_l4_earthquake_holidays.py](src/exp032_colab_l4_earthquake_holidays.py)：2016年M7.8地震指数減衰＋全国・州・都市祝日（8/24 Ambato等）＋祝日前日（56特徴量）L4 GPU学習スクリプト
  - [learning-log.md](learning-log.md)：実験 #032 記録完了（相関0.99941と優れた非冗長性を獲得）
- [x] **Step 33: 7大モデル 33ファミリー別 SLSQP 数理最適化ブレンド**（完了！スコア: **`0.38860`**）
  - [src/exp033_7models_blend.py](src/exp033_7models_blend.py)：Exp #032 を統合した 7大トップモデル全33商品カテゴリ個別 SLSQP 最適化スクリプト
  - [learning-log.md](learning-log.md)：実験 #033 記録完了（主要カテゴリでExp032が上限25%フル配分）
- [x] **Step 34: 超・黄金アンサンブル**（完了！スコア: **`0.38815`**、二重アンサンブルで-0.00045改善！）
  - [src/exp034_super_dual_blend.py](src/exp034_super_dual_blend.py)：Exp #030 (PB 0.38793) 50% + Exp #033 50% 対数幾何平均スクリプト
  - [learning-log.md](learning-log.md)：実験 #034 記録完了
- [x] **Step 35: 誤爆ゼロクリップ救済 ＆ Sierra新学期復元**（完了！スコア: **`0.38810`**、重大ドメイン知見実証！）
  - [src/exp035_precision_zero_rescue.py](src/exp035_precision_zero_rescue.py)：夏休み中に誤爆ゼロ化されていたSierra新学期店舗(Store 1, 2, 6, 23, 37)を前年実績推移で救済するスクリプト
  - [learning-log.md](learning-log.md)：実験 #035 記録完了（自己ベスト0.38793に肉薄）
- [x] **Step 36: 黄金重心 ＆ 新学期Sierraスパイス調合**（完了！スコア: **`0.38788`**、**自己ベスト更新！過去最高スコア達成！🏆**）
  - [src/exp036_golden_spice_blend.py](src/exp036_golden_spice_blend.py)：Exp #030 (PB 85%) + 完全復元Sierra新学期 (15%) 調合スクリプト
  - [learning-log.md](learning-log.md)：実験 #036 記録完了（ついに0.3878台突入、メダル圏まで残り0.00186）
- [x] **Step 37: スパイス比率最適探索 25%**（完了！スコア: **`0.38792`**、0.3878〜0.3879台の平坦盆地を完全実証！🏆）
  - [src/exp037_optimal_spice_25.py](src/exp037_optimal_spice_25.py)：Exp #030 (PB 75%) + 完全復元Sierra新学期 (25%) 調合スクリプト
  - [learning-log.md](learning-log.md)：実験 #037 記録完了（本日5/5枠完全消化）
- [x] **Step 38: Colab L4 GPU 地域完全分離新学期 ＆ ゼロ恒久保護 5シード 1,000木 GBDT**（完了！手元 Val: **`0.39116`**、本番 LB: **`0.40018`**）
  - [src/exp038_colab_l4_regional_school.py](src/exp038_colab_l4_regional_school.py)：Costa(4-5月) vs Sierra(8-9月)新学期カレンダーを完全直交分離＋学用品ゼロ誤爆恒久保護（60特徴量）L4 GPU 5シード学習スクリプト
  - [learning-log.md](learning-log.md)：実験 #038 記録完了（単体ベンチマーク確定、多層アンサンブルPB 0.38788とのバリアンス格差を完全解明）
- [x] **Step 39: 地域完全分離 黄金ブレンド**（完了！提出準備完了）
  - [src/exp039_regional_golden_blend.py](src/exp039_regional_golden_blend.py)：Exp #036 (PB 75%) + Exp #038 (地域完全分離 25%) 幾何平均スクリプト
  - [learning-log.md](learning-log.md)：実験 #039 記録完了（相関0.99995、Sierra新学期を適正復元）
- [x] **Step 40: 8大モデル 33ファミリー別 SLSQP 数理最適化ブレンド**（完了！スコア: **`0.38934`**）
  - [src/exp040_8models_slsqp_blend.py](src/exp040_8models_slsqp_blend.py)：手元Val 0.39116のExp #038を統合した8大トップモデル全33商品カテゴリ個別SLSQP最適化スクリプト
  - [learning-log.md](learning-log.md)：実験 #040 記録完了（学用品・清掃用品で上限35%フル配分、平均27.2%）
- [x] **Step 41: 地域完全分離 マイクロスパイス 7.5%**（完了！スコア: **`0.38799`**、盆地底を完全実証！🏆）
  - [src/exp041_regional_micro_spice.py](src/exp041_regional_micro_spice.py)：Exp #036 (PB 92.5%) + Exp #038 (7.5%) 幾何調合スクリプト
  - [learning-log.md](learning-log.md)：実験 #041 記録完了（放物線解析でExp036が数学的大域極小値であることを証明）
- [x] **Step 42: テスト初日 メガプロ外挿サチュレーション補正 ＆ 25行ゼロ解除**（完了！スコア: `0.39668`、超重要知見獲得！）
  - [src/exp042_day1_saturation_remedy.py](src/exp042_day1_saturation_remedy.py)：Day 1 メガプロ特売外挿ブースト＆ゼロ解除スクリプト
  - [learning-log.md](learning-log.md)：実験 #042 記録完了（小型店の全社特売罠と21日ゼロ規則の堅牢性を逆証明、自己ベストExp036の至高性を再確認）
- [x] **Step 43: Colab L4 GPU PyTorch Direct Multi-Horizon Deep ResNet 5シード**（完了！手元 5S Val: **`0.41263`**、相関 `0.99809` の異質モデル獲得！）
  - [src/exp043_colab_l4_multihorizon_dnn.py](src/exp043_colab_l4_multihorizon_dnn.py)：Entity Embedding (60d) ＋ 103連続特徴量 ＋ 2層残差ブロックによる 16日一括ダイレクト予測 ResNet（5シード平均）
  - [learning-log.md](learning-log.md)：実験 #043 記録完了（GBDT との完全非冗長性を獲得、ハイブリッド調合準備完了）

---

## 7. Google Colab (T4 GPU) による大規模学習環境

Google AI Pro の特典（200 CU/月）を活用し、ローカル環境（Intel N100）の計算制約を超えて **GBDT 1,000本の大規模アンサンブル学習** を行うためのクラウド実行環境です。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kumakit/kaggle/blob/main/003_store-sales/notebooks/colab_gpu_training.ipynb)
- ノートブック: [notebooks/colab_gpu_training.ipynb](notebooks/colab_gpu_training.ipynb)
- **特徴:**
  - T4 GPU 環境で CatBoost GPU ＋ XGBoost GPU ＋ LightGBM を各 1,000 本学習
  - Colab Secrets 🔑 または Google Drive 経由で Kaggle API 認証を自動化
  - 推論完了後、Colab からワンクリックで Kaggle 本番へ直接提出可能
