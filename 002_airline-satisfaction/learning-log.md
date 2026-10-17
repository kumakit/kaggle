# 学習・実験ログ (Learning Log): Airline Passenger Satisfaction

- **コンペ名:** [Playground Series - Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10)
- **関連Issue:** [kumakit/mission-control#404](https://github.com/kumakit/mission-control/issues/404)
- **評価指標:** **ROC AUC (Area Under the ROC Curve)**

---

## 実験一覧サマリー

| 実験 # | 手法・モデル | 検証スコア (OOF ROC AUC) | Kaggle本番スコア (Public LB) | 順位 / 参加数 | 備考 |
|---|---|---|---|---|---|
| **#001** | **LightGBM (素のテーブル特徴量) + Stratified 5-Fold CV** | **`0.95828`** | **`0.95791`** | **1,697 位** / 2,128 チーム | **初提出完了！** 手元と本番が誤差 0.00037 で完全一致 |
| **#002** | **ドメイン特徴量 (遅延集計・サービス統計・カテゴリ交差) + パラメータ調整** | *(進行中)* | - | - | 特徴量を21列→32列へ拡張 |

---

## 実験詳細

### 実験 #001: LightGBM による素のテーブルデータ初回提出 (ベースライン)
- **実施日:** 2026-10-17
- **担当・実施者:** 学習者本人（Kaggle Notebook経由で提出）

#### 1. 仮説 (Hypothesis)
- 表形式データにおいて最も強力な決定木アンサンブルである **LightGBM** を用い、カテゴリ変数（4列）を適切に `category` 型として扱えば、複雑な前処理なしでも手堅く高いROC AUC（0.95以上）を達成できるはず。
- 層化5分割交差検証（Stratified 5-Fold CV）を組むことで、手元の模試スコア（OOF ROC AUC）と本番スコア（Public Leaderboard）のズレを最小限に抑えられるはず。

#### 2. 実験内容 (Method)
- **データ:** `train.csv` (699,635行 × 23列), `test.csv` (299,844行 × 22列)
- **前処理:**
  - 目的変数 `satisfaction`（bool）を 0 / 1 の整数に変換
  - 文字列列（`Gender`, `Customer Type`, `Type of Travel`, `Class`）を `category` 型に変換
- **データ分割:** 層化5分割交差検証（Stratified 5-Fold CV, `random_state=42`）
- **モデル設定 (LightGBM):**
  - `objective='binary'`, `metric='auc'`, `learning_rate=0.05`, `num_leaves=31`, `n_estimators=500`
  - `early_stopping_rounds=30`（過学習防止）
- **提出形式:** `id, satisfaction`（299,844行、5モデルの予測確率平均値）

#### 3. 結果 (Result)
- **手元検証スコア (OOF ROC AUC):** **`0.95828`**
  - 各Foldスコア: `[0.95839, 0.95713, 0.95896, 0.95831, 0.95865]`
- **Kaggle本番スコア (Public Leaderboard):** **`0.95791`**
- **順位:** **1,697 位** / 2,128 チーム中
- **手元と本番の誤差:** わずか **0.00037**

#### 4. 分かったこと・統計的考察 (Insights)
- **CV（交差検証）とLB（リーダーボード）の驚異的な一致:**
  - 70万行の大規模データかつ各Foldのスコアばらつきが非常に小さいため（標準偏差 約0.0006）、手元の検証結果が本番テストに対して極めて信頼性の高い「羅針盤」として機能している。
- **リアルタイムコンペの密度:**
  - トップ（1位: 0.96284）と自分のベースライン（0.95791）の差はわずか **0.00493**。
  - この 0.005 ポイントの狭い帯の中に2,000人近くがひしめいているため、特徴量エンジニアリング等で `+0.002〜0.003` 改善するだけで、順位が一気に1000人以上跳ね上がる超激戦区であることが判明した。
