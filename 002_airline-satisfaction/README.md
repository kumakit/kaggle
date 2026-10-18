# Kaggle 実践ガイド: Airline Passenger Satisfaction（航空会社満足度予測）

本ガイドは、現在世界中でリアルタイムに競い合われている **Kaggle Playground Series (Season 6, Episode 10)** に挑戦するための実践入門ドキュメントです。
GitHub Issue [#404](https://github.com/kumakit/mission-control/issues/404) を作業計画・進捗の正本として管理します。

> [!TIP]
> **本実践の方針**
> - **リアルタイムの熱気を感じる:** 過去の固定されたチュートリアルではなく、世界中のKagglerが毎日最新アイデアを投稿しているDiscussionやNotebookをリアルタイムで体感します。
> - **テーブルデータ（表形式）の最強モデルを習得:** 実務で圧倒的な実績を誇る決定木アンサンブル **`LightGBM`** を実践投入します。
> - **評価指標 ROC AUC を理解する:** 0/1の決め打ちではなく「確率の並び順」を評価する、実務（信用スコアリングやマーケティング）で最も重要な指標を学びます。

---

## 1. コンペティションの概要
- **コンペ名:** [Playground Series - Season 6, Episode 10](https://www.kaggle.com/competitions/playground-series-s6e10)
- **開催期間:** 2026年10月1日 〜 2026年10月31日（現在開催中！）
- **目的:** 航空会社の乗客アンケート・フライト属性データから、乗客が「満足（satisfied）」したか「不満・普通（neutral or dissatisfied）」かを予測する **2値分類（Binary Classification）** です。

---

## 2. 予測する対象と提出形式

### 予測対象 (Target)
- **列名:** `satisfaction`
- **値:** 
  - `1`: 満足（satisfied）
  - `0`: 不満 / 普通（neutral or dissatisfied）

### 提出形式 (Submission Format)
Kaggleにアップロードする提出ファイルは、以下の形式のCSVです。
**重要:** 今回のコンペでは、単なる `0` または `1` のラベルではなく、**「満足（1）である確率（0.0 〜 1.0 の実数値）」** を出力します。

```csv
id,satisfaction
103904,0.785
103905,0.124
103906,0.952
...
```

---

## 3. データの構造と特徴量

データは乗客の属性、旅行タイプ、機内サービスの5段階評価、遅延時間などで構成されています。

| カテゴリ | 主な列名（特徴量） | 内容 |
|---|---|---|
| **基本属性** | `Gender`, `Customer Type`, `Age` | 性別、リピーター/初回、年齢 |
| **旅行情報** | `Type of Travel`, `Class`, `Flight Distance` | ビジネス/個人旅行、座席クラス（Eco/Business）、飛行距離 |
| **サービス評価**<br>(0〜5点) | `Seat comfort`, `Inflight entertainment`, `On-board service`, `Cleanliness`, `Inflight wifi service`, `Ease of Online booking` 等 | 座席の快適さ、機内エンタメ、WiFi、清潔さなどのアンケート評価スコア |
| **遅延時間** | `Departure Delay in Minutes`, `Arrival Delay in Minutes` | 出発遅延（分）、到着遅延（分） |

---

## 4. 評価指標: ROC AUC（Area Under the ROC Curve）

本コンペの成績は、正解率（Accuracy）ではなく **ROC AUC** で競われます。

### 専門用語のやさしい解説: ROC AUC とは？
- **Accuracy（正解率）との違い:**
  - Accuracy は「確率が 0.5 以上なら 1、未満なら 0」と閾値を固定して、当たった割合を測ります。
  - しかし、「絶対に満足しそうな乗客」と「ぎりぎり満足しそうな乗客」を区別できません。
- **ROC AUC の本質:**
  - **「予測した確率の高い順に並べたとき、本当に満足した乗客が上位に正しく並んでいるか？」** を測る指標です。
  - `1.0`: 完璧に並べ替えられている（満足した人が全員、不満な人より高い確率をつけられている）。
  - `0.5`: 当てずっぽう（コイントスと同じ）。
  - `0.85〜0.95`: 実務でも非常に優秀なスコア。

---

## 5. 今回学ぶ新しい機械学習モデル: LightGBM

Digit Recognizer では画像専用の「CNN」を使いましたが、**表形式（テーブル）データにおいて現在世界最強のアルゴリズムは `LightGBM`（決定木勾配ブースティング / GBDT）** です。

- **なぜLightGBMなのか？**
  - 高速でメモリ効率が良い。
  - 欠損値（空欄）やカテゴリ変数（文字列）を柔軟に扱える。
  - 特徴量間の複雑な「掛け合わせ（例: ビジネス利用 かつ 遅延が30分以上）」を自動的に捉える。

---

## 6. 学習・挑戦のロードマップ

- [x] **Step 1: コンペ参加規約への同意**（完了！）
  - 公式ページで「Join Competition」を完了。
- [x] **Step 2: 最初のベースライン提出（Kaggle Notebook）**（完了！スコア: **`0.95791`**、順位: **1,697位** 🎉）
  - [notebooks/01-baseline.ipynb](notebooks/01-baseline.ipynb)：Kaggle上でそのまま動くLightGBMベースラインノートブック
  - [src/baseline.py](src/baseline.py)：ローカル再現用スクリプト
  - [learning-log.md](learning-log.md)：実験 #001 記録済み（手元 `0.95828` と本番 `0.95791` が完全連動）
- [x] **Step 3: リアルタイムDiscussionの探索 & ドメイン特徴量作成**（完了！）
  - Kaggle公式MCPを活用し、最新トピック「Flight Distance ＝ 路線ID」の知見を発掘！
  - [src/feature_engineering.py](src/feature_engineering.py) および [src/ensemble_blend.py](src/ensemble_blend.py)
- [x] **Step 4: スコア改善と上位へのランクアップ**（完了！スコア: **`0.95954`**、順位: **`1,058位`** 🚀）
  - [src/route_id_te.py](src/route_id_te.py)：路線頻度 ＋ In-Fold Target Encoding 実装
  - [learning-log.md](learning-log.md)：実験 #004 にて **一撃で 633人抜き（1691位 ➔ 1058位）の大躍進** を達成！
- [x] **Step 5: 3大GBDTアンサンブル（LGB + CB + XGB）と自己ベスト更新**（完了！スコア: **`0.95971`** 🏆）
  - [src/ensemble_3gbdt.py](src/ensemble_3gbdt.py)：多様性注入XGBoost ＋ 絶妙CatBoost ＋ 精密LightGBM
  - [learning-log.md](learning-log.md)：実験 #009 にて **自己ベスト更新（0.95971）** を達成！Discussionの限界を突破。


