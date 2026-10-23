# Kaggle 学習・コンペティション挑戦リポジトリ

本リポジトリは、Kaggleのコンペティションを通じて機械学習、データ分析、および統計モデリングの実践力を段階的に高めていくための学習リポジトリです。
GitHub Issue（[kumakit/mission-control#402](https://github.com/kumakit/mission-control/issues/402)）を作業計画・進捗の正本として管理しています。

---

## 1. 基本方針・ルール

- **小さく着実に進める:** 1回あたり **30〜60分程度**、「今やることは1つだけ」に集中します。
- **環境の優先順位:** 
  - 最初はPCへのセットアップ不要な **Kaggle Notebook（ブラウザ環境）** を最優先します。
  - ローカルで実行する場合は Windows PowerShell 7 (`pwsh`) + `uv` + Python 3.12以上（`.venv`）を使用します。
- **理解重視:** 外部の自動生成コード（Codex等）に頼り切らず、モデルの仕組みや評価指標（統計検定2級レベルの混同行列等）の意味を1つずつ理解しながら進めます。
- **過剰な自動化の防止:** 外部サービスの設定変更・自動提出・不要な定期ジョブ（常駐化）は行わず、手動での検証と考察を大切にします。

---

## 2. コンペティション一覧

今後挑戦するコンペティションは、実施順がわかるよう3桁の連番プレフィックス（`001_`, `002_`, ...）を付けた専用フォルダを作成して独立管理します。

| # | コンペ名 | フォルダ | 種別 / タスク | ステータス | ガイドリンク |
|---|---|---|---|---|---|
| **001** | **Digit Recognizer** | [`001_digit-recognizer/`](001_digit-recognizer/) | Getting Started / 手書き数字画像分類 (0〜9) | **完了・卒業 (0.99132 / 319位)** | [入門ガイド](001_digit-recognizer/README.md) |
| **002** | **Playground S6E10 (Airline Satisfaction)** | [`002_airline-satisfaction/`](002_airline-satisfaction/) | Playground / 航空会社顧客満足度予測 (2値分類・ROC AUC) | **自己ベスト 0.96084 / 3桁856位達成！🏆** | [実践ガイド](002_airline-satisfaction/README.md) |
| **003** | **Store Sales - Time Series Forecasting** | [`003_store-sales/`](003_store-sales/) | Getting Started / 店舗売上の時系列需要予測 (回帰・RMSLE) | **自己ベスト 0.45505 / 277位達成！🚀 ([#407](https://github.com/kumakit/mission-control/issues/407))** | [実践ガイド](003_store-sales/README.md) |


---

## 3. ディレクトリ構成

各コンペティションフォルダ内は、以下の標準構造で整理されます。

```text
kaggle/
├── README.md                          # 全体インデックス（本ファイル）
├── .gitignore                         # Python/データ共通の除外設定
│
├── 001_digit-recognizer/              # 実施順の連番付きコンペフォルダ
│   ├── README.md                      # コンペ入門ガイド・進行手順・総括
│   ├── rules-ja.md                    # 公式ルールの日本語要約
│   ├── community-digest.md            # 初心者向け厳選Q&A・おすすめNotebook
│   ├── poc-info-retrieval.md          # 情報取得（MCP/CLI）の検証レポート
│   ├── learning-log.md                # 実験記録・考察（#001〜#003）
│   ├── notebooks/                     # Kaggle / Jupyter ノートブック
│   │   ├── 01-baseline.ipynb          # 初回提出用ベースライン (0.96360)
│   │   ├── 02-model-comparison.ipynb # ロジスティック回帰 vs RF 比較 & 誤答可視化
│   │   └── 03-cnn.ipynb               # 2ブロックCNN (0.99132 / 319位)
│   └── src/                           # ローカル再現用スクリプト
│       ├── baseline.py
│       └── model_comparison.py
│
└── 002_[次のコンペ名]/                 # 今後追加されるコンペフォルダ
```
