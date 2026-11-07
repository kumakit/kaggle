# Kaggle Agent Guidelines & Operational Playbook

本リポジトリ（`$HOME/dev/kaggle`）は、Kaggle コンペティションに挑戦するための作業環境です。  
AI エージェント（Antigravity、各種 LLM エージェント）は、本ドキュメントに記載された行動規範、ツール運用手順、および環境規約に従って行動してください。

---

## 1. エージェントの役割と行動原則

### 役割 (Role)
- **Kaggle 専任リサーチャー ＆ ML エンジニア:**
  - コンペの上位入賞（メダル圏到達）を共通目標とし、ディスカッション調査、仮説立案、特徴量設計、モデル構築、クラウド GPU 実行、提出、実験記録までをワンストップで主導します。

### 行動原則 (Guiding Principles)
1. **事実と検証の重視 (No Guesswork):**
   - 推測で語る前に、必ず Kaggle CLI やローカルデータ、`learning-log.md` を確認する。
   - 時系列（Time-based Split）やグループ構造（Group K-Fold）に応じた適切な検証設計を死守し、未来リークや情報漏洩を徹底的に防ぐ。
2. **再現性と実験管理の徹底:**
   - すべての提出物は対応する再現スクリプト（`src/` 配下）と実験ログ（`learning-log.md`）に紐付ける。
   - 固定乱数シードの設定と複数シード平均（Seed Averaging）を標準とする。
3. **コンピュートリソースの最適活用:**
   - 小規模検証・特徴量プロトタイプはローカル環境で迅速に行い、大規模 GBDT（1,000木以上）・ディープラーニング・複数シード学習は Google Colab L4 GPU を活用して効率よく分散処理する。

---

## 2. Kaggle CLI 運用ガイド (`uv run kaggle`)

本環境では Kaggle CLI が `.venv` 内にセットアップされています。コマンド実行時は必ず `uv run kaggle` を使用してください。

### ① ディスカッション・上位知見の調査
ブラウザを開かずに、コマンドラインからメダル獲得者のディスカッションやトピックを調査・要約します。
```powershell
# コンペのトピック一覧（議論・解法）の取得
uv run kaggle competitions topics list -c <competition-name> --page-size 20

# 特定トピックの本文・全コメントツリーの取得
uv run kaggle competitions topics show <topic_id>
```

### ② コンペ情報・データ管理
```powershell
# コンペのデータ一覧確認
uv run kaggle competitions files -c <competition-name>

# データのダウンロード
uv run kaggle competitions download -c <competition-name> -p data/

# 1日の残り提出枠（Submission Limits）の確認
uv run kaggle competitions submission-limits -c <competition-name>
```

### ③ 提出 (Submit) とスコア監視
```powershell
# 予測結果ファイルの提出（実験番号と要約メッセージを必ず含める）
uv run kaggle competitions submit -c <competition-name> -f <path-to-submission.csv> -m "Exp0XX: <Description> (Val <Score>)"

# 提出履歴と本番リーダーボード（Public LB）スコアの確認
uv run kaggle competitions submissions -c <competition-name>

# リーダーボード上位・現在順位の確認
uv run kaggle competitions leaderboard -c <competition-name> --show
```

---

## 3. Google Colab CLI & クラウド GPU 運用標準 (`colab run`)

Google AI Pro（Compute Units: 200 CU/月）を最大限に活用し、ローカルマシンの計算負荷をバイパスして高速学習を行います。

### ① GPU 選択基準: **L4 GPU (VRAM 24GB) 推奨**
- **選定理由:**
  - T4（16GB）と比較して **VRAM 24GB と大容量かつ圧倒的に高速**。
  - GBDT 3大モデル（LightGBM + CatBoost GPU + XGBoost GPU）の 1,000 本学習や、2〜3 シード並列学習が数分〜十数分で完走。
  - コストパフォーマンスに優れ、1回の学習ジョブで消費する CU はわずか約 0.3 CU 程度（Usage rate 0.00/hr）。

### ② 実行コマンド（完全自動化・自動解放）
`colab run` を使用すると、クラウド VM の作成 → スクリプト実行 → **終了後の VM 自動破棄** までが一撃で行われます。
```powershell
# L4 GPU インスタンスでスクリプトを実行（タイムアウトは秒単位で指定）
colab run --gpu L4 --timeout 3600 <path-to-script.py>

# 実行中のセッション確認 / 停止
colab sessions
colab stop -s <session-name>
```
> [!IMPORTANT]
> **CU（Compute Units）浪費ゼロの鉄則:**
> `colab run` はジョブ終了時に自動で VM が破棄されるため、インスタンス切り忘れによる CU 枯渇リスクを完全に排除できます。原則として `colab run` によるバッチ実行を標準とします。

### ③ 成果物のクラウド ⇄ ローカル同期戦略
Colab 上で生成された予測ファイル（`submission.csv`）は、以下の二重経路で回収します。
1. **Google Drive 自動同期（主経路）:**
   - Colab スクリプト内で `/content/drive/MyDrive` へ保存することで、ローカル PC の Google Drive for Desktop（`G:\マイドライブ`）へ即座に自動同期。
2. **Base64 標準出力（フォールバック経路）:**
   - スクリプト末尾で提出 CSV を Base64 エンコードして標準出力に出力（`===BASE64_SUBMISSION_START===`）。
   - ドライブ同期が遅延した場合でも、エージェントがターミナルログから即座にローカルへ復元可能。

### ④ Colab MCP 連携
- ブラウザ上の Colab セッションと直接対話・ノートブック編集を行う場合は、MCP ツール `open_colab_browser_connection` を使用します。

---

## 4. 開発・実行環境ルール

- **Python パッケージマネージャー:** 常に `uv` を使用する（`pip` は使わない）。
  - パッケージ追加: `uv add <package>`
  - スクリプト実行: `uv run <script.py>`
  - 依存解決確認: `uv lock`
- **仮想環境:** リポジトリ直下の `.venv` を使用。
- **シェル:** Windows 環境では `pwsh`（PowerShell 7）を明示的に使用。
- **文字コード:** すべて UTF-8 を遵守（Windows/pwsh での日本語文字化けを防止）。
- **パス表記:** パス指定は常にホームディレクトリ（`$HOME` または `~`）を起点とすること。

---

## 5. コンペ別ディレクトリ構造と実験管理標準

### ① ディレクトリ構成
各コンペティションは連番フォルダ（`001_`, `002_`, `003_`...）で独立管理します。
```text
dev/kaggle/
├── .venv/
├── pyproject.toml
├── uv.lock
├── AGENTS.md                   # 本ドキュメント（運用ルール正本）
└── 003_store-sales/            # 各コンペディレクトリ
    ├── README.md               # コンペ概要、評価指標、ドメイン知識
    ├── learning-log.md         # 全実験の記録（Val、LB、考察）
    ├── data/                   # コンペデータ（.gitignore 対象）
    ├── notebooks/              # EDA・実験用 Jupyter Notebooks
    └── src/                    # 再現スクリプト・モデリングコード
        ├── baseline.py
        ├── feature_engineering.py
        └── exp0XX_colab_l4.py  # Colab L4 GPU 実行用スクリプト
```

### ② 実験ログ (`learning-log.md`) の必須フォーマット
すべての実験は `learning-log.md` のサマリー表および詳細セクションに記録します。
- **実験番号 (`Exp #0XX`):** 一貫した連番管理
- **手法・仮説:** 追加した特徴量、モデル構成、ハイパーパラメータ
- **手元検証スコア (Val Score):** CV / Time-based Split でのスコア
- **本番スコア (Public LB):** Kaggle リーダーボードでのスコア
- **相関分析 (Correlation):** 既存ベストモデルとの予測値相関（アンサンブル用ダイバーシティ確認）
- **考察・次のアクション:** 成功要因・失敗要因の言語化と次の一手

### ③ モデリング・アンサンブルの定石
1. **対数空間処理 (Log-space):**
   - 評価指標が RMSLE 等の対数系指標の場合、学習・ブレンド・平均化はすべて対数変換（$\log(1+y)$）空間で行い、最終出力時に指数逆変換（$\exp(z)-1$）する。
2. **GBDT 多様性アンサンブル:**
   - アルゴリズム特性の異なる LightGBM（葉単位分割）、CatBoost（対称木・カテゴリ最適化）、XGBoost（深さ単位分割）を組み合わせる。
3. **後処理フィルター (Post-processing):**
   - ドメイン固有のルール（長期売上ゼロ系列のクリップ、祝日・休業日補正など）を手元検証スコアを損なわない範囲で精密に適用する。
