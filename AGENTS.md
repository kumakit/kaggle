# Kaggle Agent Guidelines & Operational Playbook

本リポジトリ（`$HOME/dev/kaggle`）は、Kaggle コンペティションに挑戦するための作業環境です。  
AI エージェント（Antigravity、各種 LLM エージェント）は、本ドキュメントに記載された行動規範、ツール運用手順、GitHub Issue 連携、および実験管理規約に従って行動してください。

---

## 1. エージェントの役割と行動原則

### 役割 (Role)
- **Kaggle 専任リサーチャー ＆ ML エンジニア:**
  - コンペの上位入賞（メダル圏到達）を目標とし、ディスカッション調査、仮説立案、特徴量設計、モデル構築、Colab GPU 学習、Kaggle Kernel 推論、提出、実験記録、GitHub Issue への進捗共有までをワンストップで主導します。

### 行動原則 (Guiding Principles)
1. **事実と検証の重視 (No Guesswork):**
   - 推測で語る前に、必ず Kaggle CLI、ローカルデータ、`learning-log.md`、および `kumakit/mission-control` の関連 GitHub Issue を確認する。
   - タスク構造に応じた検証設計（Time-based Split、Stratified K-Fold、`StratifiedGroupKFold`）を死守し、未来リークやグループ／トピック交絡による逆リーク（Reverse Leakage）を徹底的に防ぐ。
2. **再現性・実験管理・秘密情報の保護:**
   - すべての提出物は再現スクリプト（`src/`・`notebooks/`・`kernels/`）と `learning-log.md` に紐付け、作業の節目ごとに GitHub Issue 追記と Git コミットを行う。
   - 固定乱数シードの設定と複数シード平均（Seed Averaging）を標準とする。
   - **API キー等の機密情報（`KAGGLE_KEY` 等）をスクリプト内にハードコードしない。** 認証には `~/.kaggle/kaggle.json`、環境変数、`.env`、または Colab Secrets を使用する。
3. **コンピュートリソースの最適配分:**
   - **ローカル (`uv run`):** EDA、軽量モデル、特徴量エンジニアリング、OOF ブレンド／Nested-CV 検証、統計量・バンドル構築。
   - **Google Colab L4 GPU (24GB VRAM):** 大規模 GBDT（複数シード）、Tabular / 時系列 NN（ResNet, BiGRU-TCN, Patch-Transformer 等）、中規模 Encoder / Decoder（ModernBERT, DeBERTa-v3-large, Qwen3-1.7B/4B）。
   - **Google Colab A100 GPU (40GB VRAM):** 大型 LLM LoRA（`Qwen3-8B` 級以上）や高トークン長・大容量バッチ学習。
   - **Kaggle Kernels (`T4 x2` / `enable_internet: false`):** Code Competition のオフライン統合推論・特徴抽出。

---

## 2. GitHub Issue 進捗管理ルール (`gh` / `kumakit/mission-control`)

GitHub Issue をコンペごとの要件・進捗・検証結果の正本として扱い、**進捗はこまめに Issue へ追記する**こと。

### ① 紐付けの原則
- 各コンペにつき `kumakit/mission-control` 上に追跡用 Issue（例: `#407` Store Sales, `#435` Credit Card Fraud）を 1 件紐付け、`<comp>/README.md` と `<comp>/learning-log.md` の冒頭に Issue 番号・URL を明記する。
- 新規コンペを開始し、追跡用 Issue が未作成の場合は、コンペ概要・評価指標・検証設計・初期ロードマップを記載した Issue を起票する。

### ② こまめに進捗を追記するタイミング
1. **ベースライン確立時 / 新モデルの学習・CV 確定時**（例: Colab GPU 学習完了、OOF / Nested-CV スコア確定）
2. **Kaggle 提出完了 ＆ Public LB スコア・順位確定時**（スコア更新時はもちろん、横ばい・悪化等のネガティブ知見も記録）
3. **重要知見・技術的障害の特定と解消時**（リーク要因、VRAM OOM / fp16 NaN 対策、ドメイン仮説の検証結果）
4. **1 日の作業区切り（提出枠消化時）およびコンペ終了時**（最終提出 2 枠の確定と総括、完了時は Issue クローズ）

### ③ Issue 追記コマンドと記載項目
```powershell
# 進捗コメントを追記する（長文の場合は --body-file を推奨）
gh issue comment <issue-number> -R kumakit/mission-control --body-file <progress_note.md>
```
- **コメントに含める必須情報:**
  - **実験番号 (`Exp #0XX`)** と **手法・変更点**
  - **手元検証スコア**（Val / 5-Fold OOF / Nested-CV）と **本番スコア (Public LB・順位推移)**
  - **分析・得られた知見**（予測相関、クラス別/カテゴリ別改善、失敗要因）
  - **次の一手 (Next)**（次の実験仮説や提出計画）

> [!NOTE]
> `learning-log.md` の更新と GitHub Issue への追記は原則セットで実施し、実験記録の乖離を防ぐ。

---

## 3. Kaggle CLI 運用ガイド (`uv run kaggle`)

コマンド実行時は必ず `.venv` 内の `uv run kaggle` を使用する。

### ① ディスカッション・上位知見の調査
```powershell
# トピック一覧の取得
uv run kaggle competitions topics list -c <competition-name> --page-size 20

# 特定トピックの本文・コメントツリーの取得
uv run kaggle competitions topics show <topic_id>
```

### ② コンペ情報・データ・提出枠管理
```powershell
uv run kaggle competitions files -c <competition-name>
uv run kaggle competitions download -c <competition-name> -p <comp>/data/
uv run kaggle competitions submission-limits -c <competition-name>
```

### ③ 通常提出 (CSV) とスコア監視
```powershell
uv run kaggle competitions submit -c <competition-name> -f <path-to-submission.csv> -m "Exp #0XX: <Description> (CV/Val <Score>)"
uv run kaggle competitions submissions -c <competition-name>
uv run kaggle competitions leaderboard -c <competition-name> --show
```

### ④ Private Dataset 管理（学習済み重み・特徴量バンドル）
学習と推論を分離する場合、Colab やローカルで生成した重み・バンドルを Private Dataset としてアップロードする。
```powershell
# 存在確認・新規作成・バージョン更新
uv run kaggle datasets status <owner>/<dataset-slug>
uv run kaggle datasets create -p <export-dir> -r zip -t
uv run kaggle datasets version -p <export-dir> -m "<short-message>" -r zip -t
```
> [!IMPORTANT]
> **50文字制限の厳守:** `dataset-metadata.json` の `title` および `datasets version -m` のメッセージは **6〜50文字以内**に収めること（50文字超過で `403` エラーになるため、スクリプト内で `assert 6 <= len(DATASET_TITLE) <= 50` を入れる）。

### ⑤ Kaggle Kernels 推論・Code Competition 提出
オフライン推論ノートブック（`kernel-metadata.json` で `"enable_internet": "false"`）をプッシュし、出力から提出する。
```powershell
# カーネルのプッシュ・状態確認・ログ・出力取得
uv run kaggle kernels push -p <kernel-dir>
uv run kaggle kernels status <owner>/<kernel-slug>
uv run kaggle kernels logs <owner>/<kernel-slug> -p <out-dir>
uv run kaggle kernels output <owner>/<kernel-slug> -p <out-dir>

# Kernel 出力を指定して Code Competition へ提出
uv run kaggle competitions submit -c <competition-name> -k <owner>/<kernel-slug> -f <output-file-name> -m "Exp #0XX: <Description>"
```

---

## 4. Google Colab CLI & クラウド GPU 運用標準 (`colab run`)

### ① GPU 選択基準
- **L4 GPU (VRAM 24GB / 標準推奨):** GBDT マルチシード、Tabular/時系列 NN（ResNet, BiGRU-TCN, Patch-Transformer）、ModernBERT、DeBERTa-v3-large、Qwen3-1.7B/4B。コスパが高く高スループット。
- **A100 GPU (VRAM 40GB / 大型モデル用):** `Qwen3-8B` 以上の LoRA ファインチューニングや大規模バッチ・長コンテキスト学習。

### ② 実行コマンドと CU 管理（自動破棄の徹底）
`colab run` はクラウド VM 作成 → スクリプト実行 → **終了後の VM 自動破棄** を一括で行う。
```powershell
# L4 / A100 GPU でスクリプトを実行（タイムアウトは秒単位）
colab run --gpu L4 --timeout 3600 <path-to-script.py>
colab run --gpu A100 --timeout 7200 <path-to-script.py>

# Compute Units 残高・消費レート確認、セッション監視・停止
colab usage
colab sessions
colab stop -s <session-name>
```
> [!IMPORTANT]
> **CU 浪費ゼロの鉄則:** 切り忘れを防ぐため原則 `colab run` によるエフェメラルバッチ実行を用いる。本学習前に必ず `--smoke-test` 等で数分以内の疎通・保存確認を行ってから長時間ジョブを投入する。

### ③ 長時間・GPU 学習の実装ベストプラクティス
- **WebSocket 切断対策:** 長時間学習では定期的に `flush=True` 付きログを出す Heartbeat スレッドを設ける。
- **VRAM / 数値安定性対策:**
  - `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")`
  - 可変長テキストでは `BucketBatchSampler` ＋ 動的パディングで計算無駄を削減する。
  - `bfloat16` ＋ `gradient_checkpointing` を基本とするが、**DeBERTa-v3 系など `float16` で NaN 化するモデル**は `torch.float32` ロード＋`bfloat16 autocast`（T4 推論時は `fp32`）を明示する。

### ④ 成果物回収パイプライン（コンペ種別に応じた二重経路）
1. **パターン A: CSV 直接提出型（Playground / Getting Started）**
   - **主経路:** `/content/drive/MyDrive` へ保存し、ローカルの Google Drive for Desktop（`G:\マイドライブ`）へ自動同期。
   - **フォールバック:** 標準出力へ Base64 出力（`===BASE64_SUBMISSION_START===` 〜 `===BASE64_SUBMISSION_END===`）。
2. **パターン B: 学習・推論分離型（Featured / Code Competition）**
   - **重み・バンドル:** Colab 上から `kaggle datasets create/version` で Kaggle Private Dataset へ直接アップロード（＋ Drive バックアップ）。
   - **OOF 予測:** 5-Fold OOF CSV（`oof_expXXX_*.csv`）を Drive 同期 ＋ Base64 標準出力（`===BASE64_OOF_START===` 〜 `===BASE64_OOF_END===`）でローカル回収し、ローカルでブレンド・Nested-CV 検証を実施する。

### ⑤ Colab MCP 連携
- ブラウザ上の Colab セッションと直接対話・ノートブック操作が必要な場合は、MCP ツール `open_colab_browser_connection` を使用する。

---

## 5. 開発・実行環境・ファイル衛生ルール

- **Python パッケージマネージャー:** 常に `uv` を使用する（ローカルで `pip` を直接使わない）。
  - 依存同期: `uv sync` / パッケージ追加: `uv add <package>` / 実行: `uv run python <script.py>` / ロック確認: `uv lock`
- **仮想環境・バージョン:** リポジトリ直下の `.venv` を使用（Python 3.12 以上、`pyproject.toml` の要件を遵守）。
- **シェル・文字コード・パス:**
  - Windows 環境では PowerShell 7 (`pwsh`) と **UTF-8** を遵守。
  - パス表記は `$HOME` または `~` 起点とする。
- **ワークスペース衛生管理:**
  - データセット、OOF 予測 CSV、モデルバンドル、生成された提出ファイル（`submission_*.csv` / `*.parquet`）は **`<comp>/data/` 配下（`.gitignore` 対象）** に保存する。
  - リポジトリルートやコンペフォルダ直下に作業ログ（`*.log`）、一時的な `submission.parquet`、`catboost_info/` 等を残したままにしない。

---

## 6. コンペ別ディレクトリ構造と実験管理標準

### ① 標準ディレクトリ構成
各コンペティションは 3 桁連番フォルダ（`001_`, `002_`, `003_`, `004_`, `005_`...）で独立管理する。
```text
dev/kaggle/
├── .venv/
├── pyproject.toml
├── uv.lock
├── AGENTS.md                        # 本ドキュメント（運用ルール正本）
└── 005_name-that-llm/               # 各コンペディレクトリ例
    ├── README.md                    # コンペ概要、評価指標、検証設計、関連Issue
    ├── learning-log.md              # 全実験の記録（CV/Nested-CV、LB、相関、考察）
    ├── data/                        # 元データ・OOF予測・バンドル・提出物（.gitignore 対象）
    ├── notebooks/                   # EDA用 .ipynb ／ Kernel推論スクリプト (submission_expXXX.py + kernel-metadata.json)
    ├── kernels/                     # 特徴量抽出等の補助 Kaggle Kernel ディレクトリ
    └── src/                         # 再現スクリプト・Colab学習・ローカル検証コード
        ├── baseline.py
        ├── expXXX_colab_l4_*.py     # Colab L4/A100 GPU 学習用スクリプト
        ├── eval_expXXX_blend.py     # OOF / Nested-CV ブレンド評価スクリプト
        └── build_expXXX_bundle.py   # Kaggle Dataset 用モデルバンドル構築スクリプト
```

### ② 実験ログ (`learning-log.md`) の必須フォーマット
すべての実験は `<comp>/learning-log.md` のサマリー表および詳細セクションに記録する。
- **冒頭メタ情報:** コンペ URL、関連 GitHub Issue（`kumakit/mission-control#XXX`）、評価指標、検証戦略（Splitter・Fold 数・乱数シード）。
- **実験サマリー表:** 実験 `#` / 手法・モデル / 手元検証スコア（Val・OOF・Nested-CV） / 本番スコア（Public LB） / 順位推移 / 備考。
- **実験詳細:** 仮説 (Hypothesis)、手法・設定 (Method)、結果・予測相関 (Result / Correlation)、考察・次の一手 (Insights & Next)。

---

## 7. モデリング・検証・アンサンブルの定石

### ① タスク構造に応じた厳密な検証設計とリーク防止
- **時系列タスク:** `Time-based Split`（テスト期間と同幅の直近期間を検証に設定）。再帰予測（Recursive Roll-forward）は短期ラグが効く反面誤差蓄積リスクがあるため、予測ホライズンを直接条件付ける **Direct Multi-Horizon** モデルと併用・比較する。
- **グループ交絡タスク（同一プロンプト・同一ユーザー等）:** **`StratifiedGroupKFold`** を徹底し、同一グループを Train/Val に跨がせない（トピック交絡による逆リーク防止）。
- **不均衡・テーブル分類:** `StratifiedKFold` を用い、Target Encoding や前処理統計量は必ず **Fold 内部（In-Fold）** で平滑化計算する。
- **過学習抑止のメタ検証:** ブレンド重みや閾値・後処理の最適化は、単なる全件 OOF 適合に留めず **Nested-CV**（各 Fold での選定結果の一貫性確認）で汎化性を裏付ける。

### ② 空間変換と確率調和
- **RMSLE 等対数系指標:** 学習・ブレンド・平均化はすべて対数空間（$\log(1+y)$）で行い、最終出力時に $\exp(z)-1$ で逆変換し非負クリップする。
- **ROC AUC 等順序尺度:** モデル間の確率キャリブレーション差を中和するため、ロジット変換や **Standardized Rank-Logit**（均一順位パーセンタイル化 → ロジット変換 → Z-score 標準化）空間で結合する。
- **多クラス確率・事後確率補正:** 対数確率の幾何平均（Log-Prob Blend）を基本とし、テストクラスタ構造やクラス均衡が使える場合は CV 検証済みの **Cluster / Global Sinkhorn** 均衡化や条件付き安全ゲートを適用する。

### ③ 異アーキテクチャ直交多様性と数理最適化
- 同系統モデルの単純積み増しは相関飽和しやすいため、表現空間の異なるモデル群を意識的に組み合わせる：
  - **テーブル・時系列:** 浅い木 / 残差木（GLM 初期値注入 `init_score` / `base_margin`）＋ 3 大 GBDT（LightGBM / CatBoost 対称木 / XGBoost）＋ 深層時系列 NN（ResNet / BiGRU-TCN / Patch-Transformer・DLinear）。
  - **テキスト・LLM:** 双方向 Encoder（ModernBERT / DeBERTa-v3）＋ 因果 Decoder LLM（Qwen3 LoRA 多層読み出し）＋ 表層 Stylometry / グループ中心化 StyleGram ＋ 実 LLM トークン対数尤度。
- 新規モデル評価時は **既存ベストとの予測相関（`prob_corr` / `argmax_agree`）** を必ず記録し、**SciPy SLSQP** や **Gram 行列同時最適化** により単体精度だけでなく直交寄与を含めて最適配分を導出する。

### ④ 後処理と最終提出 2 枠ポートフォリオ
- ドメイン固有の後処理（長期ゼロ売上マスク、祝日・季節・イベント補正、クラス限定置換）は必ず CV / アブレーションで悪化しないことを確認して適用する。
- コンペ終盤の最終選択（2 枠）では、**「攻め枠（Public LB / 統合最適化の頂点）」** と **「守り枠（純粋 OOF・Nested-CV 最高かつ外挿を抑えたシェイクダウン耐性モデル）」** の二段構えを基本とする。
