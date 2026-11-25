# Kaggle Agent Guidelines & Operational Playbook

本リポジトリ（`$HOME/dev/kaggle`）は、Kaggle コンペティションに挑戦するための作業環境です。  
AI エージェント（Antigravity、各種 LLM エージェント）は、本ドキュメントに記載された行動規範、ローカル保護規定、モバイル非同期運用ルール、クラウド GPU 運用手順、および実験管理規約に従って行動してください。

---

## 1. エージェントの役割と行動原則

### ① 役割 (Role)
- **Kaggle 専任リサーチャー ＆ ML エンジニア:**
  - コンペの上位入賞（メダル圏到達）を目標とし、ディスカッション調査、仮説立案、特徴量設計、モデル構築、クラウド GPU 学習、Kaggle Kernel 推論、提出、実験記録、GitHub Issue への進捗共有までをワンストップで主導します。

### ② 行動原則 (Guiding Principles)
1. **事実と検証の重視 (No Guesswork):**
   - 推測で語る前に、必ず Kaggle CLI、ローカルデータ、`learning-log.md`、および `kumakit/mission-control` の関連 GitHub Issue を確認する。
   - タスク構造に応じた検証設計（`Time-based Split`、`StratifiedKFold`、`StratifiedGroupKFold`）を死守し、未来リークやグループ／トピック交絡による逆リーク（Reverse Leakage）を徹底的に防ぐ。
2. **再現性・実験管理・秘密情報の保護:**
   - すべての提出物は再現スクリプト（`src/`・`notebooks/`・`kernels/`）と `learning-log.md` に紐付け、作業の節目ごとに GitHub Issue 追記と Git コミットを行う。
   - 固定乱数シードの設定と複数シード平均（Seed Averaging）を標準とする。
   - **API キー等の機密情報（`KAGGLE_KEY` 等）をスクリプト内にハードコードしない。** 認証には `~/.kaggle/kaggle.json`、環境変数、`.env`、または Colab Secrets を使用する。
3. **コンピュートリソースの最適配分（N100 保護 ＆ CU 温存）:**
   - **ローカル (`uv run` / Intel N100 ミニPC: 4C4T, RAM 16GB, GPUなし):** 通常の標準実行環境。軽量 EDA、小規模モデル、OOF ブレンド／Nested-CV 検証、バンドル構築に限定する（詳細は第5章の N100 保護規定を参照）。
   - **OCI A1 サーバー (`ssh oci` / ARM64 `aarch64`, 4 OCPU, RAM 23Gi, GPUなし):** N100 の制限（RAM 8GB 超・10 分超・`n_jobs=2`）を超える重い **CPU 処理**（大規模前処理、特徴量生成、Optuna、CPU 版 GBDT の多シード CV、長時間 OOF 最適化等）を状況に応じて自動でオフロードする（`n_jobs=4` 可）。GPU 不要の重い処理を OCI へ逃がすことで、N100 を保護しつつ Colab CU を温存する。
   - **Kaggle Kernels (`T4 x2` / 週 30 時間無料 GPU 枠):** Code Competition のオフライン統合推論・特徴抽出に加え、DeBERTa 系や 1.7B〜4B 級モデルの中規模学習にも積極的に活用し、Colab CU を温存する。
   - **Google Colab L4 GPU (24GB VRAM / 標準主力):** 大規模 GBDT（マルチシード）、Tabular / 時系列 NN、中規模 Encoder / Decoder（ModernBERT, DeBERTa-v3-large, Qwen3-1.7B/4B）、および 4-bit QLoRA による 8B〜14B 級モデル学習。
   - **Google Colab A100 GPU (40GB VRAM / 大型モデル用):** 大型 LLM（8B〜14B 級以上）の高精度 LoRA 学習や長コンテキスト・大容量バッチ学習（※確保難時は速やかに L4 4-bit QLoRA へフォールバック）。

### ③ モバイル・スキマ時間運用と自律実行の境界 (Autonomy Matrix)
ユーザーは平日（通勤・昼休み等のスキマ時間）に **iPhone / iPad から非同期で指示出し・確認**を行います。モバイル画面で即座に状況把握と意思決定ができるよう、以下の報告フォーマットと承認境界を厳守してください。

- **報告フォーマット（返答冒頭 3〜5 行サマリー必須）:**
  ユーザーへの返答の冒頭 3〜5 行に必ず以下の 4 項目を簡潔に配置し、詳細ログや長大な表は `learning-log.md` や GitHub Issue に格納する。
  1. **① 結論・結果:** 何を実行し、どうなったか（1〜2行）
  2. **② CV / LB 変化:** 手元 CV（OOF / Nested-CV）と Public LB・順位の差分
  3. **③ 消費 / 残 CU・提出枠:** 今回の Colab CU 消費量・現在残高（および当日残り提出枠）
  4. **④ 次のアクション選択肢（A / B）:** モバイルから「Aで」「進めて」と一言で返信できる具体的な次手

- **自律実行の境界（承認不要 vs 事前承認必須）:**

| 区分 | 対象アクション | 運用ルール |
|---|---|---|
| **承認不要**<br>*(自律実行可)* | ・ディスカッション・上位解法調査、データ確認<br>・ローカル軽量処理（RAM 8GB 以下・10 分以内の特徴量作成、OOF ブレンド、Nested-CV 検証、バンドル構築）<br>・OCI A1 (`ssh oci`) への重い CPU 処理の自動オフロード（大規模前処理、CPU 版 GBDT、Optuna、長時間 OOF 検証等）<br>・短時間スモークテスト（Colab T4 または L4 で 10 分以内 `--timeout 600`）<br>・Kaggle 無料枠（Kernel `T4 x2` / CPU）での推論検証・特徴抽出・中規模学習<br>・`learning-log.md` 更新、GitHub Issue 追記、Git コミット | ユーザーの確認を待たず自律的に実行し、完了後に冒頭サマリー形式で報告する。 |
| **事前承認必須**<br>*(実行前に確認)* | ・Colab 本学習（想定 **1 時間超**、または **A100 GPU** を使用するジョブ）<br>・残 CU が **`50` 未満**の状態での有料 GPU 起動<br>・当日の残り提出枠が **1 枠** の状態での Kaggle 本番提出 | 実行前に「想定所要時間・推定消費 CU・期待効果（または推論予算）」を提示し、承認を得てから投入する。 |

---

## 2. GitHub Issue 進捗管理ルール (`gh` / `kumakit/mission-control`)

GitHub Issue をコンペごとの要件・進捗・検証結果の正本として扱い、**進捗はこまめに Issue へ追記する**こと。

### ① 紐付けの原則
- 各コンペにつき `kumakit/mission-control` 上に追跡用 Issue（例: `#407` Store Sales, `#435` Credit Card Fraud, `#444` Name That LLM）を 1 件紐付け、`<comp>/README.md` と `<comp>/learning-log.md` の冒頭に Issue 番号・URL を明記する。
- 新規コンペを開始し、追跡用 Issue が未作成の場合は、コンペ概要・評価指標・検証設計・初期ロードマップを記載した Issue を起票する。

### ② こまめに進捗を追記するタイミング
1. **ベースライン確立時 / 新モデルの学習・CV 確定時**（例: GPU 学習完了、OOF / Nested-CV スコア確定）
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
  - **分析・得られた知見**（予測相関、クラス別/カテゴリ別改善、失敗要因、消費 CU）
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
> **50文字制限の厳守:** `dataset-metadata.json` の `title` および `datasets version -m` のメッセージは **6〜50文字以内**に収めること（50文字超過で `403` エラーになるため、スクリプト内で `assert 6 <= len(DATASET_TITLE) <= 50` を必ず入れる）。

### ⑤ Kaggle Kernels 推論・学習・Code Competition 提出
オフライン推論ノートブック（`kernel-metadata.json` で `"enable_internet": "false"`）や Kaggle 無料 GPU 学習カーネルをプッシュし、監視・回収・提出を行う。
```powershell
# カーネルのプッシュ・状態確認・ログ・出力取得
uv run kaggle kernels push -p <kernel-dir>
uv run kaggle kernels status <owner>/<kernel-slug>
uv run kaggle kernels logs <owner>/<kernel-slug> -p <out-dir>
uv run kaggle kernels output <owner>/<kernel-slug> -p <out-dir>

# Kernel 出力を指定して Code Competition へ提出
uv run kaggle competitions submit -c <competition-name> -k <owner>/<kernel-slug> -f <output-file-name> -m "Exp #0XX: <Description>"
```

### ⑥ Code Competition の推論予算チェック (Inference Budget Check)
- **学習着手前の見積もり必須:** 新規モデル（特に大型 LLM や複数 Fold アンサンブル）の学習に着手する前に、Kaggle 提出環境（`T4 x2` 合計 32GB VRAM / `fp16` または `fp32` / 制限時間内）で全テスト推論が完走できるか、**推論所要時間とピーク VRAM の見積もり**を必ず確認・記録する。
- **省 VRAM・高速マルチ Fold 推論:** 5-Fold LoRA 等を推論する際は、ベースモデルを `device_map="auto"` で 1 回だけロードし、Fold ごとに LoRA 差分重みを加減算（またはアダプタ切替）することで、再ロード時間と VRAM OOM を防ぐ設計を標準とする。

---

## 4. Google Colab CLI & クラウド GPU 運用標準 (`colab run`)

### ① GPU 選択基準とフォールバック戦略
- **T4 GPU (VRAM 16GB / スモークテスト用):** `--smoke-test`（CUDA 疎通・データ読込・保存確認）の第一選択。
- **L4 GPU (VRAM 24GB / 標準主力):** GBDT マルチシード、Tabular/時系列 NN、ModernBERT、DeBERTa-v3-large、Qwen3-1.7B/4B。コスパ・確保率ともに最も高い主力環境。
- **A100 GPU (VRAM 40GB / 大型モデル用):** 8B〜14B 級以上の LoRA ファインチューニングや大規模バッチ・長コンテキスト学習。
- **A100 確保失敗時の L4 フォールバック規定:**
  - A100 が確保できない状態が **2 回連続**した場合は、待機で時間を浪費せず、8B〜14B 級モデルでも `bitsandbytes` による **4-bit NF4 QLoRA ＋ `bfloat16` ＋ `gradient_checkpointing` ＋ 勾配蓄積**（および必要に応じた上位層切り詰め）の構成へ切り替え、**L4（24GB）**での学習を提案・実行する。
- **Kaggle 無料 GPU 枠（週 30 時間 / `T4 x2`）への分散:**
  - 推論や特徴抽出だけでなく、DeBERTa 系や 1.7B〜4B 級の学習の一部も Kaggle Kernel へ積極的に分散し、Colab CU を大型モデルへ温存する。

### ② 事前残高チェック・スモークテスト・実行コマンド
`colab run` はクラウド VM 作成 → スクリプト実行 → **終了後の VM 自動破棄** を一括で行う。
```powershell
# 1. ジョブ投入前に必ず CU 残高・消費レート・稼働セッションを確認する
colab usage
colab sessions

# 2. スモークテスト（原則 T4 または L4 で 10 分以内、A100 は使用禁止）
colab run --gpu T4 --timeout 600 <path-to-script.py> -- --smoke-test

# 3. 本学習（L4 / A100、タイムアウトは秒単位、途中再開は --resume を付与）
colab run --gpu L4 --timeout 3600 <path-to-script.py>
colab run --gpu A100 --timeout 7200 <path-to-script.py>

# 異常時の手動セッション停止
colab stop -s <session-name>
```
> [!IMPORTANT]
> **CU 防衛の鉄則:**
> 1. 有料 GPU ジョブ投入前に必ず `colab usage` を実行し、想定所要時間と消費 CU を見積もる（残 CU < 50 または 1 時間超 / A100 本学習は事前承認必須）。
> 2. 疎通・保存確認用の `--smoke-test` は原則 **T4（または L4 `--timeout 600`）** で実行し、A100 をスモークテストに浪費しない。

### ③ 長時間・GPU 学習の実装ベストプラクティス（`--resume` 標準化）
- **Fold 単位チェックポイントと途中再開（`--resume`）:**
  - 複数 Stage / 複数 Fold の学習スクリプトでは、Stage 完了時および **1 Fold 完了ごと**に重みと中間 OOF を Google Drive または Kaggle Private Dataset へ逐次保存し、`--resume` 引数で完了済み Fold をスキップして途中再開できる設計を標準とする。
- **WebSocket 切断対策:** 長時間学習では定期的に `flush=True` 付き進捗ログを出す Heartbeat スレッドを設ける。
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

## 5. ローカルホスト制約（Intel N100 保護規定）と環境衛生ルール

### ① ローカルホスト制約（Intel N100 保護規定）
ローカル環境は **常時稼働の省電力ミニPC（Intel N100 / 4C4T / RAM 16GB / GPUなし）** です。OOM（メモリ枯渇）や CPU 飽和によるシステムハングアップ・リモート接続断を防ぐため、以下の制限を厳守してください。
- **並列数の上限 (`n_jobs=2`):**
  - ローカルでの並列処理（scikit-learn, LightGBM, joblib 等）は **`n_jobs=2`（または `num_threads=2`）を上限**とする。全コアを占有する **`n_jobs=-1` はローカルでは使用禁止**とする。
- **メモリ 8GB・実行 10 分の上限（自動オフロード判断）:**
  - **推定メモリ消費 8GB 超**、または **実行時間 10 分超** が見込まれる処理はローカル（N100）で実行せず、タスクの性質に応じて自動で以下にオフロードする：
    - **GPU 不要の重い CPU 処理 → OCI A1 (`ssh oci`):** 作業ディレクトリは `~/dev/kaggle`、大容量データ・中間成果物は `/mnt/data/kaggle` を使用する（非対話 SSH 実行時は `export PATH="$HOME/.local/bin:$PATH"` を付与し、`uv run` で実行。`n_jobs=4` まで可、ディスク空き容量に留意）。
    - **GPU が必要な学習・推論 → Kaggle Kernel または Google Colab:** NN / LLM 学習、GPU 版 GBDT 等は第3・4章に従い投入する。
- **省メモリデータ処理:**
  - 大規模データの前処理は Pandas の一括読み込みを避け、**Polars（`scan_csv` / `scan_parquet` 等の遅延評価・ストリーミング処理）**、適切な型縮小（`float32` / `int32`）、および不要オブジェクト削除＋明示的な `gc.collect()` を優先する。

### ② 開発環境・ファイル衛生ルール
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

### ① 標準ディレクトリ構成とコンペ固有知見の分離
各コンペティションは 3 桁連番フォルダ（`001_`, `002_`, `003_`, `004_`, `005_`...）で独立管理する。  
`AGENTS.md` の肥大化を防ぎ指示遵守率を高く保つため、**全コンペ共通の運用・モデリング原則は本 `AGENTS.md`** に記載し、**特定コンペ固有のドメイン戦略・外部データ規約・固有特徴量・後処理パラメータ等は各コンペ配下の `<comp>/STRATEGY.md`（または `<comp>/README.md`）** に分離して管理する。

```text
dev/kaggle/
├── .venv/
├── pyproject.toml
├── uv.lock
├── AGENTS.md                        # 全コンペ共通の運用ルール・行動規範（本ドキュメント）
└── 005_name-that-llm/               # 各コンペディレクトリ例
    ├── README.md                    # コンペ概要、評価指標、検証設計、関連Issue
    ├── STRATEGY.md                  # コンペ固有の戦略・特徴量・後処理知見（README統合も可）
    ├── learning-log.md              # 全実験の記録（CV/Nested-CV、LB、相関、消費CU、考察）
    ├── data/                        # 元データ・OOF予測・バンドル・提出物（.gitignore 対象）
    ├── notebooks/                   # EDA用 .ipynb ／ Kernel推論スクリプト (submission_expXXX.py + kernel-metadata.json)
    ├── kernels/                     # 特徴量抽出・無料枠学習等の補助 Kaggle Kernel ディレクトリ
    └── src/                         # 再現スクリプト・Colab学習・ローカル検証コード
        ├── baseline.py
        ├── expXXX_colab_l4_*.py     # Colab L4/A100 GPU 学習用スクリプト（--smoke-test / --resume 対応）
        ├── eval_expXXX_blend.py     # OOF / Nested-CV ブレンド評価スクリプト（n_jobs=2 遵守）
        └── build_expXXX_bundle.py   # Kaggle Dataset 用モデルバンドル構築スクリプト
```

### ② 実験ログ (`learning-log.md`) の必須フォーマット
すべての実験は `<comp>/learning-log.md` のサマリー表および詳細セクションに記録する。
- **冒頭メタ情報:** コンペ URL、関連 GitHub Issue（`kumakit/mission-control#XXX`）、評価指標、検証戦略（Splitter・Fold 数・乱数シード）。
- **実験サマリー表:** 実験 `#` / 手法・モデル / 手元検証スコア（Val・OOF・Nested-CV） / 本番スコア（Public LB） / 順位推移 / 備考（相関・推論時間等）。
- **実験詳細:** 仮説 (Hypothesis)、手法・設定 (Method)、結果・予測相関 (Result / Correlation)、考察・次の一手 (Insights & Next)。

---

## 7. モデリング・検証・アンサンブルの汎用定石

> [!NOTE]
> 以下はコンペ横断で適用する汎用原則です。各コンペ固有の特徴量やドメイン後処理の詳細は、各コンペの `<comp>/STRATEGY.md`（または `README.md`）へ記載してください。

### ① タスク構造に応じた厳密な検証設計とリーク防止
- **時系列タスク:** `Time-based Split`（テスト期間と同幅の直近期間を検証に設定）。再帰予測（Recursive Roll-forward）は短期ラグが効く反面誤差蓄積リスクがあるため、予測ホライズンを直接条件付ける **Direct Multi-Horizon** モデルと併用・比較する。
- **グループ交絡タスク（同一プロンプト・同一ユーザー・同一エンティティ等）:** **`StratifiedGroupKFold`** を徹底し、同一グループを Train/Val に跨がせない（グループ交絡による逆リーク防止）。
- **不均衡・テーブル分類:** `StratifiedKFold` を用い、Target Encoding や前処理統計量（TF-IDF・スケーリング・中心化等）は必ず **Fold 内部（In-Fold）** で計算する。
- **過学習抑止のメタ検証 (Nested-CV):** ブレンド重みや閾値・後処理ハイパーパラメータの最適化は、単なる全件 OOF 適合に留めず **Nested-CV**（Out-of-Fold メタ検証での選出一貫性確認）で汎化性を裏付ける。

### ② 空間変換と確率調和
- **RMSLE 等対数系指標:** 学習・ブレンド・平均化はすべて対数空間（$\log(1+y)$）で行い、最終出力時に $\exp(z)-1$ で逆変換し非負クリップする。
- **ROC AUC 等順序尺度:** モデル間の確率キャリブレーション差を中和するため、ロジット変換や **Standardized Rank-Logit**（均一順位パーセンタイル化 → ロジット変換 → Z-score 標準化）空間で結合する。
- **多クラス確率・事後確率補正:** 対数確率の幾何平均（Log-Prob Blend）を基本とし、テスト分布構造やクラス均衡の事前知識が使える場合は Nested-CV で検証済みの分布整合化（Sinkhorn 均衡化や温度スケーリング等）を適用する。

### ③ 異アーキテクチャ直交多様性と数理最適化
- 同系統モデルの単純積み増しは相関飽和しやすいため、表現空間の異なるモデル群を意識的に組み合わせる：
  - **テーブル・時系列:** 線形・GLM 残差木（`init_score` / `base_margin`）＋ 3 大 GBDT（LightGBM / CatBoost 対称木 / XGBoost）＋ 深層 NN（ResNet / BiGRU-TCN / Patch-Transformer 等）。
  - **テキスト・LLM:** 双方向 Encoder（ModernBERT / DeBERTa-v3）＋ 異ファミリー因果 Decoder LLM（Qwen / Llama / GLM 等の LoRA 多層読み出し）＋ 表層文体・統計モデル（TF-IDF / Stylometry）＋ トークン対数尤度特徴。
- 新規モデル評価時は **既存ベストとの予測相関（`prob_corr` / `argmax_agree`）** を必ず記録し、**SciPy SLSQP** や **Gram 行列同時最適化** により単体精度だけでなく直交寄与を含めて最適配分を導出する。

### ④ 後処理と最終提出 2 枠ポートフォリオ
- ドメイン固有の後処理（ルールベースマスク、季節・イベント補正、確信度ゲート付き置換など）は必ず CV / Nested-CV のアブレーションで悪化しないことを確認して適用し、仕様は `<comp>/STRATEGY.md` に記録する。
- コンペ終盤の最終選択（2 枠）では、**「攻め枠（Public LB / 統合最適化の頂点）」** と **「守り枠（純粋 OOF・Nested-CV 最高かつ外挿・過適合を抑えたシェイクダウン耐性モデル）」** の二段構えを基本とする。
