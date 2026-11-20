# 学習・実験ログ (Learning Log): Name That LLM

- **コンペ名:** [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
- **関連Issue:** [kumakit/mission-control#444](https://github.com/kumakit/mission-control/issues/444)
- **評価指標:** **Balanced Accuracy**（※各クラスの再現率の算術平均、満点 1.0、ランダム推測は約 0.0556）
- **検証戦略 (Validation):** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `prompt_id`

---

## 実験一覧サマリー

| 実験 # | 手法・モデル | 手元検証スコア (5-fold CV Balanced Acc) | Kaggle本番スコア (Public LB) | 順位推移 | 備考 |
|---|---|---|---|---|---|
| **#016** | **7バックボーン ＋ `Gauss Head` (Two-Way話題除去＋Ledoit-Wolf共分散: `Llama3_S1`+`Qwen3-8B_Base`, `d=256, T=3.0, w_g=1.8`) ＋ `Qwen3-Emb-8B`クラスタ (`thr=0.65`) ＋ `Global Sinkhorn(0.30)` (`c_sk=0.00`) ＋ `GLM-5.3/Flash`** | **`0.75130`** (Nested-CV **`0.75094`** / 単体Gauss: `Llama_S1` `0.62201`, `Q8_Base` `0.61014`, `Q8_OOF` `0.64757` / c_sk無しベース `0.69387` 比 **`+0.05743`**) | *(Commit 実行中)* | - | **`Cluster Sinkhorn(0.85)` 完全廃止**＋`Qwen3-Emb` (ARI `0.9852`)＋未学習/S1特徴による純粋確率周辺化・反発EM統合。50%分断クラスタ耐性 (`0.69389`) も確認済み |
| **#015** | **7バックボーン (`+10% Llama-3.1-8B 24L 8層読み出し`) ＋ `SGv1(0.35)`＋`SGv2(0.35)` ＋ `Family Gates(lam=0.15)` ＋ `Cluster Sinkhorn(0.85)` ＋ `GLM-5.3/Flash`** | **`0.73165`** (Val-Split **`0.73202`** / Neural `0.64350→0.64943` / `Llama3-8B`単体 `0.57164`) | **`0.660`** (`Ref 58606868`) | **`2,616 位`** / 6,544 チーム | **Exp #013/#014/#015 が 3 連続 `0.660`（プラトー確定）**。OOF `+0.00259` に対し `Cluster Sinkhorn(0.85)`・TF-IDF クラスタ中心化依存が未知テスト分布と乖離している可能性が高く、次手は **`gauss.py`（埋め込み Two-Way 共分散ヘッド）＆ `Qwen3-Embedding` クラスタリング** へ移行 |
| - | (参考) Exp #015 単体: Stage 1 In-Style優先 36k + 5-Fold Stage 2 `Meta-Llama-3.1-8B (24L)` QLoRA | `0.57164` (`[0.57222, 0.58519, 0.59259, 0.55096, 0.55741]`) | - | - | Stage 1 Zero-Shot `0.51087` (`Pool-Val 0.52654`)。`nobuyukikumagai/name-that-llm-exp015-llama3-8b-lora` 保存完了 |
| **#014** | **6バックボーン（Exp #013同一）＋ `SGv1(0.35)`＋`SGv2(0.35)`併用 ＋ `Family Gates(lam=0.15)` ＋ `Cluster Sinkhorn(alpha=0.85)` ＋ `GLM-5.3/Flash`** | **`0.72906`** (Nested-CV **`0.72794`**, 3/5 Fold一致選出) | **`0.660`** (`Ref 58526363`) | **`2,517 位`** / 6,422 チーム | **Exp #013 と同点 `0.660` プラトー確認**（Nested-CV は `+0.00111` 改善）。後処理・TF-IDF 配合の上限に達し、次手は大型・異ファミリーモデル＆`GAUSS_HEAD` へ移行 |
| **#013** | **6バックボーン (`25% Qwen3-8B_v2[6..36] + 22.5% 4B + 22.5% MB-L + 15% Deb-L + 11.25% 1.7B + 3.75% MB-B`) ＋ `StyleGram v2(0.65)` ＋ `Family Gates(lam=0.25)` ＋ 緩和 Sinkhorn (`0.65/0.35`) ＋ `Exp #010 GLM-5.3/Flash`** | **`0.72979`** (Nested-CV **`0.72683`**, Qwen3-8B単体 `0.59570` / SGv2単体 **`0.49377`**) | **`0.660`** (`Ref 58503553`) | **`2,437 位`** / 6,323 チーム | **+0.006 自己ベスト更新（Exp #013a 比 +0.008、`+53`順位上昇）！** Top 38.5%。`Qwen3-8B v2` を `T4x2 device_map="auto"` で `198.8s` 推論、Kernel `42.6m` 完走 |
| **#013a** | (先行検証) 5バックボーン（Exp #012構成）＋ `StyleGram v2(0.50)` ＋ `Family Gates(lam=0.35)` ＋ 緩和 Sinkhorn (`0.65/0.35`) ＋ `Exp #010 GLM-5.3/Flash` | **`0.72350`** (`StyleGram v2` 単体 **`0.49377`**) | **`0.652`** (`Ref 58497893`) | - | `Exp #012(0.654)` 比 `-0.002`。`c_sk 0.85→0.65` 緩和（OOFでも `-0.00148`）や `Gates` 強度の影響を確認、v2本命は `lam=0.25` ＆ `Qwen3-8B` 統合へ |
| **#012** | **5バックボーン幾何平均 (`30% 4B + 30% MB-L + 20% Deb-L[6,24] + 15% 1.7B + 5% MB-B`) ＋ `Exp #011 StyleGram(0.35)` ＋ `capped_agglo` Cluster/Global Sinkhorn (`0.85/0.35`) ＋ `Exp #010 GLM-5.3/Flash` (`cov=0.30, w=1.15`)** | **`0.72128`** (Nested-CV **`0.72127`**, 4/5一致) | **`0.654`** (`Ref 58483891`) | **`2,490 位`** / 6,249 チーム | **+0.020 大幅更新（Exp #010 比 +314位上昇）！** Top 39.8%到達。`DeBERTa-v3-large [6,24]` `0.58236` + StyleGram + Sinkhorn 統合 |
| **#011** | **Multi-View StyleGram (`char/word/shape/function`＋`capped_agglo` 中心化) ＋ Cluster/Global Sinkhorn** | 単体 `0.40601` / +Exp008+GLM **`0.71979`** | - | - | 中心化で `0.35783→0.40675`。`capped_agglo(cap=18, thr=0.08)` で `ARI=0.9750` 復元・`prob_corr=0.5461` |
| **#010** | **Exp #008 4モデル幾何平均 ＋ 実 `GLM-5.3` & `GLM-5.3-Flash`（`L160`）対数尤度アンサンブル (`0.50*LR + 0.25*LGBM + 0.25*CB`, `cov=0.30, w=1.15`, GLM限定置換)** | **`0.68016`** (Nested-CV **`0.68016`**, 5/5一致) | **`0.634`** (`Ref 58446766`) | **`2,804 位`** / 6,175 チーム | **+0.021 大幅更新（Exp #008 比）！** `glm-5.3` Recall `0.287→0.667`、`glm-5.3-flash` `0.413→0.713`。T4x2 Commit `39.8m` で完全通過 |
| **#009** | (検証記録) `Qwen3-8B-Base` LoRA + Multi-Layer Readout (`[9, 18, 27, 36]`) | `0.60571` 単体 / Blend **`0.64869`** (Nested **`0.64313`**) | - | - | Colab A100 完走（OOF回収済み）も Dataset Title 53文字超過でアップロード失敗。案A（Exp #010）へ優先切替 |
| **#008** | **4モデル幾何平均融合: 25% Qwen3-4B + 40% ModernBERT-large + 30% Qwen3-1.7B + 5% ModernBERT-base** | `0.64312` (Nested-CV `0.63831`) | **`0.613`** (`Ref 58365321`) | **`2,854 位`** / 5,951 チーム | **+0.002 更新**（CV `0.643` 比で LB 伸び小: 4B比重低下 45→25%＆`train.csv`とのドメイン差影響）。単体 `ModernBERT-large` `0.58790` |
| - | (参考) Exp #008 単体: 忠実度フィルタ済み OpenPool Stage1 + 5-Fold Stage2 ModernBERT-large | `0.58790` | - | - | Stage 1 Zero-Shot `0.55826`。ModernBERT-base (`0.51675`) 比 **`+0.07115`**、Qwen3-4B と相関 `0.7713` |
| **#007** | **3モデル異アーキテクチャ融合: Qwen3-4B LoRA (45%) + Qwen3-1.7B (30%) + ModernBERT (25%)** | `0.63052` (Nested-CV `0.62201`) | **`0.611`** (`Ref 58331912`) | **`2,794 位`** / 5,843 チーム | **+0.045 更新（Top 50% `0.600` 突破＆+424人抜き）！** 忠実度 `<0.45` 除外 + 99,233件Stage1で単体 `0.61460` |
| - | (参考) Exp #007 単体: 忠実度フィルタ済み OpenPool Stage1 + 5-Fold Stage2 Qwen3-4B-Base LoRA | `0.61460` | - | - | Stage 1 Zero-Shot `0.59571`。相関: vs 1.7B `0.8080`, vs ModernBERT `0.7010` |
| **#006** | **異アーキテクチャ融合: OpenPool Qwen3-1.7B-Base LoRA (65%) + ModernBERT-base (35%)** | `0.59606` | **`0.566`** (`Ref 58322845`) | **`3,218 位`** / 5,812 チーム | **+0.052 大幅更新（Exp #005比 +373人抜き）！** デコーダー×エンコーダーの補完効果実証 |
| - | (参考) Exp #006 単体: Open Pool Stage 1 + 5-Fold Stage 2 Qwen3-1.7B-Base LoRA | `0.57606` | - | - | Stage 1 Zero-Shot `0.56385`。予測相関 `0.7307` とModernBERTと良好に補完 |
| **#005** | **Open Pool 2段階学習 ModernBERT-base (90%) + Exp002 LR (10%)（推論分離構成）** | `0.51787` | **`0.514`** (`Ref 58310104`) | **`3,591 位`** / 5,765 チーム | **+0.138 大躍進（Exp #003 比 +1,619人抜き）！** CV `0.5179` に対し LB `0.514` と完全連動！ |
| - | (参考) Exp #005 単体: Open Pool Stage 1 + 5-Fold Stage 2 ModernBERT-base | `0.51675` | - | - | Stage 1 Zero-Shotのみで `0.51048` 達成。全18クラスで+6〜32pt大幅改善 |
| **#004** | **3大系統完全融合アンサンブル (52% ModernBERT + 45% Exp002 LR + 3% LightGBM)** | `0.37820` | **`0.376`** (`Ref 58307541`) | (約 `4,431 位` 相当) | **+0.060 改善！** CV `0.3782` に対し LB `0.376` と高精度相関 |
| - | (参考) Exp #004 単体: ModernBERT-base (Colab L4 GPU, 5-Fold, bfloat16) | `0.32931` | - | - | 最新エンコーダーの文脈理解。単体で0.33目前。アンサンブルに絶大な多様性をもたらす |
| **#003** | **線形 ＆ 決定木 多様性アンサンブル (75% Exp002 LR + 25% LightGBM SVD)** | `0.34077` | **`0.316`** | **`5,210 位`** / 5,620 チーム | **初提出成功！公式LBランクイン🎉** 手元0.34に対し本番0.316とコミュニティ基準線と完全に整合 |
| - | (参考) Exp #003 単体: Stylometry + TruncatedSVD(128) + LightGBM | `0.27482` | - | - | SVD圧縮による単語直接性の希釈はあるが、決定木のルール抽出でアンサンブルパーツとして機能 |
| **#002** | **TF-IDF + 63次元タイポグラフィ・文体統計量 + ロジスティック回帰 (C=2.0)** | `0.33486` | - | - | **自己ベスト大幅更新 (+0.02778)！** 0.30708➔0.33486へ急伸。GLMの識別率が8%➔24%へ3倍化！ |
| **#001** | **TF-IDF (Word & Char n-gram) + Logistic Regression (StratifiedGroupKFold)** | `0.30708` | - | - | **初回ベースライン確立！** ランダム比5.5倍。gemini-3.1-proは再現率0.693と高精度判別 |

---

## 実験詳細

### 実験 #016: Two-Way ガウス共分散ヘッド (`gauss_head.py`) ＋ `Qwen3-Embedding-8B` クラスタリング ＋ `Cluster Sinkhorn` 廃止
- **実施日:** 2026-11-20〜2026-11-21
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab L4 特徴抽出 → ローカル 5-Fold In-Fold CV / Nested-CV → Kaggle `T4 x2` 推論カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #013〜#015 で CV `0.729→0.732` と改善しつつ LB `0.660` で横ばいだった主因は、`Cluster Sinkhorn(0.85)` が `train.csv` 特有の 18 択均等クラスタ構造を強制しすぎていた点にある。
- `Qwen/Qwen3-Embedding-8B` による高精度クラスタリング（`thr=0.65`）と、既存学習済み重み（未調整 `Qwen3-8B-Base` と Stage-1 未見 `Llama-3.1-8B 24L`）の隠れ表現に対する **Two-Way 話題除去＋Ledoit-Wolf 共分散 (`gauss_head.py`) ＋ 厳密周辺化 / 反発 EM** を統合し、`Cluster Sinkhorn` を廃止（`Global Sinkhorn 0.30` のみ）にすれば、リークなし・ドメイン堅牢性を保ったまま大幅な CV/LB 更新が見込める。

#### 2. 実験内容 (Method)
- **Colab L4 特徴抽出 (`exp016_colab_l4_extract_assets.py`):**
  - 再学習なしで `train.csv` 2,699 行の埋め込み `fp16` を抽出し、`nobuyukikumagai/name-that-llm-exp016-gauss-assets` へ登録（`max_vram=18.01GB`, ~14分完了）。
- **ローカル 5-Fold In-Fold & Nested-CV (`gauss_head.py`, `eval_exp016_gauss_cv.py`):**
  - `Qwen3-Embedding-8B` クラスタリング精度（vs 真 `prompt_id`）：`thr=0.60` で **ARI `0.9873`**、`thr=0.65` で **ARI `0.9852`**（`grp2..18=99.7%`、TF-IDF `0.9750` を上回る）。
  - Strict In-Fold `gauss_head.fit` (4 folds 学習 → held-out fold の教師なし `Qwen3-Emb` クラスタ評価, `d=256, T=3.0`) 単体 OOF: `Z_llama_s1` **`0.62201`**（Stage-2 線形ヘッド `0.57164` 比 `+0.05037`）、`Z_qwen8b_base` **`0.61014`**、`Z_qwen8b_oof` **`0.64757`**。
  - `Cluster Sinkhorn=0.00, Global Sinkhorn=0.30` 統合で、Exp #015 ベース (`0.69387`) → `0.5 * Llama_S1 + 0.5 * Qwen8B_Base` (`w_g=1.8`) 統合時に **Fold-OOF `0.75130` (`+0.05743`)**、Strict **5-Fold Nested-CV `0.75094`**（`[2.0, 1.4, 2.0, 2.0, 2.0]`）を達成（`train.csv` に一切適合していない純粋表現同士のみで構成）。
  - また 50% の行を強制シングルトン化する分断ストレス検証でも `0.69389`（Exp #015 ベース `0.69387` 以上）を維持し、クラスタ欠損耐性を確認。
- **統合推論 (`submission_exp016.py`):**
  - `Qwen3-Embedding-8B` でテスト文をクラスタリングし、`LLAMA3-8B` ロード時に Stage-1 特徴パス、`QWEN3-8B` ロード時に Base 特徴パスを追加（再ロードなし）。CPU で `gauss_fit`＋`gauss_scores` を実行し、`nobuyukikumagai/name-that-llm-exp016-full-ensemble`（v1）をプッシュ・実行中。

#### 3. 結果 (Result) & 次の一手 (Next)
- **手元検証:** `c_sk=0.00` 環境下で `0.69387 → 0.75130`（Nested-CV **`0.75094`**）、従来ベスト OOF (`0.73165`) 比でも **`+0.01965`** 更新。
- **次の一手:** `nobuyukikumagai/name-that-llm-exp016-full-ensemble` v1 の Commit 完了後、提出して **`0.660` プラトーの突破 (Public LB 検証)** を確認する。

---

### 実験 #015: 異ファミリー `Meta-Llama-3.1-8B (24L)` 8層読み出し LoRA 追加（7バックボーン統合）
- **実施日:** 2026-11-19〜2026-11-20
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab L4 GPU 4-bit NF4 QLoRA 学習 → ローカル OOF / Nested-CV 検証 → Kaggle `T4 x2` 推論カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #014（`0.660` LB）までのニューラルバックボーンは `Qwen3` 系 3 モデルと BERT/DeBERTa 系の 2 系統のみであり、Tiktoken 128k 語彙・Meta 事前学習コーパスを持つ完全異ファミリーの **`Meta-Llama-3.1-8B`**（先頭 24 層切り詰め＋`[2, 4, 6, 8, 12, 16, 20, 24]` 8層適応読み出し）を加えれば、低相関・直交特徴の寄与で `0.660` プラトーを突破できるはず。

#### 2. 実験内容 (Method)
- **Colab L4 GPU 学習 (`exp015_colab_l4_llama3_8b.py`):**
  - `NousResearch/Meta-Llama-3.1-8B`（32層→24層、`5.79B` パラメータ、`4-bit NF4 QLoRA`、`max_vram=10.64GB`）に対し、`tier0_instyle` 優先の 36,000 件（`tier0=32,455`、`cap/cls=2,000`）で Stage 1 (1 epoch) 学習後、`stage1_best/` チェックポイントを Kaggle Private Dataset へ保存。
  - `--resume-s1` ＋ Fold 毎逐次保存（`run-62369b`）で `StratifiedGroupKFold(5)` Stage 2（5 epochs）を全 Fold 完走し、`nobuyukikumagai/name-that-llm-exp015-llama3-8b-lora` へ 5 Fold 重み＋OOF を登録。
- **ブレンド最適化 & 相関確認 (`eval_exp015_blend.py`):**
  - `Llama3-8B` 単体 OOF は **`0.57164`**（予測相関: vs `Qwen3-4B` **`0.7870`**, vs `ModernBERT-L` **`0.7649`**, vs `DeBERTa-v3-L` **`0.7698`**, vs `Qwen3-8B` **`0.8539`**）。
  - ニューラル平均 (`Exp #014` ベース 90% + `Llama3-8B` 10%: `0.225 Q3-8B + 0.2025 Q3-4B + 0.2025 MB-L + 0.135 Deb-L + 0.10125 Q3-1.7B + 0.03375 MB-B + 0.10 Llama3-8B`) により、ニューラル単体平均 OOF が `0.64350 → 0.64943`（15% 時 `0.65128`）、フルパイプライン OOF が `0.72906 → 0.73165`（Val-Split OOF `0.73016 → 0.73202`）へ向上。Nested-CV でも 5/5 全 Fold が `w_ll > 0` を選出。
- **統合推論 (`submission_exp015.py`):**
  - Kaggle Models `alexchilton/llama-3-1-8b-base/Transformers/default/1` を読み込み時に `num_hidden_layers=24` へ切り詰め、`T4 x2` で `fp16` `device_map="auto"` ＋ Fold ごとの差分 LoRA 加減算で推論（`nobuyukikumagai/name-that-llm-exp015-full-ensemble` v1 実行中）。

#### 3. 結果 (Result) & 次の一手 (Next)
- **手元 OOF / クラス別再現率:** フルパイプライン OOF **`0.73165`**（`w_ll=0.10`、Val-Split **`0.73202`**、`Exp #014` `0.72906` 比 **`+0.00259`**）。難関クラス `claude-fable-5` (`0.450→0.470`) や `gpt-6-astra` (`0.907→0.933`) 等の再現率が向上。
- **Kaggle Commit & Public LB (`Ref 58606868`):** Commit `37.1m` 完走（`[LLAMA3-8B] 103.3s`）→ **Public LB `0.660`**（**`2,616 位 / 6,544 チーム`**, Top 40.0%。Top 33% `0.668`, Top 25% `0.684`, Bronze `0.702`, Silver `0.703`）。
- **重要知見 (`0.660` プラトー特定):** Exp #013 (`0.72979`), #014 (`0.72906`), #015 (`0.73165`) と CV は着実に伸ばしたものの LB は **3 連続 `0.660`** で完全に横ばい（CV–LB 乖離 `0.072`）。要因として ① TF-IDF ベースの `capped_agglo` + `Cluster Sinkhorn(0.85)` が `train.csv`（均等18択クラスタ）に特化しすぎている点、② テストでは `Qwen3-Embedding` クラスタ＋**Two-Way 中心化ガウス共分散ヘッド (`gauss.py`)**＋**`Qwen3-14B (27L)`**＋弱めの `Global Sinkhorn(0.30)` が上位解法（`0.696` 帯）の核心であることを確認。
- **次の一手:** ① **既存学習済み重み（`Qwen3-8B` / `Llama3-8B` / `MB-L` 等）＋ `gauss.py`（Two-Way Prompt 除去＋Ledoit-Wolf 共分散＋EM/Exact 周辺化）＋ `Qwen3-Embedding-8B` クラスタ** の OOF ＆ LB 検証（GPU 再学習不要で CV–LB ギャップを一気に詰める本命）、および ② A100 空き次第での `Qwen3-14B (27L)`（[exp015_colab_a100_qwen3_14b.py](file:///C:/Users/kumat/dev/kaggle/005_name-that-llm/src/exp015_colab_a100_qwen3_14b.py)）追加を実施する。

---

### 実験 #014: StyleGram v1+v2 併用 ＋ Family Gate 調整 (`lam=0.15`) ＋ Cluster Sinkhorn (`0.85`) 戻し
- **実施日:** 2026-11-18〜2026-11-19
- **担当・実施者:** 学習者本人 ＆ Antigravity（追加学習なし、Nested-CV 検証 → Kaggle `T4 x2` 推論カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #013a (`0.652`) vs Exp #012 (`0.654`) vs Exp #013 (`0.660`) のアブレーションから、`Qwen3-8B v2` 単体は LB `+0.008` と大きく効いた一方、`Cluster Sinkhorn alpha` を `0.85 → 0.65` に下げたこと・`train.csv` 特化の `SGv1` を完全に外して `SGv2` のみに切り替えたことがわずかに LB を抑えていた可能性がある。
- `SGv1(0.35)` ＋ `SGv2(0.35)` の併用・`Family Gate(lam=0.15)` 抑制・`Cluster Sinkhorn alpha=0.85` 戻しにより、ドメイン整合性と外部データ汎化を両立できるはず。

#### 2. 実験内容 & 結果 (`submission_exp014.py`)
- **Nested-CV:** **`0.72794`**（`Exp #013` `0.72683` 比 **`+0.00111`**、5 Fold 中 Fold 2/4/5 が一致して `(sg1=0.35, sg2=0.35, lam=0.15, c_sk=0.85)` を選出、同構成 OOF **`0.72906`**）。
- **Kaggle Commit (`nobuyukikumagai/name-that-llm-exp014-full-ensemble` v1):** **`32.8m`** 完走（`[QWEN3-8B] 134.1s`, `[StyleGram v2] 9.8s`, `[StyleGram v1] 10.4s`, `Part 1+2 367.0s`, `Flash 617s`, `GLM-5.3 979s`）、`submission.parquet (3, 2)` 検証通過。
- **Public LB 結果 (`Ref 58526363`):** **`0.660`**（**`2,517 位 / 6,422 チーム`**, 上位 `39.2%`, Exp #013 `0.660` と完全同点タイ。Top 33% `0.666`, Top 25% `0.682`, Bronze `0.702`）。

#### 3. 考察 & 次の一手 (Insights & Next)
- 後処理（`SGv1+SGv2` 併用・`c_sk=0.85` 戻し・`lam=0.15`）では Nested-CV が `0.72683→0.72794` へ伸びた一方、本番 LB は `0.660` の平坦プラトー（同一構成近傍の極大値）を確認。
- これ以上の大幅更新（`0.68+`〜`0.70+` Bronze 圏）には、現行 6 モデル（線形ヘッド）の枠を超える **大型・異ファミリーモデル (`Qwen3-14B` / `GLM-4-9B` / `Llama-3.1-8B`)** および **埋め込み空間 Two-Way 除去＋ガウス共分散ヘッド (`GAUSS_HEAD`)** の導入が必要。

---

### 実験 #013: Qwen3-8B-Base 6層 LoRA v2 ＋ StyleGram v2 (44.7k外部プール＋Family Gates) ＋ 緩和 Sinkhorn 統合
- **実施日:** 2026-11-18
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab A100 学習 + ローカル StyleGram v2 ビルド → Kaggle `T4 x2` 統合推論）

#### 1. 仮説 (Hypothesis)
- Exp #012 (`LB 0.654` / `OOF 0.72165`) の主要ボトルネックは `claude-fable-5 (0.443)`, `gemini-3.7-flash (0.500)`, `kimi-k3 (0.573)`, `gpt-5.6-sol (0.587)` であり、`StyleGram` が `train.csv` 2,699行のみで学習されていた点、および `Qwen3-8B` 系モデルが未組込みだった点が伸び代となっていた。
- Open Pool + `prvsiyan/ntl-public-in-style-pool-v1` + `ntl-extra-api-passages` + `ntl-extra-gpt-passages` を活用して **① `StyleGram v2` (plain/centred ハイブリッド + Family Gates)** と **② `Qwen3-8B-Base 6層 LoRA v2`** を構築し、さらにテストクラスタ不整合リスク軽減のため `Cluster Sinkhorn` を緩和 (`alpha=0.85→0.65`) すれば汎化性能が向上するはず。

#### 2. 実験内容 (Method)
- **StyleGram v2 + Family Gates (`exp013_stylegram_v2_cv.py` / `build_exp013_stylegram_v2_bundle.py`)**:
  - `flag_kb_proxy_after_ruling=False` かつ 1プロンプト ≥12ラベルのクリーン外部プール **44,672行**（3,132 prompts）を `EXTRA_SHARE=0.50` 重みで組み込み、`style_groups` クラスタ外行は `svc_plain`、サイズ 2〜18 クラスタ行は `svc_centred` を用いるハイブリッド推論を構築（`nobuyukikumagai/name-that-llm-exp013-stylegram-v2`）。
  - 加えて `claude/gpt/gemini/glm` 各ファミリー内の僅差行（margin < 0.45）を補正する条件付き `Family Gates` を統合。
- **Qwen3-8B-Base 6層 LoRA v2 (`exp013_colab_a100_qwen3_8b_v2.py`)**:
  - `tier0_instyle=38,435` を優先しつつクリーン Open Pool で補完した **74,919行**（`cap/cls=4,200`）で Stage 1 (2 epochs)、`train.csv` 5-Fold で Stage 2 (5 epochs) を Colab A100 (`run-893af0`) にて学習（`LAYER_INDICES=[6, 12, 18, 24, 30, 36]`）。`nobuyukikumagai/name-that-llm-exp013-qwen3-8b-lora` へ登録。
- **統合推論 (`submission_exp013.py`)**:
  - Kaggle `T4 x2` で `Qwen3-8B` を `device_map="auto"` + Fold ごとの差分 LoRA 加減算で OOM ゼロ推論。
  - 先行検証 `Exp #013a` (v1: 5バックボーン + SGv2 + Gates `0.35` + `c_sk=0.65`) と本命 `Exp #013` (v2: `25% Qwen3-8B_v2` 含む6バックボーン + SGv2 `0.65` + Gates `0.25` + `c_sk=0.65`) を実施。

#### 3. 結果 (Result)
- **StyleGram v2 単体 OOF:** **`0.49377`**（Exp #011 `0.40601` 比 **`+0.08776`**、`svc_plain=0.44267`）。
- **Qwen3-8B v2 単体 OOF:** Stage 1 Pool-Val `0.59055` / Zero-Shot `0.56870` / Stage 2 5-Fold OOF **`0.59570`**（`[0.57593, 0.59815, 0.62778, 0.59355, 0.58333]`、対 `Qwen3-4B` 相関 `0.8456`）。
- **先行版 `Exp #013a`:** OOF `0.72350` → Commit `36.1m` 通過 → **Public LB `0.652` (`Ref 58497893`)**（`Exp #012` `0.654` 比 `-0.002`。事後切り分けにて `c_sk=0.85→0.65` 緩和のみで OOF `-0.00148` 低下を確認）。
- **本命 `Exp #013 Full (v2)` (`eval_exp013_blend.py`):** Nested-CV **`0.72683`**、採用構成 OOF **`0.72979`**。Commit **`42.6m`** 完走（`[QWEN3-8B] 198.8s`, `[StyleGram v2] 10.4s`, `Part 1+2 424.6s`, `Flash 722s`, `GLM-5.3 1401s`）、`submission.parquet (3, 2)` 検証通過。
- **Public LB 結果 (`Ref 58503553`):** **`0.660`**（**`2,437 位 / 6,323 チーム`**, Exp #012 `0.654` 比 **`+0.006`** / Exp #013a `0.652` 比 **`+0.008`** 更新・Top 38.5%！ Top 33% `0.665`, Top 25% `0.681`, Bronze `0.701`）。

#### 4. 次の一手 (Next)
- `Exp #013a (0.652)` vs `Exp #013 (0.660)` から `Qwen3-8B v2` が `+0.008` 効いた一方、`c_sk=0.65` 緩和は控えていた可能性があるため、まずは追加学習なしの **`CLUSTER_SK_ALPHA=0.85` 戻し検証** を検討しつつ、上位帯（Bronze `0.701`）に向けた **大型・異ファミリーモデル (`Qwen3-14B` / `GLM-4-9B` / `Llama-3.1-8B`)** や **ガウス共分散ヘッド (`GAUSS_HEAD`)** 導入を検討する。

---

### 実験 #012: DeBERTa-v3-large (`[6, 24]` 多層平均) ＋ StyleGram ＋ クラスタ/全体 Sinkhorn ＋ 実 GLM-5.3 統合
- **実施日:** 2026-11-17〜2026-11-18
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab L4 GPU 学習 → ローカル CV / Nested-CV 検証 → Kaggle `T4 x2` 推論カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #010 で `GLM-5.3` / `GLM-5.3-Flash` 識別が改善し LB `0.634` に到達した一方、前段ベース（`0.613` 相当）は `Qwen3` + `ModernBERT` のみに依存しており、`claude-fable-5` (`0.443`), `gemini-3.7-flash` (`0.500`), `gpt-5.6-*` 等が伸び悩んでいた。
- トークナイザ（SentencePiece 128k）・Disentangled Attention が全く異なる **`microsoft/deberta-v3-large`**（多層 `[6, 24]` MeanPool）を追加し、さらに同一プロンプト文体クラスタ内の中心化＆分布均衡（**StyleGram + `capped_agglo` + Cluster/Global Sinkhorn**）を統合すれば、`0.72+` OOF 水準へ一気に引き上げられるはず。

#### 2. 実験内容 (Method)
- **Colab L4 学習 (`exp012_colab_l4_deberta_v3_large.py`)**:
  - 初期検証で `transformers 5.x` が `DeBERTa-v3-large` を `float16` 自動ロードして `AdamW eps=1e-8` で step 1 NaN 化する問題を特定・解消（`torch_dtype=torch.float32` 明示＋`bfloat16 autocast`＋`grad_ckpt` で `VRAM 11.31GB` 安定動作）。
  - 忠実度 `<0.45` 除外 Open Pool 99,233 件（2 epochs, `max_tokens=6144`）→ Stage 2 5-Fold CV（5 epochs）。計 `5.88 CU` 消費（残 `304.72 CU`）。`nobuyukikumagai/name-that-llm-exp012-deberta-large` へ保存。
- **統合推論 (`submission_exp012.py`)**:
  - Part 1: 5モデル幾何平均 `(0.30*4B + 0.30*MB-L + 0.20*Deb-L + 0.15*1.7B + 0.05*MB-B)`（`DeBERTa-v3-large` は T4 数値安定のため `fp32` 推論）。
  - Part 2: `Exp #011 StyleGram` (`0.35`) ＋ `capped_agglo(cap=18, thr=0.08)` Cluster Sinkhorn (`0.85`) ＋ Global Sinkhorn (`0.35`)。
  - Part 3: 上位 30% に `Exp #010 GLM-5.3/Flash`（`w=1.15`, GLM限定置換）。

#### 3. 結果 (Result)
- **DeBERTa-v3-large 単体:** Stage 1 Zero-Shot `0.58055` / Stage 2 5-Fold OOF **`0.58236`**（相関: vs 4B `0.7706`, vs MB-L `0.7993`, vs StyleGram `0.5154`）。
- **5-Fold Nested-CV (`eval_exp012_blend.py`):** **`0.72127`**（4/5 Fold が `(0.30, 0.30, 0.20, 0.15, 0.05)` を独立選出）。Global OOF `0.72128` (`c_sk=1.0`) / `0.72090` (`c_sk=0.75`)（Exp #010 `0.68016` 比 **`+0.04112`**）。
- **Kaggle Commit 検証 (`nobuyukikumagai/name-that-llm-exp012-full-ensemble`):** `T4 x2` で **`26.7m`** 完走（`Part 1+2` `217.7s` [Deb-L `27.4s`, StyleGram `10.7s`] + `Flash` `615s` + `GLM-5.3` `764s`）。`submission.parquet` `(3, 2)` 検証通過。
- **Public LB 結果 (`Ref 58483891`):** **`0.654`**（**`2,490 位 / 6,249 チーム`**, Exp #010 `0.634` 比 **`+0.020`** 更新・Top 39.8%！ Top 33% `0.664`, Top 25% `0.680`, Bronze `0.701`）。

---

### 実験 #011: Multi-View StyleGram (`char/word/shape/function` ＋ `capped_agglo` 中心化) ＆ Sinkhorn
- **実施日:** 2026-11-17
- **担当・実施者:** 学習者本人 ＆ Antigravity（ローカル CPU 5-Fold 検証 → `nobuyukikumagai/name-that-llm-exp011-stylegram`）
- **内容・結果:** 4視点 TF-IDF（`char 3-5`, `word 1-3`, 大文字小文字構造 `shape 2-6`, 機能語構造 `function 1-3`）に対し、`prompt_id` 非使用で `style_groups` / `capped_agglo(cap=18, thr=0.08)` クラスタ平均を減算して話題バイアスを除去。
  - 単体 5-Fold OOF: 中心化なし `0.35783` → 中心化 LR **`0.40675`**（`LR+SVC` `0.40601`, `Exp #008` 相関 `0.5461`）。
  - 全 `2,699` 件一括 `capped_agglo(cap=18, thr=0.08)` で `ARI = 0.9750`（`182` クラスタ、サイズ18完全一致 `131`）を達成し、`Exp #008 + Exp #010 GLM` OOF を `0.68016 → 0.71979`（`+0.03963`）へ引き上げた。

---

### 実験 #010: 実 `GLM-5.3` & `GLM-5.3-Flash`（IQ3_XXS 分割GGUF）対数尤度統合アンサンブル
- **実施日:** 2026-11-16〜2026-11-17
- **担当・実施者:** 学習者本人 ＆ Antigravity（Kaggle `T4 x2` で `train.csv` 特徴量抽出 → ローカル5-Fold CV検証・バンドル構築 → Kaggle `T4 x2` 推論カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #008（OOF `0.64312`, LB `0.613`）のクラス別 Recall 分析で、`glm-5.3`（`0.287`）と `glm-5.3-flash`（`0.413`）が極端に低く全18クラス平均を押し下げていた一方、`P_Exp008(glm-5.3) + P_Exp008(glm-5.3-flash)` 上位 30% で真の GLM サンプルの **99.7%**（上位 40% で **100.0%**）を捕捉できていることが判明した。
- したがって、オープン重みである実 `GLM-5.3`（19セグメント）および `GLM-5.3-Flash`（6セグメント）の `ntl-segrun` トークン対数尤度・エントロピー・Fast-DetectGPT 曲率（`fdg`）・ランク統計量を `Exp #008` 候補上位 30% に適用し、**置換前後いずれかが GLM の場合のみ変更を許可**すれば、`+0.03+` の大幅改善が見込める。

#### 2. 実験内容 (Method)
- **Phase 1（Kaggle `T4 x2` 特徴抽出 `exp010_extract_train_glm.py`）**: `train.csv` 全 2,699 件を `GLM-5.3-Flash`（`5940s / 1.65h`）と `GLM-5.3`（`10929s / 3.04h`）でスコア化し、`L320` / `L160` の特徴量・トークン尤度を抽出。
  - `Flash` 平均尤度 `lp_mean` で `glm-5.3-flash` が全18クラス1位（`-2.1733`, `fdg=-5.0062`）、`GLM-5.3` では `glm-5.3` が1位（`-2.3069`, `fdg=-4.0900`）となることを実測確認。
- **Phase 2（5-Fold CV 最適化 `eval_exp010_glm_cv.py`, `build_exp010_glm_bundle.py`）**:
  - `L160`（160トークン上限）の 63+ 特徴量（Flash 24 + GLM53 24 + 差分 12 + 前後半ドリフト 3）に対し、5-Fold `StratifiedGroupKFold` で **`0.50 * LogReg(C=0.05) + 0.25 * LightGBM + 0.25 * CatBoost`** を学習し、Private Dataset `nobuyukikumagai/name-that-llm-exp010-glm-models` へ保存。
- **Phase 3（推論パイプライン `submission_exp010.py`）**: `T4 x2`（`cap_major >= 8` 以外はネイティブ `float16` 使用）で `Exp #008` ベース推論（約3分）後に `ntl-segrun` で上位 `30%` に `GLM-5.3-Flash` + `GLM-5.3` 尤度抽出を実行し、`base_logp + 1.15 * ll_logp` に GLM 限定置換を適用（8.65h 自動時間制御・フォールバック付き）。

#### 3. 結果 (Result)
- **5-Fold OOF Balanced Accuracy (`Exp #008` → `Exp #010`):** **`0.64312` → `0.68016`（`+0.03704`）**
  - Nested-CV でも全 5 Fold が `(cov=0.30, w=1.15)` を全会一致で選出し、**Nested-CV `0.68016`**（変更率 `8.7%`）。
  - `glm-5.3` Recall: `0.287 → 0.667`（`+0.380`） / `glm-5.3-flash` Recall: `0.413 → 0.713`（`+0.300`）。
  - 参考比較：`L160 LogReg-Ext` 単体ブレンド `0.67571`、`0.6*LR + 0.4*LGBM` `0.67942`、`L320 LogReg-Ext` Nested-CV `0.67051`（短長ノイズ少ない `L160` が優位かつ高速）。
- **Kaggle Commit 検証 (`nobuyukikumagai/name-that-llm-exp010-glm-ensemble`):** `T4 x2` で `39.8m`（`Exp #008` `180.4s`, `Flash` `834s`, `GLM-5.3` `1364s`）で `KernelWorkerStatus.COMPLETE`。`submission.parquet` `(3, 2)`、Null 0、ID一致を確認済み。
- **Public LB 結果 (`Ref 58446766`):** **`0.634`**（**`2,804 位 / 6,175 チーム`**, Exp #008 `0.613` 比 **`+0.021`** 更新！ Top 50% `0.606`, Top 25% `0.679`, Bronze `0.699`）。

---

### 実験 #009 (検証記録): Qwen3-8B-Base LoRA + Multi-Layer Readout (`[9, 18, 27, 36]`)
- **実施日:** 2026-11-15〜2026-11-16
- **結果:** Colab A100 (`run-252b02`) で Stage 1 (`Pool-Val 0.61838`, `Zero-Shot 0.59239`) および Stage 2 5-Fold (`0.60571`: `[0.59444, 0.59630, 0.62407, 0.60447, 0.60926]`) が完走し、[oof_exp009_qwen3_8b_lora.csv](file:///C:/Users/kumat/dev/kaggle/005_name-that-llm/data/oof_exp009_qwen3_8b_lora.csv) を回収。4モデルブレンドで Nested-CV `0.64313`（Global `0.64869` / Log-Prob `0.64906`）まで改善する一方、Dataset タイトル 53文字（上限50文字超過）バグにより重みアップロードが `403` 失敗（修正済み：`DATASET_TITLE` 39文字化＋`assert`）。コスト効率から、+0.037級改善の Exp #010 を優先して保留。

---

### 実験 #008: ModernBERT-large 2段階学習 ＋ 4モデル幾何平均（対数確率）アンサンブル
- **実施日:** 2026-11-14〜2026-11-15
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab L4 GPU 学習 → Kaggle Private Dataset → 推論専用カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #007 まで、エンコーダー枠は `ModernBERT-base` (`0.51675`) のみであり、デコーダー群 (`Qwen3-4B` `0.61460` / `1.7B` `0.57606`) との精度差が広がっていた（ただし予測相関は `0.7010` と最も低い）。
- バックボーンを **`answerdotai/ModernBERT-large`**（395M パラメータ）へ拡大し、さらに Exp #007 と同じ **忠実度 `<0.45` 除外済み Open Pool（99,233件, `cap=6,000`）** × 3 epochs + `BucketBatchSampler` を組み合わせれば、単体 `0.58+` かつ 4モデル幾何平均ブレンドで `0.64+` を達成できるはず。

#### 2. 実験内容 (Method)
- **学習環境 (`exp008_colab_l4_modernbert_large.py`)**: Colab L4 (23.7GB VRAM), `bfloat16`, `BucketBatchSampler`（Stage 1 `max_tokens=6144, max_bs=48` / Stage 2 `max_tokens=3072, max_bs=24`）。最大 VRAM **`13.79 GB`**・わずか **約 4.30 CU**（`344.52 → 340.22 CU`）で完走。
- **Stage 1（99,233件: `Train=94,263, Pool-Val=4,970`）**: 3 epochs（各約1,942s / ~50 samples/s, `LR=3.0e-5`, `cls_weights ** 0.75`, `label_smoothing=0.05`）。
- **Stage 2 5-Fold CV**: `StratifiedGroupKFold(n_splits=5)` on `prompt_id` で 6 epochs（約57s/epoch, `LR=1.2e-5`, `weight_decay=0.02`, `label_smoothing=0.02`）。`bfloat16` 重み（各 791.7 MB × 5）を Private Dataset `nobuyukikumagai/name-that-llm-exp008-modernbert-large` へ保存。
- **推論高速化 (`submission_exp008.py`)**: Hidden Test 高速化のため全4モデルに **Length-sorted バッチ化 + 元順復元 (`inv_order`)** および `torch.inference_mode()` を実装。

#### 3. 結果 (Result)
- **Stage 1 経過**: Pool-Val Balanced Acc `0.52499` → `0.58450` → **`0.58829`**
- **Stage 1 Zero-Shot 診断 (`train.csv` 全件、参考値)**: **`0.55826`**（Exp #005 ModernBERT-base `0.51048` 比 `+0.04778`）
- **Stage 2 5-Fold OOF Balanced Accuracy (`ModernBERT-large` 単体)**: **`0.58790`**
  - 各Fold: `0.58519`, `0.58333`, `0.58519`, `0.59106`, `0.59444` (平均 `0.58784 +/- 0.00420`、極めて安定)
  - Exp #005 (`ModernBERT-base`, `0.51675`) 比 **`+0.07115`**、`Qwen3-1.7B` (`0.57606`) をも上回る！
- **相関 (`eval_exp008_blend.py`)**: vs `Qwen3-4B` = `0.7713` (`argmax_agree=0.6102`) / vs `Qwen3-1.7B` = `0.7718` / vs `ModernBERT-base` = `0.7352`
- **4モデル対数確率（幾何平均）ブレンド最適化 (`eval_exp008_blend.py`)**:
  - **`0.25 * log(Qwen3-4B) + 0.40 * log(ModernBERT-large) + 0.30 * log(Qwen3-1.7B) + 0.05 * log(ModernBERT-base)`**
  - **OOF Balanced Accuracy: `0.64312`**（**Nested-CV: `0.63831`** — 5 Fold 中 4 Fold が同一比率 `(0.25, 0.40, 0.30, 0.05)` を独立選出！）
  - （参考: `ModernBERT-base` 抜き 3 モデル幾何平均は Global `0.64053` / Nested-CV `0.63720`、線形ブレンド 4 モデル Nested-CV `0.63274`）
- **差分**: Exp #007 (`0.63052` / Nested `0.62201`) 比、Global **`+0.01260`** / Nested-CV **`+0.01630`** の大幅伸長！

#### 4. 次の一手 (Next)
- Exp #008 推論カーネル完了確認・本番提出（目標 LB `0.625+`）。
- **Exp #009 候補**: 残り Colab 残高 (`340.22 CU`) を活かし、**`Qwen3-8B-Base` LoRA（A100）** または **別系統デコーダー** を投入して Bronze ライン (`0.684+`) を狙う。

---

### 実験 #007: Qwen3-4B-Base LoRA ＋ 忠実度フィルタ済み Open Pool ＆ 3モデル異アーキテクチャ融合
- **実施日:** 2026-11-14
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab A100 40GB GPU 学習 → Kaggle Private Dataset → 推論専用カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #006 (`Qwen3-1.7B-Base`, 単体 OOF `0.57606`, Blend LB `0.566`) で因果 LLM バックボーンの有効性が実証された。
- バックボーンを **`Qwen/Qwen3-4B-Base`**（36層, `hidden_size=2560`, 4.055B params）へ拡大し、さらに `fidelity_by_source.csv` で **`fidelity_ratio < 0.45`** の低忠実度ソース（例：`claude-opus-4-8` のノイズソース等）を動的に除外＋Stage 1 上限を **6,000件/クラス**（計 99,233件）へ増強することで、単体 `0.61+`・3モデル融合で `0.63+` へ到達できるはず。

#### 2. 実験内容 (Method)
- **学習環境・VRAM対策 (`exp007_colab_qwen3_4b_lora.py`)**: Colab A100 40GB (`39.49 GiB`), `bfloat16` + `fp32` LoRA (`r=16, lora_alpha=32`, 33.03M trainable = `0.8145%`)。初回は `grad_ckpt=False, max_tokens=8192` で OOM 発生のため **`grad_ckpt=True`**（`use_reentrant=False`）+ `BucketBatchSampler`（Stage 1 `max_tokens=6144, max_bs=48` / Stage 2 `max_tokens=4096, max_bs=32`）に変更し最大 VRAM **`11.79 GB`** で完走。
- **Stage 1 データ精選**: `fidelity_ratio < 0.45` の 38 ペアから該当行 `8,287` 件を除外、`kimi-k3` は `gen_reasoning == "high"` のみ、`prvsiyan`・`qwen3.8-max (low)`・リークフラグを除外し、計 **99,233件**（min `2,673`, max `6,000`; `Train=94,263, Pool-Val=4,970`）で 2 epochs 学習（各約4,251s, `LR=1.2e-4 / 2.5e-4`, `label_smoothing=0.05`）。
- **Stage 2 5-Fold CV**: `StratifiedGroupKFold(n_splits=5)` on `prompt_id` で 5 epochs（約110s/epoch, `LR=3.5e-5 / 1.0e-4`, `weight_decay=0.03`, `label_smoothing=0.02`）。
- **成果物 & デプロイ**: 5-Fold LoRA（各132MB）＋ヘッドを `nobuyukikumagai/name-that-llm-exp007-qwen3-4b-lora` へ保存。推論カーネル `nobuyukikumagai/name-that-llm-exp007-qwen3-ensemble-inf` は `Qwen3-4B` → `Qwen3-1.7B` → `ModernBERT-base` を逐次実行・解放。

#### 3. 結果 (Result)
- **Stage 1 経過**: Pool-Val Balanced Acc `0.57004` → **`0.61961`**
- **Stage 1 Zero-Shot 診断 (`train.csv` 全件、参考値)**: **`0.59571`**（既に Exp #006 5-Fold 後 `0.57606` を凌駕）
- **Stage 2 5-Fold OOF Balanced Accuracy (`Qwen3-4B-Base LoRA` 単体)**: **`0.61460`**
  - 各Fold: `0.60926`, `0.63704`, `0.62778`, `0.59132`, `0.60741` (平均 `0.61456 +/- 0.01612`)
- **相関・多様性 (`eval_exp007_blend.py`)**:
  - vs `Exp #006 Qwen3-1.7B`: `prob_corr=0.8080`, `argmax_agree=0.6551`
  - vs `Exp #005 ModernBERT`: `prob_corr=0.7010`, `argmax_agree=0.5413`
- **3モデルアンサンブル最適化 (`eval_exp007_blend.py`)**:
  - **線形確率ブレンド `0.45 * Qwen3-4B + 0.30 * Qwen3-1.7B + 0.25 * ModernBERT`**: OOF Balanced Accuracy **`0.63052`**（Nested-CV **`0.62201`** と最良・最安定）
  - 対数確率（幾何平均）ブレンド `(0.35, 0.40, 0.25)`: OOF `0.63164` / Nested-CV `0.62164`（Nested-CV で線形ブレンドが上回ったため線形 45:30:25 を採用）
- **差分**: Exp #006 (`0.59606`) 比 **`+0.03446`**！

#### 4. 分かったこと・次の一手 (Insights & Next)
- Open Pool のソース忠実度フィルタ + `4B` スケールアップが直結し、Zero-Shot 時点で `+0.03186`、5-Fold で `+0.03854` 単体精度が伸長した。
- **次の一手 (Exp #008 候補)**: Exp #007 提出 LB 確認後、① **`Qwen3-8B-Base` LoRA（4bit QLoRA / A100）** または **`Gemma-3-4B` / `Llama-3.2` 等 別LLMファミリー** の投入、② **後処理（クラス事前確率補正 / Temperature Scaling）** の検証を進める。

---

### 実験 #006: Qwen3-1.7B-Base LoRA 2段階学習 ＋ 異アーキテクチャ深層アンサンブル (Qwen3 + ModernBERT)
- **実施日:** 2026-11-14
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab L4 GPU 学習 → Kaggle Private Dataset → 推論専用カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #005 (`ModernBERT-base`, `0.51675`) は双方向エンコーダー型モデルであり、同じ系統のモデル同士では誤差相関が高くなりやすい。
- 因果デコーダー型 LLM（**`Qwen/Qwen3-1.7B-Base`**、`bfloat16` + `fp32` LoRA `r=16`）はトークン生成分布に密接な文体・推論リズムを捉えられるため、単体性能の大幅向上（`0.57+`）に加え、`ModernBERT` との **真の異アーキテクチャ・アンサンブル効果** が期待できる。

#### 2. 実験内容 (Method)
- **バックボーン & プーリング**: `Qwen/Qwen3-1.7B-Base`（全パラメータ 1.738B 中、LoRA 17.43M = 1.003% 学習）。最終有効トークン表現 ＋ マスク平均プーリングの結合（`4096` 次元）→ `LayerNorm` → `Dropout(0.1)` → `Linear(4096, 18)`。
- **高速化・VRAM最適化**: トークン上限バケッティング `BucketBatchSampler`（Stage 1: `max_tokens=4096`, Stage 2: `max_tokens=3072`, `MAX_LEN=416`）＋ `expandable_segments:True`（最大 VRAM `21.04 GB / 22.03 GB` で安定完走）。
- **Stage 1（Open Pool 69,948件）**: Exp #005 と同一精選データ（`kimi-k3: high`, `prvsiyan`除外, `match_weight`抽出）で 2 epochs（各3,805s、`LR=1.5e-4 / 3e-4`）。
- **Stage 2（`train.csv` 5-Fold `StratifiedGroupKFold` on `prompt_id`）**: 5 epochs（各135s/epoch、`LR=5e-5 / 1.5e-4`）。
- **デプロイ**: 5-Fold の LoRA アダプター（`69.8MB × 5`）＋ 分類ヘッドを `nobuyukikumagai/name-that-llm-exp006-qwen3-lora` に保存し、推論時は `peft` 依存ゼロの純 PyTorch マージ推論を実行。

#### 3. 結果 (Result)
- **Stage 1 経過**: Pool-Val Balanced Acc `0.52625` → **`0.57507`**
- **Stage 1 Zero-Shot 診断 (`train.csv` 全件、参考値)**: **`0.56385`**（Exp #005 Stage 1比 `+0.05337`）
- **Stage 2 5-Fold OOF Balanced Accuracy (`Qwen3-1.7B-Base LoRA` 単体)**: **`0.57606`**
  - 各Fold: `0.54815`, `0.57037`, `0.59074`, `0.58582`, `0.58519` (平均 `0.57605 +/- 0.01553`)
- **クラス別相補性 (`eval_exp006_blend.py`)**:
  - `Qwen3` が大幅優位: `gpt-5.6-terra` (`+0.1467`), `qwen3.8-max` (`+0.1400`), `claude-fable-5` (`+0.1275`), `kimi-k3` (`+0.1133`), `gpt-5.6-luna` (`+0.1067`)
  - `ModernBERT` が優位: `glm-5.3` (`+0.1133`), `gpt-5.6-sol` (`+0.0400`), `gemini-3.7-flash` (`+0.0200`)
  - 予測確率相関: **`0.7307`** / argmax一致率: `0.5728`
- **2系統深層アンサンブル**:
  - **`0.65 * Qwen3-1.7B + 0.35 * ModernBERT-base` OOF Balanced Accuracy: `0.59606`**（単体最良比 **`+0.02000`**、Nested-CV も `0.59087` と安定）
  - ※ `LR (0.335)` との 3系統混合は `0.59569`（推論+63s）と僅差または低下したため、より高速・高精度な 2系統純粋ニューラル融合（65:35）を採用。
- **差分**: Exp #005 (`0.51787`) 比 **`+0.07819`** の特大ジャンプ！

#### 4. 次の一手 (Next)
- Exp #006 推論専用カーネル完了確認・本番提出（LB `0.59+` 確認）。
- **Exp #007**: メダル圏 (`0.684+`) へ向け、より上位規模モデル（例：`Qwen3-4B-Base` / `Qwen3-8B` QLoRA または `ModernBERT-large` + 学習データ倍増 / クラス事後確率最適化）の投入を計画。

---

### 実験 #005: Open Pool v14 精選データによる 2段階学習 (ModernBERT-base) ＋ 学習/推論分離アーキテクチャ
- **実施日:** 2026-11-13
- **担当・実施者:** 学習者本人 ＆ Antigravity（Colab L4 GPU 学習 → Kaggle Private Dataset → 推論専用カーネル）

#### 1. 仮説 (Hypothesis)
- Exp #004 では `train.csv`（約2,160件/fold）のみで ModernBERT-base を学習したため、1クラスあたり約120件しかなく表現学習が不足（単体 `0.32931`）。
- 公開データセット `crtproduction/name-that-llm-open-pool`（223,840件、`CC-BY-SA-4.0`、ホワイトリスト済・`train` プロンプト非重複）から、忠実度監査 (`fidelity_by_reasoning.csv`, `fidelity_by_source.csv`) に基づく高品質サブセットを抽出し **Stage 1 ドメイン適応 → Stage 2 `train.csv` 5-Fold CV** を行うことで、CV `0.50+` へ飛躍できるはず。
- 学習と推論を分離し、5-Fold重みを Kaggle Private Dataset へ保存すれば、提出時の Kaggle ノートブックは推論のみ（数分以内）で完結する。

#### 2. 実験内容 (Method)
- **Stage 1 データ精選 (`curate_open_pool`)**:
  - `prvsiyan/prvsiyans-extended-name-that-llm-dataset`（`gemini-3.7-flash` 低忠実度 `0.3342` の 266件）を除外。
  - `kimi-k3` は `gen_reasoning == "high"` の 2,673件に限定（`fidelity_ratio = 0.8092` vs 他設定 `0.11〜0.36`）。
  - `qwen3.8-max` は `gen_reasoning == "low"`（70件、忠実度 `0.0`）を除外。
  - 各ラベル上限 4,000件で `match_weight`（書式・記号・長さ補正）付き非復元抽出を実施（計 69,948件：`Train=66,485`, `Pool-Val=3,463` を `prompt_canon` グループ分割）。
  - クラス数の微差（`kimi-k3`: 2,673件 / `gemini-3.7-flash`: 3,275件 / 他 4,000件）を補正するため均衡クラス重み＋`label_smoothing=0.05` を適用。
- **Stage 1 学習**: `answerdotai/ModernBERT-base`, `MAX_LEN=416`（動的パディング）, `batch_size=48`, `lr=4e-5`, 3 epochs, Colab L4 `bfloat16`。
- **Stage 2 学習**: Stage 1 最良チェックポイントから `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `prompt_id` で 5-Fold 学習（`batch_size=16`, `lr=1.5e-5`, 6 epochs, `label_smoothing=0.02`）。
- **成果物回収**: 5-Fold 重みを `float16` 化（各285MB）し Private Dataset `nobuyukikumagai/name-that-llm-exp005-weights` へアップロード。

#### 3. 結果 (Result)
- **Stage 1 経過**: Pool-Val Balanced Acc `0.45948` → `0.50333` → **`0.52145`**（各epoch約1,134s）
- **Stage 1 Zero-Shot 診断 (`train.csv` 全件、参考値)**: **`0.51048`**（`train.csv` 未学習で既に Exp #004 単体比 `+0.18117`）
- **Stage 2 5-Fold OOF Balanced Accuracy (`ModernBERT-base` 単体)**: **`0.51675`**
  - 各Fold: `0.52222`, `0.52407`, `0.51111`, `0.50939`, `0.51667` (平均 `0.51669 +/- 0.00582`)
- **アンサンブル評価 (`eval_exp005_blend.py`)**:
  - `0.90 * Exp005 ModernBERT + 0.10 * Exp002 LR`: **`0.51787`**（線形ブレンド `0.51787` / 対数確率ブレンド `0.51786 @ 95:5`）
  - ※ Exp002 LR (`0.33486`) や Exp003 LGBM (`0.27482`) は単独精度差が ~0.18 と大きいため、過学習リスクのある高比率混合（Nested-CV `0.51231`）を避け、堅実な 90:10 を採用。
- **差分**: Exp #004 アンサンブル (`0.37820`) 比 **`+0.13967`**、初回ベースライン (`0.30708`) 比 **`+0.21079`** の大幅更新。

#### 4. 分かったこと・考察・次のアクション (Insights & Next)
- **データ規模と忠実度フィルタの威力**: 全18クラスで再現率が `+0.060`〜`+0.327` 向上（特に `gpt-5.6-sol` `0.093→0.420`, `claude-opus-4-8` `0.293→0.567`, `gemini-3.6-flash` `0.327→0.593`, `gemini-3.1-pro-preview` `0.607→0.813`）。
- **次の一手 (Exp #006 以降)**:
  1. Exp #005 推論専用カーネルの完了確認・本番提出（LB確認）。
  2. より大容量なバックボーン（例: **`ModernBERT-large`** / **`DeBERTa-v3-large`** / **`Qwen3-1.7B/4B LoRA`**）を Open Pool 全量または高上限（例: 8,000〜全件 + サブサンプリング）で学習し、同水準（CV `0.55〜0.65+`）の別系統モデル群で真の多様性アンサンブルを構築する。
  3. Open Pool + `train.csv` 双方で学習する TF-IDF + LR（Open Pool補強版線形モデル）や文体特徴モデルの強化も検討。

---

### 実験 #004: Colab L4 GPU ModernBERT-base 5-Fold ファインチューニング ＆ 3大系統完全融合アンサンブル
- **実施日:** 2026-11-12
- **担当・実施者:** 学習者本人 ＆ Antigravity（Google Colab L4 GPU 実行）

#### 1. 仮説 (Hypothesis)
- 統計量や n-gram だけではテキスト全体の「論理構成」「トーン」「推論の深さ」などの深層セマンティクスを捉えられない。
- 本コンペで最高峰の実績を持つ最新エンコーダー **`answerdotai/ModernBERT-base`** を Google Colab L4 GPU 上で 5-Fold StratifiedGroupKFold ファインチューニングする。
- 異なる世界観を持つ **「深層トランスフォーマー（ModernBERT）」＋「高次元線形モデル（TF-IDF + 文体）」＋「非線形決定木（LightGBM 記号不変則）」** の 3 大系統を融合することで、飛躍的なアンサンブル多様性を獲得できるはず。

#### 2. 実験内容 (Method)
- **学習環境:** Google Colab L4 GPU (24GB VRAM), `bfloat16` 混合精度
- **モデル設定:** `answerdotai/ModernBERT-base` (max_length=512, batch_size=16, lr=3e-5, Cosine Warmup, 4 epochs)
- **検証設計:** 厳格 `StratifiedGroupKFold(n_splits=5)` on `prompt_id`
- **成果物回収:** Google Drive 自動同期 ＋ Base64 完全復元

#### 3. 結果 (Result)
- **単体モデル比較:**
  - ModernBERT-base (深層言語モデル): **`0.32931`**
  - Logistic Regression (文体 + TF-IDF): `0.33486`
  - LightGBM (文体 + SVD): `0.27482`
- **3大系統アンサンブル重み探索:**
  - 最適比率: **ModernBERT 52% ＋ Logistic Regression 45% ＋ LightGBM 3%**
  - **🏆 3大系統アンサンブル OOF Balanced Accuracy:** **`0.37820` (37.82%)**
- **差分:** Exp #003 (`0.34077`) 比 **`+0.03743`**（+3.74%）の特大ジャンプ！初回ベースライン (`0.30708`) 比 **`+0.07112`**（+7.11%）！

#### 4. 分かったこと・統計的考察 (Insights)
- **深層学習 × スタイロメトリーの完璧な相補性:**
  - ModernBERT 単体（0.329）と線形モデル単体（0.335）はスコアが近いものの、互いの予測相関が低く、直交性が極めて高かった。
  - トランスフォーマーが「文脈の意味」を判断し、線形・決定木モデルが「エムダッシュやクォート等の生成エンジンの癖」を補完した結果、アンサンブルが 0.378 まで急伸した。
  - このスコア水準は、初期ベースライン層（5,200位帯）から一気に 1,000 人以上をごぼう抜きできるポテンシャルを持つ。

---

### 実験 #003: 決定木 (LightGBM) 導入 ＆ 線形・非線形 多様性アンサンブル (初提出)
- **実施日:** 2026-11-12
- **手元検証スコア:** **`0.34077`**
- **本番スコア:** **`0.31600`** (公式 5,210 位)
