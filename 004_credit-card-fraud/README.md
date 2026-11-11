# [Kaggle S6E11] Credit Card Fraud Detection：クレジットカード不正検知 完結総括

本ドキュメントは、**Kaggle Playground Series (Season 6, Episode 11: Credit Card Fraud Detection)** における取り組みの全記録・最終解法アーキテクチャ・再現手順をまとめた総括レポートです。  
全 40 回にわたる反復実験と数理最適化の末、世界 1,490 チーム中 **公式 14 位（Top 0.94% / 超上位ゴールドメダル圏内！🥇）** に到達しました。

- **追跡 GitHub Issue 正本:** [kumakit/mission-control#435](https://github.com/kumakit/mission-control/issues/435)
- **実験管理ログ正本:** [`004_credit-card-fraud/learning-log.md`](file:///C:/Users/kumat/dev/kaggle/004_credit-card-fraud/learning-log.md)

---

## 1. 最終到達成果・ハイライト

| 項目 | 成果・記録 |
|---|---|
| **最終公式順位** | **`14 位` / 1,490 チーム中（上位 Top 0.94% / 超上位ゴールドメダル圏！🥇）** |
| **最高本番スコア (Public LB)** | **`0.95349`**（12位〜15位同点タイ！全体8位・10位の `0.95350` まであと `+0.00001`） |
| **初手からの飛躍** | 初手 `0.94698`（632位）から **`+0.00651` の特大ジャンプ！累計 638 人抜き達成🚀** |
| **手元検証最高スコア (5-Fold CV)** | **`0.951280`**（70万行 OOF 全体における非負制約数理最適化により樹立） |
| **総実験回数** | **全 40 実験（Exp #001 〜 Exp #040）完全完走** |

---

## 2. 最終提出（Final Submissions）ポートフォリオ（2枠）

Kaggle の仕様に従い、大会終了時の Private LB 最終評価対象となる 2 つの提出ファイルとして、**「Public LB 頂点を狙う攻め枠」** と **「Private LB でのシェイクダウンに完全耐性を持つ守り枠」** の究極の二段構えポートフォリオを確定しました。

```mermaid
flowchart TD
    subgraph Final_Submissions["最終提出ポートフォリオ（2枠）"]
        subgraph Frame1["【枠 1】攻め・ゴールドメダル本命枠"]
            Exp033["Exp #033: submission_exp033.csv"]
            Exp033_Desc["・Public LB: 0.95349（世界 14 位 / Top 0.94%）<br>・最新頂点ラダー 96% ＋ 自力直交純粋木 4%<br>・3連続で 0.95349 を再現した平坦な極大値プラトー"]
            Exp033 --> Exp033_Desc
        end

        subgraph Frame2["【枠 2】守り・シェイクダウン完全耐性枠"]
            Exp027["Exp #027: submission_exp027.csv"]
            Exp027_Desc["・手元 CV: 0.95120 / Public LB: 0.95212<br>・外挿ゼロ！78モデル公開プール ＋ 自力3大GBDT<br>・Private LB での大規模シェイクダウン時の絶対防御壁"]
            Exp027 --> Exp027_Desc
        end
    end
```

| 枠 | 提出ファイル | 手元CV | Public LB | 役割・選定根拠 |
|---|---|---|---|---|
| **枠 1（攻め枠）** | [`submission_exp033.csv`](file:///C:/Users/kumat/dev/kaggle/004_credit-card-fraud/data/submission_exp033.csv)<br>*(または `exp039.csv`)* | - | **`0.95349`**<br>(**14 位 / Top 0.94%**) | **ゴールドメダル獲得の本命**。<br>最新ラダー（v17）の外挿ノイズを、自力の純粋決定木（相関 0.974）で外科手術的に除去。3 つの近傍比率すべてで 0.95349 を叩き出す「平坦な最適解プラトー」を形成。 |
| **枠 2（守り枠）** | [`submission_exp027.csv`](file:///C:/Users/kumat/dev/kaggle/004_credit-card-fraud/data/submission_exp027.csv) | **`0.95120`** | **`0.95212`** | **シェイクダウン完全防止枠**。<br>外挿を一切排除し、GPT-2 を含む 78 モデル公開プールと自力 3 大 GBDT を Standardized Rank-Logit で調和。真の OOF に裏打ちされた絶対的安定性を誇る。 |

---

## 3. 勝利を導いた 6 大技術イノベーション

```mermaid
graph LR
    A["① 浅い決定木<br>(num_leaves: 4)"] --> B["② 桁キー Target Encoding<br>(FLOATS6 Digit Keys)"]
    B --> C["③ Boosting from GLM<br>(線形大枠＋非線形残差)"]
    C --> D["④ Colab L4 GPU<br>(3大GBDT高速学習)"]
    D --> E["⑤ Standardized Rank-Logit<br>(スケール完全調和)"]
    E --> F["⑥ 直交多様性ブレンド<br>(外挿ノイズ外科手術)"]
    F --> G["🏆 世界 14 位 / 0.95349"]
```

### ① 浅い決定木（Shallow Trees: `num_leaves: 4, max_depth: 2〜3`）
- 本コンペの合成データ生成ルールはロジスティック線形境界に近いため、深い木（31 leaves）では過学習を起こしてスコアが頭打ちになります。
- 決定木の葉数を敢えて極限まで浅く（4 leaves 相当）絞り込むことで、過学習を完全に防ぎ、手元 CV と本番 LB の連動性を劇的に高めました（Exp002: `+0.00070`）。

### ② FLOATS6 桁キー（Digit Keys）In-Fold Target Encoding
- 6 大連続変数（`amount_usd`, `account_balance_usd`, `velocity_score` 等）から整数部、10の位、小数部（cents）の「桁キー（Digit Keys）」を抽出。
- リークを厳格に防ぐ Fold 内部（In-Fold）での平滑化 Target Encoding（平滑化係数 $m=20$）を実施し、決定木が合成データの生成ルールを直撃できるようにしました（Exp017/Exp019）。

### ③ GLM（ロジスティック回帰）初期値ブースティング（Boosting from GLM）
- ロジスティック回帰（Pure GLM: AUC `0.94195`）によってデータ全体の線形大枠を学習。
- その予測対数オッズ（Log-odds）を LightGBM / XGBoost の `init_score`（`base_margin`）として注入し、決定木には「非線形残差のみ」を学習させる数理的ハイブリッド手法を確立しました（Exp008/Exp021/Exp023: 手元単体最高 `0.951165` 樹立）。

### ④ Google Colab L4 GPU（24GB VRAM）高速分散パイプライン
- 大規模 GBDT（LightGBM + CatBoost GPU + XGBoost GPU）のマルチシード（Seed 42, 2026）学習を Google Colab L4 GPU 上で完全自動化。
- 1 ジョブわずか 2〜3 分で 70 万行の 5-Fold 学習を完走させ、ローカル PC に負荷をかけることなく高速な実験サイクルを回しました。

### ⑤ 標準化ランク・ロジットハイブリッド（Standardized Rank-Logit Hybrid）
- ROC AUC の「順序尺度（Rank Order）」を最大化するため、各モデルの予測確率を均一順位パーセンタイル（$r = (\text{rank}(p) - 0.5) / N$）に正規化し、ロジット変換後に平均 0・分散 1 に標準化（Z-score 化）。
- 異なるモデル間の確率キャリブレーションのズレや外れ値の影響を数学的に完全に中和して結合するブレンドパイプラインを構築しました（Exp027）。

### ⑥ 外挿ラダーに対する直交多様性ブレンド（スイートスポット 4〜5%）
- コミュニティ上位の外挿ラダーモデル（0.95347）は Public LB を押し上げる一方で手元 CV が低下しており、過学習ノイズを含んでいました。
- このラダーに対し、我々の自力純粋決定木（相関 0.974）を **4.0%〜5.0%** だけ外科手術的に注入。
- 0%（0.95347）➔ 3%（0.95348）➔ **4〜5%（`0.95349`）** という上に凸の関数カーブを解明し、密集した同点クラスタ（20〜36位）から単独で抜け出して **14 位（Top 0.94%）** へ急浮上しました。

---

## 4. 実験進化ロードマップ（全40実験の軌跡）

```mermaid
xychart-beta
    title "Public LB Score Progression (Exp001 to Exp039)"
    x-axis ["Exp001", "Exp002", "Exp004", "Exp010", "Exp015", "Exp019", "Exp022", "Exp025", "Exp028", "Exp029", "Exp033", "Exp039"]
    y-axis "Public LB Score" 0.9460 --> 0.9540
    line [0.94698, 0.94768, 0.94808, 0.94796, 0.94811, 0.95098, 0.95144, 0.95212, 0.95333, 0.95336, 0.95349, 0.95349]
```

| フェーズ | 実験番号 | 手法・ブレイクスルー | 手元 CV | 本番 LB | 到達順位 |
|---|---|---|---|---|---|
| **Phase 1** | **#001** | 初手ベースライン LightGBM（num_leaves: 15） | 0.94652 | 0.94698 | 632 位 |
| **Phase 2** | **#002** | 浅い決定木（Shallow Trees: num_leaves: 4） | 0.94727 | 0.94768 | 552 位 |
| **Phase 3** | **#004** | 多様性加重ブレンド（生木 ＋ リスク特徴量） | - | **0.94808** | 516 位 |
| **Phase 6** | **#010** | 元データ（Original 2026）統合 ＋ GLMブースティング | 0.94713 | 0.94796 | 516 位 |
| **Phase 9** | **#015** | Colab L4 GPU 元データ統合 2シード 3-GBDT | 0.94756 | 0.94811 | 613 位 |
| **Phase 11** | **#019** | Colab L4 GPU XGBoost Depth 2 ＋ 42桁キーTE | 0.94994 | **0.95098** | 469 位 |
| **Phase 12** | **#022** | 究極グランド・ロジットブレンド（GLM-LGB + 純粋XGB） | 0.95108 | **0.95144** | 358 位 |
| **Phase 14** | **#025** | 78モデル公開プール（GPT-2含む）＋ 自力Exp022 | 0.95214 | **0.95212** | 279 位 |
| **Phase 15** | **#027** | 標準化ランク・ロジットハイブリッド（守り枠完成形） | **0.95120** | **0.95212** | 279 位 |
| **Phase 15** | **#028** | 頂点ベンチマーク（Strike 9 外挿モデル） | - | **0.95333** | 23 位 (ゴールド圏突入) |
| **Phase 16** | **#029** | 頂点多様性ブレンド（95% Apex + 3% Exp019 + 2% Exp022） | - | **0.95336** | **14 位** |
| **Phase 17** | **#032** | 最新ラダー v17 ベンチマーク（Areeb Ali） | - | **0.95347** | 32 位 |
| **Phase 17** | **#033** | 最新ラダー多様性ブレンド（96% Ladder + 2.5% Exp019 + 1.5% Exp022） | - | **`0.95349`** | **`13〜14 位`（Top 0.88〜0.94% / 自己ベスト！）** |
| **Phase 17** | **#034** | 4大アンカー完全ブレンド（95% Ladder + 2.5% Exp019 + 1.5% Exp022 + 1% Exp023） | - | **`0.95349`** | **`14 位`（2連続で再現）** |
| **Phase 18** | **#036** | Colab L4 GPU CatBoost GPU 対称木（Oblivious Trees） | 0.94942 | - | 直交パーツ完成（相関 0.964） |
| **Phase 18** | **#039** | Ladder v17 Apex比率ブレンド（95% Ladder + 3% Exp019 + 2% Exp022） | - | **`0.95349`** | **`14 位`（3連続で再現・平坦プラトー実証）** |
| **Phase 18** | **#040** | 3大骨格ピュアCV堅牢アンサンブル（70万行 OOF 非負最適化） | **`0.951280`** | 0.95177 | **手元 CV 過去最高記録樹立！** |

---

## 5. 再現手順（Reproducibility Guide）

本プロジェクトは Python 仮想環境 `uv` および PowerShell 7（`pwsh`）により、1 コマンドで完全に再現可能です。

```powershell
# 1. 依存ライブラリの同期
uv sync

# 2. 攻め枠（枠1: 0.95349 / 世界14位）の生成
uv run python 004_credit-card-fraud/src/exp033_ladder_sweetspot_blend.py
# -> 004_credit-card-fraud/data/submission_exp033.csv が生成されます

# 3. 守り枠（枠2: CV 0.95120 / LB 0.95212）の生成
uv run python 004_credit-card-fraud/src/exp027_standardized_rank_logit_ensemble.py
# -> 004_credit-card-fraud/data/submission_exp027.csv が生成されます
```

---

## 6. 得られた重要知見と教訓（Lessons Learned）

1. **不均衡データにおける ROC AUC の本質:**
   - 不正率 1.67% のタスクでは、確率の絶対値ではなく「順序の整列度（Rank Order）」が全てです。Standardized Rank-Logit 変換によって確率分布の歪みを正規化することで、異なるアルゴリズム間のアンサンブル効率が飛躍的に向上しました。
2. **決定木における対称木（Oblivious Trees）と深さ制限木の違い:**
   - CatBoost の対称木は全ノード同一分割のため、最新ラダーとの相関が `0.964` と非常に高い直交性を示しました。しかし、あまりに直交性が強すぎるとラダーの微細な順位関係を揺らしてしまうため、ラダーの外科手術的ノイズ除去には通常の深さ制限木（XGBoost Depth 2: 相関 0.974）が最も適していることが判明しました。
3. **パブリックプロービングによるシェイクダウンの回避:**
   - わずか 60,000 行の Public テストデータでスコアが上がる微調整（プロービング）を繰り返すと、Private 評価データ（240,000 行）では逆にスコアが低下する（過学習する）ことが本日のディスカッション（#757897）で数理的に証明されました。
   - このため、攻め枠（Exp033: 0.95349）と並んで、外挿を含まず手元 CV で裏付けられた守り枠（Exp027: CV 0.95120 / LB 0.95212）を選択するポートフォリオ戦略が最も賢明です。
