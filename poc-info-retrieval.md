# 情報取得の疎通確認（PoC）レポート

- **対象コンペ:** [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer)
- **調査実施日:** 2026-10-16
- **担当:** Antigravity (Gemini Flash)
- **関連Issue:** [kumakit/mission-control#402](https://github.com/kumakit/mission-control/issues/402) (Phase 1)

---

## 1. 調査の目的
Kaggle公式が提供する情報取得手段（Kaggle公式MCP、Kaggle公式CLI）の機能、認証方法、およびAntigravity環境からの接続可能性を検証し、コンペ情報の取得可否を整理する。

---

## 2. Kaggle公式MCP (`https://www.kaggle.com/docs/mcp`) の検証結果

### (1) 仕様と機能
- **エンドポイント:** `https://www.kaggle.com/mcp`（Remote Streamable HTTP / SSE形式）
- **利用可能な主なツール群:**
  - `Competitions`: コンペ検索、メタデータ取得、データファイルダウンロード、予測提出
  - `Notebooks`: ノートブック管理、実行開始、停止、実行出力取得
  - `Datasets`: データセット一覧、メタデータ取得
  - `Models`: モデル一覧、メタデータ取得

### (2) 認証方式
- **OAuth 2.0:** クライアントがサポートしている場合、対話型認可フローで認証可能。
- **Personal API Token（トークン認証）:** Kaggleアカウントの「Settings > Generate New Token」から発行される、プレフィックス `KGAT_` で始まるアクセストークンをHTTPヘッダー（`Authorization: Bearer KGAT_...`）に付与して通信する。

### (3) Antigravityからの接続可否
- Antigravityは `~/.gemini/config/mcp_config.json` にてStreamable HTTP形式のMCPサーバー（`"url": "https://www.kaggle.com/mcp"`）を登録する仕組みを備えています。
- 一方で、現時点では「外部サービスの設定変更・認証トークンの自動設定は行わない」という方針に従い、トークン発行および設定ファイルの書き換えは保留としています。

---

## 3. Kaggle公式CLI (`kaggle-cli` / `kaggle`) の検証結果

### (1) 実行可能性の確認
- ローカル環境（Windows PowerShell 7）において、`uvx kaggle --version` を実行し、**`Kaggle CLI 2.2.4` が即座に実行可能** であることを確認しました（グローバル環境を汚さずにオンデマンド実行可能）。

### (2) 各種情報の取得可否まとめ

| 情報種別 | Kaggle MCP | Kaggle CLI (v2.2.4) | Web参照（検索含む） | 判定と備考 |
|---|---|---|---|---|
| **Overview / Evaluation** | 取得可能 | **取得不可** | **取得可能** | CLIには概要説明文（Markdown）をそのままダンプするコマンドが存在しないため「取得不可」。Web参照で取得。 |
| **Rules（規約全文）** | 取得可能 | **取得不可** | **取得可能** | CLIには規約全文をダンプするコマンドが存在しないため「取得不可」。Web参照で取得。 |
| **Data（ファイル一覧・DL）** | 取得可能 | **取得可能** | **取得可能** | `kaggle competitions files -c digit-recognizer` / `download` で取得可能（APIトークン要）。 |
| **参考Notebook（Code）** | 取得可能 | **取得可能** | **取得可能** | `kaggle kernels list -s digit-recognizer` / `pull` で取得可能。 |
| **Discussion（トピック）** | 一部対応 | **取得可能** | **取得可能** | CLI v2.2.0以降、`kaggle competitions topics list digit-recognizer` および `topics show` により取得可能。 |

---

## 4. 資格情報（APIトークン）の安全管理方針
- APIトークン（`kaggle.json` や `KGAT_...`）は、Gitリポジトリ（`.gitignore` に明記済み）、ターミナル出力ログ、GitHub Issue本文には一切記録・出力しません。
- 資格情報の登録が必要になった場合は、OS標準の安全な保管場所（`$HOME/.kaggle/kaggle.json`）または環境変数に限定して配置します。

---

## 5. 結論と次のステップ（Phase 2）への推奨アプローチ
1. **Phase 2（コミュニティダイジェスト作成）への進行:**
   - Discussionの良質な質問・回答（最大5件）および参考Notebook（最大3件）の選定・日本語解説は、トークン未設定のままでもWeb公開情報から高品質に収集・作成可能です。
2. **Phase 3（初提出）への進め方:**
   - 初回提出はKaggle Web画面またはKaggle Notebookからの手動提出を想定しているため、現時点で無理にAPIトークンを発行する必要はありません。
   - 学習者が「ローカルからCLIで直接ファイルをダウンロード／提出したい」と考えた段階で、APIトークンの発行をご案内します。
