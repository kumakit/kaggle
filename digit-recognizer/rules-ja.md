# Kaggle Digit Recognizer 公式ルール要約（日本語版）

> [!NOTE]
> 本ドキュメントは、Kaggle公式の [Digit Recognizer Rules](https://www.kaggle.com/competitions/digit-recognizer/rules) を初学者の学習用途向けに要約・意訳したものです。正式な規約および文言は必ず公式ページ（英語）が正本となります。

---

## 1. コンペティションの概要と種別
- **コンペ名:** Digit Recognizer（手書き数字認識）
- **公式URL:** https://www.kaggle.com/competitions/digit-recognizer
- **種別:** Getting Started / Knowledge Competition（初学者向け学習用コンペ）
- **期限:** 常設（Rolling / No deadline）
- **賞金・メダル:** なし（Kaggleメダルやランキングポイントの付与対象外。純粋な技術習得とコミュニティ交流を目的としたコンペです）

---

## 2. 参加条件とアカウント
- **アカウントの制限:** 1人の参加者が複数のKaggleアカウントを所持して参加・提出することは固く禁止されています（One account per participant）。
- **参加手続き:** Kaggleにログイン後、コンペページの「Join Competition（または Understand and Accept）」をクリックしてルールに同意することで参加できます。

---

## 3. チーム編成とコード共有
- **チーム参加:** 個人参加、または複数人でのチーム参加が可能です（チームマージにも所定のルール・期限があります）。
- **非公開での共有禁止:** チーム外の参加者と、非公開（メール、DM、非公開リポジトリなど）でコードやデータを共有することは禁止されています。
- **公開共有は歓迎:** Kaggleの公開Notebook（Code）やDiscussionフォーラムを通じて、すべての参加者に均等にアクセス可能な形で公開することは認められており、むしろ学習を深めるために推奨されています。

---

## 4. 提出（Submission）の制限
- **1日の提出上限:** **1日最大10回まで**（Daily submission limit: 10）。提出回数はUTC基準の0時（日本時間 朝9時）にリセットされます。
- **最終評価対象の選択:** 最終スコア（リーダーボード）の評価用として、過去の提出の中から**最大2件**を選択できます。

---

## 5. 外部データおよび手作業ラベリングの禁止
- **手作業ラベリング（Hand-labeling）の禁止:** テストデータを人間が見て手作業でラベルを付与したり修正したりする行為は明示的に禁止されています。
- **正解データのリーク利用禁止:** Digit Recognizerで使われているデータは有名な「MNIST」データセットです。インターネット上にはMNISTのテストデータの正解（Ground Truth）が出回っていますが、これをそのまま外部から読み込んで提出することは不正行為であり、学習の趣旨にも反するため禁止されています。機械学習モデルを通じて予測を行ってください。

---

## 6. 生成AIおよびツールの利用方針
- **AIコーディング補助ツールの利用:** モデルのコード作成、データの前処理、エラーのデバッグ、統計概念の理解などにAIを活用することは差し支えありません。
- **自動化・常駐化についての学習方針:** 本プロジェクトでは、過剰な自動化やAPIによる自動提出・不正スクレイピングは行わず、手動での理解と検証を優先します。

---

## 7. 出典・参考リンク
- [Digit Recognizer Overview (Kaggle公式)](https://www.kaggle.com/competitions/digit-recognizer/overview)
- [Digit Recognizer Rules (Kaggle公式)](https://www.kaggle.com/competitions/digit-recognizer/rules)
- [Digit Recognizer Data (Kaggle公式)](https://www.kaggle.com/competitions/digit-recognizer/data)
