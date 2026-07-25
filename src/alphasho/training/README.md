# AlphaSho 学習システム

学習 pipeline は、再開可能な独立 stage で構成されています。

1. 任意の KIF/CSA 教師棋譜の取込と warm-up
2. best model を使った MCTS 自己対局
3. 圧縮 disk replay からの候補モデル学習
4. 先後を交換した candidate 対 best の評価
5. checkpoint、昇格、モデル公開、JSONL metrics

## 実ルールと診断用短局

実験用の学習では、純粋な `SearchPosition` を `TrainingPosition` で包みます。
cshogi が返す実際の終局を常に最優先します。CPU profile では、その次に累積駒取り
threshold と、最大手数での駒得差裁定を適用できます。Baseline は実ルールのみを
使用します。

短局ルール、初期局面集合、検索ルールの意味論 version から安定した `ruleset_id` を
生成します。自己対局 shard と replay buffer はこの ID で分離されるため、駒取り
threshold、局面集合、反復局面の裁定方法を変更しても、過去の異なる教師信号は
混入しません。千日手は UI と検索の両方で同一局面の4回目に成立します。

## Entry point と責務

利用者が直接編集するのは、ルートの `training_pipeline.py` です。このファイルには
頻繁に変更する設定だけを置きます。

この package が担当するものは次のとおりです。

- `AlphaZeroTrainer` による lifecycle と stage transition
- 自己対局、教師データ取込、replay shard
- network optimization、候補評価、昇格
- atomic checkpoint、process lock、中断と再開
- 多言語の可読ログと構造化イベント

すべての PyCharm 学習構成は同じ entry point を使用します。固定 profile は実行構成の
環境変数で選択します。デスクトップ UI は training package を import しません。

## ログと checkpoint

可読メッセージは `messages.py` で一元管理され、日本語・中国語・英語に対応します。
内部の `stage`、`event`、`message_key`、metric field は英語で固定されています。

各 profile は独立した process lock を持ちます。checkpoint は完全な temporary file
を先に作り、Windows のファイル占有に対する上限付き retry の後で atomic replace
します。model と optimizer を先に保存し、`pipeline_state.json` を最後に commit
することで、状態ファイルを checkpoint 完了の印として扱います。起動時は revision を
比較し、正式ファイルより新しい正常な temporary state があれば復元します。
