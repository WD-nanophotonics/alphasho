# 実行時に生成される成果物

このディレクトリには、モデル、自己対局データ、ログなどの実行時成果物を保存します。
この README 以外は Git の管理対象外です。実行中ではない profile のディレクトリを
削除してもソースコードには影響しませんが、その profile の学習進捗は失われます。

- `cpu-debug/`、`tactical-overfit/`、`smoke-short/`、`baseline/`
  - 相互に分離された学習 profile
- `<profile>/checkpoints/best.pt`
  - その profile の自己対局で使用する現在の best model
- `<profile>/checkpoints/candidate.pt`
  - 直近で学習した候補モデル
- `<profile>/checkpoints/training_state.pt`
  - network、optimizer、scheduler、乱数状態などの再開情報
- `<profile>/checkpoints/pipeline_state.json`
  - commit 済みの pipeline stage と進捗
- `<profile>/selfplay/*.npz`
  - replay buffer を構成する自己対局 shard。単なる一時テストファイルではありません
- `<profile>/logs/`
  - 言語別の可読ログ、構造化 JSONL、現在の status
- `<profile>/diagnostics/`
  - 戦術 benchmark や収束診断
- `checkpoints/debug-best.pt`
  - 短局実験から公開された診断用モデル。正式な棋力を表しません
- `checkpoints/legacy-debug-best-repetition-v1.pt`
  - 反復局面の裁定修正前に学習した保存用モデル。比較目的にのみ使用します
- `archive/cpu-debug-repetition-v1-*/`
  - 反復局面を2回目で早期終了していた旧 ruleset の非破壊 archive
- `checkpoints/best.pt`
  - 実ルールの Baseline だけが更新できる正式モデル。UI の既定読込先です
- `book_ingest/`
  - ローカルに保存した書籍の読取資料。プログラムの実行には使用しません

Smoke の成果物は削除しても20秒以内に再生成できます。`manual`、`pytest`、
`test-temp` を名前に含むディレクトリは正式な profile ではなく、ここに残さないで
ください。
