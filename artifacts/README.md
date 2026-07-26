# 実行時に生成される成果物

このディレクトリには、モデル、自己対局データ、ログなどの実行時成果物を保存します。
この README 以外は Git の管理対象外です。

UI 専用環境で必要なのは次の二つだけです。

- `checkpoints/best.pt`
  - UI が既定で読み込む正式モデルです。
- `models/*.pt`
  - 利用者が比較用 checkpoint を置くモデルライブラリです。対局設定を開くたびに
    ファイル名順で再検出されます。training pipeline はこの directory を変更しません。

学習サーバーでは、これに加えて profile ごとの artifact を保持します。これらを削除すると
学習の再開に必要な状態が失われるため、UI 用コピーへ持ち込まずサーバー側で管理します。

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
Smoke の成果物は削除しても再生成できます。`manual`、`pytest`、`test-temp` を名前に
含む directory は正式な profile ではなく、長期保存しないでください。
