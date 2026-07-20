# 学習 profile

通常の学習では、ルートの `training_pipeline.py` を開いてください。先頭の一画面に
まとめた quick settings が、このディレクトリの profile 既定値を上書きします。
TOML には virtual loss、shard size、weight decay、教師棋譜、ログ周期、
checkpoint 周期など、変更頻度の低い設定を置きます。

設定の優先順位は次のとおりです。

1. profile TOML
2. `training_pipeline.py` の quick settings
3. 実際の適用結果を保存した `effective_config.json`

Smoke、Tactical Overfit、Baseline の PyCharm 実行構成も同じ Python entry point を
使用しますが、対応する TOML を直接読み込み、quick settings は適用しません。
デスクトップ UI はこの設定ディレクトリを import しません。

## Profile 一覧

- `smoke.toml`
  - 非ゼロ value を含む20秒以内の tactical pipeline check
- `cpu_debug.toml`
  - 標準局面を使う CPU convergence experiment
- `tactical_overfit.toml`
  - 固定局面で1枚の駒取りを学ぶ overfit experiment
- `baseline.toml`
  - 実ルールによる production training

`[monitoring]` の `output_language` は `en / zh / ja` を受け付けます。
通常の学習では `training_pipeline.py` の `OUTPUT_LANGUAGE` がこれを上書きします。

`tactical_positions.jsonl` には、先後を入れ替えた戦術局面と、検証済みの合法な駒取り
手を保存しています。
