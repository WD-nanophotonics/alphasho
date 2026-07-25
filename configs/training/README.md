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

## 並列探索

`[parallel]` は複数の対局プロセスから一つの GPU 推論サーバーへ局面を集約します。
`selfplay_workers` と `evaluation_workers` の `0` は自動選択で、最大8プロセスを使い、
CPUを2論理コア残します。`gpu_batch_size` は集約後の上限、`batch_wait_ms` は追加要求を
待つ短い時間です。単一の木に複数スレッドから触れることはなく、モデルも worker
プロセスにはコピーされません。

Baseline は自動並列、最大 GPU batch 128 を使用します。Smoke は問題の切り分けと
20秒制限を安定させるため、1 worker のままです。

## Throughput benchmark

Production の replay、pipeline state、公開モデルを変更せず、同じ検索実装の
throughput を測定する場合は、独立した benchmark entry point を使用します。

```bash
python -m alphasho.training.throughput_benchmark \
  --config configs/training/baseline.toml \
  --checkpoint artifacts/checkpoints/best.pt \
  --mode selfplay \
  --games 32 \
  --workers 12 \
  --gpu-batch-size 128 \
  --batch-wait-ms 2 \
  --output artifacts/benchmarks/rtx3090/run-001/workers-12
```

出力先が profile の正式 `artifact_root` 内にある場合、実行は拒否されます。
各 output directory は新規でなければならず、`report.json` と `events.jsonl` に
playouts/s、GPU batch、CPU/GPU sampling、checkpoint hash、Git commit を保存します。
`--mode evaluation` は同じ checkpoint を candidate と best の両方に読み込み、
正式評価と同じ先後ペアと二モデル routing を測定します。
