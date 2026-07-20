# 設定ファイル

学習 profile は `training/` に保存されています。

- `training/smoke.toml`
  - 現在の CPU 環境で最小経路を確認する短時間 profile
- `training/cpu_debug.toml`
  - 標準局面から短局裁定を使って収束を確認する CPU profile
- `training/tactical_overfit.toml`
  - 固定した戦術局面で駒取り方策を学習できるか確認する profile
- `training/baseline.toml`
  - 実ルールと CUDA 環境を想定した本格学習の出発点

実験 profile が公開できるのは `artifacts/checkpoints/debug-best.pt` だけです。
UI が既定で読み込む `artifacts/checkpoints/best.pt` は、実ルールを使う Baseline
のみ更新できます。
