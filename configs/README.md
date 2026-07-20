# Configuration

`training/smoke.toml` 是当前CPU环境的短流程验证配置；
`training/cpu_debug.toml` 和 `training/tactical_overfit.toml` 用于短局收敛诊断；
`training/baseline.toml` 是未来CUDA正式训练的起点。

实验profile只发布 `artifacts/checkpoints/debug-best.pt`。只有使用真实规则的Baseline
可以发布桌面UI默认读取的 `artifacts/checkpoints/best.pt`。
