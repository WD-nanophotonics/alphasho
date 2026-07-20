# Runtime artifacts

这里全部是运行产物，除本说明外均被 Git 忽略。删除某个未在运行的profile目录不会影响
源码，但会失去该profile的训练进度。

- `cpu-debug/`、`tactical-overfit/`、`smoke-short/`、`baseline/`：相互隔离的训练；
- `<profile>/checkpoints/best.pt`：该profile当前用于自我对弈的最佳模型；
- `<profile>/checkpoints/candidate.pt`：最近训练的候选模型；
- `<profile>/checkpoints/training_state.pt`：网络、优化器和随机数恢复状态；
- `<profile>/checkpoints/pipeline_state.json`：已提交的阶段和进度；
- `<profile>/selfplay/*.npz`：回放池样本，训练时需要，不是临时测试文件；
- `<profile>/logs/` 与 `diagnostics/`：按语言分开的可读日志、统一 JSONL 事件流和
  收敛诊断；
- `checkpoints/debug-best.pt`：短局实验发布模型，不代表正式棋力；
- `checkpoints/best.pt`：只允许真实规则Baseline发布，桌面UI默认读取它；
- `book_ingest/`：保留的本地书籍阅读资料，不参与程序运行。

Smoke输出可随时删除并在20秒内重新生成。名称包含 `manual`、`pytest` 或 `test-temp`
的目录不是正式训练profile，不应出现在本目录。
