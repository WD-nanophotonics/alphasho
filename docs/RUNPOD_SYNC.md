# Runpod 训练成果同步

AlphaSho 使用本地 Windows 计划任务主动连接 Runpod。它不依赖 Direct TCP、
VS Code Remote SSH、SCP 或 SFTP，也不要求保留一个打开的 SSH 窗口。

## 同步内容

普通同步每 15 分钟执行一次：

- 云端正式 `artifacts/checkpoints/best.pt`
- 当前 `artifacts/baseline/checkpoints/candidate.pt`
- `pipeline_state.json` 和 `effective_config.json`
- `console.log`、`baseline.jsonl`、`status.json` 和当前多语言日志的最后 8 MiB
- 云端 Git commit、20 局评估配置、当前阶段、iteration、比分和模型 SHA-256

本地文件保存在：

```text
artifacts/cloud-sync/latest/
├── manifest.json
├── models/
│   ├── best.pt
│   └── candidate.pt
├── state/
└── logs/
```

只有云端正式 `best.pt` 的 SHA-256 发生变化时，同步器才会原子更新本地
`artifacts/checkpoints/best.pt`。`candidate.pt` 永远不会覆盖正式模型；需要调试候选模型时，
在 UI 模型选择器中浏览
`artifacts/cloud-sync/latest/models/candidate.pt`。

模型没有变化时，云端不会重复发送模型；新的快照会继承并校验本地已有副本。

## 首次安装

在本地项目的 PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --install-runpodctl
```

这会从 Runpod 官方发布地址下载 Windows `runpodctl.exe` 到：

```text
%LOCALAPPDATA%\AlphaSho\tools\runpodctl.exe
```

随后先执行一次人工同步：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --once
```

确认成功后安装每 15 分钟运行的用户级任务：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --install-task
```

任务名称是 `AlphaSho Runpod Sync`。它使用当前 Python 环境和当前项目路径，不需要管理员
权限。Pod 离线或网络中断时，本轮会重试三次并保留上一次完整快照。

## 状态与日志

查看最近一次同步状态：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --status
```

本地状态和历史错误位于：

```text
artifacts/cloud-sync/sync_status.json
artifacts/cloud-sync/sync.log
```

同步过程先在临时目录接收文件，再检查安全路径、文件清单、大小和 SHA-256。全部通过后才
切换 `latest/`。SSH、传输、解包或校验失败都不会覆盖上一次快照或 UI 正式模型。

## 完整恢复包

关停 Pod、迁移机器或做阶段性归档前执行：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --full-backup
```

完整包保存在 `artifacts/cloud-sync/full-backups/`，包含整个
`artifacts/baseline/`，包括 `training_state.pt`、完整日志和 `selfplay/*.npz`，以及正式
发布模型。它可能很大，不适合每 15 分钟执行。

## 配置和移除

非敏感设置位于 `configs/runpod_sync.toml`。Pod 更换后只需修改：

```toml
[connection]
ssh_target = "<新的 Pod SSH proxy ID>@ssh.runpod.io"
remote_project_root = "/workspace/alphasho-v3"
```

专用密钥仍放在用户 `.ssh` 目录，不加入 Git。移除计划任务：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --uninstall-task
```

源码继续以 GitHub `origin/master` 为正本；此工具只同步训练产物，不负责部署源码，也不
引入 Colab、Google Drive 或云存储凭据。
