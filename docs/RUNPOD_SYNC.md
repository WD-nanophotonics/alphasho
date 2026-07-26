# Runpod 训练成果同步

AlphaSho 提供一个本地 Windows 控制窗口，主动连接已经由用户开启的 Runpod。它不依赖
Direct TCP、VS Code Remote SSH、SCP 或 SFTP，也不要求保留一个打开的 SSH 窗口。
控制程序不会调用 Runpod API，也不会启动、停止或计费管理 Pod。

## 日常使用：本地控制程序

先在 Runpod 网页中启动 Pod，然后在项目根目录双击静默启动入口：

```text
runpod_control.vbs
```

这个入口直接隐藏启动 `.venv\Scripts\pythonw.exe`，启动和后台检查都不会弹出命令窗口。
需要观察启动错误时，可以在项目终端运行：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_control_app
```

程序刚打开时不会自动联网、启动训练或开启同步。推荐的操作顺序是：

1. 点击“检查连接”，确认 Pod、GPU、远端 Git 和训练状态。
2. Pod 重启或训练已暂停时，点击“启动/恢复训练”。程序只在配置的 tmux session 中启动
   baseline，并会拒绝创建重复训练进程。
3. 点击“开启每15分钟同步”。程序会立即同步一次，之后从每次同步完成时起计算15分钟。
4. 不需要看窗口时直接关闭；程序会缩到系统托盘并继续已开启的同步。托盘菜单中的“真正
   退出”才会停止本地监控，云端 tmux 训练不会受到影响。

“安全暂停训练”会先让训练器保存 checkpoint。只有安全暂停超时且用户再次确认时，程序才
允许强制结束配置的 tmux session。控制程序只比较本地和远端 Git，不会自动上传、拉取或
修改源码。

## 同步内容

开启定时同步后，普通同步每 15 分钟执行一次：

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

## 首次准备

在本地项目的 PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --install-runpodctl
```

这会从 Runpod 官方发布地址下载 Windows `runpodctl.exe` 到：

```text
%LOCALAPPDATA%\AlphaSho\tools\runpodctl.exe
```

控制程序会复用这个 `runpodctl.exe`。也可以通过 CLI 执行一次人工同步来排查传输问题：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --once
```

控制程序本身不会创建 Windows 登录启动任务。如果明确需要无人值守的登录后自动同步，
仍可选择安装旧的用户级计划任务：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --install-task
```

任务名称是 `AlphaSho Runpod Sync`。不要同时开启控制程序的定时同步和计划任务；文件锁会
阻止数据损坏，但重复调度没有必要。Pod 离线或网络中断时，本轮会重试三次并保留上一次
完整快照。

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

训练控制设置位于同一文件的 `[control]`，包括 tmux session、训练 profile、日志路径、状态
文件和轮询/暂停超时。所有远端文件路径必须相对于 `remote_project_root`，不能使用 `..`。

专用密钥仍放在用户 `.ssh` 目录，不加入 Git。移除计划任务：

```powershell
.\.venv\Scripts\python.exe -m alphasho.runpod_sync `
  --config configs\runpod_sync.toml --uninstall-task
```

源码继续以 GitHub `origin/master` 为正本；此工具只同步训练产物，不负责部署源码，也不
引入 Colab、Google Drive 或云存储凭据。
