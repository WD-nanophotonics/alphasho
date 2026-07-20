# AlphaSho

AlphaSho 目前是一个可玩的将棋桌面程序，也是后续 AlphaZero 风格自我对弈训练项目的基础。
规则、合法手和终局判定由
[cshogi](https://github.com/TadaoYamaoka/cshogi) 提供；应用层负责安全适配、对局流程和 UI。

## 在 PyCharm 中运行

1. 打开项目后确认解释器为 `.venv\Scripts\python.exe`（Python 3.13）。
2. 在右上角选择共享配置 **AlphaSho UI**。
3. 点击 Run。

也可以直接右键根目录的 `main.py` 并选择 Run。程序不需要命令行参数。

## 当前功能

- 本地双人和人类对随机合法手 AI；
- 普通移动、吃子、持驹打入、可选/强制升变；
- 棋盘旁贴图式升变选择器，不需要来回操作确认窗口；
- 对角双驹台、手动棋盘翻转，以及日本語/中文/English 三语；
- 合法位置、上一手、选中棋子和王手高亮；
- 悔棋、重开、认输、入玉宣胜；
- SFEN 复制与载入；
- 将死、四次同局面的千日手和连续王手千日手判定。

首次启动使用日本語。语言与棋盘视角会自动记住；可以随时在顶部工具栏切换。

随机 AI 仅用于验证 Player 接口，不代表棋力。项目现在也包含：

- cshogi/dlshogi 兼容的119平面输入与2187动作标签；
- 双输入残差策略价值网络、批量 PUCT MCTS 和 AlphaZero 玩家；
- 可选 KIF/CSA 监督预热、自我对弈、磁盘回放池、训练、评估和最佳模型晋级。

对局设置中的双方都可以选择 **AlphaZero（学習済みモデル）**，模型来源可选正式
`best.pt`、CPU Debug最佳、当前候选或任意 `.pt` 文件。缺失模型会在开始前拦截；
AlphaZero思考在后台线程运行，不会阻塞窗口。实验模型只用于接口和收敛验证，不代表棋力。

## 在 PyCharm 中训练

- **AlphaSho Training**：当前CPU环境的短局收敛实验，直接运行根目录
  [`training_pipeline.py`](training_pipeline.py)；
- **AlphaSho Training Tactical Overfit**：从18个对称战术局面检验能否学会吃子；
- **AlphaSho Training Smoke**：数秒级、20秒硬超时的非零value流程检查；
- **AlphaSho Training Baseline**：真实规则的正式云端训练入口。

`training_pipeline.py` 的第一屏就是用户常调参数，分为 Run、Short-game rules、
MCTS / self-play、Network training、Evaluation / promotion 五组。训练 batch 和推理
batch 使用不同名称；`OUTPUT_LANGUAGE = "en"` 可切换为 `zh` 或 `ja`，
且不影响训练状态或数据兼容性；
模型规模只需选择 `smoke / baseline / large`。自动保存和恢复不占用这个参数区。

第一次请先运行 Smoke。它会完成自我对弈、一次网络更新、候选评估和 checkpoint 保存；
该共享运行配置内置 20 秒 watchdog，异常变慢时会安全停止并报错。
CPU Debug 默认从标准局面运行64手，某方累计先吃3枚棋子即胜；若走满则使用
`tanh(材料差 / 12)` 作为连续value。这个规则只用于验证闭环能否收敛，不代表正式棋力。

正式训练前应为当前 Python 3.13 环境安装匹配的 CUDA Torch，并在
`configs/training/baseline.toml` 中填写可选的 KIF/CSA 棋谱目录。

每个配置使用独立的 `artifacts/<profile>/` 目录。CPU、战术和Smoke模型发布为
`artifacts/checkpoints/debug-best.pt`；只有Baseline会更新正式的
`artifacts/checkpoints/best.pt`。训练被中断后，再次点击同一运行配置会从
`training_state.pt` 和 `pipeline_state.json` 继续。修改 playout、两种 batch、
buffer、每轮局数或学习率后可以继续恢复；若修改模型结构，Trainer 会明确拒绝
不兼容的 checkpoint。

训练进度会持续显示在 PyCharm Terminal，同时写入：

- `artifacts/<profile>/logs/training.en.log`：英文可读日志；选择中文或日语时分别写入
  `training.zh.log` 或 `training.ja.log`，三者独立轮转；
- `artifacts/<profile>/logs/<profile>.jsonl`：连续、语言无关的结构化事件与指标，
  同时保留当次渲染文本；
- `artifacts/<profile>/logs/status.json`：当前事件键、语言、阶段与最近进度；
- `artifacts/<profile>/effective_config.json`：本次真正生效的最终配置。

自我对弈、网络训练和候选评估都会输出进度、速度和 ETA；长步骤默认每 10 秒输出
heartbeat。PyCharm Stop 或 `Ctrl+C` 会保存最近完成的局、训练 step 和评估比分，
未完成的单局会被丢弃并在恢复后重做。

日志句子由 `src/alphasho/training/messages.py` 集中维护。Trainer 只写英文
`stage`、`event`、`message_key` 和结构化 fields；因此修改中日英模板不会改动训练
流程，也可以根据 JSONL 数据重新渲染展示。

同一profile同时只能运行一个Trainer。模型和状态先写入完整临时文件，再使用有界重试
提交；Windows短暂占用文件不会立即终止训练。若进程在提交期间退出，下次启动会恢复
revision更新的合法临时状态。修改检查频率也不会跳过已经开始的候选评估。

实验日志还会记录真实终局、吃子阈值、材料截断、平均吃子数、材料差、非零value比例，
以及固定战术局面的吃子top-1准确率和策略概率。不同短局规则具有独立 `ruleset_id`，
修改阈值或局面池后不会混入旧的自我对弈分片。

## 开发命令

以下命令都明确使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

当前安装的 Torch 是 CPU 版本。GPU/CUDA 环境应在进入正式训练阶段后再单独配置。

## 第三方许可

`cshogi` 以 GPL-3.0 发布。棋子图片来自用户提供的 `hitomoji.zip`，压缩包内没有发现明确的
授权文件。公开发布或再分发本项目之前，需要分别确认代码组合方式和图片再发布权限。
