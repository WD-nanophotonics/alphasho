# AlphaSho NN 第一代（预览）

冻结7200轮冠军，64,216,967字节，HalfKP 256×2–32–32，FV_SCALE=16。
SHA256：`8835e7a98c790a7e15b36f05384df645dacd806934151cfd9f8c9d73400fe466`。
这是推理网络，不包含训练检查点、优化器、教师权重或训练数据。后台蒸馏继续；后来的冠军不会自动进入此版本。

## 入口与发布边界

- 独立预览：https://wd-nanophotonics.github.io/alphasho-nn-preview/
- 手机基准：https://wd-nanophotonics.github.io/alphasho-nn-preview/benchmark.html
- 正式站：https://wd-nanophotonics.github.io/alphasho/ （验收前保持原版）

选择“人机对战 → AlphaSho NN第一代 → 1—10档”。ABP仍为独立选项，保留原有三档，双人和在线房间不变。网络只在第一次NN思考时下载；下载期间显示进度，成功校验后按SHA256缓存。首次Service Worker接管最多刷新一次，以获得跨源隔离。Worker在一盘棋内复用；悔棋、重开、离开对局或页面后台化会取消思考并丢弃旧回复。回到前台需显式继续NN。

若能力不足、模型损坏、缓存空间不足或Worker失败，界面会显示错误，允许重试或明确切换ABP；不静默替换棋手。更新不自动激活，不覆盖进行中的对局。离线使用需要在线成功加载过模型、引擎及页面。

每盘独立抽取居/振目标各50%；前40半手使用短开局根着引导，被将军放开合法防守，原飞车捕获/升变保留此前路线并结束引导。抽签目标、实际原飞车轨迹和无控制自然比例是三件事。对局面板可导出实际棋局、目标、路线和每手耗时记录。

## 十档（同一网络）

|档|总预算ms|最大深度|MultiPV|最大损失cp|温度cp|
|---|---:|---:|---:|---:|---:|
|1|100|1|8|1000|600|
|2|150|2|8|800|450|
|3|250|3|6|600|300|
|4|400|4|4|400|200|
|5|600|6|4|250|120|
|6|900|8|3|150|80|
|7|1300|12|2|80|40|
|8|2000|16|2|40|20|
|9|3000|64|1|—|—|
|10|5000|128|1|—|—|

低档仅比较同一次搜索、同深度、无上下界的完整候选，按温度抽样；将死分数独立处理。候选不足回退普通推荐着。总预算包括准备、选择和通信，模型加载和预热另计。预留50ms停止，超预算250ms终止Worker。9/10档没有主动放弱。10档不标注未经认证的段位/Elo。相邻档配对只用于筛选，不保证所有相邻差异显著。

## 可复现构建与对应源码

引擎来源：[官方YaneuraOu](https://github.com/yaneurao/YaneuraOu)，V9.00提交`a5ee2786c0030edc7d4a1cdfe94b04dffec55493`，GPLv3。发布二进制旁边包含[完整修补源码](public/nnue/engine/source.tar.gz)、[许可](public/nnue/engine/COPYING.txt)和[构建清单](public/nnue/engine/build.json)。网页项目原有许可不代替引擎的GPLv3许可。

需要Python3.12+、GNU make、官方emsdk **3.1.74**、Node22+。在仓库根目录：

```text
git clone https://github.com/emscripten-core/emsdk.git .cache/emsdk
.cache/emsdk/emsdk install 3.1.74
.cache/emsdk/emsdk activate 3.1.74
python web/scripts/build-nnue.py
python web/scripts/package-nnue-source.py
cd web
npm ci
npm test
npm run build
```

Windows使用`emsdk.bat`，将make所在目录加入PATH或设置`NNUE_MAKE`，可用`EMSDK`指定SDK目录。脚本首次从官方固定提交下载源码到`.cache/yaneuraou-v900`。网络由冻结清单单独提供，构建不生成或替换模型。`build.json`登记引擎和源码压缩包哈希。工具链和源码固定，但不同宿主的绝对路径/工具版本可能影响二进制哈希；发布前仍需重新做整数一致性验收。

必要修补包括：保留USI打入着大写；启用持久USI导出；初始化WASM工作线程；使用与原生相同的NNUE文件名；Threads默认1；缩小内存和pthread池；用原生SSE整数核映射WASM SIMD128。V9旧的专用WASM矩阵核未能通过整数一致性，因此未使用。未修改模型结构、评分尺度或搜索算法。

搜索线程1、pthread池2、置换表16MiB、栈8MiB、初始线性内存192MiB、上限512MiB。浏览器JS下载缓冲及缓存另占内存，线性内存不是总进程内存。关闭定跡和后台预思考。

## 测试与数据

- `nnue-reports/freeze.json`：网络冻结来源。
- `public/nnue/parity.json`：1024合法局面及原生整数评分，包含完整走子历史。
- `public/nnue/benchmark.json`：48固定局面；开局10、中盘12、残局10、被将军8、打入着8。
- `nnue-reports/wasm-parity.json`：1024/1024整数评分一致，零差异。
- `nnue-reports/browser-quick.json`：桌面浏览器24次搜索，包含通信开销。
- `nnue-reports/difficulty/`：冻结赛程、逐手数据、JSON/KIF和进度；未结束前不把中途结果称为通过。
- `nnue-reports/release-acceptance.json`：真实手机验收闸门，当前pending。

完整基准：48局面×6预算×3重复（864次），预算为1万/5万节点及0.1/0.5/2/5秒。引擎NPS使用引擎时间；实际NPS使用完整搜索请求时间，不含模型加载/预热。快速基准只覆盖8局面，不代替完整基准。

后台难度比赛命令（串行，两个单线程WASM实例，断点自动复用完成局）：

```text
node node_modules/tsx/dist/cli.mjs scripts/nnue-matches.ts
node node_modules/tsx/dist/cli.mjs scripts/export-nnue-kif.ts
```

每组相邻档24局，1/5/10交叉各24局，10档对原网页ABP困难档40局，共328局。最多512半手记实验和棋，超时单列。ABP困难档2秒而NN10档5秒，预算不同；不能把结果换算官方段位。Node WASM比赛验证难度筛选，不能冒充手机浏览器性能。校准如需改变参数，最多两轮，必须重新冻结独立验收，不能以已观察的对局追认通过。

## 手机人工验收

在**真实iPhone Safari**及**真实Android Chrome**分别：

1. 打开独立预览基准页，运行完整48局面基准与10分钟压力测试；保持前台，导出JSON，填真实设备型号并勾选真实设备。页面进入后台会取消测试。
2. 选择NN1档与10档各完成至少一盘，导出两份棋局JSON；检查无非法着、崩溃、卡死或每手重复加载。
3. 10档前台选着时间95分位≤5.25秒。手机尺寸模拟只验布局，不算真机通过。
4. 提交设备JSON、两盘棋局JSON及异常说明；在`release-acceptance.json`登记对应文件、设备/引擎/模型哈希和审阅结果。

生产Pages工作流执行`node scripts/check-nnue-release.mjs`。缺少任一手机证据或桌面验收时会拒绝发布。验收通过后才合并发布，检查正式站加载、缓存升级、ABP及在线房间；失败回滚到上一网页构建。

独立预览构建：设置`ALPHASHO_BASE=/alphasho-nn-preview/`后运行`vite build --mode nn-preview`；正式构建默认`/alphasho/`。二者Service Worker作用域隔离，不注册第二个同作用域Worker。
