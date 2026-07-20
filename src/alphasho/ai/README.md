# AlphaSho AI

本目录包含与 Qt 无关的将棋 AI：

- `encoding.py`：cshogi/dlshogi 的 119 平面与 2187 动作标签；
- `model.py`：双输入残差策略价值网络与批量推理；
- `mcts.py`：PUCT、根节点噪声、批量叶节点评价和子树推进；
- `player.py`：实现桌面程序 `Player` 接口的 AlphaZero 玩家。

这里没有围棋的 pass、贴目、数子或旋转增强。规则和终局均来自
`alphasho.engine.SearchPosition`。
