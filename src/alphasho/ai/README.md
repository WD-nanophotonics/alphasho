# AlphaSho AI

この package には、Qt に依存しない将棋 AI の実装を配置します。

- `encoding.py`
  - cshogi/dlshogi 互換の119特徴平面と2187行動 label
  - 現局面の合法手から policy mask を生成
- `model.py`
  - 二入力 Residual Policy-Value Network
  - CPU/CUDA の batch inference と inference checkpoint
- `mcts.py`
  - PUCT、root Dirichlet noise、batch leaf evaluation、subtree reuse
  - cancellation と progress callback
- `player.py`
  - デスクトップアプリケーションの `Player` interface を実装する
    `AlphaZeroPlayer`

囲碁固有の pass、コミ、地の計算、rollout、8方向の回転・反転 data augmentation は
使用しません。合法手、局面履歴、終局判定は
`alphasho.engine.SearchPosition` と cshogi に委譲します。

UI との境界では `GameSnapshot`、`MoveOption`、USI 文字列だけを使用し、cshogi の
内部 move integer を公開しません。
