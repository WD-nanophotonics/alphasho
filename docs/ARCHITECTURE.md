# AlphaSho アーキテクチャ

この文書は、プロジェクトを引き継ぐ開発者が「どの layer が何を所有し、どこから変更
すべきか」を短時間で把握するためのものです。

## 設計原則

1. **将棋ルールの唯一の正は cshogi**
   - 合法手、成り、持ち駒、王手、千日手、入玉、SFEN/USI を独自実装しません。
2. **UI と学習コードは cshogi の内部 move integer を共有しない**
   - module 境界では `MoveOption` と USI 文字列を使用します。
3. **局面の所有者を明確にする**
   - UI 対局では `ShogiGame`、探索では `SearchPosition` がそれぞれ独立した
     `cshogi.Board` を所有します。
4. **短局実験と正式な将棋モデルを分離する**
   - `ruleset_id`、artifact directory、公開 checkpoint 名を分けます。
5. **長時間処理を UI thread で実行しない**
   - AlphaZero 探索は worker thread で実行し、古い結果は generation token で破棄します。

## Layer の責務

| Layer | 主な責務 | 依存してよいもの |
|---|---|---|
| `domain` | immutable な型、enum、Player interface | Python 標準ライブラリ |
| `engine` | cshogi adapter、対局状態、棋譜、探索局面 | `domain`、cshogi |
| `players` | Human / Random Player | `domain` |
| `ui` | PySide6 widget、dialog、対局 controller | `domain`、`engine`、`players`、`ai` |
| `ai` | encoding、network、MCTS、AlphaZero Player | `domain`、`engine`、Torch、cshogi |
| `training` | self-play、replay、optimization、evaluation | `ai`、`engine`、Torch |

`domain` と `engine` は Qt を import しません。`training` はデスクトップ UI から
独立しており、UI 起動時に Trainer や replay data を読み込みません。

## UI で1手を指す流れ

```text
BoardWidget / KomadaiWidget
        │ 盤上 square または持ち駒を選択
        ▼
GameController
        │ 現局面の MoveOption と照合
        │ 任意成りの場合は盤上 selector を表示
        ▼
ShogiGame.play(USI)
        │ 現在の legal_moves に再照合
        ▼
cshogi.Board.push()
        │
        ▼
GameSnapshot を生成 → UI を再描画 → 次の Player を schedule
```

UI から渡された未検証の値を `cshogi.Board.push()` に直接送ってはいけません。
`ShogiGame` が合法手との照合を終えた後だけ push します。

## AlphaZero が1手を選ぶ流れ

```text
GameSnapshot + legal MoveOption
        ▼
SearchPosition を履歴付きで再構築
        ▼
119-plane encoding + legal policy mask
        ▼
Policy-Value Network
        ▼
PUCT MCTS
        ▼
visit count から USI を選択
        ▼
GameController が generation と SFEN を再確認
        ▼
ShogiGame.play(USI)
```

network の value は常に「現局面で手番を持つ側」の視点です。tree backup では ply ごとに
符号を反転します。方策 label は `make_move_label(move, board.turn)` によって生成し、
合法手 mask の外側を選択できないようにします。

## 学習 data flow

```text
best.pt
   ▼
MCTS self-play
   ▼
compressed NPZ shards ── ruleset_id
   ▼
disk replay window
   ▼
candidate training
   ▼
candidate.pt
   ▼
paired-color evaluation
   ├─ score < threshold → best を維持
   └─ score ≥ threshold → best.pt を更新
```

短局 profile の公開先は `debug-best.pt` です。実ルールの Baseline だけが UI の正式な
既定モデル `best.pt` を更新できます。

## 変更時に守る invariant

- `GameSnapshot` と `MoveOption` は UI・Player 間で安全に渡せる値 object とする。
- SFEN、USI、棋譜履歴は表示言語を変更しても変換しない。
- `push()` と `pop()` の後で SFEN、手番、履歴、実験用駒取り count が完全に復元される。
- checkpoint は model/optimizer を先に保存し、pipeline state を最後に commit する。
- 同一 profile の Trainer を複数 process から同時に書き込まない。
- short-game data を real-rules replay に混ぜない。
- UI の言語 catalog と training log の言語 catalog は独立して維持する。

## 拡張ポイント

### 新しい Player

`choose_move(snapshot, legal_moves)` を実装し、`ParticipantKind` と対局 dialog の選択肢を
追加します。Player から `ShogiGame` や Qt widget を直接操作しません。

### 新しい棋譜形式

parser で読み込んだ指し手を USI に変換し、必ず `ShogiGame` を通して1手ずつ合法性を
検証します。失敗した場合は、現在の対局を変更せずに読み込み全体を中止します。

### 新しい network

input encoding と action space の互換性を checkpoint metadata に明記してください。
構造が異なる checkpoint を暗黙に読み込まず、起動時に明確なエラーを返します。

