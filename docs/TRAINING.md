# 学習運用ガイド

この文書は、CPU での pipeline 確認から CUDA 環境での本格学習へ移行する際の運用手順を
まとめたものです。

## 最初に実行する profile

1. **Smoke**
   - 最小の戦術局面で自己対局、学習、評価、checkpoint を確認します。
   - 20秒の hard timeout を超えた場合は異常として停止します。
2. **Tactical Overfit**
   - 明確な駒取り手の policy probability が上昇するか確認します。
3. **CPU Debug**
   - 標準局面と短局裁定で、複数 iteration の replay と昇格 loop を確認します。
4. **Baseline**
   - CUDA 環境で実ルールの本格学習を行います。CPU での長時間実行は推奨しません。

Smoke、Tactical Overfit、CPU Debug は仕組みの診断用であり、生成されたモデルの棋力を
保証しません。

## Quick settings

通常はルートの `training_pipeline.py` の先頭だけを編集します。

| 設定 | 意味 | 調整時の注意 |
|---|---|---|
| `PROFILE` | TOML の基本 profile | artifact と resume state の単位 |
| `DEVICE` | `cpu`、`cuda`、`auto` | CUDA Torch がない状態で `cuda` を指定しない |
| `OUTPUT_LANGUAGE` | `en`、`zh`、`ja` | 表示だけを変更し、学習 data には影響しない |
| `ITERATIONS` | self-play から評価までの反復回数 | resume 時に増やすことができる |
| `MODEL_PRESET` | `smoke`、`baseline`、`large` | checkpoint と構造互換性が必要 |
| `START_POSITION_MODE` | `standard` または `tactical` | 局面集合は `ruleset_id` に含まれる |
| `ADJUDICATION_MODE` | `real_rules` または `capture_material` | Baseline は `real_rules` 固定 |
| `MAX_MOVES` | 1局の最大 ply | 1 ply は一方が1回指すこと |
| `CAPTURE_THRESHOLD` | 先に取った駒数による勝利条件 | 診断用。変更すると ruleset が変わる |
| `PLAYOUTS` | 1手あたりの MCTS simulation 数 | 棋力と計算量に最も強く影響する |
| `C_PUCT` | prior と探索回数の balance | 大きいほど未探索手を選びやすい |
| `INFERENCE_BATCH_SIZE` | MCTS leaf の推論 batch | GPU utilization に関係する |
| `GAMES_PER_ITERATION` | iteration ごとの自己対局数 | replay の多様性と所要時間に関係する |
| `DIRICHLET_ALPHA` | root noise の形状 | 将棋用の既定値を起点に調整する |
| `DIRICHLET_WEIGHT` | prior に混ぜる noise の割合 | 評価対局では使用しない |
| `TRAINING_BATCH_SIZE` | optimizer step の sample 数 | inference batch とは別設定 |
| `REPLAY_BUFFER_SIZE` | 現 ruleset から保持する最大 sample 数 | disk shard 自体の削除数ではない |
| `TRAIN_STEPS_PER_ITERATION` | iteration ごとの更新 step | data 量との balance を確認する |
| `LEARNING_RATE` | optimizer の学習率 | resume 時の変更をログに記録する |
| `CHECK_FREQUENCY` | 候補評価を行う iteration 間隔 | 進行中の評価は変更後も最後まで再開する |
| `EVALUATION_GAMES` | candidate 対 best の対局数 | 先後を組にするため偶数にする |
| `PROMOTION_SCORE` | best model への昇格 score | 引き分けは0.5として計算する |

channel 数、residual block 数、weight decay、checkpoint 周期などは
`configs/training/*.toml` に置きます。

## 短局裁定

終局判定の優先順位は次のとおりです。

1. cshogi による詰み、千日手、連続王手、入玉
2. `capture_material` で一方が先に `CAPTURE_THRESHOLD` に到達
3. `MAX_MOVES` で打ち切り

最大手数では、盤上と持ち駒の駒得差から連続 value を生成します。

```text
black_value = tanh((black_material - white_material) / 12)
```

各 training sample には、その局面で手番を持つ側から見た符号へ変換して保存します。
候補評価では、最大手数時の駒得差の正負を勝敗に変換し、完全に同点の場合だけ引き分けと
します。

## 1 iteration の流れ

```text
self-play
  → shard を atomic write
  → replay window を読込
  → candidate network を更新
  → candidate.pt を保存
  → tactical benchmark
  → check frequency に応じて candidate 対 best を評価
  → threshold 到達時だけ best を更新
  → pipeline state を commit
```

`CHECK_FREQUENCY` に達しない iteration では評価を省略します。途中で中断された評価は、
次回起動時に frequency の設定にかかわらず続きから実行します。

## ログの読み方

Terminal と言語別 text log には、次の情報が継続的に出力されます。

- self-play の局数、ply、playouts/s、ETA、終局理由
- replay sample 数、非ゼロ value 比率、平均駒取り数、平均駒得差
- policy/value loss、entropy、KL、gradient norm、samples/s
- candidate/best の勝敗、累積 score、昇格 threshold
- checkpoint retry、temporary state recovery、process lock conflict

`<profile>.jsonl` は機械処理用の正規ログです。`stage`、`event`、`message_key` と
構造化 fields を使用し、可読メッセージだけを選択言語で rendering します。

## 中断と再開

- PyCharm Stop や `Ctrl+C` は `request_stop()` を設定します。
- 完了した対局、training step、評価 score を checkpoint に保存します。
- 未完了の自己対局1局は破棄し、次回同じ番号からやり直します。
- 同一 profile の二重起動は `.trainer.lock` で拒否します。
- model preset や input encoding が異なる checkpoint は読み込みを拒否します。
- playouts、batch size、buffer size、game 数、learning rate などの変更は許可し、
  前回値との差を startup log に記録します。

## CUDA / cloud への移行

1. Python 3.13 と cshogi が利用できる環境を準備します。
2. GPU と driver に合う CUDA 版 PyTorch を公式手順で導入します。
3. repository を clone し、必要な profile artifact を安全な storage からコピーします。
4. `DEVICE = "cuda"` または `"auto"` を設定します。
5. Smoke を実行して encoding、MCTS、checkpoint write を確認します。
6. Baseline の `PLAYOUTS`、batch、games/iteration を GPU memory と throughput に合わせます。
7. `artifacts/` を定期的に永続 storage へ backup します。

`artifacts/` は Git 管理外です。GitHub へ push しただけでは、モデル、optimizer、
replay shard、学習進捗は保存されません。

## よくある問題

### 長時間ログが出ない

`heartbeat_seconds` と process の CPU/GPU 使用率を確認してください。同じ profile の別
process が動いていないか、`.trainer.lock` のログも確認します。

### Replay が空

現 `ruleset_id` と一致する self-play shard が必要です。threshold や初期局面集合を
変更した直後は、過去 shard が意図的に除外されます。

### Candidate が昇格しない

非ゼロ value 比率、policy entropy、tactical benchmark、candidate 対 best の色別結果を
確認します。短い評価対局だけで棋力を判断しないでください。

### UI でモデルを選べない

checkpoint metadata と現在の model preset が一致するか確認してください。
短局モデルは `debug-best.pt`、正式モデルは `best.pt` を使用します。
