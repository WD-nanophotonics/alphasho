# AlphaSho

AlphaSho は、将棋の対局用デスクトップアプリケーションと、AlphaZero 方式の
自己対局学習基盤を統合したプロジェクトです。

盤面、合法手、成り、持ち駒、王手、終局判定、SFEN/USI などの将棋ルールは
[cshogi](https://github.com/TadaoYamaoka/cshogi) を唯一の正として扱います。
AlphaSho 側は、安全なルールアダプター、対局制御、UI、MCTS、方策価値ネットワーク、
自己対局および学習パイプラインを担当します。

## 動作環境とセットアップ

- Python 3.13
- Windows 11
- PySide6
- cshogi 1.0.4
- NumPy / PyTorch（学習時のみ）

PyCharm でプロジェクトを開き、インタープリターとして
`.venv\Scripts\python.exe` を選択してください。新しい環境を作る場合は、プロジェクト
ルートで次を実行します。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,training]"
```

現在のローカル環境では CPU 版 PyTorch を使用しています。本格的な学習を行う場合は、
利用する GPU に対応した CUDA 版 PyTorch を別途導入してください。

## デスクトップ UI の起動

PyCharm の共有実行構成から **AlphaSho UI** を選び、Run をクリックします。
ルートの `main.py` を直接実行しても起動できます。コマンドライン引数は不要です。

現在の主な機能は次のとおりです。

- 人間同士、人間対ランダム AI、ランダム AI 同士の対局
- 学習済み AlphaZero モデルを先手または後手に指定した対局
- 通常の移動、駒取り、持ち駒の打ち、任意成りおよび強制成り
- 盤上の駒画像を使った非モーダルな成り選択
- 先手・後手の対角配置の駒台と盤面反転
- 日本語・中国語・英語の UI（初回起動時は日本語）
- 合法手、直前の指し手、選択中の駒、王手のハイライト
- 待った、投了、入玉宣言、対局のやり直し
- 主時間＋秒読み、Fischer 加算、時間無制限
- SFEN のコピー・読込、KIF/KIFU・CSA 棋譜の読込と保存
- 詰み、千日手、連続王手の千日手、時間切れ、手数上限の処理

言語と盤面の向きは `QSettings` に保存され、次回起動時に復元されます。
ランダム AI は Player インターフェースの動作確認用であり、棋力を目的としたものでは
ありません。

## AlphaZero 実装

将棋向け AI は cshogi/dlshogi と互換性のある形式を使用します。

- 入力：`62 × 9 × 9` と `57 × 9 × 9`、合計119特徴平面
- 方策出力：`27 × 81 = 2187` の固定行動空間
- モデル：二入力の Residual Policy-Value Network
- 探索：合法手マスク付き PUCT MCTS、root Dirichlet noise、batch inference
- データ：圧縮された自己対局 shard と `ruleset_id` で分離された replay buffer
- 学習：任意の KIF/CSA 教師あり warm-up、自己対局、候補モデル学習、評価、昇格

対局設定では、正式モデル `artifacts/checkpoints/best.pt`、CPU Debug の
`debug-best.pt`、現在の候補モデル、または任意の `.pt` ファイルを選択できます。
AI の探索は Qt worker thread で実行されるため、UI をブロックしません。
局面変更後に返ってきた古い探索結果は適用されません。

## PyCharm からの学習

共有実行構成はすべてルートの
[`training_pipeline.py`](training_pipeline.py) を使用します。

- **AlphaSho Training**：CPU 上で行う短局の収束確認
- **AlphaSho Training Tactical Overfit**：対称な戦術局面で駒取りを学習できるか確認
- **AlphaSho Training Smoke**：20秒の hard timeout を持つ最小パイプライン確認
- **AlphaSho Training Baseline**：実ルールによる本格学習用の入口

`training_pipeline.py` の先頭には、頻繁に変更する設定だけを一画面にまとめています。

- Run
- Short-game rules
- MCTS / self-play
- Network training
- Evaluation / promotion

`TRAINING_BATCH_SIZE` と `INFERENCE_BATCH_SIZE` は別の設定です。
モデル規模は `smoke / baseline / large` から選択します。
`OUTPUT_LANGUAGE` は `en / zh / ja` に変更でき、チェックポイントや学習データとの
互換性には影響しません。

初回は **AlphaSho Training Smoke** を実行してください。自己対局、ネットワーク更新、
候補評価、チェックポイント保存までの最小経路を確認できます。

CPU Debug では学習ループの収束を短時間で診断するため、実際の将棋とは異なる短局裁定を
使用できます。標準設定では、一方が先に一定枚数を取ると勝ちとなり、最大手数に達した
場合は次の連続 value を使用します。

```text
tanh(先手の駒得差 / 12)
```

この短局モデルは学習基盤の検証専用で、実戦棋力を表すものではありません。
`baseline.toml` は実ルールのみを使用し、正式モデルだけが
`artifacts/checkpoints/best.pt` を更新できます。

## ログ、再開、チェックポイント

各 profile は独立した `artifacts/<profile>/` を使用します。

- `logs/training.en.log`：英語の可読ログ
- `logs/training.zh.log`：中国語の可読ログ
- `logs/training.ja.log`：日本語の可読ログ
- `logs/<profile>.jsonl`：言語非依存の構造化イベント
- `logs/status.json`：現在の stage、event、message key、進捗
- `effective_config.json`：実際に適用された最終設定
- `checkpoints/training_state.pt`：モデル、optimizer、乱数状態
- `checkpoints/pipeline_state.json`：commit 済みの pipeline stage と進捗

可読メッセージは `src/alphasho/training/messages.py` で一元管理されています。
Trainer は英語の `stage`、`event`、`message_key` と構造化 fields を記録します。

PyCharm Stop または `Ctrl+C` を受けると、完了済みの対局、training step、評価結果を
保存します。未完了の1局だけを破棄し、次回は保存済み stage から再開します。
同一 profile は process lock により同時起動できません。Windows の一時的なファイル
占有に対しては、上限付き retry と revision ベースの temporary-state recovery を
行います。

## プロジェクト構成

詳細な設計と運用については、次の文書も参照してください。

- [アーキテクチャ](docs/ARCHITECTURE.md)
- [学習運用ガイド](docs/TRAINING.md)

```text
alphasho/
├── main.py                    # デスクトップ UI の入口
├── training_pipeline.py       # 一画面の学習設定
├── configs/training/          # profile と高度な学習設定
├── assets/pieces/hitomoji/    # 駒画像
├── src/alphasho/
│   ├── domain/                # UI・cshogi 非依存の型
│   ├── engine/                # cshogi adapter と対局状態
│   ├── players/               # Human / Random Player
│   ├── ui/                    # PySide6 UI
│   ├── ai/                    # encoding、network、MCTS、AI Player
│   └── training/              # self-play、replay、Trainer、evaluation
├── tests/                     # ルール、AI、学習、UI の回帰テスト
└── artifacts/                 # Git 管理外のモデル、データ、ログ
```

## 開発用コマンド

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,training]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

## ライセンスと素材

`cshogi` は GPL-3.0 で公開されています。本プロジェクトを公開または再配布する前に、
cshogi と組み合わせた成果物のライセンス条件を確認してください。

駒画像はユーザー提供の `hitomoji.zip` から取り込まれましたが、元アーカイブには
明確なライセンスファイルがありません。画像の出典と再配布許可を確認するまでは、
第三者への公開を避けてください。
