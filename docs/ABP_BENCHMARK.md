# ABP strength benchmark

この benchmark は heuristic ABP の変更を比較するための開発用 tool です。人間、MCTS、
外部 USI engine に対する棋力を保証するものではありません。

## 比較条件

- control は commit `3262cc8` の `HeuristicPlayer` を benchmark directory に固定したものです。
- opening は `configs/training/evaluation_positions.jsonl` の10局面を使用し、同一局面で双方の
  player color を交換します。
- 既定は1秒/手、最大256手です。5秒確認は局面数を明示的に減らして実行します。
- checkmate、repetition、perpetual check、nyugyoku など `ShogiGame` が確定した結果だけを
  勝敗または rule draw とします。手数上限は `unresolved` であり、得点に含めません。
- candidate または control が fallback を使った対局、または予算を
  `max(50ms, 5%)` より超過した対局は棋力得点から除外します。

```powershell
# 正式な1秒 paired run。出力先は自動作成される
.\.venv\Scripts\python.exe benchmarks\heuristic_strength.py `
  --seconds 1 --positions 10 --max-plies 256

# 5秒の小規模確認
.\.venv\Scripts\python.exe benchmarks\heuristic_strength.py `
  --seconds 5 --positions 4 --max-plies 256

# 全消融。非常に時間がかかるため自動 test では実行しない
.\.venv\Scripts\python.exe benchmarks\heuristic_strength.py --all-profiles

# 敗局の critical position を旧版・新版で1秒/5秒再探索
.\.venv\Scripts\python.exe benchmarks\heuristic_strength.py --diagnose-losses

# 既存 audit の先頭2 opening（勝局を含む）を再探索し、元 run 内へ診断 JSON を保存
.\.venv\Scripts\python.exe benchmarks\diagnose_existing_audit.py `
  artifacts\reports\abp-incremental-priority-gate-20260802 --openings 2
```

`--profile` は `legacy`、`tt_pv`、`q_lite`、`positional_lite`、
`king_safety_lite`、`full_scan`、`incremental_fast`、`lazy_full`、`full`
に加えて、`incremental_baseline`、`root_tactical`、`pvs_aspiration`、
`see_movepicker`、`lmr`、`null_move`、`shallow_pruning`、`full_candidate`
を受け付けます。後半は成熟した alpha-beta 技術を一段ずつ有効化する実験 profile であり、
消融設定は benchmark からだけ注入され、desktop UI の設定には追加されません。
`full_scan` は逐盤評価 reference、`incremental_fast` は増分 material/PST、`lazy_full` は
±360 の安全界限による実験設定です。現在の `full` は性能門の結果に従い増分評価を使用しますが、
Lazy Full は既定経路では無効です。

次世代の熱経路消融は `hotpath_baseline`、`pvs_hotpath`、`see_hotpath`、
`lmr_hotpath`、`mate_probe`、`next_full_candidate` の順です。前4つは native integer move、
8段 history bucket、占有 mask 評価、Lazy Full に PVS/aspiration、SEE、LMR を段階的に加えます。
`mate_probe` は短詰証明器だけを比較する独立 profile です。これらは benchmark-only で、存在する
だけでは UI 既定へ昇格したことを意味しません。

固定局面だけを短時間で比較する場合:

```powershell
.\.venv\Scripts\python.exe benchmarks\search_efficiency.py `
  --seconds 1 --positions 10

# 1/3/5/7手詰、不詰、打歩詰を50ms/10,000 nodeで検証
.\.venv\Scripts\python.exe benchmarks\mate_solver_benchmark.py
```

## Real-time 監査と保存

引数なしで PyCharm から実行しても、最初に `AUDIT START dir=...` が表示され、同じ directory に
次の file が作成されます。console を閉じても監査データは失われません。

- `events.jsonl`: 各手の完全な診断と run event。各 event で flush されます。
- `games.jsonl`: 完了した対局。1局終了するたびに追記・flush されます。
- `summary.json`: その時点の勝敗、未解決、fallback、信頼区間。1局ごとに置換されます。
- `console.log`: console に表示した heartbeat と status の永続 copy。
- `manifest.json`: 実行条件、状態、開始・終了時刻。

Windows では benchmark process だけに Above Normal priority を best-effort で要求し、終了時に
元の priority を復元します。適用失敗は audit を停止せず manifest に記録されます。Normal で比較する
場合は `--normal-priority` を指定します。

`artifacts/reports/latest.json` は最新 run directory と `running`、`completed`、`early_stopped`、
`interrupted`、`failed` の状態を指します。実行中でも `summary.json` と `events.jsonl` を読めるため、
全対局の終了を待たずに状況を評価できます。console の表示頻度だけを減らす場合は
`--heartbeat-plies 10` のように指定します。構造化 event は表示頻度に関係なく毎手保存されます。

Stop / Ctrl+C は完了済みの対局と途中までの move event を残します。再開時は、最初に表示された
directory を指定し、元と同じ実行条件を渡します。

```powershell
.\.venv\Scripts\python.exe benchmarks\heuristic_strength.py `
  --run-dir artifacts\reports\abp-20260801-120000 --resume
```

manifest は heuristic search、評価、ordering、benchmark の source fingerprint も保存します。
`--resume` は同じ fingerprint の場合だけ許可され、修正前後の対局を同じ統計へ混在させません。
失敗した局は `game_failed` event として ply、SFEN、history、player、例外を即時保存し、
`summary.json` は `failed`、完了局数、未完了局、次に再開する game key を記録します。再開時は
`games.jsonl` の完了局だけを skip し、途中だった1局は opening から再実行します。

既定では、最初の fallback または timing-invalid game、4局以上ですべて未解決、または
6 eligible game以上で95%信頼区間が
50%を完全に上回る・下回る場合に暫定的に早期停止します。これは時間を
無駄にしないための診断であり、「strictly stronger」の正式な証明ではありません。必ず全局を
走らせる場合は `--no-early-stop` を指定します。

## Report の読み方

- `candidate_wins` / `legacy_wins`: fallback の有無とは別に、rule 上実際に決着した局数。
- `rule_draws`: repetition など rule 上の和棋だけ。
- `unresolved`: benchmark の最大手数までに決着しなかった局数。
- `provisional_leader`: unresolved 時の参考用 full evaluation。勝敗裁定や得点には使用しません。
- `fallback_games`: どちらかが完成深さ0の着手を含んだ局数。
- `timing_invalid_games`: deadline tolerance を超えた着手を含んだ局数。
- `eligible_games`: `unresolved`、fallback、timing-invalid game を除いた得点対象。
- `by_color`: candidate の執色別 eligible 勝敗。
- `paired_results`: 完全な換先 pair の sweep、split、invalid と pair score。
- `search_metrics`: candidate の深さ、main/q node、ordering/evaluation cost と tactical hit。
- `relative_to_previous`: `--all-profiles` 時の直前消融 profile に対する各 metric の差。
- `candidate_score`: eligible game の勝ち1、rule draw 0.5、負け0による平均。
- `superiority_claim_supported`: eligible game が100局以上あり、95%信頼区間の下限が50%を
  超えた場合だけ true。

`events.jsonl` には各手の SFEN、history、PV、完成深さ、完成 root move 数、fallback、main/qsearch
node、qsearch truncation、TT hit、評価分解が保存されます。敗局診断は first fallback、評価値の
最初の大幅下落、または最後の candidate 着を critical position とし、1秒/5秒で再探索します。

Heuristic search は最初に quiescence を使わない root safety scan を完了し、それを depth 1 と
します。`attempted_depth` は超時した探索層、`fallback_reason` は depth 1 未完了の理由、
`deadline_overshoot_ms` は hard deadline 超過、`q_depth_truncations`、`q_leaf_truncations`、
`q_budget_truncations` は quiescence の制限要因を表します。従来の
`quiescence_truncations` は3種類の合計として維持されます。`search_elapsed_seconds` は player
内部、`elapsed_seconds` は benchmark 呼出全体、`wall_overshoot_ms` は後者の予算超過です。
これにより search regression と OS scheduling / 呼出側 overhead を分離できます。
`setup_elapsed_seconds` は snapshot replay、`position_history_plies` は再構築した履歴長、
`timeout_stage` は position setup、root scan、または中断した iterative depth を示します。
`q_evasion_nodes` は被将軍時の強制応手 node、`legal_generation_seconds`、
`move_ordering_seconds`、`evaluation_seconds` は主要なCPU costです。王手の事前 probe は既定経路
から削除されているため `checking_probe_seconds` は0になります。
`cpu_elapsed_seconds` と `scheduler_wait_seconds` は search CPU cost と wall-clock scheduling stall
を分離し、timing-invalid は `search_cpu_overrun`、`scheduler_stall`、`mixed` に分類されます。
`incremental_evaluation_calls`、`full_evaluation_calls`、`lazy_alpha_skips`、`lazy_beta_skips` は評価
消融の実コストを示し、最終局面の `evaluation` は常に正確な full breakdown です。
`pvs_null_window_searches` / `pvs_researches`、aspiration の fail-low/high、SEE、LMR、
null-move、razoring、reverse-futility の各 counter は、選択的探索が実際に発動した頻度を示します。
`move_picker_stage_counts` / `move_picker_stage_seconds` は TT、good capture、promotion、
killer/countermove、history quiet、bad capture の各 stage を分離します。
packed picker では `move_picker_generated`、`move_picker_yielded` と
`move_picker_yielded_by_stage` が cutoff までに実際に生成・供給した数を示します。
`full_evaluation_bound_skips` は `fast ± 360` で攻撃図を安全に省略した回数、
`occupied_piece_iterations` は full evaluation が実際に訪問した盤上駒数です。
`mate_status`、`mate_nodes`、`mate_elapsed_seconds`、`mate_principal_variation` は短詰 probe の
proven/disproven/unknown と証明列を保存します。
`cpu_see_seconds`、`cpu_tt_seconds` と `cpu_search_overhead_seconds` は、SEE の合法手検証、TT、
それ以外の Python 再帰・制御処理を評価・ordering から分離するための診断値です。

選択的探索は profile の正しさ・固定局面の性能・換先対局の順に門を通過した場合だけ既定経路へ
昇格します。`full_candidate` が存在すること自体は、その組合せが UI 既定であることや棋力向上を
意味しません。

## 任意の YaneuraOu oracle

YaneuraOu は製品依存ではなく、利用者がローカルに用意した実行ファイルを明示した場合だけ使う
benchmark oracle です。この repository は実行ファイル、評価ファイル、GPL source を download・
bundle しません。固定 node の MultiPV と `go mate` corpus は次の command で生成できます。

```powershell
.\.venv\Scripts\python.exe benchmarks\build_yaneuraou_corpus.py `
  --engine C:\path\to\YaneuraOu.exe `
  --input configs\training\evaluation_positions.jsonl `
  --output artifacts\corpora\yaneuraou_multipv.jsonl `
  --nodes 1000000 --multipv 3 --mate-ms 10000
```

出力は binary SHA-256、engine ID、options、固定 node 数、各 MultiPV score/PV を保持し、隣接する
manifest に同じ実行条件を固定します。エンジンが未設定でも UI と通常 test はすべて動作します。
現在 repository 内の短詰 corpus は audit 由来と決定的 test 局面だけです。YaneuraOu 検証済み
50局面は実行ファイルなしに捏造せず、この command を実行した後に人手 review して freeze します。

任意の paired game は同じ local binary を使い、candidate の執色を交換して JSON report に保存します。

```powershell
.\.venv\Scripts\python.exe benchmarks\yaneuraou_match.py `
  --engine C:\path\to\YaneuraOu.exe --positions 4 `
  --profile full --seconds 1 --oracle-nodes 100000 `
  --output artifacts\reports\yaneuraou_pair.json
```

固定 node の oracle と wall-clock candidate は同じ資源尺度ではないため、この結果は第三者標尺で
あって直接の Elo 換算ではありません。外部 engine 側にも時間制限を使う正式比較は、実行環境と
USI options を固定した別 audit として扱います。
