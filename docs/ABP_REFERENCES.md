# ABP algorithm references and adoption policy

AlphaSho の heuristic search は公開された algorithm idea を参考にした独立 Python 実装です。
Stockfish または YaneuraOu の GPL code を copy しません。成熟 engine の技法は、将棋で有名だから
という理由だけで既定にせず、固定戦術、同一時間性能、換先対局の順に消融します。

## 参照している設計

- [Stockfish search.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp):
  PVS、aspiration、staged move ordering、history、LMR などの構成を比較する一次資料。
- [YaneuraOu](https://github.com/yaneurao/YaneuraOu): 将棋固有の USI search と評価を確認する
  一次資料。[探索部の計測資料](https://github.com/yaneurao/YaneuraOu/wiki/%E6%8E%A2%E7%B4%A2%E9%83%A8%E3%81%AE%E8%A8%88%E6%B8%AC%E8%B3%87%E6%96%99)
  は hot path の測定単位を決める参考にする。
- [df-pn の将棋への応用](https://ipsj.ixsq.nii.ac.jp/records/32069) と
  [循環局面を含む proof-number search](https://ojs.aaai.org/index.php/AAAI/article/download/7534/7395):
  AND/OR proof、proof/disproof number、resource exhaustion を unknown とする短詰探索の背景。
- [YaneuraOu 詰将棋 engine](https://github.com/yaneurao/YaneuraOu/wiki/%E3%82%84%E3%81%AD%E3%81%86%E3%82%89%E7%8E%8B%E8%A9%B0%E5%B0%86%E6%A3%8B%E3%82%A8%E3%83%B3):
  外部 `go mate` oracle の相互運用資料。

`BoundedDfpnMateSolver` は現時点では df-pn-inspired の保守的な bounded AND/OR proof search です。
攻方は王手だけ、守方は全合法応手を探索し、資源切れ・経路反復・判断不能は常に `unknown` に
します。`proven` の PV は cshogi で逐手再検証します。本格的な threshold update を備えた df-pn
と同等だとは主張しません。

ProbCut、singular extension、複雑な check extension、NNUE は今回の既定候補ではありません。
Python engine の実戦深さ、SEE の保守性、短詰 corpus が先に安定してから、独立 profile として
同じ acceptance gate を通します。
