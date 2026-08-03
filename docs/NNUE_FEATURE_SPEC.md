# Future CPU NNUE feature contract

This document reserves an evaluator boundary; it does not enable a neural
evaluator in the desktop player.

## Version 1 input

- Two accumulators, each oriented around one king square.
- Sparse board features identify perspective king square, owner, piece type,
  and square for every non-king piece.
- Hand features identify owner, droppable piece type, and a cumulative count
  bucket for each held piece.
- Side to move is supplied to the output layer instead of rotating state.

The serialized feature version is `alphasho-halfkp-shogi-v1`. King moves force
one accumulator refresh; ordinary moves, captures, promotions, and drops update
only dirty sparse features. Inference must use bounded integer arithmetic and
return the same side-to-move score convention as `HandcraftedEvaluator`.

## Admission criteria

A network remains benchmark-only until incremental CPU inference is faster
than the exact handcrafted evaluation on the fixed suite, network and feature
fingerprints are persisted in audit manifests, and paired games pass the same
fallback, timing, and confidence rules as heuristic ABP.

Training, network distribution, and UI selection are outside the current
search-efficiency change.
