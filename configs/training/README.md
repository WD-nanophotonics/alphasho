# Training profiles

Open the root `training_pipeline.py` for normal training. Its one-screen quick
settings override the profile defaults here. TOML keeps advanced settings such
as virtual loss, shard size, weight decay, supervised records, logging, and
checkpoint intervals.

Precedence is profile TOML, then quick pipeline settings, then the resulting
`effective_config.json` snapshot. Fixed Smoke, Tactical Overfit, and Baseline
PyCharm configurations use the same entry point but load their TOML directly.
The desktop UI does not import this directory.

Available profiles:

- `smoke.toml`: a 20-second tactical, non-zero-value pipeline check;
- `cpu_debug.toml`: a standard-position CPU convergence experiment;
- `tactical_overfit.toml`: one-capture learning on fixed tactical positions;
- `baseline.toml`: production training with real shogi rules.

`output_language` in `[monitoring]` accepts `en`, `zh`, or `ja`. The quick
pipeline's `OUTPUT_LANGUAGE` overrides it. `tactical_positions.jsonl` stores
paired positions and verified legal capture moves.
