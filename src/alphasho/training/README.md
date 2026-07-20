# AlphaSho training

The training pipeline has independent stages:

1. optional KIF/CSA supervised record ingestion;
2. MCTS self-play using the best model;
3. compressed disk replay and candidate training;
4. paired-color evaluation against the best model;
5. checkpointing, promotion, and JSONL metrics.

Experimental training wraps the pure-rule `SearchPosition` in
`TrainingPosition`. Real cshogi terminals have priority. CPU profiles may end at
a capture threshold or material-scored move limit, while Baseline always uses
real rules. Shards are isolated by `ruleset_id`.

Users start from the root `training_pipeline.py`, which contains only frequent
settings. `AlphaZeroTrainer`, logs, checkpoints, shards, and error handling stay
inside this package. Every PyCharm training configuration uses that entry point;
fixed profiles are selected through the run configuration. The desktop UI does
not import this package.

Human-readable messages live in `messages.py` and support English, Chinese, and
Japanese. Events remain structured and language-neutral. Each profile has an
independent process lock. Checkpoints use complete temporary files, bounded
Windows contention retries, revision recovery, and commit pipeline state last.
