"""AlphaSho training: edit this one-screen block, then click Run in PyCharm."""

from alphasho.training import run_training_entry

# Run
PROFILE = "cpu_debug"
DEVICE = "cpu"
OUTPUT_LANGUAGE = "zh"  # en / zh / ja
ITERATIONS = 2000
MODEL_PRESET = "smoke"  # smoke / baseline / large

# Short-game rules
START_POSITION_MODE = "standard"  # standard / tactical
ADJUDICATION_MODE = "capture_material"  # real_rules / capture_material
MAX_MOVES = 80
CAPTURE_THRESHOLD = 4

# MCTS / self-play
PLAYOUTS = 200
C_PUCT = 1.5
INFERENCE_BATCH_SIZE = 8
GAMES_PER_ITERATION = 8
SELFPLAY_WORKERS = 0  # 0 = auto
EVALUATION_WORKERS = 0  # 0 = auto
GPU_BATCH_SIZE = 128
GPU_BATCH_WAIT_MS = 2.0
DIRICHLET_ALPHA = 0.15
DIRICHLET_WEIGHT = 0.25

# Network training
TRAINING_BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50_000
TRAIN_STEPS_PER_ITERATION = 32
LEARNING_RATE = 0.001

# Evaluation / promotion
CHECK_FREQUENCY = 10
EVALUATION_GAMES = 20
PROMOTION_SCORE = 0.55

if __name__ == "__main__":
    run_training_entry(globals())
