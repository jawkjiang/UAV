"""Central configuration for the MATLAB-based UAV spoofing detection pipeline."""
import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'matlab_sim', 'output')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'matlab_pipeline', 'output')

# ── Feature columns (9-d state vector) ───────────────────────────────────────
FEATURE_COLS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
N_FEATURES   = len(FEATURE_COLS)

# ── Windowing ─────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 50   # samples  (~5 s at 10 Hz)
STEP_SIZE    = 5    # samples  (~0.5 s stride)
SAMPLE_RATE  = 10   # Hz

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
MAX_EPOCHS      = 100
PATIENCE        = 10
LEARNING_RATE   = 1e-3
SEEDS           = [0, 1, 2, 3, 4]   # 5 independent runs per model

# ── Deployability thresholds ──────────────────────────────────────────────────
DR_TIME_THRESHOLD = 5.0   # seconds
DR_PASS_THRESHOLD = 0.85  # 85%
MTBFA_THRESHOLD   = 0.1   # hours

# ── Statistical tests ─────────────────────────────────────────────────────────
N_BOOTSTRAP   = 1000
CI_LEVEL      = 0.95
