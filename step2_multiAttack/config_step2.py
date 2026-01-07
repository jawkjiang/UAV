"""
Configuration for Multi-Attack GPS Spoofing Detection

Extends step1_benchmark with multiple attack types:
- Drift Spoofing
- Delay/Replay
- Consistent Takeover
"""
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Skip existing experiments (if best_model.pth exists)
SKIP_EXISTING = False

# Data paths
DATA_PATH = "../data/src/flights.csv"
OUTPUT_DIR = "./output"
MODEL_DIR = "./models"

# Data split ratios (by flight)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Window parameters
WINDOW_SIZE = 50  # Number of samples per window
STEP_SIZE = 5     # Step size for sliding window

# ============================================================================
# UNIFIED ATTACK CONSTRAINTS
# ============================================================================

# Time window constraints
ATTACK_START_BUFFER = 10.0  # seconds after takeoff
ATTACK_END_BUFFER = 10.0    # seconds before landing

# Attack magnitudes (meters)
ATTACK_MAGNITUDES = [5.0, 15.0, 30.0]

# Direction modes
DIRECTION_MODES = [
    'random_xy',
    'along_track_xy',
    'cross_track_xy',
    'fixed_east',
    'fixed_north'
]

# Consistency modes
# Changed to only pos_vel_acc for fair comparison and realistic attack scenarios
# Original: ['pos_only', 'pos_vel', 'pos_vel_acc']
CONSISTENCY_MODES = ['pos_vel_acc']

# Velocity epsilon for along_track/cross_track direction computation
VELOCITY_EPSILON = 0.01  # m/s

# ============================================================================
# ATTACK A: DRIFT SPOOFING
# ============================================================================

# Drift profiles
DRIFT_PROFILES = ['ramp', 'sigmoid']

# Drift durations (seconds)
DRIFT_DURATIONS = [5.0, 10.0, 20.0]

# Sigmoid steepness parameter
SIGMOID_K = 10.0

# ============================================================================
# ATTACK B1: DELAY (Fixed Delay)
# ============================================================================

# Delay amounts (seconds)
DELAY_SECONDS = [1.0, 3.0, 5.0]

# ============================================================================
# ATTACK B2: REPLAY (Segment Replay)
# ============================================================================

# Replay segment durations (seconds)
REPLAY_SEGMENT_DURATIONS = [5.0, 10.0, 15.0]

# Donor sources
REPLAY_DONOR_SOURCES = ['same_flight_earlier', 'same_route_other_flight']

# Stitching modes
REPLAY_STITCHING_MODES = ['hard', 'soft']

# Soft stitching transition duration (seconds)
REPLAY_TRANSITION_SECONDS = 0.5

# Gap between donor and attack segment (for same_flight_earlier)
REPLAY_GAP_SECONDS = 1.0

# ============================================================================
# ATTACK C: CONSISTENT TAKEOVER
# ============================================================================

# Takeover offset profiles
TAKEOVER_OFFSET_PROFILES = ['step', 'ramp', 'sigmoid']

# Takeover dynamics (alpha values)
TAKEOVER_ALPHAS = [0.3, 0.5, 0.8]

# Takeover dynamics (tau values in seconds)
TAKEOVER_TAUS = [0.5, 1.0, 2.0]

# Takeover ramp/sigmoid durations (seconds)
TAKEOVER_DURATIONS = [3.0, 5.0, 10.0]

# ============================================================================
# ORIGINAL STEP1 ATTACK (for comparison)
# ============================================================================

# Original step1 attack parameters (for baseline comparison)
STEP1_ATTACK_DURATION = 3.0  # seconds for velocity/acceleration transient
STEP1_ATTACK_TIME_CONSTANT = 1.0  # seconds for exponential decay
STEP1_VELOCITY_TRANSIENT_MIN = 0.5  # m/s
STEP1_VELOCITY_TRANSIENT_MAX = 2.0  # m/s

# ============================================================================
# SAMPLING & LABELING
# ============================================================================

# Positive sample ratios (window-level)
TRAIN_POS_RATIO_MIN = 0.30
TRAIN_POS_RATIO_MAX = 0.40
VAL_TEST_POS_RATIO_MIN = 0.01
VAL_TEST_POS_RATIO_MAX = 0.05

# ============================================================================
# FEATURES
# ============================================================================

# Position features
POSITION_FEATURES = ['position_x', 'position_y']

# Velocity features
VELOCITY_FEATURES = ['velocity_x', 'velocity_y']

# Acceleration features
ACCELERATION_FEATURES = ['linear_acceleration_x', 'linear_acceleration_y']

# Time feature
TIME_FEATURE = 'time'

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10

# Evaluation thresholds
FPR_THRESHOLDS = [0.01, 0.05, 0.10]  # For computing Recall @ FPR
