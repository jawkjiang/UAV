"""
Configuration for Multi-Model Training with Mixed Attacks

Each model is trained on a dataset with mixed attack types (excluding replay).
This provides better generalization compared to single-attack training.
"""
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Skip existing experiments (if best_model.pth exists)
SKIP_EXISTING = False

# Data paths
DATA_PATH = "../data/src/flights.csv"
OUTPUT_DIR = "./output"

# Data split ratios (by flight)
# Strategy C: Conservative Hybrid with Flight Reuse
# - Train/Val will be expanded via reuse (each flight × 7 versions)
# - Test remains original (no reuse) with stratified sampling
# Adjusted to 60/20/20 for more reasonable validation set size
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Window parameters
WINDOW_SIZE = 50  # Number of samples per window
STEP_SIZE = 5     # Step size for sliding window

# Sampling rate (for time-aware evaluation)
SAMPLING_RATE = 100  # Hz (100 samples per second)

# Feature columns
POSITION_FEATURES = ['position_x', 'position_y', 'position_z']
VELOCITY_FEATURES = ['velocity_x', 'velocity_y', 'velocity_z']
ACCELERATION_FEATURES = ['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']

# Training parameters
BATCH_SIZE = 64
MAX_EPOCHS = 100
NUM_EPOCHS = 100  # Alias for MAX_EPOCHS (for compatibility)
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# ============================================================================
# MIXED ATTACK CONFIGURATION
# ============================================================================

# Attack types to include (excluding replay attacks due to poor performance)
ATTACK_TYPES = [
    'step',
    'drift_ramp',
    'drift_sigmoid',
    'delay',
    'takeover_step',
    'takeover_ramp'
]

# Attack injection ratios
# For main.py (without flight reuse):
# - TRAIN/VAL: Use standard mixed injection (random distribution)
# - TEST: Use guaranteed coverage injection to ensure ALL attack types are tested
# For main_strategyC.py (with flight reuse):
# - TRAIN/VAL: Each flight generates all attack versions (flight reuse)
# - TEST: Stratified sampling ensures all attack types are represented
TRAIN_ATTACK_RATIO = 0.60   # 60% of training flights will be attacked
VAL_ATTACK_RATIO = 0.50     # 50% of validation flights will be attacked
TEST_ATTACK_RATIO = 0.70    # 70% of test flights will be attacked (ensures enough samples per attack)

# Minimum samples per attack type in test set (for guaranteed coverage)
MIN_TEST_FLIGHTS_PER_ATTACK = 3     # At least 3 flights per attack type in test set

# Flight reuse strategy (Strategy C)
USE_FLIGHT_REUSE = True              # Enable flight reuse for train/val
REUSE_TEST_SET = False               # Do NOT reuse test set (maintain independence)
# MIN_TEST_FLIGHTS_PER_ATTACK = 8     # Minimum flights per attack type in test set (for strategyC)

# Model architectures to train
MODEL_TYPES = [
    'cnn',
    'lstm',
    'bilstm',
    'gru',
    'cnn_lstm',
    'tcn',
    'transformer'
]

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

# Consistency modes (only pos_vel_acc for realistic scenarios)
CONSISTENCY_MODES = ['pos_vel_acc']

# Velocity epsilon for along_track/cross_track direction computation
VELOCITY_EPSILON = 0.01  # m/s

# ============================================================================
# ATTACK-SPECIFIC PARAMETERS
# ============================================================================

# Step attack parameters
STEP1_ATTACK_DURATION = 5.0  # Duration of step attack transient (seconds)
STEP1_ATTACK_TIME_CONSTANT = 1.0  # Time constant for velocity decay (seconds)
STEP1_VELOCITY_TRANSIENT_MIN = 0.5  # Min velocity during transient (m/s)
STEP1_VELOCITY_TRANSIENT_MAX = 2.0  # Max velocity during transient (m/s)

# Drift profiles
DRIFT_PROFILES = ['ramp', 'sigmoid']
DRIFT_DURATIONS = [5.0, 10.0, 20.0]
SIGMOID_K = 10.0

# Delay amounts (seconds)
DELAY_SECONDS = [1.0, 3.0, 5.0]

# Replay parameters
REPLAY_SEGMENT_DURATIONS = [5.0, 10.0, 15.0]
REPLAY_DONOR_SOURCES = ['same_flight_earlier', 'same_route_other_flight']
REPLAY_STITCHING_MODES = ['hard', 'soft']
REPLAY_TRANSITION_SECONDS = 0.5
REPLAY_GAP_SECONDS = 1.0

# Takeover offset profiles
TAKEOVER_OFFSET_PROFILES = ['step', 'ramp']
TAKEOVER_DURATIONS = [0.5, 1.0, 2.0]
TAKEOVER_ALPHAS = [0.3, 0.5, 0.7]  # Tracking gain
TAKEOVER_TAUS = [0.5, 1.0, 2.0]  # Time constant for tracking

# ============================================================================
# WINDOW BALANCING
# ============================================================================

# Target positive ratios for window balancing
TRAIN_POS_RATIO = 0.35  # Higher ratio for training to learn attack patterns
VAL_POS_RATIO = 0.03    # Lower ratio to match real-world distribution
TEST_POS_RATIO = 0.03   # Lower ratio to match real-world distribution

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# FPR thresholds for computing recall at fixed FPR
FPR_THRESHOLDS = [0.001, 0.01, 0.05]

# ============================================================================
# ATTACK TYPE MAPPING
# ============================================================================

# Map attack names to injection types
ATTACK_TYPE_MAP = {
    'step': 'step',
    'drift_ramp': 'drift',
    'drift_sigmoid': 'drift',
    'delay': 'delay',
    'takeover_step': 'takeover',
    'takeover_ramp': 'takeover'
}

# Attack-specific parameters for injection
ATTACK_PARAMS = {
    'step': {},
    'drift_ramp': {'profile': 'ramp'},
    'drift_sigmoid': {'profile': 'sigmoid'},
    'delay': {},
    'takeover_step': {'offset_profile': 'step'},
    'takeover_ramp': {'offset_profile': 'ramp'}
}
