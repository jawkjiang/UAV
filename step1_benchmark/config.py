"""
Configuration for GPS Spoofing Detection Model
"""
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

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

# Attack injection parameters
ATTACK_START_BUFFER = 10.0  # seconds after takeoff
ATTACK_END_BUFFER = 10.0    # seconds before landing
ATTACK_MAGNITUDES = [5.0, 15.0, 30.0]  # meters
ATTACK_DURATION = 3.0  # seconds for velocity/acceleration transient
ATTACK_TIME_CONSTANT = 1.0  # seconds for exponential decay
VELOCITY_TRANSIENT_MIN = 0.5  # m/s
VELOCITY_TRANSIENT_MAX = 2.0  # m/s

# Positive sample ratios (window-level)
TRAIN_POS_RATIO_MIN = 0.30
TRAIN_POS_RATIO_MAX = 0.40
VAL_TEST_POS_RATIO_MIN = 0.01
VAL_TEST_POS_RATIO_MAX = 0.05

# Features (excluding 'route')
POSITION_FEATURES = ['position_x', 'position_y']
VELOCITY_FEATURES = ['velocity_x', 'velocity_y']
ACCELERATION_FEATURES = ['linear_acceleration_x', 'linear_acceleration_y']
TIME_FEATURE = 'time'

# Model parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Evaluation thresholds
FPR_THRESHOLDS = [0.01, 0.05, 0.10]  # For computing Recall @ FPR
