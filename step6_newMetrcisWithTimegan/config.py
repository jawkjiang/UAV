"""
Step6: TimeGAN模型评估 - 时间感知指标
整合Step5的模型和数据 + Step4b的时间感知评估方法
"""
import os

# ============================================================================
# BASIC CONFIGURATION
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Step5 paths (数据和模型来源)
STEP5_BASE = "../step5_timegan"
STEP5_OUTPUT_DIR = os.path.join(STEP5_BASE, "output")
STEP5_DATA_PATH = os.path.join(STEP5_BASE, "../data/src/flights.csv")

# Step6 output
OUTPUT_DIR = "./output"
TIME_AWARE_OUTPUT = os.path.join(OUTPUT_DIR, "time_aware_metrics")
VISUALIZATION_OUTPUT = os.path.join(OUTPUT_DIR, "visualizations")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# 所有待评估的模型（与step5训练的模型一致）
MODEL_NAMES = [
    'cnn',
    'lstm', 
    'bilstm',
    'gru',
    'cnn_lstm',
    'tcn',
    'transformer'
]

# Alias
MODELS = MODEL_NAMES

# ============================================================================
# ATTACK CONFIGURATION
# ============================================================================

# Attack types (from step5)
ATTACK_TYPES = [
    'step',
    'drift_ramp',
    'drift_sigmoid', 
    'delay',
    'takeover_step',
    'takeover_ramp'
]

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================

# Feature columns (compatible with step3b)
POSITION_FEATURES = ['position_x', 'position_y', 'position_z']
VELOCITY_FEATURES = ['velocity_x', 'velocity_y', 'velocity_z']
ACCELERATION_FEATURES = ['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']

# ============================================================================
# TIME-AWARE METRICS CONFIGURATION
# ============================================================================

# Detection Rate within Δt thresholds (seconds)
DELTA_T_VALUES = [1, 2, 5, 10, 15, 30]

# Sampling rate (from step5)
# 注意：实际数据的采样间隔不是固定的，平均约0.156秒
# 这不是传统意义的采样率，而是控制指令的更新频率
SAMPLING_RATE = 6.4  # Hz (约1/0.156)
TIME_PER_POINT = 0.156  # 每个采样点0.156秒 (实际数据delta_t的中位数)

# Window parameters (from step5)
WINDOW_SIZE = 50  # 窗口包含50个采样点
STEP_SIZE = 5     # 窗口步长5个采样点

# MTBFA Configuration
FALSE_ALARM_GAP_THRESHOLD = 1.0  # seconds

# TIME_PER_SAMPLE for timestamp calculation
TIME_PER_SAMPLE = TIME_PER_POINT

# ============================================================================
# SCENARIO ANALYSIS CONFIGURATION
# ============================================================================

# Scenario 1: Safety-Critical (安全关键场景)
SCENARIO_SAFETY_CRITICAL = {
    'name': 'Safety-Critical',
    'description': 'Quick response required, tolerates some false alarms',
    'constraints': {
        'DR@5s': 0.95,      # Must detect 95% of attacks within 5 seconds
        'ADD': 3.0,          # Average delay must be under 3 seconds
        'MTBFA': 5.0         # At least 5 hours between false alarms
    },
    'weights': {
        'DR@5s': 0.5,
        'ADD': 0.3,
        'MTBFA': 0.2
    }
}

# Scenario 2: Monitoring (长期监控场景)
SCENARIO_MONITORING = {
    'name': 'Long-term Monitoring',
    'description': 'Low false alarm rate, slower detection acceptable',
    'constraints': {
        'DR@10s': 0.90,      # Must detect 90% within 10 seconds
        'MTBFA': 24.0,       # At least 24 hours between false alarms
        'ADD': 10.0          # Average delay under 10 seconds
    },
    'weights': {
        'MTBFA': 0.5,
        'DR@10s': 0.3,
        'ADD': 0.2
    }
}

# Scenario 3: Balanced (均衡场景)
SCENARIO_BALANCED = {
    'name': 'Balanced',
    'description': 'Balance between detection speed and false alarm rate',
    'constraints': {
        'DR@5s': 0.90,       # Must detect 90% within 5 seconds
        'MTBFA': 12.0,       # At least 12 hours between false alarms
        'ADD': 5.0           # Average delay under 5 seconds
    },
    'weights': {
        'DR@5s': 0.4,
        'ADD': 0.3,
        'MTBFA': 0.3
    }
}

SCENARIOS = [SCENARIO_SAFETY_CRITICAL, SCENARIO_MONITORING, SCENARIO_BALANCED]

# ============================================================================
# OUTPUT FILE NAMES
# ============================================================================

# CSV files
OVERALL_METRICS_CSV = 'overall_metrics.csv'
PER_ATTACK_METRICS_CSV_TEMPLATE = '{model}_per_attack_metrics.csv'
DETAILED_DELAYS_CSV = 'detailed_delays.csv'

# Visualization files
DR_VS_DELAY_PNG = 'dr_vs_delay.png'
DR_VS_MTBFA_PNG = 'dr_vs_mtbfa.png'
DELAY_DISTRIBUTION_PNG = 'delay_distribution.png'
PER_ATTACK_HEATMAP_PNG = 'per_attack_heatmap.png'
MTBFA_COMPARISON_PNG = 'mtbfa_comparison.png'
MODEL_COMPARISON_PNG = 'model_comparison.png'

# Report files
SCENARIO_ANALYSIS_TXT = 'scenario_analysis.txt'
EVALUATION_SUMMARY_MD = 'evaluation_summary.md'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_output_dir():
    """Create output directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TIME_AWARE_OUTPUT, exist_ok=True)
    os.makedirs(VISUALIZATION_OUTPUT, exist_ok=True)


def get_step5_model_path(model_name: str) -> str:
    """Get path to step5 model weights."""
    return os.path.join(STEP5_OUTPUT_DIR, f'best_model_{model_name}.pth')


def get_step5_test_data_path() -> str:
    """Get path to step5 test data."""
    return os.path.join(STEP5_OUTPUT_DIR, 'test_with_attacks.csv')


def get_step5_test_attack_info_path() -> str:
    """Get path to step5 test attack info."""
    return os.path.join(STEP5_OUTPUT_DIR, 'test_attack_info.csv')


def get_step5_normalization_path() -> str:
    """Get path to step5 normalization stats."""
    return os.path.join(STEP5_OUTPUT_DIR, 'normalization_stats.json')


def get_predictions_path(model_name: str) -> str:
    """Get path to save/load predictions."""
    return os.path.join(OUTPUT_DIR, f'{model_name}_test_predictions.npz')
