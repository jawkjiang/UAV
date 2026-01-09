"""
Configuration for Time-Aware Evaluation Framework

This module defines all parameters for evaluating GPS spoofing detection models
using time-aware metrics (DR@Δt, ADD, MTBFA).
"""
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths
STEP3B_BASE = "../step3b_multiModelGeneral"
STEP3B_OUTPUT_DIR = os.path.join(STEP3B_BASE, "output")
STEP3B_DATA_PATH = os.path.join(STEP3B_BASE, "../data/src/flights.csv")
OUTPUT_DIR = "./output/time_aware_metrics"
TIME_AWARE_OUTPUT = OUTPUT_DIR

# Model names (matching step3b directory structure)
MODEL_NAMES = [
    'cnn',
    'lstm',
    'bilstm',
    'gru',
    'cnn_lstm',
    'tcn',
    'transformer'
]

# Alias for compatibility
MODELS = MODEL_NAMES

# Attack types (from step3b config)
ATTACK_TYPES = [
    'step',
    'drift_ramp', 
    'drift_sigmoid',
    'delay',
    'takeover_step',
    'takeover_ramp'
]

# ============================================================================
# TIME-AWARE METRICS CONFIGURATION
# ============================================================================

# Detection Rate within Δt thresholds (seconds)
DELTA_T_VALUES = [1, 2, 5, 10, 15, 30]

# Sampling rate (from step3b)
SAMPLING_RATE = 100  # Hz
TIME_PER_POINT = 0.01  # 每个采样点0.01秒 (1/100 Hz)

# Window parameters (from step3b)
WINDOW_SIZE = 50
STEP_SIZE = 5

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

# Scenario 3: Balanced (平衡场景)
SCENARIO_BALANCED = {
    'name': 'Balanced',
    'description': 'Balance between detection speed and false alarm rate',
    'constraints': {
        'DR@5s': 0.90,       # 90% detection within 5 seconds
        'MTBFA': 12.0,       # At least 12 hours between false alarms
        'ADD': 5.0           # Average delay under 5 seconds
    },
    'weights': {
        'DR@5s': 0.33,
        'MTBFA': 0.33,
        'ADD': 0.34
    }
}

SCENARIOS = [
    SCENARIO_SAFETY_CRITICAL,
    SCENARIO_MONITORING,
    SCENARIO_BALANCED
]

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Figure settings
FIGURE_DPI = 300
FIGURE_SIZE_SINGLE = (10, 6)
FIGURE_SIZE_DOUBLE = (12, 5)
FIGURE_SIZE_HEATMAP = (12, 8)

# Color scheme for models (consistent across all plots)
MODEL_COLORS = {
    'cnn': '#1f77b4',
    'lstm': '#ff7f0e',
    'bilstm': '#2ca02c',
    'gru': '#d62728',
    'cnn_lstm': '#9467bd',
    'tcn': '#8c564b',
    'transformer': '#e377c2'
}

# Line styles for models
MODEL_LINESTYLES = {
    'cnn': '-',
    'lstm': '--',
    'bilstm': '-.',
    'gru': ':',
    'cnn_lstm': '-',
    'tcn': '--',
    'transformer': '-.'
}

# Markers for models
MODEL_MARKERS = {
    'cnn': 'o',
    'lstm': 's',
    'bilstm': '^',
    'gru': 'v',
    'cnn_lstm': 'D',
    'tcn': 'p',
    'transformer': '*'
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Output file names
OVERALL_METRICS_CSV = "overall_metrics.csv"
DR_VS_DELAY_PNG = "dr_vs_delay.png"
DR_VS_MTBFA_PNG = "dr_vs_mtbfa.png"
DELAY_DISTRIBUTION_PNG = "delay_distribution.png"
PER_ATTACK_HEATMAP_PNG = "per_attack_heatmap.png"
MTBFA_COMPARISON_PNG = "mtbfa_comparison.png"
SCENARIO_ANALYSIS_TXT = "scenario_analysis.txt"
EVALUATION_SUMMARY_MD = "evaluation_summary.md"
DETAILED_DELAYS_CSV = "detailed_delays.csv"
PER_ATTACK_METRICS_CSV_TEMPLATE = "{model}_per_attack_metrics.csv"

# Per-model output subdirectories
PER_MODEL_SUBDIR = "{model}"
PER_ATTACK_METRICS_CSV = "per_attack_metrics.csv"
DELAY_HISTOGRAM_PNG = "delay_histogram.png"

# ============================================================================
# VISUALIZATION CONFIGURATION (Extended)
# ============================================================================

FIGSIZE_SINGLE = FIGURE_SIZE_SINGLE
FIGSIZE_DOUBLE = FIGURE_SIZE_DOUBLE
FIGSIZE_HEATMAP = FIGURE_SIZE_HEATMAP
FIGURE_DPI = FIGURE_DPI

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_step3b_model_dir(model_name: str) -> str:
    """Get the full path to a model's output directory in step3b."""
    return os.path.join(STEP3B_OUTPUT_DIR, model_name)

def get_step3b_predictions_file(model_name: str) -> str:
    """Get the full path to a model's test predictions file."""
    return os.path.join(get_step3b_model_dir(model_name), "test_predictions.npz")

def get_step3b_attack_info_file(model_name: str) -> str:
    """Get the full path to a model's test attack info file."""
    return os.path.join(get_step3b_model_dir(model_name), "test_attack_info.csv")

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    # Test configuration
    print("Step 4b Configuration:")
    print(f"  Models: {MODELS}")
    print(f"  Attack Types: {ATTACK_TYPES}")
    print(f"  Delta-t values: {DELTA_T_VALUES}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"\nStep 3b paths:")
    print(f"  Output dir: {STEP3B_OUTPUT_DIR}")
    print(f"  Data path: {STEP3B_DATA_PATH}")
    
    # Test helper functions
    print(f"\nHelper functions test:")
    print(f"  Model dir: {get_step3b_model_dir('cnn')}")
    print(f"  Predictions: {get_step3b_predictions_file('cnn')}")
    print(f"  Attack info: {get_step3b_attack_info_file('cnn')}")
    
    # Create output dir
    ensure_output_dir()
    print(f"\n✓ Output directory created/verified: {OUTPUT_DIR}")
