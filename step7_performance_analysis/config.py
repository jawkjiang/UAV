"""
Step7配置文件
"""
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

STEP5_BASE = "../step5_timegan"
STEP5_OUTPUT = os.path.join(STEP5_BASE, "output")
STEP6_BASE = "../step6_newMetrcisWithTimegan"
STEP6_OUTPUT = os.path.join(STEP6_BASE, "output/time_aware_metrics")

OUTPUT_DIR = "./output"
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAMES = [
    'cnn', 'lstm', 'bilstm', 'gru', 
    'cnn_lstm', 'tcn', 'transformer'
]

# ============================================================================
# INFERENCE PROFILING CONFIGURATION
# ============================================================================

# 测试配置
WARMUP_ITERATIONS = 50          # 预热次数
BENCHMARK_ITERATIONS = 500      # 基准测试次数
BATCH_SIZES = [1, 8, 16, 32, 64, 128]  # 测试不同批大小

# 延迟阈值（ms）
LATENCY_THRESHOLDS = {
    'real_time': 100,        # 实时系统要求
    'near_real_time': 500,   # 准实时
    'offline': 1000          # 离线分析
}

# 设备配置
DEVICES = ['cpu', 'cuda']   # 测试CPU和GPU

# ============================================================================
# MODEL COMPLEXITY CONFIGURATION
# ============================================================================

# 输入维度（用于FLOPs计算）
WINDOW_SIZE = 50
N_FEATURES = 16

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

FIGSIZE_SINGLE = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_LARGE = (12, 10)

MODEL_COLORS = {
    'cnn': '#1f77b4',
    'lstm': '#ff7f0e',
    'bilstm': '#2ca02c',
    'gru': '#d62728',
    'cnn_lstm': '#9467bd',
    'tcn': '#8c564b',
    'transformer': '#e377c2'
}

# ============================================================================
# DEPLOYMENT SCENARIOS
# ============================================================================

DEPLOYMENT_SCENARIOS = {
    'edge_device': {
        'name': '边缘设备（嵌入式）',
        'constraints': {
            'max_latency_ms': 100,      # 最大延迟
            'max_params_m': 1.0,        # 最大参数量（百万）
            'max_memory_mb': 512,       # 最大内存
            'min_dr_5s': 0.90           # 最低DR@5s
        }
    },
    'edge_server': {
        'name': '边缘服务器（小型GPU）',
        'constraints': {
            'max_latency_ms': 50,
            'max_params_m': 5.0,
            'max_memory_mb': 2048,
            'min_dr_5s': 0.95
        }
    },
    'cloud': {
        'name': '云端（大型GPU）',
        'constraints': {
            'max_latency_ms': 20,
            'max_params_m': 50.0,
            'max_memory_mb': 8192,
            'min_dr_5s': 0.98
        }
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
