"""
Step8 Painter - Configuration
全局配置文件：颜色、字体、路径等
"""
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
STEP5_OUTPUT = BASE_DIR / 'step5_timegan' / 'output'
STEP6_OUTPUT = BASE_DIR / 'step6_newMetrcisWithTimegan' / 'output'
STEP7_OUTPUT = BASE_DIR / 'step7_performance_analysis' / 'output'
PAPER_ANALYSIS_OUTPUT = BASE_DIR / 'paper_analysis' / 'output'

OUTPUT_DIR = Path(__file__).parent / 'output'
TABLES_DIR = OUTPUT_DIR / 'tables'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# 创建输出目录
for dir_path in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

for section in ['section_b', 'section_c', 'section_d', 'section_e', 'section_f', 'section_i']:
    (FIGURES_DIR / section).mkdir(exist_ok=True)

# ============================================================================
# 模型配置
# ============================================================================
MODELS = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
MODEL_ORDER = MODELS  # 模型显示顺序

MODEL_DISPLAY_NAMES = {
    'cnn': 'CNN',
    'lstm': 'LSTM',
    'bilstm': 'BiLSTM',
    'gru': 'GRU',
    'cnn_lstm': 'CNN-LSTM',
    'tcn': 'TCN',
    'transformer': 'Transformer'
}
MODEL_LABELS = MODEL_DISPLAY_NAMES  # 别名

# ============================================================================
# 颜色方案 - 高对比学术配色（Nature/Science 风格）
# ============================================================================
MODEL_COLORS = {
    'cnn': '#4477AA',        # 蓝色
    'lstm': '#EE6677',       # 红色
    'bilstm': '#228833',     # 绿色
    'gru': '#CCBB44',        # 黄色
    'cnn_lstm': '#66CCEE',   # 青色
    'tcn': '#AA3377',        # 紫红色
    'transformer': '#BBBBBB' # 灰色
}

# 性能等级颜色
PERFORMANCE_COLORS = {
    'excellent': '#2ecc71',  # 绿色
    'good': '#3498db',       # 蓝色
    'medium': '#f39c12',     # 橙色
    'poor': '#e74c3c'        # 红色
}

# 热力图颜色
HEATMAP_CMAP = 'RdYlGn'  # 红-黄-绿
HEATMAP_CMAP_REVERSE = 'RdYlGn_r'  # 绿-黄-红（用于延迟等指标）

# ============================================================================
# 图表尺寸
# ============================================================================
FIGURE_SIZES = {
    'single': (8, 6),        # 单图
    'double': (12, 5),       # 双图并排
    'triple': (15, 5),       # 三图并排
    'wide': (14, 5),         # 宽图
    'tall': (8, 10),         # 高图
    'grid_2x2': (12, 10),    # 2x2网格
    'grid_3x3': (15, 12),    # 3x3网格
    'square': (8, 8)         # 正方形
}

# ============================================================================
# 样式文件配置
# ============================================================================
STYLES_DIR = BASE_DIR / 'styles'
# 使用大图12px样式作为默认样式
DEFAULT_STYLE = STYLES_DIR / 'sci_large_14px.mplstyle'

# 可选样式（根据需要切换）
AVAILABLE_STYLES = {
    'large_14px': STYLES_DIR / 'sci_large_14px.mplstyle',
    'large_12px': STYLES_DIR / 'sci_large_12px.mplstyle',
    'large_10px': STYLES_DIR / 'sci_large_10px.mplstyle',
    'small_12px': STYLES_DIR / 'sci_small_12px.mplstyle',
    'small_10px': STYLES_DIR / 'sci_small_10px.mplstyle',
    'small_8px': STYLES_DIR / 'sci_small_8px.mplstyle',
    'square_large_14px': STYLES_DIR / 'sci_square_large_14px.mplstyle',
    'square_large_12px': STYLES_DIR / 'sci_square_large_12px.mplstyle',
    'square_large_10px': STYLES_DIR / 'sci_square_large_10px.mplstyle',
    'square_small_12px': STYLES_DIR / 'sci_square_small_12px.mplstyle',
    'square_small_10px': STYLES_DIR / 'sci_square_small_10px.mplstyle',
    'square_small_8px': STYLES_DIR / 'sci_square_small_8px.mplstyle',
}

# ============================================================================
# 字体设置（从样式文件继承，这里作为备用）
# ============================================================================
FONT_SIZES = {
    'title': 12,
    'subtitle': 12,
    'label': 12,
    'tick': 10,
    'legend': 10,
    'annotation': 9
}

FONT_FAMILY = 'DejaVu Sans'  # 支持多语言

# ============================================================================
# 输出格式
# ============================================================================
OUTPUT_FORMATS = ['pdf']  # 输出PDF格式
DPI = 300  # 高分辨率

# ============================================================================
# 绘图风格
# ============================================================================
PLOT_STYLE = 'seaborn-v0_8-paper'  # 论文风格

# 线条样式
LINE_STYLES = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
    'dashdot': '-.'
}

LINE_WIDTH = 2.5
MARKER_SIZE = 8

# 网格
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'

# ============================================================================
# 指标配置
# ============================================================================
METRICS_CONFIG = {
    'traditional': {
        'precision': {'label': 'Precision', 'format': '.4f', 'range': [0, 1]},
        'recall': {'label': 'Recall', 'format': '.4f', 'range': [0, 1]},
        'f1': {'label': 'F1-Score', 'format': '.4f', 'range': [0, 1]}
    },
    'time_aware': {
        'dr_5s': {'label': 'DR@5s', 'format': '.4f', 'range': [0, 1]},
        'add': {'label': 'ADD (s)', 'format': '.2f', 'range': [0, None]},
        'mtbfa': {'label': 'MTBFA (h)', 'format': '.2f', 'range': [0, None]}
    },
    'efficiency': {
        'latency': {'label': 'Latency (ms)', 'format': '.2f', 'range': [0, None]},
        'throughput': {'label': 'Throughput (windows/s)', 'format': '.1f', 'range': [0, None]},
        'params': {'label': 'Parameters (M)', 'format': '.2f', 'range': [0, None]},
        'memory': {'label': 'Memory (MB)', 'format': '.2f', 'range': [0, None]}
    }
}

# DR@Δt时间阈值
DR_THRESHOLDS = [0.5, 1, 1.5, 2, 3, 5, 10, 15, 30]  # 秒

# ============================================================================
# 表格配置
# ============================================================================
TABLE_FORMATS = ['csv', 'latex']  # 输出格式

LATEX_TABLE_TEMPLATE = r"""\begin{{table}}[htbp]
\centering
\caption{{{caption}}}
\label{{tab:{label}}}
{content}
\end{{table}}
"""

# ============================================================================
# 统计显著性
# ============================================================================
SIGNIFICANCE_LEVELS = {
    'ns': 0.05,      # not significant
    '*': 0.05,       # p < 0.05
    '**': 0.01,      # p < 0.01
    '***': 0.001     # p < 0.001
}

# ============================================================================
# 场景配置（用于模型推荐）
# ============================================================================
SCENARIOS = {
    'safety_critical': {
        'name': 'Safety-Critical',
        'priority': ['dr_5s', 'add'],
        'description': 'Prioritize detection speed'
    },
    'low_false_alarm': {
        'name': 'Low False Alarm',
        'priority': ['mtbfa', 'precision'],
        'description': 'Prioritize low false alarm rate'
    },
    'balanced': {
        'name': 'Balanced',
        'priority': ['f1', 'dr_5s', 'mtbfa'],
        'description': 'Balance all metrics'
    },
    'resource_constrained': {
        'name': 'Resource-Constrained',
        'priority': ['latency', 'params', 'dr_5s'],
        'description': 'Prioritize computational efficiency'
    }
}

# ============================================================================
# 攻击类型配置
# ============================================================================
ATTACK_TYPES = {
    'step': {'label': 'Step', 'color': '#e74c3c'},
    'drift': {'label': 'Drift', 'color': '#3498db'},
    'delay': {'label': 'Delay', 'color': '#f39c12'},
    'takeover': {'label': 'Takeover', 'color': '#9b59b6'}
}

# 如果有细分攻击类型
ATTACK_TYPES_DETAILED = {
    'step': {'label': 'Step', 'color': '#e74c3c'},
    'drift_ramp': {'label': 'Drift (Ramp)', 'color': '#3498db'},
    'drift_sigmoid': {'label': 'Drift (Sigmoid)', 'color': '#2980b9'},
    'delay': {'label': 'Delay', 'color': '#f39c12'},
    'takeover_step': {'label': 'Takeover (Step)', 'color': '#9b59b6'},
    'takeover_ramp': {'label': 'Takeover (Ramp)', 'color': '#8e44ad'}
}

# ============================================================================
# 图表标题和标签模板
# ============================================================================
FIGURE_TITLES = {
    'b1': 'Traditional Metrics Comparison',
    'b2': 'Time-Aware Metrics Comparison',
    'b3': 'Precision vs MTBFA: The Paradox',
    'b4': 'FP Window Aggregation Analysis',
    'c1': 'Detection Rate at Different Time Thresholds',
    'c2': 'Key Time Threshold Comparison',
    'c3': 'Detection Speed Distribution',
    'd1': 'FP Event Duration Distribution',
    'd2': 'FP Event Duration Comparison',
    'd3': 'Cumulative Distribution of FP Durations',
    'd4': '100-Hour Operation Scenario',
    'e1': 'Pareto Frontier: DR@5s vs MTBFA',
    'e2': 'ADD vs DR@5s Correlation',
    'e3': '3D Trade-off Analysis',
    'e4': 'Scenario-based Model Recommendation',
    'f1': 'Per-Attack-Type Detection Rate',
    'f2': 'Per-Attack-Type Detection Delay',
    'f3': 'Attack Detection Difficulty',
    'i1': 'Inference Latency Comparison',
    'i2': 'Model Complexity Analysis',
    'i3': 'Performance-Efficiency Trade-off',
    'i4': 'Comprehensive Model Evaluation'
}

# ============================================================================
# 辅助函数
# ============================================================================
def get_model_color(model: str) -> str:
    """获取模型颜色"""
    return MODEL_COLORS.get(model.lower(), '#333333')

def get_model_name(model: str) -> str:
    """获取模型显示名称"""
    return MODEL_DISPLAY_NAMES.get(model.lower(), model.upper())

def get_figure_path(section: str, figure_name: str, format: str = 'png') -> Path:
    """获取图表保存路径"""
    return FIGURES_DIR / section / f"{figure_name}.{format}"

def get_table_path(table_name: str, format: str = 'csv') -> Path:
    """获取表格保存路径"""
    return TABLES_DIR / f"{table_name}.{format}"
