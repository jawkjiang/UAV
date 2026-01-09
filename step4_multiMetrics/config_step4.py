"""
Step 4 配置文件 - 电力系统无人机巡检场景指标计算
Configuration for Power Grid UAV Inspection Metrics Calculation
"""

import os

# ============================================================================
# 路径配置
# ============================================================================

# Step3输出目录 (实验结果来源)
STEP3_OUTPUT_DIR = '../step3_multiModel/output'

# Step4输出目录
STEP4_OUTPUT_DIR = './output/power_grid_analysis'

# 可视化输出目录
VISUALIZATION_DIR = os.path.join(STEP4_OUTPUT_DIR, 'visualizations')


# ============================================================================
# 数据参数配置
# ============================================================================

# 窗口和采样参数
WINDOW_SIZE = 50  # 窗口大小
STEP_SIZE = 5     # 步长
SAMPLING_INTERVAL = 0.5  # 采样间隔 (秒)

# 检测阈值
DETECTION_THRESHOLD = 0.5  # 二分类阈值


# ============================================================================
# 指标体系1: 代价权衡参数
# ============================================================================

# 代价参数
COST_FP = 1    # 误报代价 (False Positive Cost)
COST_FN = 50   # 漏报代价 (False Negative Cost)

# F-beta参数
F_BETA_VALUES = [0.5, 1.0, 2.0, 5.0]

# Cost-Benefit参数
ATTACK_PROBABILITY = 0.01  # 攻击概率 (1%)
ATTACK_DAMAGE = 1000       # 攻击损失 (单位成本)
FALSE_ALARM_COST = 1       # 误报处理成本
NORMAL_OPERATION_FREQ = 1  # 正常运行频率


# ============================================================================
# 指标体系2: 时效性参数
# ============================================================================

# Rapid Response Rate的时间窗口 (单位: 窗口数)
RRR_WINDOWS = [5, 10, 20]

# 检测延迟分位数
DELAY_PERCENTILES = [50, 90, 95, 99]


# ============================================================================
# 指标体系3: 可靠性参数
# ============================================================================

# 目标TPR (True Positive Rate)
TARGET_TPR = 0.99

# 飞行参数
SAMPLES_PER_FLIGHT = 3600  # 每次飞行样本数 (30分钟 * 60秒 * 2采样/秒)
INSPECTION_TIME = 10        # 误报检查时间 (分钟)
FLIGHT_DURATION = 30        # 飞行时长 (分钟)


# ============================================================================
# 指标体系4: 风险加权参数
# ============================================================================

# 攻击严重度权重
ATTACK_WEIGHTS = {
    'replay_same_hard': 5,   # 最危险: 完全隐蔽的重放攻击
    'replay_other_soft': 4,  # 次危险: 部分隐蔽的重放攻击
    'delay': 3,              # 中等: 时间延迟攻击
    'drift_ramp': 2,         # 较低: 缓慢漂移
    'drift_sigmoid': 2,      # 较低: S型漂移
    'step': 1,               # 最低: 突变攻击
    'takeover_step': 1,      # 最低: 接管突变
    'takeover_ramp': 1       # 最低: 接管渐变
}

# 关键攻击类型 (用于CMR计算)
CRITICAL_ATTACKS = ['replay_same_hard', 'replay_other_soft']


# ============================================================================
# 电力巡检风险评估模型 (新增)
# ============================================================================

POWER_GRID_RISK_MODEL = {
    # A. 碰撞风险权重（基于偏移幅度）
    # 假设安全走廊宽度 = 20米
    'collision_risk': {
        'thresholds': [5, 10, 20],  # 米
        'weights': [1, 3, 10, 50],  # 风险倍数
        'coefficient': 0.5  # 总风险中的权重
    },
    
    # B. 任务失效风险（基于持续时间）
    # 假设每秒应检测2米线路
    'mission_failure_risk': {
        'thresholds': [10, 30],  # 秒
        'weights': [1, 5, 20],  # 风险倍数
        'coefficient': 0.3
    },
    
    # C. 操作失控风险（基于变化速率/攻击类型）
    # 人类反应时间 ≈ 1秒
    'control_loss_risk': {
        'type_weights': {
            'step': 10,           # 瞬间突变
            'ramp': 3,            # 线性渐变
            'sigmoid': 5,         # S型渐变
            'delay': 4,           # 延迟（中等）
            'replay': 8,          # 重放（较高隐蔽性）
        },
        'coefficient': 0.1
    },
    
    # D. 隐蔽累积风险（特殊攻击类型加权）
    'stealth_accumulation_risk': {
        'type_weights': {
            'replay_same_hard': 15,   # 循环检测，永久遗漏
            'replay_other_soft': 12,  # 跨飞行重放
            'delay': 8,               # 延迟发现故障
            'takeover_ramp': 10,      # 缓慢接管
            'takeover_step': 12,      # 突然接管
            'drift_sigmoid': 6,       # S型难察觉
            'drift_ramp': 4,          # 线性易察觉
            'step': 2,                # 突变明显
        },
        'coefficient': 0.1
    },
    
    # 安全关键阈值（用于CTDR指标）
    'critical_thresholds': {
        'magnitude': 10.0,  # 米（可能撞线）
        'duration': 20.0,   # 秒（遗漏大段检测）
    }
}

# 幅度分层参数（用于MSDR指标）
MAGNITUDE_BINS = {
    'low': (0, 5),      # 小幅度 (<5m)
    'mid': (5, 10),     # 中幅度 (5-10m)  
    'high': (10, float('inf'))  # 大幅度 (>10m)
}


# ============================================================================
# 指标体系5: 边缘部署参数
# ============================================================================

# 模型参数数量
MODEL_PARAMS = {
    'cnn': 263873,
    'lstm': 215873,
    'bilstm': 150337,
    'gru': 164545,
    'cnn_lstm': 301953,
    'tcn': 77313,
    'transformer': 103489
}

# 延迟约束 (毫秒)
TARGET_LATENCY_MS = 100

# 数据类型 (用于计算模型大小)
DTYPE_BYTES = 4  # float32


# ============================================================================
# 区分度评价标准
# ============================================================================

# 变异系数 (CV) 评级阈值
CV_RATING_THRESHOLDS = {
    'excellent': 0.15,  # CV > 0.15: ⭐⭐⭐⭐⭐
    'good': 0.10,       # CV > 0.10: ⭐⭐⭐⭐
    'moderate': 0.05,   # CV > 0.05: ⭐⭐⭐
    'low': 0.0          # CV <= 0.05: ⭐⭐
}


# ============================================================================
# 可视化配置
# ============================================================================

# 图表DPI
FIGURE_DPI = 300
SAVE_DPI = 300

# 中文字体
CHINESE_FONTS = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

# 颜色方案
COLOR_SCHEMES = {
    'cost': 'steelblue',
    'delay': 'RdYlGn_r',
    'risk': 'tab10',
    'edge': 'RdYlGn',
    'severity': 'Reds'
}

# 图表大小
FIGURE_SIZES = {
    'single': (10, 6),
    'double': (14, 5),
    'large': (12, 8),
    'radar': (10, 10)
}


# ============================================================================
# 输出文件名配置
# ============================================================================

OUTPUT_FILES = {
    # 指标CSV文件
    'metrics1': 'metrics1_cost_sensitive.csv',
    'metrics2': 'metrics2_detection_delay.csv',
    'metrics3': 'metrics3_reliability.csv',
    'metrics3_robust': 'metrics3_robustness.csv',
    'metrics4': 'metrics4_risk_weighted.csv',
    'metrics5': 'metrics5_edge_deployment.csv',
    
    # 分析结果
    'rankings': 'comprehensive_rankings.csv',
    'discrimination': 'discrimination_analysis.csv',
    
    # 报告
    'report': 'POWER_GRID_REPORT.md',
    
    # 可视化
    'viz_cost': 'cost_sensitive_comparison.png',
    'viz_delay': 'detection_delay_distribution.png',
    'viz_risk': 'risk_weighted_performance.png',
    'viz_edge': 'edge_deployment_suitability.png',
    'viz_ranking': 'comprehensive_ranking.png',
    'viz_severity': 'attack_severity_analysis.png'
}


# ============================================================================
# 日志配置
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ============================================================================
# 实验设置
# ============================================================================

# 模型列表
MODELS = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']

# 攻击类型列表
ATTACK_TYPES = [
    'step', 'drift_ramp', 'drift_sigmoid', 
    'delay', 'replay_same_hard', 'replay_other_soft',
    'takeover_step', 'takeover_ramp'
]


# ============================================================================
# 推荐配置
# ============================================================================

# 电力巡检场景推荐的指标优先级
RECOMMENDED_METRICS = [
    ('指标体系4', 'risk_weighted', '风险加权 - 重视重放攻击检测'),
    ('指标体系2', 'detection_delay', '时效性 - 确保快速响应'),
    ('指标体系1', 'cost_sensitive', '代价权衡 - 平衡误报和漏报')
]

# 不同约束下的模型选择策略
MODEL_SELECTION_STRATEGIES = {
    'performance_first': '选择RWDR最高的模型 (风险加权检测率)',
    'realtime_first': '选择MTTD最低的模型 (平均检测延迟)',
    'resource_constrained': '选择PER最高的轻量级模型 (性能效率比)',
    'balanced': '综合考虑多个指标的平均排名'
}
