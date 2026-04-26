"""
Step5: TimeGAN数据增强配置
目标：通过扩充正常飞行数据，降低训练集攻击比例，改善FPR
"""
import numpy as np

# ============================================================================
# 基础配置
# ============================================================================
RANDOM_SEED = 42
DATA_PATH = "../data/src/flights.csv"
OUTPUT_DIR = "./output"

# ============================================================================
# TimeGAN配置
# ============================================================================
# 扩充倍数（生成多少倍的正常数据）
TIMEGAN_EXPANSION_FACTOR = 5  # 5倍扩充

# TimeGAN模型参数
TIMEGAN_SEQUENCE_LENGTH = 150  # 训练序列长度（150步约1.5秒）
TIMEGAN_HIDDEN_DIM = 32        # 隐藏层维度
TIMEGAN_NUM_LAYERS = 3         # GRU层数

# TimeGAN训练参数
TIMEGAN_BATCH_SIZE = 128
TIMEGAN_EPOCHS_AE = 100        # 自编码器训练轮数
TIMEGAN_EPOCHS_SUP = 100       # Supervisor训练轮数
TIMEGAN_EPOCHS_GAN = 200       # GAN训练轮数
TIMEGAN_LR = 0.001             # 学习率

# TimeGAN设备
TIMEGAN_DEVICE = 'cuda'  # 'cuda' or 'cpu'

# ============================================================================
# 数据集划分比例
# ============================================================================
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# ============================================================================
# 攻击注入配置（降低比例）
# ============================================================================
# 攻击类型
ATTACK_TYPES = [
    'step',
    'drift_ramp',
    'drift_sigmoid',
    'delay',
    'takeover_step',
    'takeover_ramp'
]

# 攻击比例（提高测试集攻击比例以获得足够的测试样本）
TRAIN_ATTACK_RATIO = 0.15  # 15%（vs step3b的60%）
VAL_ATTACK_RATIO = 0.10    # 10%（vs step3b的50%）
TEST_ATTACK_RATIO = 0.25   # 25%（提升以增加测试攻击数量，原5%仅51个攻击样本太少）

# 测试集保证覆盖
MIN_TEST_FLIGHTS_PER_ATTACK = 2  # 每种攻击至少2个flights

# ============================================================================
# 攻击参数配置（复用step3b）
# ============================================================================

# Time window constraints（调整为更小的缓冲区，适应合成飞行）
ATTACK_START_BUFFER = 2.0  # seconds after takeoff (减小从10.0到2.0)
ATTACK_END_BUFFER = 2.0    # seconds before landing (减小从10.0到2.0)

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
CONSISTENCY_MODES = ['pos_vel_acc']

# Velocity epsilon
VELOCITY_EPSILON = 0.01  # m/s

# Step attack parameters
STEP1_ATTACK_DURATION = 5.0
STEP1_ATTACK_TIME_CONSTANT = 1.0
STEP1_VELOCITY_TRANSIENT_MIN = 0.5
STEP1_VELOCITY_TRANSIENT_MAX = 2.0

# Drift profiles
DRIFT_PROFILES = ['ramp', 'sigmoid']
DRIFT_DURATIONS = [5.0, 10.0, 20.0]
SIGMOID_K = 10.0

# Delay amounts
DELAY_SECONDS = [1.0, 3.0, 5.0]

# Takeover parameters
TAKEOVER_OFFSET_PROFILES = ['step', 'ramp']
TAKEOVER_DURATIONS = [0.5, 1.0, 2.0]
TAKEOVER_ALPHAS = [0.3, 0.5, 0.7]
TAKEOVER_TAUS = [0.5, 1.0, 2.0]

# Attack type mapping
ATTACK_TYPE_MAP = {
    'step': 'step',
    'drift_ramp': 'drift',
    'drift_sigmoid': 'drift',
    'delay': 'delay',
    'takeover_step': 'takeover',
    'takeover_ramp': 'takeover'
}

# Attack-specific parameters
ATTACK_PARAMS = {
    'step': {},
    'drift_ramp': {'profile': 'ramp'},
    'drift_sigmoid': {'profile': 'sigmoid'},
    'delay': {},
    'takeover_step': {'offset_profile': 'step'},
    'takeover_ramp': {'offset_profile': 'ramp'}
}

# ============================================================================
# 窗口参数
# ============================================================================
WINDOW_SIZE = 50  # 窗口大小
STEP_SIZE = 5     # 滑动步长
SAMPLING_RATE = 100  # Hz

# ============================================================================
# 特征配置
# ============================================================================
POSITION_FEATURES = ['position_x', 'position_y', 'position_z']
VELOCITY_FEATURES = ['velocity_x', 'velocity_y', 'velocity_z']
ACCELERATION_FEATURES = ['linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']

# TimeGAN训练使用的核心物理特征（10个）
# 这些特征具有时序相关性，适合TimeGAN学习
TIMEGAN_FEATURES = [
    # 位置 (3个)
    'position_x', 'position_y', 'position_z',
    # 速度 (3个)
    'velocity_x', 'velocity_y', 'velocity_z',
    # 加速度 (3个)
    'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
    # 角速度 (1个) - 偏航最重要
    'angular_z'
]

# ============================================================================
# 训练配置
# ============================================================================
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# ============================================================================
# 模型配置
# ============================================================================
MODEL_TYPES = ['lstm', 'gru', 'cnn_lstm', 'transformer']
DEFAULT_MODEL = 'lstm'

# LSTM/GRU配置
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

# CNN-LSTM配置
CNN_FILTERS = [32, 64]
CNN_KERNEL_SIZE = 3

# Transformer配置
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128

# ============================================================================
# 窗口和训练配置（与step3b兼容）
# ============================================================================
WINDOW_SIZE = 50  # 窗口大小
WINDOW_STRIDE = 1  # 窗口步长（每步1个样本，最大重叠）
STEP_SIZE = 5  # 与step3b兼容

# 训练超参数
BATCH_SIZE = 64
MAX_EPOCHS = 100
NUM_EPOCHS = 100  # 与MAX_EPOCHS相同（兼容性）
LEARNING_RATE = 0.0001  # 降低学习率防止NaN
EARLY_STOP_PATIENCE = 10
GRADIENT_CLIP = 1.0  # 梯度裁剪

# 评估配置
FPR_THRESHOLDS = [0.001, 0.01, 0.05]  # FPR阈值用于评估

# 窗口平衡（不平衡，使用实际比例）
TRAIN_POS_RATIO = None  # None表示不平衡
VAL_POS_RATIO = None
TEST_POS_RATIO = None

# 设备
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# ============================================================================
# 数据泄漏修复配置（新增）
# ============================================================================

# 数据划分比例（Split-First策略）- 标准60/20/20划分
SPLIT_RATIOS = {
    'train': 0.60,  # 125 / 209 ≈ 60% (标准训练集比例)
    'val': 0.20,    # 42 / 209 ≈ 20% (标准验证集比例)
    'test': 0.20    # 42 / 209 ≈ 20% (标准测试集比例)
}

# 每个split的扩充倍数（基于原始flight数量）- 统一5x增强
AUGMENTATION_MULTIPLIERS = {
    'train': 5,     # 125 × 5 = 625 synthetic → 750 total (含原始)
    'val': 5,       # 42 × 5 = 210 synthetic → 252 total (含原始)
    'test': 5       # 42 × 5 = 210 synthetic → 252 total (含原始)
}

# 每个split的攻击注入比例
ATTACK_INJECTION_RATIOS = {
    'train': 0.08,  # 8% attack (89 flights)
    'val': 0.15,    # 15% attack (22 flights)
    'test': 0.40    # 40% attack (53 flights，约8-9每种)
}

# 测试集每种攻击类型最少实例数
TEST_ATTACKS_PER_TYPE = 8  # 6种攻击 × 8 = 48，加上余量5 → 53 total

# 合成flight ID范围（避免与原始ID冲突）
SYNTHETIC_FLIGHT_ID_RANGES = {
    'train': (1000, 2000),  # [1000, 1999] - 1000个合成IDs
    'val': (2000, 2200),    # [2000, 2199] - 200个合成IDs
    'test': (2200, 2400)    # [2200, 2399] - 200个合成IDs
}
