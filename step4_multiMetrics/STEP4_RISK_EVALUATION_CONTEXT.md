# Step4 风险导向评估指标 - 完整实现Context

## 📋 任务概述

在step4中实现**基于偏移幅度和持续时间的电力巡检风险评估体系**，作为论文的核心创新点。

### 核心创新叙事

```
标题方向：
"Risk-Aware GPS Spoofing Detection for Power Grid UAV Inspection: 
 Beyond Accuracy towards Safety-Critical Performance Evaluation"

创新点：
1. 现有指标（F1/AUC）假设所有样本同等重要，但电力巡检场景有明确的物理风险差异
2. 提出基于攻击幅度×持续时间的风险量化模型
3. 设计4个新指标：RWMR, MSDR, DS-FNR, CTDR
4. 实验表明传统最优模型在高风险场景下表现不同

为什么投电力期刊：
- 考虑了电力线路碰撞风险（偏移→撞线/撞塔）
- 考虑了巡检任务失效影响（时长→遗漏缺陷）
- 考虑了操作员反应时间（突变vs渐变）
- 提出了电力场景的安全关键阈值
```

---

## ✅ 数据可行性确认

### 已验证：数据完全支持风险评估

#### 1. attack_info.csv 包含完整攻击参数

**位置**: `step3_multiModel/output/{attack_type}/{model}/test_attack_info.csv`

**可用字段**:
```csv
- flight: 飞行编号
- attacked: 是否攻击
- attack_type: 攻击类型
- attack_start_time: 攻击开始时间 (秒)
- attack_params: 攻击参数字典（字符串形式）
- consistency_mode: 一致性模式
- reason: 未攻击原因
```

**attack_params包含的风险因子**（以字符串形式存储）:

```python
# Step攻击示例:
"{'magnitude': np.float64(30.0), 'direction_mode': 'random_xy', 
  'direction_vector': [0.90, 0.43], 'attack_duration': 3.0, 'time_constant': 1.0}"
→ magnitude = 30.0 米 ✅

# Drift攻击示例:
"{'profile': 'ramp', 'T_drift': 10.0, 'T_drift_clipped': 10.0, 'M': 15.0, 
  'direction_mode': 'random_xy', 'direction_vector': [0.71, 0.71]}"
→ M = 最终幅度15米, T_drift = 持续时间10秒 ✅

# Delay攻击示例:
"{'delay_seconds': np.float64(1.0), 'source_time_clamped_count': 0}"
→ delay_seconds = 1.0秒 ✅

# Replay攻击示例:
"{'segment_duration': 10.0, 'donor_source': 'same_flight_earlier', 
  'stitching': 'hard', 'transition_seconds': 0.5, ...}"
→ segment_duration = 10秒 ✅

# Takeover攻击示例:
"{'offset_profile': 'ramp', 'magnitude': 15.0, 'duration': 5.0, 
  'alpha': 0.5, 'tau': 1.0, ...}"
→ magnitude = 15米, duration = 5秒 ✅
```

#### 2. config_step2.py 定义的攻击参数范围

```python
# 幅度范围
ATTACK_MAGNITUDES = [5.0, 15.0, 30.0]  # 米

# 持续时间范围
DRIFT_DURATIONS = [5.0, 10.0, 20.0]  # 秒
DELAY_SECONDS = [1.0, 3.0, 5.0]  # 秒
REPLAY_SEGMENT_DURATIONS = [5.0, 10.0, 15.0]  # 秒
TAKEOVER_DURATIONS = [3.0, 5.0, 10.0]  # 秒

# 攻击开始时间约束
ATTACK_START_BUFFER = 10.0  # 起飞后10秒
ATTACK_END_BUFFER = 10.0    # 降落前10秒

# 采样间隔
SAMPLING_INTERVAL = 0.5  # 秒/样本
```

#### 3. test_predictions.npz 包含预测概率

**位置**: `step3_multiModel/output/{attack_type}/{model}/test_predictions.npz`

**字段**:
```python
- y_true: 真实标签 (0/1)
- y_pred: 预测类别 (0/1)
- y_prob: 预测概率 [0, 1] ✅ 用于计算置信度
- sample_indices: 样本索引
```

---

## 🎯 需要实现的风险模型

### 风险因子提取函数

```python
def extract_risk_factors(attack_info_row):
    """
    从attack_info.csv的attack_params字段提取风险因子
    
    Returns:
        {
            'magnitude': float,  # 米
            'duration': float,   # 秒
            'attack_type': str,
            'profile': str,      # step/ramp/sigmoid
        }
    """
    import ast
    
    attack_type = attack_info_row['attack_type']
    params_str = attack_info_row['attack_params']
    
    if attack_type == 'none':
        return None
    
    # 解析字符串为字典（处理np.float64）
    params = ast.literal_eval(params_str.replace('np.float64(', '').replace(')', ''))
    
    if attack_type == 'step':
        magnitude = params['magnitude']
        duration = params['attack_duration']  # 固定3秒
        profile = 'step'
    
    elif attack_type in ['drift_ramp', 'drift_sigmoid']:
        magnitude = params['M']
        duration = params['T_drift']
        profile = params['profile']
    
    elif attack_type == 'delay':
        magnitude = params['delay_seconds'] * 5.0  # 延迟秒数×典型速度 估算偏移
        duration = 999.0  # 持续到飞行结束（特殊处理）
        profile = 'delay'
    
    elif attack_type in ['replay_same_hard', 'replay_other_soft']:
        magnitude = 10.0  # replay没有明确幅度，用中等值
        duration = params['segment_duration']
        profile = 'replay'
    
    elif attack_type in ['takeover_step', 'takeover_ramp']:
        magnitude = params['magnitude']
        duration = params['duration']
        profile = params['offset_profile']
    
    return {
        'magnitude': magnitude,
        'duration': duration,
        'attack_type': attack_type,
        'profile': profile
    }
```

### 电力巡检风险评分模型

```python
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

def calculate_attack_risk_score(magnitude, duration, attack_type, profile):
    """
    计算单个攻击的风险分数
    
    ARS = α×CR + β×MFR + γ×CLR + δ×SAR
    
    Returns:
        float: 攻击风险分数 [0, 100+]
    """
    model = POWER_GRID_RISK_MODEL
    
    # A. 碰撞风险
    cr_thresholds = model['collision_risk']['thresholds']
    cr_weights = model['collision_risk']['weights']
    if magnitude < cr_thresholds[0]:
        cr = cr_weights[0]
    elif magnitude < cr_thresholds[1]:
        cr = cr_weights[1]
    elif magnitude < cr_thresholds[2]:
        cr = cr_weights[2]
    else:
        cr = cr_weights[3]
    cr = cr * magnitude  # 乘以实际偏移量
    
    # B. 任务失效风险
    mfr_thresholds = model['mission_failure_risk']['thresholds']
    mfr_weights = model['mission_failure_risk']['weights']
    if duration < mfr_thresholds[0]:
        mfr = mfr_weights[0]
    elif duration < mfr_thresholds[1]:
        mfr = mfr_weights[1]
    else:
        mfr = mfr_weights[2]
    mfr = mfr * duration  # 乘以实际时长
    
    # C. 操作失控风险
    clr = model['control_loss_risk']['type_weights'].get(profile, 5)
    
    # D. 隐蔽累积风险
    sar = model['stealth_accumulation_risk']['type_weights'].get(attack_type, 5)
    
    # 加权求和
    α = model['collision_risk']['coefficient']
    β = model['mission_failure_risk']['coefficient']
    γ = model['control_loss_risk']['coefficient']
    δ = model['stealth_accumulation_risk']['coefficient']
    
    ars = α * cr + β * mfr + γ * clr + δ * sar
    
    return ars
```

---

## 🎯 需要实现的4个创新指标

### 指标1: Risk-Weighted Miss Rate (RWMR)

```python
def calculate_rwmr(y_true, y_pred, attack_risk_scores):
    """
    风险加权漏报率
    
    RWMR = Σ(未检出攻击的风险分数) / Σ(所有攻击的风险分数)
    
    Args:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        attack_risk_scores: 每个样本的风险分数数组
    
    Returns:
        float: RWMR值 [0, 1]
    
    意义：
        - 传统FNR只看漏了多少个（数量）
        - RWMR看漏掉的攻击造成的风险有多大（危害）
        - 如果漏掉的都是低风险攻击 → RWMR低（可接受）
        - 如果漏掉的都是高风险攻击 → RWMR高（危险）
    """
    attack_mask = (y_true == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    total_attack_risk = attack_risk_scores[attack_mask].sum()
    missed_attack_risk = attack_risk_scores[fn_mask].sum()
    
    if total_attack_risk == 0:
        return 0.0
    
    return missed_attack_risk / total_attack_risk
```

### 指标2: Magnitude-Stratified Detection Rate (MSDR)

```python
def calculate_msdr(y_true, y_pred, magnitudes):
    """
    按幅度分层的检出率
    
    MSDR = {
        "小幅度 (<5m)":   TPR_low,
        "中幅度 (5-10m)": TPR_mid,
        "大幅度 (>10m)":  TPR_high
    }
    
    期望：大幅度攻击必须100%检出（安全要求）
    """
    bins = [0, 5, 10, float('inf')]
    labels = ['low', 'mid', 'high']
    
    msdr = {}
    attack_mask = (y_true == 1)
    
    for i, label in enumerate(labels):
        bin_mask = (magnitudes >= bins[i]) & (magnitudes < bins[i+1]) & attack_mask
        
        if bin_mask.sum() == 0:
            msdr[label] = None
        else:
            tp = ((y_pred == 1) & bin_mask).sum()
            fn = ((y_pred == 0) & bin_mask).sum()
            msdr[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return msdr
```

### 指标3: Duration-Sensitive False Negative Rate (DS-FNR)

```python
def calculate_ds_fnr(y_true, y_pred, durations):
    """
    时长敏感的漏报率
    
    DS-FNR = Σ(FN样本的持续时间) / Σ(所有攻击样本的持续时间)
    
    vs 传统FNR = FN数量 / 攻击数量
    
    意义：
        - 漏了1个30秒攻击 vs 漏了3个5秒攻击
        - FNR相同（都是漏了样本）
        - 但DS-FNR前者更高（30秒 vs 15秒）→ 更危险
    """
    attack_mask = (y_true == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    total_attack_duration = durations[attack_mask].sum()
    missed_attack_duration = durations[fn_mask].sum()
    
    if total_attack_duration == 0:
        return 0.0
    
    return missed_attack_duration / total_attack_duration
```

### 指标4: Critical Threshold Detection Rate (CTDR)

```python
def calculate_ctdr(y_true, y_pred, magnitudes, durations):
    """
    关键阈值检出率
    
    超过安全阈值的攻击检出率：
    - 幅度 > 10米（可能撞线）
    - 或 时长 > 20秒（遗漏检测）
    
    CTDR = 检出的高危攻击 / 所有高危攻击
    
    vs TPR = 检出的所有攻击 / 所有攻击
    
    强调：不是所有攻击都同等重要，高危攻击必须检出
    """
    critical_mag_threshold = 10.0
    critical_dur_threshold = 20.0
    
    critical_mask = (y_true == 1) & (
        (magnitudes > critical_mag_threshold) | 
        (durations > critical_dur_threshold)
    )
    
    if critical_mask.sum() == 0:
        return None
    
    critical_detected = ((y_pred == 1) & critical_mask).sum()
    critical_total = critical_mask.sum()
    
    return critical_detected / critical_total
```

---

## 📊 输出文件规范

### 1. risk_weighted_performance.csv

```csv
model_type,attack_type,RWMR,MSDR_low,MSDR_mid,MSDR_high,DS_FNR,CTDR,avg_risk_score
cnn,step,0.023,1.0,1.0,1.0,0.015,1.0,125.3
lstm,step,0.0,1.0,1.0,1.0,0.0,1.0,125.3
...
cnn,replay_same_hard,0.782,0.45,0.52,0.68,0.691,0.601,287.6
...
```

### 2. magnitude_duration_analysis.csv

```csv
attack_type,magnitude_bin,duration_bin,attack_count,avg_tpr,avg_risk_score
step,0-5,0-10,150,0.98,45.2
step,5-10,0-10,142,1.0,98.5
step,10-20,0-10,138,1.0,215.7
...
replay_same_hard,0-5,10-20,85,0.32,178.4
...
```

### 3. 可视化

#### magnitude_vs_tpr.png
- 8个子图（每个攻击类型）
- X轴：攻击幅度（米）
- Y轴：TPR
- 7条曲线（每个模型）
- 标注临界阈值线（10米）

#### duration_vs_tpr.png
- 8个子图
- X轴：持续时间（秒）
- Y轴：TPR
- 标注临界阈值线（20秒）

#### risk_heatmap.png
- 幅度×时长的2D热图
- 颜色表示TPR
- 每个模型/攻击一个子图

#### rwmr_comparison.png
- 柱状图：模型×攻击类型的RWMR值
- 对比传统FNR和RWMR的差异

---

## 📝 报告输出：RISK_ANALYSIS.md

```markdown
# 电力巡检场景GPS欺骗检测风险评估报告

## 执行摘要

本报告从电力系统安全角度评估各模型的检测性能，重点关注：
1. 高风险攻击（大幅度偏移、长时间攻击）的检出能力
2. 漏报造成的物理风险（碰撞、任务失效）
3. 传统指标无法揭示的安全盲区

## 风险模型说明

### 物理风险因子
- 碰撞风险（CR）：偏移距离 × 风险权重
- 任务失效风险（MFR）：持续时间 × 遗漏检测权重
- 操作失控风险（CLR）：攻击类型固有风险
- 隐蔽累积风险（SAR）：特殊攻击类型加权

### 安全关键阈值
- 幅度阈值：10米（超过可能撞击高压线/铁塔）
- 时长阈值：20秒（遗漏大段线路检测）

## 关键发现

### 发现1: 传统最优模型在高风险场景下表现不同
- CNN-LSTM在整体F1最高（0.95）
- 但在大幅度攻击（>10m）的TPR仅0.82
- LSTM虽然整体F1稍低（0.93），但MSDR_high=0.98

### 发现2: 重放攻击的风险被低估
- 传统FNR = 35%（看起来中等）
- 但RWMR = 68%（风险加权后很高）
- 原因：重放攻击持续时间长，累积风险大

### 发现3: 时长敏感性揭示新洞察
- Step攻击：FNR = DS-FNR（瞬时攻击）
- Drift攻击：DS-FNR > FNR × 2（长时间攻击漏报更严重）

## 部署建议

### 场景A: 高压线密集区（碰撞风险优先）
推荐模型：LSTM（MSDR_high=0.98, CTDR=0.97）

### 场景B: 长距离巡检（任务失效风险优先）
推荐模型：BiLSTM（DS-FNR最低，长时间攻击检测好）

### 场景C: 综合场景
推荐模型：TCN（RWMR平衡，参数量小）

## 与传统指标对比

| 模型 | F1 (传统) | RWMR (风险) | 排名变化 |
|------|-----------|-------------|---------|
| CNN-LSTM | 0.952 (1st) | 0.245 (3rd) | ↓2 |
| LSTM | 0.948 (2nd) | 0.187 (1st) | ↑1 |
| BiLSTM | 0.945 (3rd) | 0.201 (2nd) | ↑1 |

→ 风险导向评估改变了模型选择！
```

---

## 🔧 实现步骤

### Step 1: 增强power_grid_metrics.py

在PowerGridMetrics类中添加：

```python
def calculate_risk_metrics(self, experiment_dir: str) -> Dict:
    """
    计算风险导向指标
    
    Returns:
        {
            'RWMR': float,
            'MSDR': {'low': float, 'mid': float, 'high': float},
            'DS_FNR': float,
            'CTDR': float,
            'risk_distribution': {...}
        }
    """
    # 1. 加载attack_info.csv和test_predictions.npz
    # 2. 提取风险因子（magnitude, duration）
    # 3. 计算风险分数
    # 4. 计算4个指标
    # 5. 返回结果
```

### Step 2: 创建risk_analysis.py

新文件专门处理风险可视化和报告：

```python
# risk_analysis.py

def generate_magnitude_vs_tpr_plot(...)
def generate_duration_vs_tpr_plot(...)
def generate_risk_heatmap(...)
def generate_rwmr_comparison(...)
def generate_risk_report_md(...)
```

### Step 3: 更新config_step4.py

添加风险模型配置（上述POWER_GRID_RISK_MODEL）

### Step 4: 更新main.py

```python
# 在现有流程后添加
print("\n=== 计算风险导向指标 ===")
from risk_analysis import RiskAnalyzer
risk_analyzer = RiskAnalyzer()
risk_analyzer.analyze_all_experiments()
risk_analyzer.generate_visualizations()
risk_analyzer.generate_report()
```

---

## ⚠️ 注意事项

### 1. attack_params字符串解析

attack_params是字符串，包含`np.float64()`：

```python
# 错误解析会报错
import json
params = json.loads(attack_params)  # ❌ 会失败

# 正确解析方法
import ast
params_str = attack_params.replace('np.float64(', '').replace(')', '')
params = ast.literal_eval(params_str)  # ✅ 正确
```

### 2. Delay和Replay的幅度处理

这两种攻击没有直接的magnitude参数：

```python
# Delay: 用延迟秒数 × 典型速度估算
magnitude = delay_seconds * 5.0  # 假设UAV速度5m/s

# Replay: 用中等值或根据donor segment计算
magnitude = 10.0  # 保守估计中等幅度
```

### 3. 窗口级别 vs 攻击级别

当前predictions是窗口级别，需要聚合到攻击级别：

```python
# 对于每个攻击段，统计：
- 有多少窗口被检出（窗口级TPR）
- 攻击是否被检出（至少1个窗口检出 = 攻击检出）

# 建议：使用攻击级别评估（更符合实际）
```

### 4. 正常样本的风险分数

正常飞行段risk_score = 0（用于归一化）

---

## 🎯 预期成果

实现后应该能够回答：

1. **哪些模型更适合高风险场景部署？**
   - 传统F1最优 ≠ 风险场景最优

2. **模型的安全盲区在哪里？**
   - 大幅度攻击检出率如何？
   - 长时间攻击检出率如何？

3. **如何设置阈值和报警策略？**
   - 高风险区域降低阈值
   - 低风险区域提高阈值

4. **论文创新点的数据支撑**
   - 风险加权后排名变化（表格）
   - 幅度-TPR曲线（图）
   - RWMR vs FNR对比（图）

---

## ✅ 检查清单

开始实现前确认：

- [x] 数据可用性验证（attack_info.csv有完整参数）
- [x] 风险模型设计完成（4个维度）
- [x] 指标定义明确（4个新指标）
- [x] 输出格式规范（CSV + 可视化 + MD报告）
- [ ] 代码实现（待完成）
- [ ] 测试验证（待完成）
- [ ] 结果解读（待完成）

---

## 📌 给下一个窗口的指令

**你的任务**：

1. 阅读本context，理解风险评估体系
2. 实现`extract_risk_factors()`和`calculate_attack_risk_score()`
3. 实现4个指标函数（RWMR, MSDR, DS-FNR, CTDR）
4. 在power_grid_metrics.py中集成
5. 创建risk_analysis.py生成可视化
6. 运行并生成RISK_ANALYSIS.md报告

**数据路径**：
- 输入：`../step3_multiModel/output/{attack_type}/{model}/`
  - test_attack_info.csv
  - test_predictions.npz
- 输出：`./output/risk_analysis/`

**代码风格**：
- 遵循现有代码规范
- 添加详细注释和docstring
- 处理edge cases（如空数据、缺失字段）

**开始实现！**
