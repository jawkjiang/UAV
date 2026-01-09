# Time-Aware Evaluation Framework - Implementation Context

## 📋 项目背景

### 当前状态
- **项目**: UAV GPS Spoofing Detection using Deep Learning
- **模型**: 7种架构 (CNN, LSTM, BiLSTM, GRU, CNN-LSTM, TCN, Transformer)
- **攻击类型**: 6种 (step, drift_ramp, drift_sigmoid, delay, takeover_step, takeover_ramp)
- **当前评估**: ROC-AUC, PR-AUC, F1, Precision, Recall (传统ML指标)

### 性能现状
```
Model Performance (Traditional Metrics):
├── ROC-AUC: 0.991 - 0.998 (接近完美)
├── PR-AUC:  0.936 - 0.970 (优秀)
└── F1:      0.358 - 0.611 (区分度较好)
```

### 问题识别
1. **传统指标局限性**: 
   - 忽略检测延迟 (Detection Delay)
   - 忽略误报时间间隔 (False Alarm Interval)
   - 无法评估实时检测性能

2. **评估方法不匹配**:
   - 训练: 连续时间序列
   - 评估: 滑动窗口离散采样
   - **不反映真实部署场景** (需要逐点实时检测)

3. **Takeover攻击案例**:
   - ROC-AUC = 1.0, Recall = 1.0 (完美检测)
   - Precision = 0.16, F1 = 0.27 (大量误报)
   - **根本原因**: 标签只标记t_s时刻，模型检测到整个攻击持续期

---

## 🎯 新评估框架规格

### 核心指标定义

#### 1. Detection Rate within Δt (DR@Δt)
```
定义: 在攻击开始后Δt秒内被检测到的攻击比例

DR@Δt = (# of attacks detected within Δt seconds) / (# of total attacks)

参数:
- Δt ∈ {1s, 2s, 5s, 10s, 15s, 30s} (时间窗口)

计算逻辑:
for each attack instance:
    t_attack = attack start time
    t_detect = first detection time (model output = 1)
    if t_detect - t_attack <= Δt:
        count as detected within Δt
```

#### 2. Average Detection Delay (ADD)
```
定义: 从攻击开始到首次检测的平均时间延迟

ADD = mean(t_detect - t_attack) for all detected attacks

单位: 秒 (seconds)

计算逻辑:
delays = []
for each attack instance:
    if detected:
        delay = t_detect - t_attack
        delays.append(delay)
ADD = mean(delays)
```

#### 3. Mean Time Between False Alarms (MTBFA)
```
定义: 正常飞行段中，两次误报之间的平均时间间隔

MTBFA = (Total normal flight time) / (# of false alarms)

单位: 小时 (hours)

计算逻辑:
total_normal_time = sum(duration of all normal flight segments)
false_alarms = 0
for each normal flight segment:
    for each time point t:
        if model_output[t] == 1 and true_label[t] == 0:
            false_alarms += 1
            
MTBFA = total_normal_time / false_alarms (in hours)

注意: 连续的false positive应只计为1次false alarm
```

---

## 📊 实现需求

### 输入数据格式

#### 从现有模型输出中提取
```python
# 数据源: output/{model_name}/test_predictions.npz
test_predictions = np.load('output/{model}/test_predictions.npz')

所需字段:
- y_true: shape (N,) - 真实标签 (0=normal, 1=attack)
- y_pred: shape (N,) - 模型预测 (0/1 binary)
- y_prob: shape (N,) - 预测概率 [0, 1]
- timestamps: shape (N,) - 时间戳 (秒)
- flight_ids: shape (N,) - 飞行ID
- attack_types: shape (N,) - 攻击类型标签
```

#### 攻击片段识别
```python
需要识别每个攻击片段:
attack_segment = {
    'flight_id': int,
    'attack_type': str,
    't_start': float,      # 攻击开始时间 (相对时间戳)
    't_end': float,        # 攻击结束时间
    'indices': [int],      # 该片段在全局数组中的索引
}

识别逻辑:
- 扫描y_true，识别0→1转换点 (攻击开始)
- 识别1→0转换点 (攻击结束)
- 每个连续的1片段视为一个attack instance
```

### 核心算法实现

#### 检测延迟计算
```python
def calculate_detection_delay(y_true, y_pred, timestamps, attack_segments):
    """
    计算每个攻击片段的检测延迟
    
    Returns:
        delays: List[float] - 每个攻击的检测延迟 (秒)
        detected: List[bool] - 是否被检测到
    """
    delays = []
    detected_flags = []
    
    for seg in attack_segments:
        t_attack = timestamps[seg['indices'][0]]  # 攻击开始时间
        attack_indices = seg['indices']
        
        # 查找首次检测时间
        t_detect = None
        for idx in attack_indices:
            if y_pred[idx] == 1:
                t_detect = timestamps[idx]
                break
        
        if t_detect is not None:
            delay = t_detect - t_attack
            delays.append(delay)
            detected_flags.append(True)
        else:
            delays.append(None)  # 未检测到
            detected_flags.append(False)
    
    return delays, detected_flags
```

#### DR@Δt计算
```python
def calculate_dr_at_delta_t(delays, delta_t_values):
    """
    计算不同Δt下的检测率
    
    Args:
        delays: List[float] - 检测延迟列表 (None表示未检测)
        delta_t_values: List[float] - Δt阈值列表 (如[1,2,5,10,15,30])
    
    Returns:
        dr_dict: Dict[float, float] - {Δt: DR@Δt}
    """
    total_attacks = len(delays)
    dr_dict = {}
    
    for delta_t in delta_t_values:
        detected_within = sum(
            1 for d in delays 
            if d is not None and d <= delta_t
        )
        dr_dict[delta_t] = detected_within / total_attacks
    
    return dr_dict
```

#### MTBFA计算
```python
def calculate_mtbfa(y_true, y_pred, timestamps, flight_ids):
    """
    计算平均误报时间间隔
    
    Returns:
        mtbfa_hours: float - MTBFA (小时)
        false_alarm_count: int - 误报次数
    """
    # 识别正常飞行段
    normal_segments = []
    current_seg = None
    
    for i in range(len(y_true)):
        if y_true[i] == 0:  # 正常点
            if current_seg is None:
                current_seg = {'start': i, 'flight_id': flight_ids[i]}
        else:  # 攻击点
            if current_seg is not None:
                current_seg['end'] = i - 1
                normal_segments.append(current_seg)
                current_seg = None
    
    # 计算总正常时间和误报次数
    total_normal_time = 0  # 秒
    false_alarms = 0
    
    for seg in normal_segments:
        duration = timestamps[seg['end']] - timestamps[seg['start']]
        total_normal_time += duration
        
        # 统计该段中的误报 (连续FP算1次)
        in_false_alarm = False
        for i in range(seg['start'], seg['end'] + 1):
            if y_pred[i] == 1:  # 预测为攻击
                if not in_false_alarm:
                    false_alarms += 1
                    in_false_alarm = True
            else:
                in_false_alarm = False
    
    # 转换为小时
    mtbfa_hours = (total_normal_time / 3600) / false_alarms if false_alarms > 0 else float('inf')
    
    return mtbfa_hours, false_alarms
```

---

## 📈 输出要求

### 1. 性能指标表格
```
文件: output/time_aware_metrics/overall_metrics.csv

格式:
model_name,ADD,MTBFA,DR@1s,DR@2s,DR@5s,DR@10s,DR@15s,DR@30s
cnn,3.2,18.5,0.65,0.78,0.92,0.96,0.98,0.99
lstm,2.8,12.3,0.72,0.85,0.95,0.98,0.99,1.00
...
```

### 2. 每个攻击类型的指标
```
文件: output/time_aware_metrics/{model}/per_attack_metrics.csv

格式:
attack_type,n_instances,ADD,DR@5s,DR@10s,MTBFA
step,42,1.5,0.95,0.98,-
drift_ramp,38,4.2,0.87,0.93,-
takeover_step,35,2.1,0.91,0.97,-
...
```

### 3. 可视化图表

#### 3.1 DR vs Delay Curve
```
X轴: Δt (1-30秒)
Y轴: DR@Δt (0-1)
每条线: 一个模型
图例: 7个模型
保存: output/time_aware_metrics/dr_vs_delay.png
```

#### 3.2 DR vs MTBFA Trade-off
```
X轴: MTBFA (小时, log scale)
Y轴: DR@5s
每个点: 一个模型在特定阈值下
标注: 模型名称
保存: output/time_aware_metrics/dr_vs_mtbfa.png
```

#### 3.3 Detection Delay Distribution
```
类型: 箱线图
X轴: 模型名称
Y轴: Detection Delay (秒)
显示: 中位数、四分位数、异常值
保存: output/time_aware_metrics/delay_distribution.png
```

#### 3.4 Per-Attack Performance Heatmap
```
行: 模型名称
列: 攻击类型
值: DR@5s
颜色: 0 (白) → 1 (深绿)
保存: output/time_aware_metrics/per_attack_heatmap.png
```

### 4. 场景分析报告
```
文件: output/time_aware_metrics/scenario_analysis.txt

内容:
1. Safety-Critical Scenario (安全关键)
   - 目标: DR@5s > 0.95, ADD < 3s
   - 最佳模型: XXX
   - 性能: DR@5s=0.98, ADD=2.1s, MTBFA=8.5h

2. Monitoring Scenario (长期监控)
   - 目标: MTBFA > 24h, DR@10s > 0.90
   - 最佳模型: XXX
   - 性能: DR@10s=0.93, MTBFA=31.2h, ADD=5.8s

3. Balanced Scenario (平衡)
   - 目标: DR@5s > 0.90, MTBFA > 12h, ADD < 5s
   - 最佳模型: XXX
   - 性能: DR@5s=0.94, MTBFA=15.3h, ADD=3.2s
```

---

## 🔧 实现建议

### 代码结构
```
step3b_multiModelGeneral/
├── time_aware_evaluation.py      # 核心评估模块
├── time_aware_metrics.py         # 指标计算函数
├── time_aware_visualizations.py  # 可视化生成
└── run_time_aware_eval.py        # 主运行脚本
```

### 实现步骤

#### Phase 1: 数据准备 (1天)
```python
# 1. 加载所有模型的test predictions
# 2. 识别攻击片段
# 3. 验证数据完整性
```

#### Phase 2: 指标计算 (2天)
```python
# 1. 实现detection delay计算
# 2. 实现DR@Δt计算
# 3. 实现MTBFA计算
# 4. 验证算法正确性
```

#### Phase 3: 批量评估 (1天)
```python
# 1. 遍历7个模型
# 2. 计算所有指标
# 3. 生成CSV报告
```

#### Phase 4: 可视化 (2天)
```python
# 1. DR vs Delay curves
# 2. DR vs MTBFA scatter
# 3. Delay distribution boxplot
# 4. Per-attack heatmap
```

#### Phase 5: 场景分析 (1天)
```python
# 1. 定义场景约束
# 2. 筛选最佳模型
# 3. 生成报告
```

### 验证清单
- [ ] 所有攻击片段都被正确识别
- [ ] 检测延迟计算逻辑正确 (首次检测)
- [ ] MTBFA计算中连续FP只算1次
- [ ] DR@Δt单调递增 (Δt增大，DR增大)
- [ ] 所有模型都生成了完整报告
- [ ] 可视化图表清晰易读

---

## 📚 参考数据

### 现有数据位置
```
step3b_multiModelGeneral/output/
├── cnn/
│   ├── test_predictions.npz
│   ├── test_metrics.json
│   └── per_attack_metrics.json
├── lstm/
│   ├── test_predictions.npz
│   └── ...
└── ... (其他5个模型)
```

### 配置参数
```python
# 来自 config.py
WINDOW_SIZE = 50         # 窗口大小
STEP_SIZE = 5            # 步长
SAMPLING_RATE = 100      # 采样率 (Hz)
TIME_PER_POINT = 0.01    # 每个点0.01秒
```

### 攻击类型列表
```python
ATTACK_TYPES = [
    'step',
    'drift_ramp', 
    'drift_sigmoid',
    'delay',
    'takeover_step',
    'takeover_ramp'
]
```

---

## 🎓 研究设计说明

### 方案选择
✅ **采用方案1**: 同一套模型权重，不同评估指标对比
- 训练: 使用传统BCE Loss
- 评估: 传统指标 + 时间感知指标对比

### 论文贡献点
1. **问题识别**: 传统ML指标不适合实时检测系统评估
2. **框架提出**: Time-Aware Evaluation Framework (DR@Δt, ADD, MTBFA)
3. **实验验证**: 7种模型在6种攻击上的对比分析
4. **场景应用**: 安全关键/监控/平衡场景的模型选择指导

### 预期发现
- 传统指标高的模型，时间感知指标未必好
- DR@Δt曲线能清晰区分模型的响应速度
- MTBFA揭示误报率的真实影响
- 不同场景下最优模型可能不同

---

## ⏱️ 时间估算

```
总计: 1-2周

详细分解:
├── 代码实现: 3-5天
│   ├── 数据准备: 0.5天
│   ├── 指标计算: 1-2天
│   ├── 批量评估: 0.5天
│   ├── 可视化: 1-2天
│   └── 场景分析: 0.5天
│
├── 实验运行: 1-2天
│   ├── 首次运行: 0.5天
│   ├── Bug修复: 0.5-1天
│   └── 完整评估: 0.5天
│
└── 结果分析: 2-3天
    ├── 数据验证: 0.5天
    ├── 模式发现: 1-1.5天
    └── 报告撰写: 0.5-1天
```

---

## 📝 注意事项

### 数据处理
1. **时间戳处理**: 确保timestamps是相对时间 (从0开始)
2. **飞行ID**: 用于区分不同飞行，避免跨飞行统计
3. **窗口效应**: 滑动窗口可能导致重复计数，需去重

### 算法细节
1. **首次检测**: 只统计第一次检测时间，后续重复检测不计
2. **连续误报**: 正常段中连续的FP只算1次false alarm
3. **未检测**: DR@Δt中未检测的攻击计为0贡献

### 可视化建议
1. **DR曲线**: 使用不同颜色和标记区分模型
2. **箱线图**: 标注中位数和均值
3. **热力图**: 使用color bar显示数值范围
4. **保存格式**: PNG (300 DPI) + PDF (矢量图)

---

## 🚀 快速开始

```bash
# 1. 进入工作目录
cd C:\Users\Jawk\PycharmProjects\UAV\step3b_multiModelGeneral

# 2. 激活虚拟环境
.venv\Scripts\activate

# 3. 运行评估
python run_time_aware_eval.py

# 4. 查看结果
# - CSV: output/time_aware_metrics/*.csv
# - 图表: output/time_aware_metrics/*.png
# - 报告: output/time_aware_metrics/scenario_analysis.txt
```

---

## 📧 联系与反馈

如遇到问题或需要调整，请在vibe coding窗口中提出。

**Good luck with implementation! 🎯**
