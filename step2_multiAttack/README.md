# Step 2: Multi-Attack GPS Spoofing Detection

这是对 step1_benchmark 的扩展，实现了多种 GPS 欺骗攻击类型的注入和检测比较。

## 📋 项目概述

本项目在 step1 基础上实现了四大类 GPS 欺骗攻击：

- **攻击 A: 渐变漂移 (Drift Spoofing)** - 位置偏移从 0 逐渐累积到目标值
- **攻击 B1: 固定延迟 (Delay)** - 位置数据延迟固定时间
- **攻击 B2: 片段重放 (Replay)** - 重放历史轨迹片段
- **攻击 C: 一致性保持接管 (Consistent Takeover)** - 使用一阶动态系统接管轨迹

## 🎯 攻击类型详细说明

### 攻击 A: 渐变漂移 (Drift Spoofing)

**特点:**
- 位置偏移从 0 逐渐增长到最大值 M，然后保持
- 支持两种增长曲线：
  - `ramp`: 线性增长
  - `sigmoid`: S 型曲线增长（更平滑）

**参数:**
- `profile`: 增长曲线类型 ('ramp' | 'sigmoid')
- `T_drift`: 漂移到位时间（秒）
- `M`: 最大偏移幅度（米）
- `direction_mode`: 方向模式（见统一约束）
- `consistency_mode`: 一致性模式（'pos_only' | 'pos_vel' | 'pos_vel_acc'）

**公式:**
```
g(t) = {
  0                           if t < t_s
  (t-t_s)/T_drift            if t_s ≤ t ≤ t_s+T_drift (ramp)
  sigmoid((t-t_s)/T_drift)   if t_s ≤ t ≤ t_s+T_drift (sigmoid)
  1                           if t > t_s+T_drift
}
Δp(t) = M * g(t) * d̂
```

---

### 攻击 B1: 固定延迟 (Delay)

**特点:**
- 模拟 meaconing 攻击
- 所有位置数据延迟固定时间量
- 轨迹形态保持真实（因为来自真实数据）

**参数:**
- `delay_seconds`: 延迟时间（秒）
- `consistency_mode`: 一致性模式

**实现:**
```python
对于 t ≥ t_s:
  position(t) = interp(original_position, t - delay_seconds)
```

---

### 攻击 B2: 片段重放 (Replay Segment)

**特点:**
- 从历史轨迹中提取片段并重放
- 支持两种供体来源：
  - `same_flight_earlier`: 同一飞行的早期片段
  - `same_route_other_flight`: 同路线其他飞行
- 支持两种拼接模式：
  - `hard`: 直接替换（可能有突变）
  - `soft`: 边界处线性混合（更平滑）

**参数:**
- `segment_duration`: 重放片段长度（秒）
- `donor_source`: 供体来源
- `stitching`: 拼接模式 ('hard' | 'soft')
- `transition_seconds`: soft 模式的过渡时间（秒）
- `consistency_mode`: 一致性模式

**拼接策略:**
- **Hard**: 直接替换 `[t_s, t_e]` 区间的位置
- **Soft**: 
  - 段首过渡: `[t_s, t_s+transition]` 线性混合 0→1
  - 核心段: `[t_s+transition, t_e-transition]` 完全使用供体
  - 段尾过渡: `[t_e-transition, t_e]` 线性混合 1→0

---

### 攻击 C: 一致性保持接管 (Consistent Takeover)

**特点:**
- 模拟强攻击者的系统性轨迹篡改
- 使用一阶动态系统确保运动学自洽
- 不使用人为的指数瞬态，瞬态由差分自然产生

**参数:**
- `offset_profile`: 参考轨迹偏移曲线 ('step' | 'ramp' | 'sigmoid')
- `M`: 偏移幅度（米）
- `direction_mode`: 方向模式
- `takeover_alpha`: 一阶系数（0 < α ≤ 1）或
- `takeover_tau`: 时间常数（秒，α = 1 - exp(-Δt/τ)）
- `T_takeover`: ramp/sigmoid 的到位时间（秒）
- `consistency_mode`: 必须是 'pos_vel' 或 'pos_vel_acc'

**一阶跟踪公式:**
```
p_ref(t) = p_orig(t) + M * g(t) * d̂
p'[i] = p'[i-1] + α[i] * (p_ref[i] - p'[i-1])

其中:
- α[i] = constant (如果使用 takeover_alpha)
- α[i] = 1 - exp(-Δt[i]/τ) (如果使用 takeover_tau)
```

---

## 🔧 统一约束（所有攻击共享）

### 方向模式 (direction_mode)

支持的方向模式：

| 模式 | 说明 | 实现 |
|------|------|------|
| `random_xy` | 随机水平方向 | θ ~ U(0, 2π), d̂ = (cos θ, sin θ) |
| `fixed_east` | 固定东向 | d̂ = (1, 0, 0) |
| `fixed_north` | 固定北向 | d̂ = (0, 1, 0) |
| `along_track_xy` | 沿航迹方向 | 基于 t_s 时刻的速度向量 |
| `cross_track_xy` | 垂直航迹方向 | 航迹方向旋转 90° |

### 一致性模式 (consistency_mode)

| 模式 | 说明 |
|------|------|
| `pos_only` | 仅修改位置，速度和加速度保持原值 |
| `pos_vel` | 修改位置后，通过差分重算速度 |
| `pos_vel_acc` | 修改位置后，重算速度和加速度 |

**差分重算公式:**
```
velocity[i] = (position[i] - position[i-1]) / delta_t[i]
acceleration[i] = (velocity[i] - velocity[i-1]) / delta_t[i]
```

### 时间窗口采样

- 每个 flight 仅注入一次攻击
- 攻击起点 `t_s` 在 `[t_min + start_buffer, t_max - end_buffer]` 均匀采样
- 如果飞行时长不足，标记为 `flight_too_short`

---

## 📁 项目结构

```
step2_multiAttack/
├── config.py                    # 配置文件（扩展 step1 配置）
├── multi_attack_injector.py     # 多攻击注入器实现
├── main.py                      # 主运行脚本
├── compare_attacks.py           # 攻击对比分析脚本
├── README.md                    # 本文档
└── output/                      # 输出目录（自动生成）
    ├── flight_splits.json       # 数据集划分信息
    ├── attack_comparison.csv    # 攻击类型比较结果
    ├── analysis_report.md       # 详细分析报告
    ├── comparison_*.png         # 可视化图表
    ├── drift/                   # 漂移攻击结果
    ├── drift_sigmoid/           # Sigmoid 漂移攻击结果
    ├── delay/                   # 延迟攻击结果
    ├── replay_same/             # 同飞行重放结果
    ├── replay_other_soft/       # 跨飞行软拼接重放结果
    ├── takeover_step/           # 阶跃接管结果
    └── takeover_ramp/           # 斜坡接管结果
```

每个攻击类型的输出目录包含：
- `train_attack_info.csv` - 训练集攻击信息
- `val_attack_info.csv` - 验证集攻击信息
- `test_attack_info.csv` - 测试集攻击信息
- `normalization_stats.json` - 特征归一化统计
- `training_history.json` - 训练历史
- `best_model.pth` - 最佳模型权重
- `test_metrics.json` - 测试集评估指标

---

## 🚀 使用方法

### 1. 环境准备

确保已安装 step1_benchmark 的所有依赖：
```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn scipy
```

### 2. 运行完整实验

运行所有攻击类型的实验：
```bash
cd step2_multiAttack
python main.py
```

这将：
1. 加载并预处理数据
2. 对每种攻击类型：
   - 注入攻击到训练/验证/测试集
   - 训练检测模型
   - 评估性能指标
3. 保存所有结果到 `output/` 目录

**预计运行时间**: 根据数据量和硬件，可能需要数小时。

### 3. 分析和比较

运行比较分析脚本：
```bash
python compare_attacks.py
```

这将生成：
- 性能比较表格
- 检测难度排名
- 可视化图表（条形图、热图、雷达图）
- 详细分析报告（Markdown 格式）

---

## 📊 输出指标

### 性能指标

对每种攻击类型，系统计算以下指标：

| 指标 | 说明 |
|------|------|
| **AUC-ROC** | ROC 曲线下面积（越高越好） |
| **AUC-PR** | 精确率-召回率曲线下面积 |
| **F1 Score** | 精确率和召回率的调和平均 |
| **Precision** | 精确率（正确检测 / 总检测） |
| **Recall** | 召回率（正确检测 / 总攻击） |
| **Accuracy** | 准确率（正确分类 / 总样本） |

### 检测难度分类

根据 AUC-ROC 分数：

- **Very Easy**: AUC-ROC ≥ 0.95
- **Easy**: 0.90 ≤ AUC-ROC < 0.95
- **Moderate**: 0.80 ≤ AUC-ROC < 0.90
- **Hard**: 0.70 ≤ AUC-ROC < 0.80
- **Very Hard**: AUC-ROC < 0.70

---

## 🔬 实验配置

### 攻击参数配置

在 `config.py` 中可以调整：

```python
# 攻击幅度（米）
ATTACK_MAGNITUDES = [5.0, 15.0, 30.0]

# 漂移攻击
DRIFT_DURATIONS = [5.0, 10.0, 20.0]
DRIFT_PROFILES = ['ramp', 'sigmoid']

# 延迟攻击
DELAY_SECONDS = [1.0, 3.0, 5.0]

# 重放攻击
REPLAY_SEGMENT_DURATIONS = [5.0, 10.0, 15.0]
REPLAY_STITCHING_MODES = ['hard', 'soft']

# 接管攻击
TAKEOVER_ALPHAS = [0.3, 0.5, 0.8]
TAKEOVER_TAUS = [0.5, 1.0, 2.0]
```

### 数据划分

```python
TRAIN_RATIO = 0.70  # 70% 飞行用于训练
VAL_RATIO = 0.15    # 15% 飞行用于验证
TEST_RATIO = 0.15   # 15% 飞行用于测试
```

### 攻击比例

- **训练集**: 50% 飞行被攻击（用于学习攻击模式）
- **验证/测试集**: 3% 飞行被攻击（模拟真实低攻击率场景）

---

## 🔍 关键实现细节

### 1. Delta-t 处理

所有攻击都考虑非均匀采样，使用真实的 `delta_t[i] = time[i] - time[i-1]`。

对于无效的 `delta_t ≤ 0`:
- 速度/加速度重算时跳过
- 一阶动态中 `α = 0`
- 在 `attack_params` 中记录计数

### 2. 边界处理

- 时间超出范围时使用夹紧（clamping）
- 插值使用 scipy 的 `interp1d` 确保平滑性
- 所有边界情况都在 `attack_params` 中记录

### 3. 可重现性

- 所有随机数生成使用固定种子
- 飞行划分、攻击参数采样都是确定性的
- 训练过程使用固定随机种子

---

## 📈 预期结果

基于攻击特性，预期的检测难度排序（从难到易）：

1. **Replay (soft stitch, other flight)** - 最难检测
   - 轨迹完全真实，来自真实飞行
   - 软拼接使边界平滑
   
2. **Delay** - 较难检测
   - 轨迹真实，仅时间偏移
   
3. **Takeover (ramp/sigmoid)** - 中等难度
   - 一阶动态确保平滑，但有系统性偏移
   
4. **Drift (sigmoid)** - 较易检测
   - S 型曲线较平滑但仍有异常加速度
   
5. **Drift (ramp)** - 容易检测
   - 线性增长产生恒定异常速度
   
6. **Takeover (step)** - 最易检测
   - 阶跃产生明显的瞬态响应

---

## 🛠️ 自定义攻击

要添加新的攻击类型：

1. 在 `multi_attack_injector.py` 中添加新方法：
```python
def _inject_custom_attack(self, df, t_s, **params):
    # 实现攻击逻辑
    ...
    return df, attack_info
```

2. 在 `inject_attack_to_flight` 中添加路由：
```python
elif attack_type == 'custom':
    df, attack_info = self._inject_custom_attack(df, t_s, **attack_params)
```

3. 在 `config.py` 中添加参数配置

4. 在 `main.py` 中添加实验调用

---

## 📝 引用

如果使用本代码，请引用：

```bibtex
@software{uav_gps_spoofing_detection_step2,
  title = {Multi-Attack GPS Spoofing Detection System},
  author = {Your Name},
  year = {2026},
  note = {Extension of step1_benchmark with multiple attack types}
}
```

---

## 📧 联系方式

如有问题或建议，请联系项目维护者。

---

## 🔄 与 Step1 的关系

本项目完全兼容 step1_benchmark：
- 复用所有数据加载、特征工程、模型训练模块
- 仅替换攻击注入模块
- 输出格式保持一致
- 评估指标完全相同

可以直接比较 step1 的单一攻击（step + exponential transient）与 step2 的多种攻击类型。

---

## ⚠️ 注意事项

1. **内存消耗**: 同时训练多个模型需要充足内存
2. **计算时间**: 完整实验可能需要数小时
3. **数据要求**: 确保数据包含必要字段（flight, time, position_*, velocity_*, etc.）
4. **路由信息**: Replay 的 `same_route_other_flight` 需要 `route` 列

---

## 📚 参考文献

- GPS Spoofing Attack Modeling
- Meaconing and Replay Attacks
- Kinematic Consistency in GNSS
- Machine Learning for Intrusion Detection

---

**最后更新**: 2026-01-04
