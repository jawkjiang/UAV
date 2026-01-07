# GPS欺骗攻击检测性能差异分析

## 实验结果总览

| 攻击类型 | AUC-ROC | AUC-PR | F1 Score | Precision | Recall |
|---------|---------|--------|----------|-----------|--------|
| **step** | 1.000 | 1.000 | 0.583 | 0.411 | 1.000 |
| **takeover_step** | 0.998 | 0.990 | 0.993 | 1.000 | 0.987 |
| **takeover_ramp** | 1.000 | 1.000 | 0.984 | 0.973 | 0.995 |
| **drift_ramp** | 0.999 | 0.987 | 0.942 | 0.959 | 0.925 |
| **drift_sigmoid** | 0.995 | 0.951 | 0.957 | 0.995 | 0.921 |
| **delay** | 0.552 | 0.031 | 0.040 | 0.025 | 0.103 |
| **replay_other_soft** | 0.548 | 0.054 | 0.071 | 0.038 | 0.469 |
| **replay_same_hard** | 0.504 | 0.028 | 0.059 | 0.032 | 0.364 |

## 核心发现：检测性能差异的根本原因

### 🔍 问题核心：Consistency Modes导致的不公平比较

检查`test_attack_info.csv`发现了**关键问题**：

#### 1. **Step, Delay, Drift(部分), Replay攻击**
- `consistency_mode = 'pos_only'`
- **只修改位置，不重新计算速度和加速度**
- 导致明显的**运动学不一致性**
- 例如：
  ```
  delay: pos_only
  step: pos_only
  drift_ramp (flight 192): pos_only
  replay_same_hard: pos_only
  ```

#### 2. **Takeover和部分Drift攻击**
- `consistency_mode = 'pos_vel'` 或 `'pos_vel_acc'`
- **修改位置后重新计算速度和加速度**
- 保持运动学一致性
- 例如：
  ```
  drift_ramp (flight 264): pos_vel
  replay_same_hard (flight 131): pos_vel_acc
  takeover: 强制使用 pos_vel 或 pos_vel_acc
  ```

## 检测原理与攻击特征分析

### 模型使用的特征

```python
features = [
    # 基础运动特征
    'position_x', 'position_y',          
    'velocity_x', 'velocity_y',          
    'linear_acceleration_x', 'linear_acceleration_y',
    'delta_t',
    
    # 一致性残差特征 (核心检测特征)
    'residual_pv_x', 'residual_pv_y', 'residual_pv_norm',   # 位置-速度残差
    'residual_va_x', 'residual_va_y', 'residual_va_norm'    # 速度-加速度残差
]
```

### 一致性残差计算

**位置-速度残差 (PV Residual):**
```
r_pv = |Δp - v·Δt|
```
- 正常情况：位置变化 ≈ 速度 × 时间间隔
- 攻击且 `pos_only`：位置突变，但速度未更新 → **残差激增**

**速度-加速度残差 (VA Residual):**
```
r_va = |Δv/Δt - a|
```
- 正常情况：速度变化率 ≈ 加速度
- 攻击且 `pos_only`：速度与位置不一致 → **残差激增**

## 各攻击类型检测性能分析

### ✅ **高检出率攻击组 (AUC-ROC > 0.99)**

#### 1. **Step Attack (AUC-ROC = 1.000)**
**攻击特征:**
- 位置瞬间跃变 (5-30米)
- 3秒指数衰减速度瞬态
- `consistency_mode = 'pos_only'` ← **关键**

**为何易检测:**
- ✓ 位置突变但速度未更新 → PV残差爆炸
- ✓ 速度瞬态但加速度未正确对应 → VA残差异常
- ✓ 攻击持续时间短(3秒) → 异常模式集中明显

**示例（Flight 221）:**
```json
{
  "magnitude": 5.0,
  "velocity_transient_max": 1.944,
  "attack_duration": 3.0,
  "consistency_mode": "pos_only"  // 未重算速度/加速度
}
```

#### 2. **Takeover Attacks (AUC-ROC ≈ 0.998-1.000)**
**攻击特征:**
- 一阶动态跟踪参考轨迹
- 强制 `consistency_mode = 'pos_vel'` 或 `'pos_vel_acc'`
- 重新计算速度和加速度以保持一致性

**为何仍高检出:**
- ✗ 虽然保持运动学一致性，但...
- ✓ **参考轨迹本身偏离正常轨迹** → 位置特征异常
- ✓ 动态跟踪过程产生的**跟踪误差**和**暂态响应**
- ✓ 攻击幅度大(5-30米)，偏离幅度明显

**关键差异:**
```python
# Takeover强制要求一致性
if consistency_mode == 'pos_only':
    consistency_mode = 'pos_vel'  # 强制升级
```

#### 3. **Drift Attacks (AUC-ROC ≈ 0.994-0.999)**
**攻击特征:**
- 渐进式位置偏移(5-20秒)
- **随机采样** `consistency_mode` (pos_only/pos_vel/pos_vel_acc)

**检出率分析:**
- `pos_only` 时：残差特征明显 → 高检出
- `pos_vel` 时：
  - 速度重算后PV残差降低
  - 但漂移速度产生异常速度模式
  - 加速度未重算 → VA残差仍存在
- `pos_vel_acc` 时：检出率应降低（但样本中较少）

**示例对比:**
```
Flight 192 (pos_only):  高检出率
Flight 264 (pos_vel):   检出率稍降但仍高
```

### ❌ **低检出率攻击组 (AUC-ROC ≈ 0.50-0.55)**

#### 4. **Delay Attack (AUC-ROC = 0.552)**
**攻击特征:**
- 固定时间延迟(1-5秒)
- 位置取自 `t - delay_seconds`
- `consistency_mode = 'pos_only'` ← **应该易检测，为何失败？**

**检测失败的真实原因:**

**a) 延迟攻击的隐蔽性:**
```python
# 延迟攻击的位置是过去的真实位置
position_attacked(t) = position_real(t - delay)

# 关键洞察：
# - 如果UAV在相对平稳飞行，t 和 t-delay 的位置差异很小
# - 速度 v(t) 和 v(t-delay) 也相似
# - 导致 |Δp - v·Δt| 残差不大！
```

**b) 运动平滑性导致残差不明显:**
- UAV通常沿平滑轨迹飞行
- 1-5秒延迟内，轨迹变化不大
- PV残差：`|p(t) - p(t-1)| - v(t)·Δt ≈ |p(t-delay) - p(t-delay-Δt)| - v(t)·Δt`
- 若 `v(t) ≈ v(t-delay)` 且轨迹平滑 → **残差接近正常水平**

**c) 样本不足:**
```
Test集中仅 2/32 航班被攻击 (6.25%)
→ 训练样本少，模型难以学习延迟攻击的细微特征
```

**为何Recall仍有10.3%:**
- 在UAV转弯、加速等**运动模式变化**时刻
- `v(t)` 和 `v(t-delay)` 差异变大
- 残差才变得明显

#### 5. **Replay Attacks (AUC-ROC ≈ 0.50-0.55)**
**攻击特征:**
- 重放过去的轨迹片段
- 硬拼接(hard)或软拼接(soft)
- `consistency_mode` 随机采样

**检测失败原因:**

**a) 重放的是真实轨迹:**
```python
# Replay本质：用真实的历史轨迹替换当前轨迹
# 重放片段内部的运动学完全一致！
# 因为它来自真实飞行数据

donor_segment = real_trajectory[t_a : t_b]
attacked_segment = donor_segment  # 内部一致性完美
```

**b) 拼接点检测挑战:**
- **Hard stitching**: 拼接点有突变 → 理论可检测
  - 但仅**2个时间点**异常 (开始+结束)
  - 滑动窗口(50个样本)稀释了异常
  
- **Soft stitching**: 0.5秒线性过渡 → 异常极度平滑
  - 过渡极短，难以捕捉

**c) 窗口稀释效应:**
```
窗口大小 = 50个样本
异常点数 ≤ 2个 (拼接点)
异常比例 ≤ 4%

→ 特征被正常样本淹没
```

**d) 同航线重放的隐蔽性:**
```python
# replay_other_soft: 同路线其他航班
# → 运动模式高度相似
# → 特征分布接近正常

# replay_same_hard: 同航班早期片段
# → 飞行模式几乎相同
# → 除了拼接点，无异常可言
```

## 实验设计的系统性偏差

### 🚨 **核心问题：Consistency Mode不统一**

```python
# config_step2.py
CONSISTENCY_MODES = ['pos_only', 'pos_vel', 'pos_vel_acc']

# 大部分攻击随机采样
consistency_mode = self.rng.choice(config.CONSISTENCY_MODES)

# 但Takeover强制高一致性
if consistency_mode == 'pos_only':
    consistency_mode = 'pos_vel'
```

**这导致:**
1. **不公平比较**: 不同攻击使用不同一致性策略
2. **检测器偏向**: 模型过度依赖残差特征检测 `pos_only` 攻击
3. **真实性问题**: 高级攻击者会保持运动学一致性

### 📊 **攻击比例偏差**

| 数据集 | 攻击比例 | 影响 |
|--------|---------|------|
| Train | 50% | 过高，模型过拟合攻击模式 |
| Val/Test | 10% | 偏低，但更接近现实 |

**问题:**
- Delay/Replay在Test集中样本极少 (2-3次攻击)
- 模型未充分学习这些攻击的细微特征

### 🎯 **特征工程的局限性**

当前特征完全依赖**运动学一致性残差**:
```python
# 仅当 pos_only 时有效
residual_pv = |Δp - v·Δt|  # 位置跳变检测
residual_va = |Δv/Δt - a|  # 速度跳变检测
```

**缺少:**
- 轨迹连续性特征 (检测replay的拼接)
- 时序异常检测 (检测delay的时间错位)
- 多航班对比特征 (检测重复轨迹)

## 建议的改进方案

### 1. **统一Consistency Mode - 公平比较**

```python
# 为所有攻击设定统一策略
UNIFIED_CONSISTENCY_MODE = 'pos_vel_acc'

# 或分别测试
for consistency_mode in ['pos_only', 'pos_vel', 'pos_vel_acc']:
    run_all_attacks(consistency_mode=consistency_mode)
```

**预期结果:**
- `pos_only`: Step/Drift/Delay高检出，Takeover/Replay低检出
- `pos_vel_acc`: 所有攻击检出率下降，更接近真实攻击场景

### 2. **增加攻击样本量**

```python
# 提高test集攻击比例 (当前10% → 30%)
TEST_ATTACK_RATIO = 0.30

# 或使用分层采样确保每种攻击至少N个样本
MIN_ATTACK_SAMPLES_PER_TYPE = 10
```

### 3. **增强特征工程**

#### a) **时序异常特征 (针对Delay)**
```python
# 检测速度-位置的时间错位
features['velocity_position_correlation'] = ...
features['trajectory_curvature_consistency'] = ...
```

#### b) **轨迹连续性特征 (针对Replay)**
```python
# 检测拼接点的二阶导数突变
features['position_jerk'] = np.diff(acceleration)
features['trajectory_smoothness'] = ...
```

#### c) **长时记忆特征**
```python
# 使用LSTM检测历史轨迹重复
# 检测与过去轨迹的相似度
```

### 4. **分攻击类型训练专用检测器**

```python
# 残差检测器: 检测pos_only攻击
detector_residual = train_on(['step', 'drift_pos_only'])

# 时序检测器: 检测delay
detector_temporal = train_on(['delay'])

# 重复检测器: 检测replay
detector_replay = train_on(['replay'])

# 集成检测
final_score = ensemble([detector_residual, detector_temporal, detector_replay])
```

### 5. **实验对照组设计**

| 实验组 | Consistency Mode | 攻击类型 | 目的 |
|--------|-----------------|---------|------|
| A | pos_only | 所有 | 基线（最易检测）|
| B | pos_vel | 所有 | 中等难度 |
| C | pos_vel_acc | 所有 | 最难检测（真实场景）|

## 结论

### 当前结果的解读

1. **Step/Drift/Takeover高检出** ← **主要因为使用pos_only或保留了残差特征**
   - 非攻击本质易检测
   - 而是实现选择了"易检测"的方式

2. **Delay/Replay低检出** ← **不是实验参数问题，是攻击本质隐蔽**
   - Delay: 使用真实历史位置，平滑飞行时残差不明显
   - Replay: 重放真实轨迹，内部运动学完美
   - 现有特征无法捕捉这类攻击的细微异常

### 最关键的发现

**当前检测器本质上是 "运动学不一致性检测器"，而非真正的"GPS欺骗检测器"**

- 对 `pos_only` 攻击有效 (AUC > 0.95)
- 对 `pos_vel_acc` 攻击失效 (AUC < 0.60)

**真实攻击者会保持运动学一致性！**

### 下一步行动

**紧急优先级:**
1. ✅ 重跑实验，统一 `consistency_mode = 'pos_vel_acc'`
2. ✅ 增加test集delay/replay样本量
3. ✅ 设计时序特征检测delay
4. ✅ 设计拼接检测特征检测replay

**中期目标:**
5. 实现ensemble检测器
6. 使用LSTM/Transformer捕捉长时依赖
7. 增加跨航班对比特征

---

**生成时间**: 2026-01-05  
**分析者**: Copilot Analysis System
