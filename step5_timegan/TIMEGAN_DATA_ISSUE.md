# TimeGAN数据生成问题分析报告

## 问题概述
TimeGAN生成的数据导致训练时出现NaN，根本原因是**特征不匹配**。

## 问题根源

### TimeGAN训练阶段（prepare_normal_data.py）
```python
feature_cols = (
    config.POSITION_FEATURES +   # ['position_x', 'position_y', 'position_z']
    config.VELOCITY_FEATURES     # ['velocity_x', 'velocity_y', 'velocity_z']
)
# 总共只有 6 个特征
```

### TimeGAN生成阶段（generate_synthetic_flights.py）
生成的合成数据也**只包含这6个特征**：
- position_x, position_y, position_z
- velocity_x, velocity_y, velocity_z

### 训练阶段（main_simple.py）
训练时使用的是**16个特征**（来自step3b的feature_engineering）：
```python
feature_cols = get_feature_columns()  # 返回16个特征
```

## 数据流程问题

```
1. prepare_normal_data.py
   → 从原始数据提取6个特征（position + velocity）
   → 保存为timegan_training_sequences.npz

2. train_timegan.py
   → 加载6个特征训练TimeGAN

3. generate_synthetic_flights.py
   → 生成6个特征的合成数据
   → 保存为synthetic_normal_flights.csv（只有6列数据）

4. merge_and_split.py
   → 合并原始正常数据（30+列）和合成数据（6列）
   → 合成数据中缺失的24列全部是NaN！

5. inject_attacks.py
   → 读取包含大量NaN的合并数据
   → 注入攻击
   → 保存train_with_attacks.csv（71%的数据是NaN）

6. main_simple.py
   → 尝试使用16个特征训练
   → 但数据中只有6个特征有值，其余10个全是NaN
   → 导致模型输入/输出全是NaN
```

## 缺失的特征

TimeGAN生成的数据**缺少以下特征**：
1. time
2. wind_speed
3. wind_angle
4. battery_voltage
5. battery_current
6. orientation_x, orientation_y, orientation_z, orientation_w
7. angular_x, angular_y, angular_z
8. linear_acceleration_x, linear_acceleration_y, linear_acceleration_z
9. speed
10. payload
11. altitude
12. date, time_day, route
13. 其他元数据

## 数据统计证据

从`analyze_dataset.py`的输出：
```
TimeGAN训练集缺失值检查:
  ⚠️  包含NaN的列: 22个
  - wind_speed: 379,800 (71.48%)
  - battery_voltage: 379,800 (71.48%)
  - orientation_x: 379,800 (71.48%)
  ...等等
```

**71.48%的数据是NaN** = 合成数据占比（379,800条合成 / 531,333总数）

## 解决方案

### 方案1：扩展TimeGAN训练特征（推荐）
修改TimeGAN使用更多特征：
```python
# 在config.py中
TIMEGAN_FEATURES = (
    POSITION_FEATURES +      # 3个
    VELOCITY_FEATURES +      # 3个
    ACCELERATION_FEATURES +  # 3个
    ['orientation_w', 'angular_z', 'linear_acceleration_z']  # 关键特征
)
# 总共约10-12个特征
```

### 方案2：仅用真实数据训练（临时方案）
跳过TimeGAN，直接使用原始正常数据：
- 删除或不生成synthetic_normal_flights.csv
- merge_and_split.py会自动只使用原始数据

### 方案3：为合成数据填充缺失特征
在generate_synthetic_flights.py中：
```python
# 为缺失的特征生成合理的默认值或基于已有特征推导
synthetic_df['time'] = np.arange(len(synthetic_df)) * 0.156
synthetic_df['wind_speed'] = 0.0  # 或基于统计分布生成
# ...等等
```

## 建议行动

**立即行动**：使用方案2，先让流程跑通
1. 删除或重命名synthetic_normal_flights.csv
2. 重新运行merge_and_split.py
3. 重新运行inject_attacks.py
4. 运行main_simple.py验证

**长期优化**：实施方案1
1. 扩展TimeGAN训练特征到10-12个核心特征
2. 重新训练TimeGAN
3. 重新生成合成数据

## 当前状态
- ❌ 合成数据只有6个特征
- ❌ 训练需要16个特征
- ❌ 71%的数据是NaN
- ❌ 模型无法训练

## 期望状态
- ✅ 合成数据包含所有必需特征
- ✅ 无NaN值（或少于1%）
- ✅ 模型可以正常训练
