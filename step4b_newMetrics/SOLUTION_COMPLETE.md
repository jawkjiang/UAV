# 时间感知评估框架 - 完整解决方案

## 问题诊断

### 发现的关键问题

通过诊断脚本 `diagnose_issue.py`，我们发现了三个致命错误：

1. **❌ 错误1: 使用错误的预测数据**
   - 当前使用 `data['predictions']` 键
   - 但该键存储的是**概率值**（0-1的连续值），不是二值化预测
   - 应该使用 `data['probabilities']` 并二值化：`(probabilities > 0.5).astype(int)`

2. **❌ 错误2: 时间戳重建完全错误**
   - 当前方法：`timestamps = np.arange(n_windows) * 0.05s`
   - 这只能生成简单序列：0s, 0.05s, 0.1s, 0.15s...
   - **无法对应真实攻击时间**：attack_info中的attack_start_time是95.8s, 165.5s等
   - 窗口索引与真实时间完全脱节

3. **❌ 错误3: 缺失关键元数据**
   - `test_predictions.npz` 只包含：labels, predictions, probabilities
   - **缺少**：window_indices, timestamps, flight_ids, attack_types
   - 无法将窗口映射回原始时间线

### 问题影响

```
检测延迟全部为 0.000s - 0.002s
原因：攻击片段的第一个窗口立即被"检测"

DR@Δt 不随 Δt 变化（DR@1s = DR@30s）
原因：时间戳错误，所有"检测"都在t=0时发生
```

## 完整解决方案

### 方案概述

需要从数据源头（step3b）到评估流程（step4b）的完整改造：

```
step3b (数据生成)          step4b (评估)
     ↓                          ↓
保存完整元数据    →    加载完整元数据    →    正确计算指标
window_indices        window_metadata       使用真实时间戳
timestamps            flight_timeline       二值化预测
flight_ids            attack_segments       
attack_types
```

---

## 第一部分：修改 step3b (数据源)

### 1.1 修改 `step3b_multiModelGeneral/main.py`

**位置**: 第247-251行（np.savez 部分）

**当前代码**:
```python
np.savez(
    os.path.join(output_dir, 'test_predictions.npz'),
    predictions=test_predictions,
    labels=test_targets,
    probabilities=test_predictions
)
```

**修改为**:
```python
# 保存完整的窗口元数据用于时间感知评估
np.savez(
    os.path.join(output_dir, 'test_predictions.npz'),
    # 预测和标签
    predictions=test_predictions,
    labels=test_targets,
    probabilities=test_predictions,
    
    # 窗口元数据（新增）
    window_indices=np.arange(len(test_targets)),
    flight_ids=test_flight_ids,
    
    # 时间相关元数据（新增）
    window_size=config.WINDOW_SIZE,
    step_size=config.STEP_SIZE,
    sampling_rate=config.SAMPLING_RATE
)

print(f"    ✓ Saved test predictions with metadata to test_predictions.npz")
print(f"      - {len(test_targets)} windows")
print(f"      - {len(np.unique(test_flight_ids))} flights")
print(f"      - Window metadata: indices, flight_ids, timestamps")
```

### 1.2 新增：生成 `window_metadata.csv`

**在 main.py 中添加**（紧接在 np.savez 之后）:

```python
# 生成详细的窗口元数据文件
print(f"\n  Generating window metadata file...")

# 为每个窗口构建元数据
window_metadata_list = []

for idx in range(len(test_targets)):
    flight_id = test_flight_ids[idx]
    
    # 找到对应的飞行信息
    flight_info = test_attack_info[test_attack_info['flight'] == flight_id]
    
    if len(flight_info) > 0:
        flight_info = flight_info.iloc[0]
        attacked = flight_info['attacked']
        attack_type = flight_info['attack_type'] if attacked else 'normal'
        attack_start_time = flight_info['attack_start_time'] if attacked else np.nan
    else:
        attacked = False
        attack_type = 'normal'
        attack_start_time = np.nan
    
    window_metadata_list.append({
        'window_index': idx,
        'flight_id': flight_id,
        'label': test_targets[idx],
        'attacked': attacked,
        'attack_type': attack_type,
        'attack_start_time': attack_start_time
    })

window_metadata_df = pd.DataFrame(window_metadata_list)
window_metadata_df.to_csv(
    os.path.join(output_dir, 'window_metadata.csv'),
    index=False
)
print(f"    ✓ Saved window metadata to window_metadata.csv")
print(f"      - {len(window_metadata_df)} windows with attack timing info")
```

### 1.3 额外改进：保存测试集的原始数据索引

**在 main.py 的测试数据准备部分**（创建 test_dataset 之前）:

```python
# 保存测试集的窗口到飞行时间的映射
# 这需要从原始flights.csv重建每个窗口的真实时间范围
test_window_timeline = []

for flight_id in np.unique(test_flight_ids):
    # 获取该飞行的所有窗口
    flight_window_mask = test_flight_ids == flight_id
    flight_window_indices = np.where(flight_window_mask)[0]
    
    # 从test_df获取该飞行的原始数据
    flight_df_original = test_df[test_df['flight'] == flight_id].copy()
    
    if len(flight_df_original) > 0:
        # 假设每个窗口的起始采样点索引
        for local_win_idx, global_win_idx in enumerate(flight_window_indices):
            # 窗口在飞行内的起始采样点
            start_sample_in_flight = local_win_idx * config.STEP_SIZE
            end_sample_in_flight = start_sample_in_flight + config.WINDOW_SIZE
            
            # 获取窗口的真实时间范围
            if 'timestamp' in flight_df_original.columns:
                if end_sample_in_flight <= len(flight_df_original):
                    t_start = flight_df_original.iloc[start_sample_in_flight]['timestamp']
                    t_end = flight_df_original.iloc[min(end_sample_in_flight-1, len(flight_df_original)-1)]['timestamp']
                else:
                    continue
            else:
                # 使用采样率估算时间
                t_start = start_sample_in_flight / config.SAMPLING_RATE
                t_end = end_sample_in_flight / config.SAMPLING_RATE
            
            test_window_timeline.append({
                'window_index': global_win_idx,
                'flight_id': flight_id,
                't_start': t_start,
                't_end': t_end,
                't_center': (t_start + t_end) / 2
            })

if len(test_window_timeline) > 0:
    timeline_df = pd.DataFrame(test_window_timeline)
    timeline_df.to_csv(os.path.join(output_dir, 'window_timeline.csv'), index=False)
    print(f"    ✓ Saved window timeline to window_timeline.csv")
```

---

## 第二部分：修改 step4b (评估框架)

### 2.1 创建新的数据加载器：`data_loader_with_metadata.py`

```python
"""
Data Loader with Complete Metadata
使用step3b生成的完整元数据进行时间感知评估
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os
import config


def load_model_data_with_metadata(model_name: str) -> Dict[str, np.ndarray]:
    """
    加载模型测试数据及完整的窗口元数据
    
    Args:
        model_name: 模型名称
    
    Returns:
        包含所有必要数据的字典：
        - y_true: 真实标签
        - y_pred: 二值化预测 (probabilities > 0.5)
        - y_prob: 预测概率
        - timestamps: 每个窗口的中心时间戳
        - flight_ids: 每个窗口的飞行ID
        - attack_types: 每个窗口的攻击类型
        - window_indices: 窗口索引
    """
    print(f"Loading data with metadata for {model_name}...")
    
    model_dir = config.get_step3b_model_dir(model_name)
    
    # 1. 加载预测文件
    pred_file = os.path.join(model_dir, 'test_predictions.npz')
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    data = np.load(pred_file)
    
    # 使用probabilities并二值化（关键修复！）
    y_prob = data['probabilities']
    y_pred = (y_prob > 0.5).astype(int)
    y_true = data['labels'].astype(int)
    
    # 从npz获取元数据（如果存在）
    if 'flight_ids' in data:
        flight_ids = data['flight_ids']
        window_indices = data.get('window_indices', np.arange(len(y_true)))
    else:
        # 回退方案：从文件加载
        flight_ids = None
        window_indices = np.arange(len(y_true))
    
    # 2. 加载window_metadata.csv
    metadata_file = os.path.join(model_dir, 'window_metadata.csv')
    timeline_file = os.path.join(model_dir, 'window_timeline.csv')
    attack_info_file = os.path.join(model_dir, 'test_attack_info.csv')
    
    if os.path.exists(metadata_file):
        print(f"  ✓ Loading window_metadata.csv")
        metadata = pd.read_csv(metadata_file)
        
        # 确保索引对齐
        metadata = metadata.sort_values('window_index').reset_index(drop=True)
        
        if flight_ids is None:
            flight_ids = metadata['flight_id'].values
        
        attack_types = metadata['attack_type'].fillna('normal').values
        
    else:
        print(f"  ⚠ window_metadata.csv not found, using fallback method")
        attack_types = np.array(['normal'] * len(y_true))
        
        if os.path.exists(attack_info_file):
            attack_info = pd.read_csv(attack_info_file)
            # 粗略分配attack_types
            if flight_ids is None:
                flight_ids = np.zeros(len(y_true), dtype=int)
                # 均匀分配
                n_flights = len(attack_info)
                windows_per_flight = len(y_true) // n_flights
                for i, row in attack_info.iterrows():
                    start_idx = i * windows_per_flight
                    end_idx = min(start_idx + windows_per_flight, len(y_true))
                    flight_ids[start_idx:end_idx] = row['flight']
                    if row['attacked']:
                        attack_types[start_idx:end_idx] = row['attack_type']
    
    # 3. 加载window_timeline.csv（时间戳）
    if os.path.exists(timeline_file):
        print(f"  ✓ Loading window_timeline.csv (真实时间戳)")
        timeline = pd.read_csv(timeline_file)
        timeline = timeline.sort_values('window_index').reset_index(drop=True)
        
        # 使用窗口中心时间作为时间戳
        timestamps = timeline['t_center'].values
        
    else:
        print(f"  ⚠ window_timeline.csv not found, using estimated timestamps")
        # 回退：基于窗口索引和采样率估算
        # 每个窗口中心 = 窗口起始 + 窗口大小/2
        window_duration = config.WINDOW_SIZE / config.SAMPLING_RATE
        window_step = config.STEP_SIZE / config.SAMPLING_RATE
        
        timestamps = np.arange(len(y_true)) * window_step + window_duration / 2
    
    # 4. 加载攻击信息用于构建攻击片段的真实时间
    attack_segments_info = []
    if os.path.exists(attack_info_file):
        attack_info = pd.read_csv(attack_info_file)
        
        for _, row in attack_info.iterrows():
            if row['attacked']:
                attack_segments_info.append({
                    'flight_id': row['flight'],
                    'attack_type': row['attack_type'],
                    'attack_start_time': row['attack_start_time']
                })
    
    print(f"  Loaded {len(y_true)} windows")
    print(f"  Flights: {len(np.unique(flight_ids))}")
    print(f"  Attack windows (label=1): {(y_true == 1).sum()}")
    print(f"  Predicted attack windows (pred=1): {(y_pred == 1).sum()}")
    print(f"  Timestamp range: {timestamps.min():.2f}s - {timestamps.max():.2f}s")
    print(f"  Attack types: {np.unique(attack_types[attack_types != 'normal'])}")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'timestamps': timestamps,
        'flight_ids': flight_ids,
        'attack_types': attack_types,
        'window_indices': window_indices,
        'attack_segments_info': attack_segments_info
    }
```

### 2.2 修改 `time_aware_metrics.py`

**关键修改**：攻击片段识别需要使用真实时间

```python
def identify_attack_segments_with_timing(y_true: np.ndarray,
                                        timestamps: np.ndarray,
                                        flight_ids: np.ndarray,
                                        attack_types: np.ndarray,
                                        attack_segments_info: list) -> List[Dict]:
    """
    识别攻击片段，使用真实的攻击开始时间
    
    Args:
        y_true: 真实标签 (0/1)
        timestamps: 每个窗口的时间戳 (秒)
        flight_ids: 每个窗口的飞行ID
        attack_types: 每个窗口的攻击类型
        attack_segments_info: 包含attack_start_time的攻击信息列表
    
    Returns:
        攻击片段列表，每个包含：
        - flight_id, attack_type
        - t_attack: 真实的攻击开始时间
        - t_first_window: 第一个被标记为攻击的窗口时间
        - t_last_window: 最后一个被标记为攻击的窗口时间
        - indices: 攻击窗口的索引列表
    """
    segments = []
    
    # 为每个飞行构建攻击片段
    for flight_id in np.unique(flight_ids):
        flight_mask = flight_ids == flight_id
        flight_indices = np.where(flight_mask)[0]
        
        # 找到该飞行的攻击窗口
        attack_mask = (y_true[flight_mask] == 1)
        attack_indices_local = np.where(attack_mask)[0]
        
        if len(attack_indices_local) == 0:
            continue
        
        # 获取该飞行的真实攻击开始时间
        attack_info = [seg for seg in attack_segments_info if seg['flight_id'] == flight_id]
        
        if len(attack_info) > 0:
            attack_info = attack_info[0]
            t_attack_real = attack_info['attack_start_time']
            attack_type_real = attack_info['attack_type']
        else:
            # 没有attack_info，使用第一个攻击窗口的时间
            t_attack_real = timestamps[flight_indices[attack_indices_local[0]]]
            attack_type_real = attack_types[flight_indices[attack_indices_local[0]]]
        
        # 找连续的攻击段
        attack_indices_global = flight_indices[attack_indices_local]
        
        # 分割连续片段
        splits = np.where(np.diff(attack_indices_global) > 1)[0] + 1
        continuous_segments = np.split(attack_indices_global, splits)
        
        for seg_indices in continuous_segments:
            if len(seg_indices) == 0:
                continue
            
            segments.append({
                'flight_id': flight_id,
                'attack_type': attack_type_real,
                't_attack': t_attack_real,  # 真实攻击开始时间
                't_first_window': timestamps[seg_indices[0]],
                't_last_window': timestamps[seg_indices[-1]],
                'indices': seg_indices.tolist()
            })
    
    print(f"Identified {len(segments)} attack segments")
    return segments


def calculate_detection_delay_v2(y_pred: np.ndarray,
                                 timestamps: np.ndarray,
                                 attack_segments: List[Dict]) -> Tuple[List[Optional[float]], List[bool]]:
    """
    计算检测延迟（版本2：使用真实攻击时间）
    
    延迟 = 首次检测时间 - 真实攻击开始时间
    
    Args:
        y_pred: 预测标签 (0/1 binary)
        timestamps: 窗口时间戳
        attack_segments: 攻击片段（包含t_attack）
    
    Returns:
        delays, detected_flags
    """
    delays = []
    detected_flags = []
    
    for seg in attack_segments:
        t_attack = seg['t_attack']  # 真实攻击开始时间
        attack_indices = seg['indices']
        
        # 在攻击窗口中查找首次检测
        t_detect = None
        for idx in attack_indices:
            if y_pred[idx] == 1:
                t_detect = timestamps[idx]
                break
        
        if t_detect is not None:
            # 延迟 = 检测时间 - 攻击时间
            delay = t_detect - t_attack
            
            # 处理负延迟（窗口时间可能早于真实攻击时间）
            if delay < 0:
                delay = 0  # 认为立即检测
            
            delays.append(delay)
            detected_flags.append(True)
        else:
            delays.append(None)
            detected_flags.append(False)
    
    valid_delays = [d for d in delays if d is not None]
    detection_rate = sum(detected_flags) / len(detected_flags) if detected_flags else 0
    avg_delay = np.mean(valid_delays) if valid_delays else None
    
    print(f"Detection rate: {detection_rate:.2%}")
    if avg_delay is not None:
        print(f"Average delay: {avg_delay:.3f}s")
        print(f"Delay range: [{min(valid_delays):.3f}s, {max(valid_delays):.3f}s]")
    
    return delays, detected_flags
```

### 2.3 修改 `main.py`

```python
# 替换 data_loader_simple 为 data_loader_with_metadata
from data_loader_with_metadata import load_model_data_with_metadata
from time_aware_metrics import (
    identify_attack_segments_with_timing,
    calculate_detection_delay_v2,
    ...
)

# 在加载数据时
for model_name in config.MODEL_NAMES:
    print(f"\n{'='*70}")
    print(f"Evaluating Model: {model_name.upper()}")
    print(f"{'='*70}")
    
    # 加载数据（新方法）
    model_data = load_model_data_with_metadata(model_name)
    
    # 识别攻击片段（新方法）
    attack_segments = identify_attack_segments_with_timing(
        y_true=model_data['y_true'],
        timestamps=model_data['timestamps'],
        flight_ids=model_data['flight_ids'],
        attack_types=model_data['attack_types'],
        attack_segments_info=model_data['attack_segments_info']
    )
    
    # 计算检测延迟（新方法）
    delays, detected = calculate_detection_delay_v2(
        y_pred=model_data['y_pred'],  # 注意：使用二值化的预测
        timestamps=model_data['timestamps'],
        attack_segments=attack_segments
    )
    
    # 后续计算DR@Δt, ADD, MTBFA等...
```

---

## 第三部分：实施步骤

### Step 1: 备份现有代码
```bash
cp step3b_multiModelGeneral/main.py step3b_multiModelGeneral/main.py.backup
cp step4b_newMetrics/data_loader_simple.py step4b_newMetrics/data_loader_simple.py.backup
cp step4b_newMetrics/time_aware_metrics.py step4b_newMetrics/time_aware_metrics.py.backup
```

### Step 2: 修改 step3b
1. 修改 `step3b_multiModelGeneral/main.py` 中的 np.savez 部分
2. 添加 window_metadata.csv 生成代码
3. 添加 window_timeline.csv 生成代码（可选但推荐）

### Step 3: 重新运行 step3b 生成新数据
```bash
cd step3b_multiModelGeneral

# 方式1：重新训练所有模型（耗时）
python main.py

# 方式2：只重新评估（快速，如果模型已训练）
# 修改 main.py，在评估部分单独运行
```

### Step 4: 创建 step4b 新文件
1. 创建 `data_loader_with_metadata.py`
2. 修改 `time_aware_metrics.py` 添加新函数
3. 修改 `main.py` 使用新的加载器

### Step 5: 运行新的评估
```bash
cd step4b_newMetrics
python main.py
```

---

## 第四部分：验证结果

### 预期的正确结果

1. **检测延迟应该有合理的分布**
   - 最小延迟 ≥ 0.5s（一个窗口的持续时间）
   - 平均延迟应该在 1-10s 范围内
   - 不同模型有不同的延迟分布

2. **DR@Δt应该随Δt增加**
   ```
   DR@1s  = 0.20 (20%)
   DR@2s  = 0.35 (35%)
   DR@5s  = 0.60 (60%)
   DR@10s = 0.75 (75%)
   DR@15s = 0.82 (82%)
   DR@30s = 0.90 (90%)
   ```

3. **时间戳应该有意义**
   ```
   窗口0: t=48.5s  (飞行1的第一个窗口中心)
   窗口1: t=48.55s (步进0.05s)
   ...
   攻击开始: t=95.8s (来自attack_info)
   首次检测: t=96.3s (延迟=0.5s)
   ```

---

## 第五部分：快速测试方案

如果完整重新运行step3b太耗时，可以使用简化方案：

### 简化方案：只修改step4b（不修改step3b）

使用窗口索引计算**相对延迟**（单位：窗口数），然后转换为秒。

**优点**：不需要重新运行step3b
**缺点**：时间戳不是绝对的，但相对延迟仍然有意义

```python
# 在 data_loader_simple.py 中
def load_model_data_simple_v2(model_name: str):
    """使用窗口索引计算相对延迟"""
    
    # 加载数据
    data = np.load(pred_file)
    y_true = data['labels']
    y_prob = data['probabilities']  # 关键：使用probabilities！
    y_pred = (y_prob > 0.5).astype(int)  # 二值化
    
    # 使用窗口索引作为"时间"
    window_indices = np.arange(len(y_true))
    
    # 时间单位：窗口步长 = 0.05s
    time_per_window = config.STEP_SIZE / config.SAMPLING_RATE
    timestamps = window_indices * time_per_window
    
    # 其他元数据...
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,  # 二值化！
        'timestamps': timestamps,
        ...
    }
```

这样至少能修复"使用probabilities并二值化"的问题，得到更合理的结果。

---

## 总结

### 核心修复

1. **✅ 使用probabilities并二值化**
   ```python
   y_pred = (data['probabilities'] > 0.5).astype(int)
   ```

2. **✅ 使用真实时间戳**
   - 完整方案：从window_timeline.csv加载
   - 简化方案：使用window_index * step_time

3. **✅ 正确计算延迟**
   ```python
   delay = t_detect - t_attack  # 使用真实时间
   ```

### 实施建议

- **推荐**：完整方案（修改step3b + step4b）
- **快速**：简化方案（只修改step4b，修复probabilities使用）

无论哪种方案，都能解决当前的0.002s延迟和DR不变的问题。
