# 电力系统无人机巡检场景 - 指标体系计算规范

## 📋 项目背景

### 应用场景
**无人机电力巡检端侧GPS欺骗检测**
- 高度自动化电力系统中，巡检无人机依赖GNSS+IMU导航
- 端侧部署轻量级检测模型，识别GPS攻击
- 检测到攻击后通知地面站，触发人工或智能体介入

### 关键约束
- ✅ 部署在边缘设备（资源受限）
- ✅ 需要低延迟实时检测
- ✅ 误报和漏报代价不对称（漏报更严重）
- ✅ 重放攻击是最危险的攻击类型

---

## 🎯 任务目标

基于**现有的56个实验结果**，计算5套候选指标体系，评估各模型在电力巡检场景下的适用性。

### 输入数据
所有数据位于：`step3_multiModel/output/`

```
output/
├── overall_comparison.csv              # 56个实验的汇总指标
├── {attack_type}/{model_type}/
│   ├── test_metrics.json              # 详细指标（AUC, F1, Precision, Recall等）
│   ├── test_predictions.npz           # 预测结果（predictions, targets, flight_ids）
│   ├── test_attack_info.csv           # 攻击信息（attack_id, start_idx, end_idx等）
│   └── training_history.json          # 训练历史
└── visualizations/                     # 已有可视化
```

### 输出要求
生成一个新的分析脚本和报告：
```
step3_multiModel/
├── power_grid_metrics.py              # 新建：指标计算脚本
├── power_grid_analysis.py             # 新建：分析和可视化
├── output/
│   └── power_grid_analysis/           # 新建：输出目录
│       ├── metrics_comparison.csv     # 5套指标体系的结果
│       ├── metrics_rankings.csv       # 各指标下的模型排名
│       ├── attack_severity.csv        # 攻击严重度分析
│       ├── detection_delay_stats.csv  # 检测延迟统计
│       ├── POWER_GRID_REPORT.md       # 综合分析报告
│       └── visualizations/            # 新的可视化图表
│           ├── cost_sensitive_comparison.png
│           ├── detection_delay_distribution.png
│           ├── risk_weighted_performance.png
│           ├── edge_deployment_suitability.png
│           └── comprehensive_ranking.png
```

---

## 📊 5套指标体系详细定义

### 指标体系1: 基于代价的安全-效率权衡
**Cost-Sensitive Safety-Efficiency Trade-off**

#### 指标定义

**1.1 Weighted F-Score (多个β值)**
```python
# 计算不同β值下的F_β分数
beta_values = [0.5, 1.0, 2.0, 5.0]
for beta in beta_values:
    F_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
```
- β=0.5: 重视精确率（减少误报）
- β=1.0: 标准F1（平衡）
- β=2.0: 重视召回率（减少漏报）
- β=5.0: 极度重视召回率（巡检场景推荐）

**数据来源**: `test_metrics.json`中的`precision`和`recall`

**1.2 Expected Operational Cost (EOC)**
```python
# 从confusion matrix计算
TN, FP, FN, TP = confusion_matrix.ravel()
FPR = FP / (FP + TN)  # False Positive Rate
FNR = FN / (FN + TP)  # False Negative Rate

# 代价定义
C_FP = 1    # 误报代价（人工检查成本）
C_FN = 50   # 漏报代价（事故风险，建议范围: 10-100）

EOC = C_FP * FPR + C_FN * FNR
```

**数据来源**: 从`test_metrics.json`中的`precision`, `recall`反推confusion matrix

**1.3 Cost-Benefit Score (CBS)**
```python
# 收益 = 成功检测避免的损失
benefit = recall * attack_probability * attack_damage
# 成本 = 误报处理成本
cost = FPR * normal_operation_frequency * false_alarm_cost

CBS = benefit - cost
```

**参数建议**:
- `attack_probability = 0.01` (假设1%飞行遭遇攻击)
- `attack_damage = 1000` (单位成本)
- `false_alarm_cost = 1` (单位成本)

---

### 指标体系2: 基于时效性的实时检测能力
**Real-time Detection Capability**

#### 指标定义

**2.1 Mean Time to Detection (MTTD)**
```python
# 需要从predictions和attack_info计算
for each_attack_segment:
    attack_start = attack_info['start_idx']
    attack_end = attack_info['end_idx']
    predictions_in_attack = predictions[attack_start:attack_end]
    
    # 找到首次检测点（prediction > threshold）
    first_detection = np.argmax(predictions_in_attack > threshold)
    
    # 计算延迟（窗口数）
    delay_windows = first_detection
    delay_seconds = delay_windows * STEP_SIZE * SAMPLING_INTERVAL
    
MTTD = mean(delay_seconds for all attacks)
```

**数据来源**: 
- `test_predictions.npz`: predictions数组
- `test_attack_info.csv`: start_idx, end_idx
- 配置常量: WINDOW_SIZE=50, STEP_SIZE=5, SAMPLING_INTERVAL=0.5s

**2.2 Detection Delay Distribution**
```python
# 计算分位数
delay_percentiles = {
    'P50': np.percentile(delays, 50),
    'P90': np.percentile(delays, 90),
    'P95': np.percentile(delays, 95),
    'P99': np.percentile(delays, 99)
}
```

**2.3 Rapid Response Rate (RRR@k)**
```python
# k个时间步内检测到的比例
k_values = [5, 10, 20]  # 窗口数
for k in k_values:
    RRR_k = sum(delay <= k) / total_attacks
```

**2.4 Inference Latency (需要实际测量)**
```python
# 在同一硬件上测试
import time
model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        output = model(test_input)
    end = time.time()
    
avg_latency_ms = (end - start) / 100 * 1000
```

---

### 指标体系3: 基于可靠性的运行质量
**Operational Reliability**

#### 指标定义

**3.1 False Alarm Rate @ Operating Point**
```python
# 在达到99% TPR时的FPR
# 方法1: 从ROC曲线数据找
threshold_99 = find_threshold_for_tpr(roc_curve, target_tpr=0.99)
FAR_99 = FPR_at_threshold(predictions, targets, threshold_99)

# 方法2: 如果没有ROC数据，从当前指标估算
# 假设当前threshold达到的TPR
current_TPR = recall
if current_TPR >= 0.99:
    FAR_99 = FPR  # 使用当前FPR
else:
    # 需要调整阈值（降低阈值以提高TPR）
    # 这会增加FPR，需要实际计算
```

**3.2 Mean Time Between False Alarms (MTBFA)**
```python
# 假设：
# - 每次飞行时长: 30分钟 = 1800秒
# - 采样间隔: 0.5秒
# - 每次飞行样本数: 3600个

samples_per_flight = 3600
normal_samples = (1 - attack_probability) * total_samples
FP_count = FPR * normal_samples

MTBFA_samples = normal_samples / FP_count
MTBFA_flights = MTBFA_samples / samples_per_flight
```

**3.3 Availability Score**
```python
# 系统可用性 = 1 - 误报导致的停机时间比例
inspection_time = 10  # 每次误报需要10分钟人工检查
flight_duration = 30  # 每次飞行30分钟

downtime_ratio = FPR * (inspection_time / flight_duration)
Availability = 1 - downtime_ratio
```

**3.4 Robustness Coefficient**
```python
# 跨攻击类型的性能稳定性
auc_values = [auc for each attack type]
mean_auc = np.mean(auc_values)
std_auc = np.std(auc_values)

Robustness = 1 - (std_auc / mean_auc)  # 变异系数的倒数
```

**数据来源**: `overall_comparison.csv`中每个模型在8种攻击上的AUC

---

### 指标体系4: 基于风险的关键任务保障
**Risk-Based Mission Assurance**

#### 指标定义

**4.1 Risk-Weighted Detection Rate (RWDR)**
```python
# 攻击严重度权重定义
attack_weights = {
    'replay_same_hard': 5,      # 最危险：完全隐蔽
    'replay_other_soft': 4,     # 次危险：部分隐蔽
    'delay': 3,                  # 中等：时间偏移
    'drift_ramp': 2,             # 较低：缓慢偏移
    'drift_sigmoid': 2,
    'step': 1,                   # 最低：明显突变
    'takeover_step': 1,
    'takeover_ramp': 1
}

# 计算加权检测率
RWDR = sum(weight[attack] * recall[attack] for attack in attacks) / sum(weights)
```

**理由**:
- 重放攻击使用真实GPS数据，最难检测
- 渐变攻击有时间窗口积累特征
- 突变攻击在residual中立即显现

**4.2 Critical Miss Rate (CMR)**
```python
# 对高危攻击的漏报率
critical_attacks = ['replay_same_hard', 'replay_other_soft']

CMR = sum(FN[attack] for attack in critical_attacks) / 
      sum(TP[attack] + FN[attack] for attack in critical_attacks)
```

**4.3 Safety Margin (SM)**
```python
# 检测样本的置信度安全裕度
detected_samples = predictions[predictions > threshold]
confidence_p95 = np.percentile(detected_samples, 95)

SM = confidence_p95 - threshold
# SM越大，说明检测越有把握
```

**数据来源**: `test_predictions.npz`

**4.4 Worst-Case Performance (WCP)**
```python
# 在最难攻击上的F1分数
f1_scores = [f1 for each attack type]
WCP = min(f1_scores)
```

---

### 指标体系5: 基于资源约束的边缘部署适配性
**Edge Deployment Suitability**

#### 指标定义

**5.1 Performance-Efficiency Ratio (PER)**
```python
# 性能/资源消耗的综合指标
model_params = {
    'cnn': 263873,
    'lstm': 215873,
    'bilstm': 150337,
    'gru': 164545,
    'cnn_lstm': 301953,
    'tcn': 77313,
    'transformer': 103489
}

model_size_MB = params * 4 / (1024**2)  # float32, 4 bytes per param

# 需要测量推理延迟
latency_ms = measure_inference_time(model)

PER = AUC / (model_size_MB * latency_ms)**0.5
```

**5.2 Throughput at Target Latency**
```python
# 在100ms延迟约束下的吞吐量
target_latency_ms = 100
actual_latency_ms = measure_inference_time(model)

if actual_latency_ms <= target_latency_ms:
    throughput = 1000 / actual_latency_ms  # samples/second
else:
    throughput = 0  # 不满足延迟要求
```

**5.3 Memory Footprint Score**
```python
# 归一化的内存占用评分
max_params = max(model_params.values())
min_params = min(model_params.values())

Memory_Score = 1 - (params - min_params) / (max_params - min_params)
# Score越高，内存占用越小
```

**5.4 Energy-Performance Trade-off**
```python
# 估算能耗效率
avg_inference_per_attack = MTTD / STEP_SIZE  # 检测一次攻击需要的推理次数

Energy_Score = AUC / (params * avg_inference_per_attack / 1e6)
# 归一化到合理范围
```

---

## 🔧 实现步骤

### Step 1: 数据加载工具函数

```python
# power_grid_metrics.py

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_experiment_results(output_dir='./output'):
    """加载所有实验结果"""
    results = []
    
    # 加载overall_comparison.csv
    overall_df = pd.read_csv(os.path.join(output_dir, 'overall_comparison.csv'))
    
    # 遍历所有实验
    for _, row in overall_df.iterrows():
        attack_type = row['attack_type']
        model_type = row['model_type']
        
        exp_dir = os.path.join(output_dir, attack_type, model_type)
        
        # 加载详细指标
        metrics_path = os.path.join(exp_dir, 'test_metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # 加载预测结果
        predictions_path = os.path.join(exp_dir, 'test_predictions.npz')
        predictions_data = np.load(predictions_path)
        
        # 加载攻击信息
        attack_info_path = os.path.join(exp_dir, 'test_attack_info.csv')
        attack_info = pd.read_csv(attack_info_path)
        
        results.append({
            'attack_type': attack_type,
            'model_type': model_type,
            'metrics': metrics,
            'predictions': predictions_data['predictions'],
            'targets': predictions_data['targets'],
            'flight_ids': predictions_data['flight_ids'],
            'attack_info': attack_info
        })
    
    return results

def get_confusion_matrix(precision, recall, total_positive, total_negative):
    """从precision和recall反推confusion matrix"""
    TP = recall * total_positive
    FP = TP * (1/precision - 1) if precision > 0 else 0
    FN = total_positive - TP
    TN = total_negative - FP
    return int(TN), int(FP), int(FN), int(TP)
```

### Step 2: 各指标体系的计算函数

```python
def calculate_cost_sensitive_metrics(results, C_FP=1, C_FN=50):
    """指标体系1: 代价权衡"""
    metrics = []
    
    for exp in results:
        precision = exp['metrics']['precision']
        recall = exp['metrics']['recall']
        
        # Weighted F-Score
        f_scores = {}
        for beta in [0.5, 1.0, 2.0, 5.0]:
            if precision + recall > 0:
                f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
            else:
                f_beta = 0
            f_scores[f'F_{beta}'] = f_beta
        
        # EOC
        # 假设test数据中正负样本各约一半
        total_samples = len(exp['targets'])
        total_positive = exp['targets'].sum()
        total_negative = total_samples - total_positive
        
        TN, FP, FN, TP = get_confusion_matrix(precision, recall, total_positive, total_negative)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        
        EOC = C_FP * FPR + C_FN * FNR
        
        metrics.append({
            'attack_type': exp['attack_type'],
            'model_type': exp['model_type'],
            **f_scores,
            'EOC': EOC,
            'FPR': FPR,
            'FNR': FNR
        })
    
    return pd.DataFrame(metrics)

def calculate_detection_delay_metrics(results, threshold=0.5, step_size=5, sampling_interval=0.5):
    """指标体系2: 时效性"""
    metrics = []
    
    for exp in results:
        predictions = exp['predictions']
        attack_info = exp['attack_info']
        
        delays = []
        
        # 计算每个攻击段的检测延迟
        for _, attack in attack_info.iterrows():
            if attack['attack_id'] == 0:  # 跳过正常段
                continue
            
            start_idx = int(attack['start_idx'])
            end_idx = int(attack['end_idx'])
            
            # 获取该攻击段的预测
            attack_predictions = predictions[start_idx:end_idx]
            
            # 找到首次检测点
            detected_indices = np.where(attack_predictions > threshold)[0]
            
            if len(detected_indices) > 0:
                first_detection = detected_indices[0]
                delay_windows = first_detection
            else:
                delay_windows = len(attack_predictions)  # 未检测到
            
            delay_seconds = delay_windows * step_size * sampling_interval
            delays.append(delay_seconds)
        
        if len(delays) > 0:
            MTTD = np.mean(delays)
            P50 = np.percentile(delays, 50)
            P90 = np.percentile(delays, 90)
            P95 = np.percentile(delays, 95)
            P99 = np.percentile(delays, 99)
            
            # RRR@k
            RRR_5 = sum(d <= 5 * step_size * sampling_interval for d in delays) / len(delays)
            RRR_10 = sum(d <= 10 * step_size * sampling_interval for d in delays) / len(delays)
            RRR_20 = sum(d <= 20 * step_size * sampling_interval for d in delays) / len(delays)
        else:
            MTTD = P50 = P90 = P95 = P99 = float('inf')
            RRR_5 = RRR_10 = RRR_20 = 0
        
        metrics.append({
            'attack_type': exp['attack_type'],
            'model_type': exp['model_type'],
            'MTTD': MTTD,
            'P50_delay': P50,
            'P90_delay': P90,
            'P95_delay': P95,
            'P99_delay': P99,
            'RRR@5': RRR_5,
            'RRR@10': RRR_10,
            'RRR@20': RRR_20
        })
    
    return pd.DataFrame(metrics)

def calculate_risk_weighted_metrics(results, attack_weights=None):
    """指标体系4: 风险加权"""
    if attack_weights is None:
        attack_weights = {
            'replay_same_hard': 5,
            'replay_other_soft': 4,
            'delay': 3,
            'drift_ramp': 2,
            'drift_sigmoid': 2,
            'step': 1,
            'takeover_step': 1,
            'takeover_ramp': 1
        }
    
    # 按模型汇总
    model_metrics = {}
    
    for exp in results:
        model = exp['model_type']
        attack = exp['attack_type']
        recall = exp['metrics']['recall']
        f1 = exp['metrics']['f1']
        
        if model not in model_metrics:
            model_metrics[model] = {
                'weighted_recalls': [],
                'weights': [],
                'f1_scores': [],
                'critical_stats': {'TP': 0, 'FN': 0}
            }
        
        weight = attack_weights.get(attack, 1)
        model_metrics[model]['weighted_recalls'].append(recall * weight)
        model_metrics[model]['weights'].append(weight)
        model_metrics[model]['f1_scores'].append(f1)
        
        # Critical Miss Rate (只统计重放攻击)
        if attack in ['replay_same_hard', 'replay_other_soft']:
            total_samples = len(exp['targets'])
            total_positive = exp['targets'].sum()
            precision = exp['metrics']['precision']
            
            TN, FP, FN, TP = get_confusion_matrix(precision, recall, total_positive, total_samples - total_positive)
            model_metrics[model]['critical_stats']['TP'] += TP
            model_metrics[model]['critical_stats']['FN'] += FN
    
    # 计算汇总指标
    metrics = []
    for model, data in model_metrics.items():
        RWDR = sum(data['weighted_recalls']) / sum(data['weights'])
        WCP = min(data['f1_scores'])
        
        TP = data['critical_stats']['TP']
        FN = data['critical_stats']['FN']
        CMR = FN / (TP + FN) if (TP + FN) > 0 else 1.0
        
        metrics.append({
            'model_type': model,
            'RWDR': RWDR,
            'CMR': CMR,
            'WCP': WCP
        })
    
    return pd.DataFrame(metrics)
```

### Step 3: 主执行脚本

```python
def main():
    """主函数"""
    print("="*80)
    print("电力系统无人机巡检场景 - 指标体系计算")
    print("="*80)
    
    # 创建输出目录
    output_dir = './output/power_grid_analysis'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 加载数据
    print("\n加载实验数据...")
    results = load_experiment_results('./output')
    print(f"已加载 {len(results)} 个实验结果")
    
    # 计算各指标体系
    print("\n计算指标体系1: 代价权衡...")
    metrics1 = calculate_cost_sensitive_metrics(results, C_FP=1, C_FN=50)
    metrics1.to_csv(os.path.join(output_dir, 'metrics1_cost_sensitive.csv'), index=False)
    
    print("计算指标体系2: 时效性...")
    metrics2 = calculate_detection_delay_metrics(results)
    metrics2.to_csv(os.path.join(output_dir, 'metrics2_detection_delay.csv'), index=False)
    
    print("计算指标体系4: 风险加权...")
    metrics4 = calculate_risk_weighted_metrics(results)
    metrics4.to_csv(os.path.join(output_dir, 'metrics4_risk_weighted.csv'), index=False)
    
    # 生成综合排名
    print("\n生成综合排名...")
    generate_comprehensive_ranking(metrics1, metrics2, metrics4, output_dir)
    
    print(f"\n✓ 所有指标已计算完成，结果保存在: {output_dir}")

if __name__ == '__main__':
    main()
```

---

## 📈 可视化要求

### 1. 代价权衡对比图
- X轴: 模型
- Y轴: EOC (越低越好)
- 不同颜色表示不同C_FN值 (10, 50, 100)

### 2. 检测延迟分布图
- 箱线图，每个模型一个箱体
- 显示MTTD的P50, P90, P99

### 3. 风险加权性能雷达图
- 雷达图，轴为: RWDR, (1-CMR), WCP, 等
- 每个模型一条线

### 4. 边缘部署适配性散点图
- X轴: 模型大小 (MB)
- Y轴: AUC
- 气泡大小: 推理延迟
- 右上角 = 最优

### 5. 综合排名热力图
- 行: 模型
- 列: 各项指标的排名
- 颜色: 排名 (1=绿, 7=红)

---

## ✅ 验证标准

### 区分度评估
对于每套指标体系，计算模型间的**变异系数** (CV):
```python
CV = std(metric_values) / mean(metric_values)
```

- CV > 0.15: ⭐⭐⭐⭐⭐ 优秀区分度
- CV > 0.10: ⭐⭐⭐⭐ 良好区分度
- CV > 0.05: ⭐⭐⭐ 中等区分度
- CV < 0.05: ⭐⭐ 区分度不足

### 一致性检验
检查各指标体系的模型排名相关性:
```python
from scipy.stats import spearmanr

# 计算Spearman秩相关系数
correlation = spearmanr(ranking1, ranking2)
```

- 如果相关性 > 0.8: 指标体系冗余，选其一
- 如果相关性 < 0.5: 指标体系互补，都有价值

---

## 📚 文献支持建议

### 代价敏感学习
1. Elkan, C. (2001). "The foundations of cost-sensitive learning"
2. Zhou & Liu (2006). "Training cost-sensitive neural networks with methods addressing the class imbalance problem"

### 实时检测
3. Hundman et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" (NASA)
4. Munir et al. (2019). "DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series"

### 风险评估
5. Barrère et al. (2018). "Cyber-Physical Security Risk Assessment for Industrial Control Systems"
6. NIST SP 800-82: "Guide to Industrial Control Systems (ICS) Security"

### 边缘部署
7. Canziani et al. (2016). "An Analysis of Deep Neural Network Models for Practical Applications"
8. Han et al. (2015). "Deep Compression: Compressing Deep Neural Networks"

---

## 🎯 预期成果

完成后应得到：

1. **5套指标的详细数值** (CSV格式)
2. **各指标下的模型排名** (表格)
3. **区分度分析报告** (哪些指标有效)
4. **5-6张专业可视化图表** (300 DPI)
5. **综合分析报告** (Markdown)，包含:
   - 每套指标的解释和计算方法
   - 模型在各指标下的表现对比
   - 推荐的最优指标体系（区分度最好的1-2套）
   - 针对电力巡检场景的模型选择建议

---

## 💡 额外建议

### 如果时间允许，可以补充：

1. **敏感性分析**
   - 改变权重参数（C_FN, 攻击权重）看排名是否稳定
   
2. **场景对比**
   - 对比"电力巡检场景"和"通用场景"的模型排名差异
   
3. **多目标优化**
   - 帕累托前沿分析（性能 vs 效率）

---

**实现优先级**:
1. ⭐⭐⭐⭐⭐ 指标体系2（时效性）+ 指标体系4（风险）
2. ⭐⭐⭐⭐ 指标体系1（代价）
3. ⭐⭐⭐ 指标体系5（边缘部署，需要测量）
4. ⭐⭐ 指标体系3（可靠性，可选）

**开始实现吧！** 🚀
