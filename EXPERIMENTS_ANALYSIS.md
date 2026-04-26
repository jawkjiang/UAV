# 论文实验需求分析

## 📊 现有数据汇总

### ✅ 已有数据

#### B. Overall Performance: Traditional vs. Time-Aware Metrics

**传统指标**（来源：step5_timegan/output/test_metrics_*.json）
- ✅ 7个模型的Precision/Recall/F1
- ✅ 详细指标：PR-AUC, ROC-AUC, TP/FP/TN/FN

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| CNN | 0.9408 | 0.9505 | 0.9456 |
| LSTM | 0.9268 | 0.9691 | 0.9475 |
| BiLSTM | 0.9161 | 0.9862 | 0.9498 |
| GRU | 0.9149 | 0.9878 | 0.9499 |
| CNN-LSTM | 0.9393 | 0.9650 | 0.9520 |
| TCN | 0.9816 | 0.8759 | 0.9258 |
| Transformer | 0.8802 | 0.9450 | 0.9115 |

**时间感知指标**（来源：step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv）
- ✅ 7个模型的DR@5s/ADD/MTBFA
- ✅ 多个时间阈值的DR：DR@0.5s, DR@1s, DR@1.5s, DR@2s, DR@3s, DR@5s, DR@10s, DR@15s, DR@30s
- ✅ 检测率和误报事件数

| Model | DR@5s | ADD (s) | MTBFA (h) | False Alarms |
|-------|-------|---------|-----------|--------------|
| CNN | 0.874 | 2.24 | 0.057 | 45 |
| LSTM | 0.874 | 2.08 | 0.123 | 21 |
| BiLSTM | 0.874 | 2.10 | 0.117 | 22 |
| GRU | 0.874 | 2.08 | 0.117 | 22 |
| CNN-LSTM | 0.874 | 2.15 | 0.096 | 27 |
| TCN | 0.631 | 3.73 | 0.287 | 9 |
| Transformer | 0.864 | 2.26 | 0.013 | 198 |

**关键发现**
- ✅ 可以计算Precision-MTBFA悖论（数据已有）
- ✅ 可以计算Recall-DR@5s差异（数据已有）

#### C. Detection Speed Analysis: DR@Δt Curves

**DR@Δt曲线数据**
- ✅ 已有9个时间点的DR：0.5s, 1s, 1.5s, 2s, 3s, 5s, 10s, 15s, 30s
- ✅ 所有7个模型的完整DR曲线数据

#### D. False Alarm Frequency Analysis

**MTBFA数据**
- ✅ 每个模型的MTBFA值（小时）
- ✅ 每个模型的误报事件数（n_false_alarms）
- ⚠️ **缺少：FP窗口数**（只有FP事件数）
- ❌ **缺少：误报事件持续时长分布**

#### E. Trade-off Analysis

**权衡分析数据**
- ✅ DR@5s和MTBFA数据（可绘制Pareto前沿）
- ✅ 所有模型的完整性能指标

#### F. Per-Attack-Type Performance

**分攻击类型数据**（来源：step6_newMetrcisWithTimegan/output/time_aware_metrics/*_per_attack_metrics.csv）
- ✅ 各攻击类型的ADD
- ✅ 各攻击类型的DR@5s和DR@10s
- ✅ 4种攻击类型：step, delay, takeover, drift
- ⚠️ **数据粒度不够细**：只有4类攻击，没有原始的6类细分（step, drift_ramp, drift_sigmoid, delay, takeover_step, takeover_ramp）

示例数据（CNN模型）：
| Attack Type | n_instances | ADD | DR@5s | DR@10s |
|-------------|-------------|-----|-------|--------|
| step | 26 | 1.93 | 0.885 | 1.0 |
| delay | 25 | 2.45 | 0.840 | 1.0 |
| takeover | 26 | 1.66 | 0.962 | 1.0 |
| drift | 26 | 2.92 | 0.808 | 1.0 |

#### I. Computational Efficiency Analysis

**性能数据**（来源：step7_performance_analysis/output/）
- ✅ 推理延迟：mean_latency_ms, p95_latency_ms, p99_latency_ms
- ✅ 吞吐量：throughput_windows_per_sec
- ✅ 内存占用：total_params, model_size_mb, memory_mb
- ✅ 计算复杂度：FLOPs
- ✅ CPU和CUDA两种设备的性能数据

---

## ❌ 缺失的实验和数据

### 🔴 高优先级（论文必需）

#### 1. **误报窗口数 vs 误报事件数对比**
**位置**：D. False Alarm Frequency Analysis - MTBFA与Precision的对比

**缺失内容**：
- FP窗口数（window-level false positives）
- 需要对比：Precision计算用的FP窗口数 vs MTBFA计算用的FP事件数

**示例表格**（目前缺失）：
| Model | Precision | FP Windows | FP Events | MTBFA (h) |
|-------|-----------|------------|-----------|-----------|
| Transformer | 0.880 | ??? | 198 | 0.013 |
| CNN | 0.941 | ??? | 45 | 0.057 |

**解决方案**：
- 需要从step5的test_metrics_*.json中提取false_positives（窗口级）
- 与step6的n_false_alarms（事件级）对比
- 计算"窗口-事件聚合率"：FP_events / FP_windows

**实验脚本**：需要新建分析脚本

---

#### 2. **误报事件持续时长分布**
**位置**：D. False Alarm Frequency Analysis - 误报事件的时间分布

**缺失内容**：
- 每个FP事件的持续时间（duration）
- FP事件持续时长的分布直方图
- 统计：平均持续时长、中位数、最长/最短

**期望输出**：
```
FP Event Duration Distribution:
- <5s: 80% (短暂误报)
- 5-10s: 15% (中等持续)
- >10s: 5% (长时间误报)
```

**解决方案**：
1. 修改time_aware_metrics.py中的calculate_mtbfa函数
2. 记录每个FP事件的start_time和end_time
3. 计算duration = end_time - start_time
4. 输出分布统计和可视化

**实验脚本**：需要修改现有代码并重新运行

---

#### 3. **细分攻击类型的性能分析**
**位置**：F. Per-Attack-Type Performance

**缺失内容**：
- 当前只有4类合并后的攻击类型（step, delay, takeover, drift）
- 需要6类原始攻击类型的详细数据：
  - step
  - drift_ramp
  - drift_sigmoid
  - delay
  - takeover_step
  - takeover_ramp

**期望表格**：
| Model | Attack Type | n_instances | ADD | DR@5s | DR@10s |
|-------|-------------|-------------|-----|-------|--------|
| CNN | step | 26 | 1.93 | 0.885 | 1.0 |
| CNN | drift_ramp | 13 | ??? | ??? | ??? |
| CNN | drift_sigmoid | 13 | ??? | ??? | ??? |
| ... | ... | ... | ... | ... | ... |

**解决方案**：
- 检查data_loader.py中的攻击类型标注
- 如果已有细分数据，修改evaluation.py保留原始攻击类型
- 如果没有细分，需要重新注入攻击并标注

**实验脚本**：可能需要修改step6代码并重新运行

---

### 🟡 中优先级（增强论文说服力）

#### 4. **攻击参数敏感性分析**（可选）
**位置**：F. Per-Attack-Type Performance - 攻击参数敏感性分析

**缺失内容**：
- 不同幅值（5m/15m/30m）对检测延迟的影响
- 不同方向模式对检测率的影响

**需要确认**：
- 当前数据集是否包含不同参数的攻击？
- 如果没有，需要额外注入攻击实验

**建议**：
- 如果数据已有：分析现有数据
- 如果数据没有：标注为"可选"或删除此部分

---

#### 5. **运行负担评估的具体案例**
**位置**：D. False Alarm Frequency Analysis - 运行负担评估

**缺失内容**：
- 具体的100小时连续运行场景分析
- 每个模型的预期误报次数

**期望输出**：
```
Scenario: 100小时连续运行
- Transformer: 预期误报 7692次 (100h / 0.013h)
- CNN: 预期误报 1754次 (100h / 0.057h)
- TCN: 预期误报 349次 (100h / 0.287h)
```

**解决方案**：
- 简单计算：100 / MTBFA
- 可以直接基于现有数据生成

**实验脚本**：简单分析脚本即可

---

### 🟢 低优先级（锦上添花）

#### 6. **可视化图表**
**位置**：多个章节

**缺失内容**：
- 传统指标对比柱状图
- 时间感知指标对比柱状图
- Precision vs MTBFA散点图
- DR@Δt曲线图
- ADD vs DR@5s散点图
- Pareto前沿图
- 分攻击类型热力图

**解决方案**：
- step6_newMetrcisWithTimegan/visualizations.py已有部分框架
- 需要完善和生成所有需要的图表

**实验脚本**：完善visualizations.py

---

#### 7. **相关性分析**
**位置**：B. Overall Performance - 关键发现

**缺失内容**：
- Recall vs DR@5s的相关性系数（文中提到r=0.742）
- Precision vs MTBFA的相关性分析

**解决方案**：
- 使用scipy.stats.pearsonr计算相关系数
- 可视化散点图和拟合线

**实验脚本**：简单统计分析

---

## 🎯 实验优先级总结

### 必须完成（论文核心数据）

1. **提取FP窗口数**（从step5数据中）
   - 对比窗口级FP vs 事件级FP
   - 生成Precision-MTBFA悖论表格

2. **计算误报事件持续时长分布**
   - 修改time_aware_metrics.py
   - 重新运行step6分析
   - 生成分布统计和直方图

3. **生成核心可视化**
   - Precision vs MTBFA散点图
   - DR@Δt曲线图
   - Pareto前沿图

### 建议完成（增强说服力）

4. **细分攻击类型分析**（如果数据支持）
   - 检查是否有6类细分攻击类型
   - 如有，重新生成per-attack表格

5. **运行负担评估**
   - 简单计算100小时场景
   - 对比不同模型的误报次数

6. **相关性分析**
   - 计算Recall-DR@5s相关系数
   - 计算Precision-MTBFA相关系数

### 可选内容

7. **攻击参数敏感性**（取决于数据可用性）
8. **完整的可视化库**（所有图表）

---

## 📝 建议的实验顺序

### Phase 1: 数据提取和补充（1-2小时）

1. 创建`extract_fp_windows.py`：提取FP窗口数
2. 修改`time_aware_metrics.py`：添加FP事件持续时长记录
3. 重新运行step6：`python main.py`

### Phase 2: 数据分析（1-2小时）

4. 创建`analyze_precision_mtbfa_paradox.py`：分析窗口-事件悖论
5. 创建`analyze_fp_duration.py`：分析误报持续时长分布
6. 创建`calculate_correlations.py`：计算相关性系数

### Phase 3: 可视化（2-3小时）

7. 完善`visualizations.py`：生成所有核心图表
   - Precision vs MTBFA散点图
   - DR@Δt曲线图（7个模型）
   - ADD vs DR@5s散点图
   - Pareto前沿图
   - 误报持续时长分布直方图

### Phase 4: 细分攻击类型（可选，1-2小时）

8. 检查攻击类型细分数据
9. 如需要，修改evaluation代码重新运行

---

## 🚀 快速启动方案

**如果时间紧张，优先完成以下3项**：

1. **提取FP窗口数**（30分钟）
   - 从step5 test_metrics提取false_positives
   - 生成对比表格

2. **计算FP持续时长**（1小时）
   - 修改time_aware_metrics.py
   - 重新运行step6

3. **生成3个核心图表**（1小时）
   - Precision vs MTBFA散点图
   - DR@Δt曲线图
   - Pareto前沿图

**总计：2.5小时即可获得论文必需的核心数据和图表**

---

## 📂 建议的文件组织

```
UAV/
├── paper_analysis/                    # 新建：论文专用分析
│   ├── extract_fp_windows.py         # 提取FP窗口数
│   ├── analyze_mtbfa_paradox.py      # 分析Precision-MTBFA悖论
│   ├── analyze_fp_duration.py        # 分析误报持续时长
│   ├── calculate_correlations.py     # 计算相关性
│   ├── generate_all_figures.py       # 生成所有论文图表
│   └── output/                       # 输出结果
│       ├── tables/                   # CSV表格
│       ├── figures/                  # 图表
│       └── statistics/               # 统计分析
└── step6_newMetrcisWithTimegan/      # 修改现有代码
    └── time_aware_metrics.py         # 添加FP持续时长记录
```
