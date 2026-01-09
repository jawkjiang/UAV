# Step 4: 电力系统无人机巡检场景 - 多指标评估体系

## 📋 项目概述

本模块实现了针对**电力系统无人机巡检场景**的GPS欺骗检测模型评估体系。基于Step 3的56个实验结果，计算5套专业指标体系，全面评估各模型在电力巡检这一特定应用场景下的适用性。

### 核心特点

- ✅ **场景化评估**: 针对电力巡检的实际需求设计指标
- ✅ **多维度分析**: 5套指标体系覆盖安全、效率、可靠性等多个维度
- ✅ **风险导向**: 重点关注高危的重放攻击检测能力
- ✅ **实用性强**: 提供明确的模型选择建议

---

## 🎯 应用场景

**无人机电力巡检端侧GPS欺骗检测**

- 高度自动化电力系统中，巡检无人机依赖GNSS+IMU导航
- 端侧部署轻量级检测模型，实时识别GPS攻击
- 检测到攻击后通知地面站，触发人工或智能体介入
- **关键约束**:
  - 资源受限的边缘设备部署
  - 低延迟实时检测要求
  - 误报和漏报代价不对称（漏报更严重）
  - 重放攻击是最危险的攻击类型

---

## 📊 5套指标体系

### 1️⃣ 基于代价的安全-效率权衡
**Cost-Sensitive Safety-Efficiency Trade-off**

- **Weighted F-Score (F_β)**: 不同β值下的F分数 (β=0.5, 1.0, 2.0, 5.0)
- **Expected Operational Cost (EOC)**: 期望运营代价 = C_FP × FPR + C_FN × FNR
- **Cost-Benefit Score (CBS)**: 成本收益分数

**适用场景**: 需要平衡误报和漏报代价的场景

---

### 2️⃣ 基于时效性的实时检测能力
**Real-time Detection Capability**

- **Mean Time to Detection (MTTD)**: 平均检测延迟
- **Detection Delay Distribution**: 检测延迟分位数 (P50, P90, P95, P99)
- **Rapid Response Rate (RRR@k)**: 快速响应率 (k=5, 10, 20窗口)

**适用场景**: 实时性要求高的关键任务

---

### 3️⃣ 基于可靠性的运行质量
**Operational Reliability**

- **False Alarm Rate @ 99% TPR**: 达到99%召回率时的误报率
- **Mean Time Between False Alarms (MTBFA)**: 平均误报间隔时间
- **Availability Score**: 系统可用性评分
- **Robustness Coefficient**: 跨攻击类型的性能稳定性

**适用场景**: 长期稳定运行的系统

---

### 4️⃣ 基于风险的关键任务保障
**Risk-Based Mission Assurance**

- **Risk-Weighted Detection Rate (RWDR)**: 风险加权检测率
- **Critical Miss Rate (CMR)**: 关键攻击漏报率 (仅统计重放攻击)
- **Worst-Case Performance (WCP)**: 最差情况性能
- **Performance Gap**: 最佳与最差性能差距

**适用场景**: 关键任务保障，重视高危攻击检测

**⭐ 电力巡检场景推荐使用此指标体系**

---

### 5️⃣ 基于资源约束的边缘部署适配性
**Edge Deployment Suitability**

- **Performance-Efficiency Ratio (PER)**: 性能效率比 = AUC / √(模型大小 × 延迟)
- **Memory Footprint Score**: 内存占用评分
- **Energy-Performance Trade-off**: 能耗性能权衡

**适用场景**: 资源受限的边缘设备部署

---

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy pandas matplotlib seaborn scipy
```

### 运行流程

#### 步骤1: 计算所有指标

```bash
cd step4_multiMetrics
python power_grid_metrics.py
```

**输出**:
```
output/power_grid_analysis/
├── metrics1_cost_sensitive.csv      # 代价权衡指标
├── metrics2_detection_delay.csv     # 时效性指标
├── metrics3_reliability.csv         # 可靠性指标
├── metrics3_robustness.csv          # 鲁棒性系数
├── metrics4_risk_weighted.csv       # 风险加权指标 ⭐
├── metrics5_edge_deployment.csv     # 边缘部署指标
└── POWER_GRID_REPORT.md             # 初步报告
```

#### 步骤2: 生成分析和可视化

```bash
python power_grid_analysis.py
```

**输出**:
```
output/power_grid_analysis/
├── comprehensive_rankings.csv       # 综合排名
├── discrimination_analysis.csv      # 区分度分析
├── POWER_GRID_REPORT.md             # 完整报告
└── visualizations/                  # 可视化图表
    ├── cost_sensitive_comparison.png
    ├── detection_delay_distribution.png
    ├── risk_weighted_performance.png
    ├── edge_deployment_suitability.png
    ├── comprehensive_ranking.png
    └── attack_severity_analysis.png
```

---

## 📂 文件结构

```
step4_multiMetrics/
├── README.md                        # 本文件
├── config_step4.py                  # 配置文件
├── power_grid_metrics.py            # 指标计算模块
├── power_grid_analysis.py           # 分析和可视化模块
└── output/
    └── power_grid_analysis/         # 输出目录
        ├── metrics*.csv             # 各指标体系结果
        ├── comprehensive_rankings.csv
        ├── discrimination_analysis.csv
        ├── POWER_GRID_REPORT.md
        └── visualizations/          # 图表
```

---

## 🔧 配置说明

### 修改配置参数

编辑 `config_step4.py` 文件：

```python
# 代价参数 (指标体系1)
COST_FP = 1    # 误报代价
COST_FN = 50   # 漏报代价 (可调整为10-100)

# 攻击权重 (指标体系4)
ATTACK_WEIGHTS = {
    'replay_same_hard': 5,   # 最危险
    'replay_other_soft': 4,
    'delay': 3,
    # ... 可根据实际情况调整
}

# 检测阈值
DETECTION_THRESHOLD = 0.5  # 二分类阈值
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `COST_FN` | 50 | 漏报代价，建议范围10-100 |
| `COST_FP` | 1 | 误报代价，作为基准 |
| `TARGET_TPR` | 0.99 | 目标召回率 |
| `DETECTION_THRESHOLD` | 0.5 | 检测阈值 |
| `RRR_WINDOWS` | [5, 10, 20] | 快速响应窗口数 |

---

## 📈 结果解读

### 综合排名 (comprehensive_rankings.csv)

- **avg_rank**: 平均排名（越小越好）
- 各列表示在不同指标下的排名
- 排名1表示该指标下的最佳模型

### 区分度分析 (discrimination_analysis.csv)

- **CV**: 变异系数（Coefficient of Variation）
- **Rating**: 区分度评级
  - ⭐⭐⭐⭐⭐ (CV > 0.15): 优秀区分度
  - ⭐⭐⭐⭐ (CV > 0.10): 良好区分度
  - ⭐⭐⭐ (CV > 0.05): 中等区分度
  - ⭐⭐ (CV ≤ 0.05): 区分度不足

### 关键指标说明

#### 风险加权检测率 (RWDR)
```
RWDR = Σ(weight_i × recall_i) / Σ(weight_i)
```
- 加权平均各攻击类型的召回率
- **推荐**: 选择RWDR最高的模型用于电力巡检

#### 关键任务漏报率 (CMR)
```
CMR = FN_critical / (TP_critical + FN_critical)
```
- 仅统计重放攻击的漏报率
- **目标**: CMR < 0.05 (95%以上检出率)

#### 性能效率比 (PER)
```
PER = AUC / √(model_size_MB × latency_ms)
```
- 综合性能和资源消耗
- **适用**: 资源受限场景

---

## 💡 模型选择建议

根据不同需求选择合适的模型：

### 1. 性能优先 (安全第一)
- **指标**: RWDR (风险加权检测率)
- **策略**: 选择RWDR最高的模型
- **适用**: 对安全性要求极高的场景

### 2. 实时性优先 (快速响应)
- **指标**: MTTD (平均检测延迟)
- **策略**: 选择MTTD最低的模型
- **适用**: 需要快速响应的实时系统

### 3. 资源受限 (边缘部署)
- **指标**: PER (性能效率比)
- **策略**: 选择PER最高的轻量级模型
- **适用**: 计算资源有限的边缘设备

### 4. 平衡选择 (综合考虑)
- **指标**: 平均排名
- **策略**: 综合考虑多个指标的排名
- **适用**: 一般应用场景

---

## 🔬 技术细节

### 混淆矩阵反推

从`precision`和`recall`反推混淆矩阵：

```python
TP = recall × total_positive
FP = TP × (1/precision - 1)
FN = total_positive - TP
TN = total_negative - FP
```

### 检测延迟计算

```python
# 对每个攻击段
attack_predictions = predictions[start_idx:end_idx]
first_detection = argmax(attack_predictions > threshold)
delay_seconds = first_detection × STEP_SIZE × SAMPLING_INTERVAL
```

### 风险加权

```python
# 按攻击严重度加权
RWDR = Σ(attack_weight × recall) / Σ(attack_weight)
```

---

## 📊 可视化图表说明

### 1. 代价权衡对比图
- 展示各模型在不同代价设置下的EOC
- 对比不同β值的F-Score

### 2. 检测延迟分布图
- 箱线图展示延迟分位数
- 各模型MTTD对比

### 3. 风险加权性能雷达图
- 多维度展示各模型性能
- 包括RWDR, 1-CMR, WCP, BCP, AVG_F1

### 4. 边缘部署适配性散点图
- X轴: 模型大小
- Y轴: 平均AUC
- 气泡大小: 参数量
- 颜色: 性能效率比

### 5. 综合排名热力图
- 行: 模型
- 列: 各项指标排名
- 颜色: 排名优劣

### 6. 攻击严重度分析
- 各攻击类型的检测难度
- 按严重度着色

---

## 🔍 常见问题

### Q1: 为什么MTTD显示为inf?
**A**: 某些模型在某些攻击类型上未能检测到任何样本，导致延迟无穷大。这表明该模型在该攻击类型上表现很差。

### Q2: 如何调整代价比例?
**A**: 修改`config_step4.py`中的`COST_FN`参数。建议范围10-100，值越大表示越重视召回率。

### Q3: 哪个指标体系最重要?
**A**: 对于电力巡检场景，推荐优先使用**指标体系4 (风险加权)**，因为它重视重放攻击的检测。

### Q4: 如何验证结果的可靠性?
**A**: 查看`discrimination_analysis.csv`中的区分度评级，CV > 0.10表示指标有良好的区分能力。

### Q5: 输出文件过多怎么办?
**A**: 重点关注以下文件：
- `metrics4_risk_weighted.csv` (核心指标)
- `comprehensive_rankings.csv` (综合排名)
- `POWER_GRID_REPORT.md` (分析报告)

---

## 📚 参考文献

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

## 🤝 贡献

如需添加新的指标体系或修改现有指标，请参考：
1. 在`power_grid_metrics.py`中添加计算函数
2. 在`power_grid_analysis.py`中添加可视化函数
3. 在`config_step4.py`中添加相关配置
4. 更新本README文档

---

## 📝 更新日志

### v1.0 (2026-01-07)
- ✅ 实现5套指标体系计算
- ✅ 生成综合排名和区分度分析
- ✅ 完成6类可视化图表
- ✅ 生成电力巡检场景分析报告

---

## 📧 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**⭐ 推荐工作流程**:
1. 运行`power_grid_metrics.py`计算所有指标
2. 运行`power_grid_analysis.py`生成分析和可视化
3. 查看`POWER_GRID_REPORT.md`了解整体分析
4. 查看`comprehensive_rankings.csv`选择合适模型
5. 根据具体约束条件参考不同指标体系的结果

**🎯 关键产出**: 明确的模型选择建议和支撑数据
