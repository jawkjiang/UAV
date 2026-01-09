# 电力系统无人机巡检场景 - 指标体系分析报告

## 📋 概述

- **分析日期**: 2026-01-07 22:16:03
- **实验数量**: 56
- **模型数量**: 7 (CNN, LSTM, BiLSTM, GRU, CNN-LSTM, TCN, Transformer)
- **攻击类型**: 8

---

## 📊 5套指标体系对比

### 1. 代价权衡指标
**Cost-Sensitive Safety-Efficiency Trade-off**

关注误报和漏报的代价差异，适用于不对称风险场景

### 2. 时效性指标
**Real-time Detection Capability**

关注检测延迟和响应速度，适用于实时性要求高的场景

### 3. 可靠性指标
**Operational Reliability**

关注系统稳定性和误报控制，适用于长期运行场景

### 4. 风险加权指标
**Risk-Based Mission Assurance**

关注高危攻击的检测能力，适用于关键任务保障

### 5. 边缘部署指标
**Edge Deployment Suitability**

关注资源消耗和性能平衡，适用于资源受限环境

---

## 🎯 关键发现

### 指标体系1: 代价权衡

- **最佳模型**: cnn_lstm (EOC = 4.2444)
- **最差模型**: transformer (EOC = 17.3461)
- **区分度**: CV = 0.6459

### 指标体系2: 时效性

- **最快检测**: cnn_lstm (MTTD = 0.21s)
- **平均检测延迟**: 0.36s

### 指标体系4: 风险加权

- **最佳模型**: cnn_lstm (RWDR = 0.8401)
- **关键任务漏报率**: 详见CSV文件

### 指标体系5: 边缘部署

- **最佳性能效率比**: tcn (PER = 1.7378)
- **最轻量模型**: tcn (0.29 MB)

---

## 💡 电力巡检场景建议

### 推荐指标体系

1. **首选**: 指标体系4 (风险加权) - 重视重放攻击检测
2. **次选**: 指标体系2 (时效性) - 确保快速响应
3. **参考**: 指标体系1 (代价权衡) - 平衡误报和漏报

### 模型选择建议

根据不同约束条件的推荐：

- **性能优先**: 选择RWDR最高的模型
- **实时性优先**: 选择MTTD最低的模型
- **资源受限**: 选择PER最高的轻量级模型
- **平衡选择**: 综合考虑多个指标的排名

---

## 📈 详细数据

所有详细指标数据已保存在以下CSV文件中：

- `metrics1_cost_sensitive.csv` - 代价权衡指标
- `metrics2_detection_delay.csv` - 时效性指标
- `metrics3_reliability.csv` - 可靠性指标
- `metrics3_robustness.csv` - 鲁棒性系数
- `metrics4_risk_weighted.csv` - 风险加权指标
- `metrics5_edge_deployment.csv` - 边缘部署指标
- `comprehensive_rankings.csv` - 综合排名

可视化图表保存在 `visualizations/` 目录下。
