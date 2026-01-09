# Step6: TimeGAN模型 + 时间感知评估

## 📋 概述

Step6整合了Step5的TimeGAN数据增强模型和Step4b的时间感知评估方法，对所有训练好的模型进行全面的性能评估。

### 核心特点

- ✅ **加载Step5模型权重**：7个模型（CNN, LSTM, BiLSTM, GRU, CNN-LSTM, TCN, Transformer）
- ✅ **使用Step5数据集**：低攻击比例测试集（5%）
- ✅ **时间感知指标**：DR@Δt, ADD, MTBFA
- ✅ **场景分析**：Safety-Critical, Monitoring, Balanced
- ✅ **完整对比**：多模型性能对比和可视化

## 🎯 评估指标

### 时间感知指标

1. **DR@Δt (Detection Rate within Δt)**
   - 在Δt秒内检测到攻击的比例
   - 评估点：Δt = {1, 2, 5, 10, 15, 30}秒

2. **ADD (Average Detection Delay)**
   - 从攻击开始到首次检测的平均延迟
   - 越低越好（更快响应）

3. **MTBFA (Mean Time Between False Alarms)**
   - 假警报事件之间的平均时间
   - 越高越好（更少干扰）

### 场景分析

评估模型在三种部署场景下的适应性：

1. **Safety-Critical（安全关键）**
   - DR@5s ≥ 0.95
   - ADD ≤ 3s
   - MTBFA ≥ 5h

2. **Long-term Monitoring（长期监控）**
   - DR@10s ≥ 0.90
   - MTBFA ≥ 24h
   - ADD ≤ 10s

3. **Balanced（均衡）**
   - DR@5s ≥ 0.90
   - MTBFA ≥ 12h
   - ADD ≤ 5s

## 🚀 快速开始

### 前置条件

确保Step5已完成训练，存在以下模型权重：

```
step5_timegan/output/
├── best_model_cnn.pth
├── best_model_lstm.pth
├── best_model_bilstm.pth
├── best_model_gru.pth
├── best_model_cnn_lstm.pth
├── best_model_tcn.pth
├── best_model_transformer.pth
├── test_with_attacks.csv
├── test_attack_info.csv
└── normalization_stats.json
```

### 运行评估

```bash
cd step6_newMetrcisWithTimegan

# 运行完整评估（使用缓存）
python main.py

# 强制重新生成预测（不使用缓存）
python main.py --no-cache
```

## 📊 输出文件

### 指标文件（CSV）

```
output/time_aware_metrics/
├── overall_metrics.csv              # 所有模型的整体指标
├── detailed_delays.csv              # 每个攻击的详细延迟信息
├── cnn_per_attack_metrics.csv       # CNN各攻击类型性能
├── lstm_per_attack_metrics.csv      # LSTM各攻击类型性能
├── bilstm_per_attack_metrics.csv    # BiLSTM各攻击类型性能
├── gru_per_attack_metrics.csv       # GRU各攻击类型性能
├── cnn_lstm_per_attack_metrics.csv  # CNN-LSTM各攻击类型性能
├── tcn_per_attack_metrics.csv       # TCN各攻击类型性能
└── transformer_per_attack_metrics.csv  # Transformer各攻击类型性能
```

### 可视化图表（PNG）

```
output/visualizations/
├── dr_vs_delay.png                  # DR@5s vs ADD散点图
├── dr_vs_mtbfa.png                  # DR@5s vs MTBFA散点图
├── delay_distribution.png           # 检测延迟分布直方图
├── per_attack_heatmap.png           # 各攻击类型性能热图
└── mtbfa_comparison.png             # MTBFA对比柱状图
```

### 报告文件

```
output/time_aware_metrics/
├── scenario_analysis.txt            # 场景分析报告
└── evaluation_summary.md            # 评估总结（Markdown）
```

## 📐 架构说明

### 数据流

```
Step5数据 → 数据加载器 → 特征工程 → 窗口创建 → 模型推理 → 时间感知评估 → 可视化报告
```

### 模块说明

- **config.py**：配置文件（路径、模型、指标参数）
- **data_loader.py**：加载Step5数据和模型，生成预测
- **evaluation.py**：评估模块（调用时间感知指标）
- **time_aware_metrics.py**：时间感知指标计算（复用自Step4b）
- **scenario_analysis.py**：场景分析（复用自Step4b）
- **visualizations.py**：可视化生成（复用自Step4b）
- **main.py**：主流程

### 关键技术点

1. **时间戳重建**：从窗口索引重建真实时间戳
2. **攻击段信息**：利用`test_attack_info.csv`提供精确的`attack_start_time`
3. **评估兼容性**：数据格式完全兼容Step4b的评估流程

## 🔬 与Step3b/Step4b的对比

| 方面 | Step3b | Step4b | Step6 |
|------|--------|--------|-------|
| 数据来源 | 原始数据 | Step3b预测 | Step5数据集 |
| 训练攻击比例 | 40% | - | 15% |
| 测试攻击比例 | 70% | 70% | 5% |
| 数据增强 | 无 | - | TimeGAN扩充5倍 |
| 评估指标 | 传统ML指标 | 时间感知 | 时间感知 |
| 模型数量 | 7 | 7 | 7 |

**Step6优势**：
- ✅ 更真实的测试场景（5%攻击比例）
- ✅ 数据增强降低FPR
- ✅ 时间感知指标更贴近实际部署

## 📈 预期改进

相比Step3b基线，Step6预期改进：

| 指标 | Step3b | Step6预期 | 改进方向 |
|------|--------|-----------|---------|
| FPR | 高 | 低 | ↓ TimeGAN扩充正常数据 |
| MTBFA | 低 | 高 | ↑ 更少假警报 |
| DR@5s | 高 | 中-高 | ≈ 保持检测能力 |
| ADD | 中 | 低 | ↓ 更快检测 |

## 🛠️ 故障排查

### 模型文件未找到

```
FileNotFoundError: Model not found: ...
```

**解决方案**：确保Step5训练已完成，检查`step5_timegan/output/`目录

### 内存不足

**解决方案**：减少批次大小，在`data_loader.py`中修改`batch_size=128`为更小值

### CUDA错误

**解决方案**：修改为CPU模式，在`data_loader.py`中强制设置`device='cpu'`

## 📝 自定义配置

在`config.py`中可调整：

```python
# 评估的时间阈值
DELTA_T_VALUES = [1, 2, 5, 10, 15, 30]

# 假警报间隔阈值
FALSE_ALARM_GAP_THRESHOLD = 1.0  # seconds

# 待评估的模型
MODEL_NAMES = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
```

## 📚 参考文档

- [Step5 TimeGAN README](../step5_timegan/README.md)
- [Step4b Time-Aware Metrics](../step4b_newMetrics/README.md)
- [Step3b Multi-Model Training](../step3b_multiModelGeneral/README.md)

## 💡 后续工作

- [ ] 与Step3b基线详细对比
- [ ] 不同TimeGAN扩充倍数的影响
- [ ] 攻击比例变化实验
- [ ] 在线检测性能分析
- [ ] 部署优化建议

---

**作者**: UAV GPS Spoofing Detection Project  
**日期**: 2026-01  
**版本**: 1.0
