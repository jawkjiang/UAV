# Step7: 性能分析与部署建议

## 📋 项目概述

本模块对Step5训练的7个模型进行全面的性能分析，包括：
- **推理性能测试**：延迟、吞吐量、实时性能力
- **模型复杂度分析**：参数量、FLOPs、内存占用
- **综合评估**：整合检测性能、推理速度、模型轻量化
- **部署建议**：针对不同场景（边缘设备、边缘服务器、云端）的最优模型推荐

---

## 🏗️ 项目结构

```
step7_performance_analysis/
├── config.py                      # 配置文件
├── inference_profiler.py          # 推理性能测试
├── model_analyzer.py              # 模型复杂度分析
├── comprehensive_evaluation.py    # 综合评估
├── deployment_advisor.py          # 部署建议生成器
├── visualizations.py              # 可视化模块
├── main.py                        # 主流程
├── README.md                      # 本文件
└── output/                        # 输出目录
    ├── performance_metrics.csv
    ├── model_complexity.csv
    ├── comprehensive_report.csv
    ├── deployment_recommendations.txt
    └── visualizations/
        ├── latency_comparison.png
        ├── accuracy_vs_latency.png
        ├── params_vs_performance.png
        ├── radar_chart.png
        └── deployment_matrix.png
```

---

## 📦 依赖项

```bash
pip install torch numpy pandas matplotlib seaborn
pip install thop  # 可选，用于精确FLOPs计算
```

---

## 🚀 使用方法

### 前置条件

1. **Step5完成**：`../step5_timegan/output/` 目录下有7个模型文件：
   - `best_model_cnn.pth`
   - `best_model_lstm.pth`
   - `best_model_bilstm.pth`
   - `best_model_gru.pth`
   - `best_model_cnn_lstm.pth`
   - `best_model_tcn.pth`
   - `best_model_transformer.pth`

2. **Step6完成**：`../step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv` 存在

### 运行

```bash
cd step7_performance_analysis
python main.py
```

---

## 📊 输出说明

### 1. performance_metrics.csv
推理性能指标：
- `model`: 模型名称
- `device`: 测试设备（cpu/cuda）
- `mean_latency_ms`: 平均延迟
- `p95_latency_ms`: P95延迟
- `p99_latency_ms`: P99延迟
- `throughput_windows_per_sec`: 吞吐量（窗口/秒）
- `meets_real_time`: 是否满足实时要求（<100ms）

### 2. model_complexity.csv
模型复杂度指标：
- `total_params`: 总参数量
- `params_millions`: 参数量（百万）
- `model_size_mb`: 模型文件大小（MB）
- `flops`: FLOPs
- `flops_millions`: FLOPs（百万）
- `memory_mb`: 推理内存占用（MB）

### 3. comprehensive_report.csv
综合评估报告（整合所有指标）：
- 检测性能：`dr_1s`, `dr_2s`, `dr_5s`, `add_seconds`, `mtbfa_hours`
- 推理性能：`latency_mean_ms`, `latency_p95_ms`, `throughput_windows_per_sec`
- 模型复杂度：`params_millions`, `model_size_mb`, `memory_mb`
- 综合得分：`composite_score`, `rank`

### 4. deployment_recommendations.txt
部署建议文本报告，包含：
- 3种部署场景的约束条件
- 每个场景推荐的模型列表（按优先级排序）
- 推荐理由

### 5. visualizations/
5张可视化图表：
- **latency_comparison.png**: 延迟对比（均值 vs P95）
- **accuracy_vs_latency.png**: 准确性-延迟权衡（气泡图）
- **params_vs_performance.png**: 参数量 vs 性能（双子图）
- **radar_chart.png**: 多维度对比雷达图（Top 3模型）
- **deployment_matrix.png**: 部署场景适配矩阵（热图）

---

## 🎯 评估维度

### 推理性能
- **单窗口延迟**：测量单个窗口的推理时间（Mean, P50, P95, P99）
- **批量吞吐量**：测量不同批大小的处理能力
- **实时性能力**：判断是否满足实时（<100ms）、准实时（<500ms）要求

### 模型复杂度
- **参数量**：模型总参数数量
- **模型大小**：.pth文件大小
- **FLOPs**：浮点运算次数（使用thop库精确计算）
- **内存占用**：推理时的内存消耗

### 综合得分
$$
\text{Composite Score} = 0.5 \times \text{DR@5s} + 0.3 \times \text{Normalized Latency} + 0.2 \times \text{Normalized Params}
$$

- 50% 检测性能权重
- 30% 推理速度权重
- 20% 模型轻量化权重

---

## 🎨 部署场景

### 1. 边缘设备（嵌入式）
- 最大延迟：100ms
- 最大参数：1.0M
- 最大内存：512MB
- 最低DR@5s：90%

**适用场景**：无人机机载计算单元、IoT设备

### 2. 边缘服务器（小型GPU）
- 最大延迟：50ms
- 最大参数：5.0M
- 最大内存：2048MB
- 最低DR@5s：95%

**适用场景**：地面站服务器、边缘计算节点

### 3. 云端（大型GPU）
- 最大延迟：20ms
- 最大参数：50.0M
- 最大内存：8192MB
- 最低DR@5s：98%

**适用场景**：云端分析平台、离线审计系统

---

## 🔧 配置说明

### 修改测试参数

编辑 [config.py](config.py#L24-L27):
```python
WARMUP_ITERATIONS = 50          # 预热次数
BENCHMARK_ITERATIONS = 500      # 基准测试次数
BATCH_SIZES = [1, 8, 16, 32, 64, 128]  # 测试批大小
```

### 修改综合得分权重

编辑 [comprehensive_evaluation.py](comprehensive_evaluation.py#L82-L86):
```python
df['composite_score'] = (
    0.5 * df['norm_dr_5s'] +        # 检测性能权重
    0.3 * df['norm_latency'] +      # 推理速度权重
    0.2 * df['norm_params']         # 模型轻量权重
)
```

### 添加新的部署场景

编辑 [config.py](config.py#L52-L80):
```python
DEPLOYMENT_SCENARIOS = {
    'your_scenario': {
        'name': '场景名称',
        'constraints': {
            'max_latency_ms': 100,
            'max_params_m': 1.0,
            'max_memory_mb': 512,
            'min_dr_5s': 0.90
        }
    }
}
```

---

## 📈 典型输出示例

### 模型排名（控制台输出）
```
================================================================================
MODEL RANKINGS (by Composite Score)
================================================================================

1. GRU
   Composite Score: 0.892
   DR@5s: 98.1%
   Latency (P95): 12.50ms
   Parameters: 0.85M

2. TRANSFORMER
   Composite Score: 0.854
   DR@5s: 96.2%
   Latency (P95): 15.20ms
   Parameters: 2.31M

3. LSTM
   Composite Score: 0.847
   DR@5s: 98.1%
   Latency (P95): 18.30ms
   Parameters: 1.24M
...
```

### 部署建议（文本报告）
```
================================================================================
Scenario: 边缘设备（嵌入式）
================================================================================

Constraints:
  • Max Latency: 100ms
  • Max Parameters: 1.0M
  • Max Memory: 512MB
  • Min DR@5s: 90.0%

✅ Recommended Models (2 found):

  1. GRU
     Score: 0.892
     DR@5s: 98.1%
     Latency: 12.50ms
     Params: 0.85M
     Reason: 优秀的检测率; 极低延迟; 轻量级模型
...
```

---

## 🛠️ 故障排除

### 问题1：找不到Step6指标文件
```
FileNotFoundError: Step6 metrics not found
```
**解决**：确保Step6已运行完成，检查 `../step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv` 是否存在

### 问题2：找不到模型文件
```
Model not found: ../step5_timegan/output/best_model_xxx.pth
```
**解决**：确保Step5训练完成，检查 `../step5_timegan/output/` 目录下是否有对应的模型文件

### 问题3：CUDA不可用
```
⚠️  CUDA not available, skipping GPU test
```
**说明**：这不是错误，系统会自动使用CPU测试。如需GPU测试，确保：
- 安装CUDA版本的PyTorch
- GPU驱动正确安装

### 问题4：thop库缺失（FLOPs计算）
```
ImportError: No module named 'thop'
```
**解决**：
```bash
pip install thop
```
或者使用粗略估算（代码会自动回退）

---

## 📚 参考资料

- [PyTorch性能测试最佳实践](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [模型压缩与加速综述](https://arxiv.org/abs/1710.09282)
- [边缘计算部署指南](https://edge-ai-vision.github.io/)

---

## 📝 更新日志

### v1.0 (2026-01-09)
- ✅ 初始版本
- ✅ 支持7种模型的推理性能测试
- ✅ 模型复杂度分析（参数、FLOPs、内存）
- ✅ 综合评估与排名
- ✅ 3种部署场景的建议生成
- ✅ 5种可视化图表

---

## 🤝 贡献

如需修改或扩展功能，请参考各模块的文档注释。

---

**作者**: GitHub Copilot  
**日期**: 2026-01-09
