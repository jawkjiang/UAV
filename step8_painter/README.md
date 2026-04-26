# Step8 Painter - 论文图表生成器

## 📋 概述

Step8 Painter 是一个完整的论文图表生成系统，用于自动生成所有论文所需的图表和表格。

## 🎯 功能特性

- ✅ **模块化设计**：每个章节独立，易于维护
- ✅ **一键生成**：支持生成所有图表或指定章节
- ✅ **多格式输出**：PNG + SVG（可选PDF）
- ✅ **LaTeX表格**：自动生成LaTeX格式表格
- ✅ **统一风格**：论文级别的绘图风格
- ✅ **数据验证**：自动检查数据完整性

## 📂 目录结构

```
step8_painter/
├── config.py                    # 全局配置
├── data_loader.py               # 数据加载器
├── main.py                      # 主入口
│
├── section_b_overall.py         # ✅ B节：整体性能对比
├── section_c_detection_speed.py # 🚧 C节：检测速度分析
├── section_d_false_alarms.py    # 🚧 D节：误报频率分析
├── section_e_tradeoff.py        # 🚧 E节：权衡分析
├── section_f_per_attack.py      # 🚧 F节：分攻击类型分析
├── section_i_efficiency.py      # 🚧 I节：计算效率分析
│
├── utils/
│   ├── plot_styles.py           # 绘图风格
│   ├── table_generator.py       # 表格生成
│   └── statistics.py            # 统计分析
│
└── output/
    ├── figures/                 # 图表输出
    │   ├── section_b/
    │   ├── section_c/
    │   ├── section_d/
    │   ├── section_e/
    │   ├── section_f/
    │   └── section_i/
    ├── tables/                  # 表格输出
    └── summary/                 # 汇总报告
```

## 🚀 快速开始

### 1. 验证数据

```bash
python main.py --validate
```

### 2. 生成所有图表

```bash
python main.py --all
```

### 3. 生成指定章节

```bash
# 只生成Section B
python main.py --section B

# 生成多个章节
python main.py --section B C D
```

### 4. 详细输出

```bash
python main.py --all --verbose
```

## 📊 已实现的图表

### Section B: Overall Performance ✅

#### 表格
- ✅ Table 1: 传统指标对比（Precision/Recall/F1）
- ✅ Table 2: 时间感知指标对比（DR@5s/ADD/MTBFA）
- ✅ Table 3: Precision vs MTBFA排名对比

#### 图表
- ✅ Figure B1: 传统指标对比柱状图
- ✅ Figure B2: 时间感知指标对比柱状图
- ✅ Figure B3: Precision vs MTBFA散点图（展示悖论）
- ✅ Figure B4: FP窗口数 vs FP事件数对比

### Section C: Detection Speed Analysis 🚧

- 🚧 Figure C1: DR@Δt曲线
- 🚧 Figure C2: 关键时间阈值对比
- 🚧 Figure C3: 检测速度分布

### Section D: False Alarm Frequency Analysis 🚧

- 🚧 Table 4: MTBFA详细分析
- 🚧 Figure D1: 误报事件持续时长分布
- 🚧 Figure D2: 误报持续时长箱线图
- 🚧 Figure D3: 误报持续时长CDF
- 🚧 Figure D4: 100小时运行场景

### Section E: Trade-off Analysis 🚧

- 🚧 Figure E1: Pareto前沿
- 🚧 Figure E2: ADD vs DR@5s相关性
- 🚧 Figure E3: 3D权衡分析
- 🚧 Figure E4: 场景化模型推荐

### Section F: Per-Attack-Type Performance 🚧

- 🚧 Table 5: 分攻击类型性能
- 🚧 Figure F1: 分攻击类型热力图（DR@5s）
- 🚧 Figure F2: 分攻击类型热力图（ADD）
- 🚧 Figure F3: 攻击检测难度对比

### Section I: Computational Efficiency Analysis 🚧

- 🚧 Table 6: 计算效率对比
- 🚧 Figure I1: 推理延迟对比
- 🚧 Figure I2: 模型复杂度对比
- 🚧 Figure I3: 性能-效率权衡
- 🚧 Figure I4: 综合评分雷达图

## 🎨 配置说明

### 颜色方案

在 `config.py` 中定义了统一的颜色方案：

```python
MODEL_COLORS = {
    'cnn': '#1f77b4',        # 蓝色
    'lstm': '#ff7f0e',       # 橙色
    'bilstm': '#2ca02c',     # 绿色
    'gru': '#d62728',        # 红色
    'cnn_lstm': '#9467bd',   # 紫色
    'tcn': '#8c564b',        # 棕色
    'transformer': '#e377c2' # 粉色
}
```

### 图表尺寸

```python
FIGURE_SIZES = {
    'single': (8, 6),
    'double': (12, 5),
    'triple': (15, 5),
    'grid_2x2': (12, 10),
    'grid_3x3': (15, 12)
}
```

### 输出格式

```python
OUTPUT_FORMATS = ['png', 'svg']  # 默认输出PNG和SVG
DPI = 300  # 高分辨率
```

## 📝 数据来源

| 数据类型 | 来源 |
|---------|------|
| 传统指标 | `step5_timegan/output/test_metrics_*.json` |
| 时间感知指标 | `step6_newMetrcisWithTimegan/output/time_aware_metrics/` |
| FP分析数据 | `paper_analysis/output/fp_duration/` |
| 计算效率 | `step7_performance_analysis/output/` |

## 🔧 开发指南

### 添加新图表

1. 在对应的 `section_*.py` 文件中添加新方法
2. 在 `generate_all()` 方法中调用
3. 使用统一的保存函数：

```python
from utils.plot_styles import save_figure

fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])
# ... 绘图代码 ...
save_figure(fig, 'fig_name', self.output_dir)
plt.close()
```

### 添加新表格

```python
from utils.table_generator import TableGenerator

table_gen = TableGenerator()
table_gen.save_table(df, 'table_name',
                    caption='Table Caption',
                    label='table_label')
```

## 📊 输出示例

### 图表文件

```
output/figures/section_b/
├── fig_b1_traditional_metrics_comparison.png
├── fig_b1_traditional_metrics_comparison.svg
├── fig_b2_time_aware_metrics_comparison.png
├── fig_b2_time_aware_metrics_comparison.svg
├── fig_b3_precision_mtbfa_paradox.png
├── fig_b3_precision_mtbfa_paradox.svg
├── fig_b4_fp_aggregation_analysis.png
└── fig_b4_fp_aggregation_analysis.svg
```

### 表格文件

```
output/tables/
├── table1_traditional_metrics.csv
├── table1_traditional_metrics.tex
├── table2_time_aware_metrics.csv
├── table2_time_aware_metrics.tex
├── table3_precision_mtbfa_comparison.csv
└── table3_precision_mtbfa_comparison.tex
```

## ✅ 测试

### 测试单个章节

```bash
python section_b_overall.py
```

### 测试数据加载

```bash
python data_loader.py
```

## 🐛 故障排除

### 数据缺失

如果出现数据缺失错误：

1. 确保已运行 `step5_timegan/main.py`
2. 确保已运行 `step6_newMetrcisWithTimegan/main.py`
3. 确保已运行 `step7_performance_analysis/main.py`
4. 确保已运行 `paper_analysis/analyze_fp_duration.py`

### 字体问题

如果出现字体警告，可以在 `config.py` 中修改：

```python
FONT_FAMILY = 'Arial'  # 或其他可用字体
```

## 📈 进度追踪

- [x] 基础设施（config, data_loader, utils）
- [x] Section B: Overall Performance
- [ ] Section C: Detection Speed Analysis
- [ ] Section D: False Alarm Frequency Analysis
- [ ] Section E: Trade-off Analysis
- [ ] Section F: Per-Attack-Type Performance
- [ ] Section I: Computational Efficiency Analysis
- [ ] 汇总报告和打包

## 🎓 论文投稿建议

### 主图（Main Figures）

建议选择以下图表作为论文主图：

1. **Figure B3**: Precision vs MTBFA散点图（核心贡献）
2. **Figure C1**: DR@Δt曲线（时间感知指标）
3. **Figure D2**: FP持续时长箱线图（误报特性）
4. **Figure E1**: Pareto前沿（权衡分析）

### 补充材料（Supplementary）

- 所有表格
- 其他图表
- 详细统计数据

## 📞 联系

如有问题，请查看 `IMPLEMENTATION_PLAN.md` 获取更多详细信息。

---

**Status**: 🚧 In Progress (Section B Complete, Others Pending)
