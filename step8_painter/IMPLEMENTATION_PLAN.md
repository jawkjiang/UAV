# Step8: Paper Figures Generation Plan

## 📋 方案概述

根据论文大纲，将所有图表生成代码整合到 `step8_painter/` 目录，按章节组织。

---

## 🎯 论文大纲对应的图表需求

### B. Overall Performance: Traditional vs. Time-Aware Metrics

#### 表格
- **Table 1**: 传统指标对比（Precision/Recall/F1）
- **Table 2**: 时间感知指标对比（DR@5s/ADD/MTBFA）
- **Table 3**: Precision vs MTBFA排名对比

#### 图表
- **Figure B1**: 传统指标对比柱状图（Precision/Recall/F1）
- **Figure B2**: 时间感知指标对比柱状图（DR@5s/ADD/MTBFA）
- **Figure B3**: Precision vs MTBFA散点图（展示悖论）
- **Figure B4**: FP窗口数 vs FP事件数对比（聚合率可视化）

---

### C. Detection Speed Analysis: DR@Δt Curves

#### 图表
- **Figure C1**: DR@Δt曲线（所有模型，Δt ∈ {1s, 2s, 5s, 10s, 15s, 30s}）
- **Figure C2**: DR@1s vs DR@5s vs DR@10s对比柱状图
- **Figure C3**: 关键时间阈值分析（1s/5s/10s的检出率对比）

---

### D. False Alarm Frequency Analysis: MTBFA Deep Dive

#### 表格
- **Table 4**: MTBFA详细分析（Precision/FP窗口/FP事件/MTBFA）

#### 图表
- **Figure D1**: 误报事件持续时长分布（直方图，7个子图）
- **Figure D2**: 误报持续时长箱线图对比
- **Figure D3**: 误报持续时长累积分布函数（CDF）
- **Figure D4**: 100小时运行场景误报次数预测

---

### E. Trade-off Analysis: Detection Speed vs. False Alarm Rate

#### 图表
- **Figure E1**: Pareto前沿（DR@5s vs MTBFA）
- **Figure E2**: ADD vs DR@5s散点图（展示相关性）
- **Figure E3**: 三维权衡分析（DR@5s, MTBFA, ADD）
- **Figure E4**: 场景化模型推荐雷达图

---

### F. Per-Attack-Type Performance

#### 表格
- **Table 5**: 分攻击类型性能（DR@5s和ADD）

#### 图表
- **Figure F1**: 分攻击类型热力图（DR@5s）
- **Figure F2**: 分攻击类型热力图（ADD）
- **Figure F3**: 攻击类型检测难度对比（柱状图）

---

### I. Computational Efficiency Analysis

#### 表格
- **Table 6**: 计算效率对比（延迟/内存/FLOPs）

#### 图表
- **Figure I1**: 推理延迟对比（CPU vs CUDA）
- **Figure I2**: 模型复杂度对比（参数量 vs 内存占用）
- **Figure I3**: 性能-效率权衡（DR@5s vs 推理延迟）
- **Figure I4**: 综合评分雷达图

---

## 📂 目录结构设计

```
step8_painter/
├── config.py                          # 全局配置（颜色、字体、路径）
├── data_loader.py                     # 数据加载器（统一接口）
├── main.py                            # 主入口（生成所有图表）
│
├── section_b_overall.py               # B节：整体性能对比
├── section_c_detection_speed.py       # C节：检测速度分析
├── section_d_false_alarms.py          # D节：误报频率分析
├── section_e_tradeoff.py              # E节：权衡分析
├── section_f_per_attack.py            # F节：分攻击类型分析
├── section_i_efficiency.py            # I节：计算效率分析
│
├── utils/
│   ├── __init__.py
│   ├── plot_styles.py                 # 统一绘图风格
│   ├── table_generator.py             # 表格生成工具
│   └── statistics.py                  # 统计分析工具
│
└── output/
    ├── tables/                        # CSV和LaTeX表格
    │   ├── table1_traditional_metrics.csv
    │   ├── table2_time_aware_metrics.csv
    │   └── ...
    ├── figures/                       # 图表（PNG + SVG）
    │   ├── section_b/
    │   │   ├── fig_b1_traditional_metrics.png
    │   │   ├── fig_b1_traditional_metrics.svg
    │   │   └── ...
    │   ├── section_c/
    │   ├── section_d/
    │   ├── section_e/
    │   ├── section_f/
    │   └── section_i/
    └── summary/
        ├── all_figures_index.md       # 所有图表索引
        └── paper_ready_package.zip    # 打包所有论文素材
```

---

## 🎨 设计规范

### 1. 统一配置（config.py）

```python
# 颜色方案
MODEL_COLORS = {
    'cnn': '#1f77b4',
    'lstm': '#ff7f0e',
    'bilstm': '#2ca02c',
    'gru': '#d62728',
    'cnn_lstm': '#9467bd',
    'tcn': '#8c564b',
    'transformer': '#e377c2'
}

# 图表尺寸
FIGURE_SIZES = {
    'single': (8, 6),
    'double': (12, 5),
    'triple': (15, 5),
    'grid': (12, 10)
}

# 字体设置
FONT_SIZES = {
    'title': 14,
    'label': 12,
    'tick': 10,
    'legend': 10
}

# 输出格式
OUTPUT_FORMATS = ['png', 'svg', 'pdf']  # 可选
DPI = 300
```

### 2. 数据加载器（data_loader.py）

```python
class PaperDataLoader:
    """统一的数据加载接口"""
    
    def load_traditional_metrics(self) -> pd.DataFrame:
        """加载传统指标（Precision/Recall/F1）"""
        
    def load_time_aware_metrics(self) -> pd.DataFrame:
        """加载时间感知指标（DR@Δt/ADD/MTBFA）"""
        
    def load_fp_window_event_comparison(self) -> pd.DataFrame:
        """加载FP窗口vs事件对比数据"""
        
    def load_fp_duration_data(self) -> Dict:
        """加载FP持续时长数据"""
        
    def load_per_attack_metrics(self) -> pd.DataFrame:
        """加载分攻击类型指标"""
        
    def load_efficiency_metrics(self) -> pd.DataFrame:
        """加载计算效率指标"""
```

### 3. 绘图风格（utils/plot_styles.py）

```python
def setup_paper_style():
    """设置论文级别的绘图风格"""
    plt.style.use('seaborn-v0_8-paper')
    # 设置字体、线宽、网格等
    
def save_figure(fig, name, output_dir, formats=['png', 'svg']):
    """统一的图表保存函数"""
    
def add_significance_markers(ax, data, pairs):
    """添加统计显著性标记"""
```

---

## 🚀 实施步骤

### Phase 1: 基础设施（30分钟）
1. ✅ 创建目录结构
2. ✅ 实现 `config.py`
3. ✅ 实现 `data_loader.py`
4. ✅ 实现 `utils/plot_styles.py`

### Phase 2: 核心图表（2小时）
5. ✅ 实现 `section_b_overall.py`（4个图表）
6. ✅ 实现 `section_c_detection_speed.py`（3个图表）
7. ✅ 实现 `section_d_false_alarms.py`（4个图表）
8. ✅ 实现 `section_e_tradeoff.py`（4个图表）

### Phase 3: 补充图表（1小时）
9. ✅ 实现 `section_f_per_attack.py`（3个图表）
10. ✅ 实现 `section_i_efficiency.py`（4个图表）

### Phase 4: 表格生成（30分钟）
11. ✅ 实现 `utils/table_generator.py`
12. ✅ 生成所有LaTeX表格

### Phase 5: 整合和打包（30分钟）
13. ✅ 实现 `main.py`（一键生成所有图表）
14. ✅ 生成图表索引和打包

---

## 📊 图表清单（共22个图表 + 6个表格）

### 图表（Figures）
- Section B: 4个
- Section C: 3个
- Section D: 4个
- Section E: 4个
- Section F: 3个
- Section I: 4个

### 表格（Tables）
- Section B: 3个
- Section D: 1个
- Section F: 1个
- Section I: 1个

---

## 🎯 关键特性

### 1. 一键生成
```bash
python step8_painter/main.py --all
python step8_painter/main.py --section B  # 只生成B节
python step8_painter/main.py --figure B1  # 只生成特定图表
```

### 2. 格式灵活
```bash
python step8_painter/main.py --format png svg pdf
python step8_painter/main.py --dpi 600  # 高分辨率
```

### 3. 样式定制
```bash
python step8_painter/main.py --style ieee  # IEEE风格
python step8_painter/main.py --style nature  # Nature风格
```

### 4. 自动打包
```bash
python step8_painter/main.py --package  # 生成论文素材包
```

---

## 📝 输出示例

### 图表文件命名规范
```
fig_b1_traditional_metrics_comparison.png
fig_b2_time_aware_metrics_comparison.png
fig_b3_precision_mtbfa_paradox.png
fig_c1_dr_delta_t_curves.png
fig_d1_fp_duration_histograms.png
fig_e1_pareto_frontier.png
...
```

### 表格文件命名规范
```
table1_traditional_metrics.csv
table1_traditional_metrics.tex
table2_time_aware_metrics.csv
table2_time_aware_metrics.tex
...
```

---

## ✅ 验证清单

生成完成后，自动验证：
- [ ] 所有图表文件存在
- [ ] 所有表格文件存在
- [ ] 图表尺寸符合规范
- [ ] 文字清晰可读
- [ ] 颜色一致性
- [ ] 图例完整
- [ ] 坐标轴标签正确

---

## 🔄 与现有代码的关系

### 数据来源
- `step5_timegan/output/` → 传统指标
- `step6_newMetrcisWithTimegan/output/` → 时间感知指标
- `step7_performance_analysis/output/` → 计算效率
- `paper_analysis/output/` → FP分析数据

### 代码复用
- 从 `paper_analysis/analyze_fp_duration.py` 提取FP可视化代码
- 从 `paper_analysis/quick_fp_analysis.py` 提取对比分析代码
- 从 `step6/visualizations.py` 提取DR@Δt曲线代码

---

## 💡 优势

1. **模块化**：每个章节独立，易于维护
2. **可复现**：一键生成所有图表
3. **灵活性**：支持单独生成特定图表
4. **规范化**：统一的风格和命名
5. **可扩展**：易于添加新图表
6. **论文就绪**：直接输出高质量图表

---

## 🎓 论文投稿建议

### 主图（Main Figures）
- Figure B3: Precision vs MTBFA散点图（核心贡献）
- Figure C1: DR@Δt曲线（时间感知指标）
- Figure D2: FP持续时长箱线图（误报特性）
- Figure E1: Pareto前沿（权衡分析）

### 补充材料（Supplementary）
- 所有表格
- 其他图表
- 详细统计数据

---

## 📅 时间估算

- **Phase 1-2**: 2.5小时（核心功能）
- **Phase 3-4**: 1.5小时（补充内容）
- **Phase 5**: 0.5小时（整合打包）
- **总计**: 4.5小时

---

## 🚦 下一步行动

**立即开始？**
1. 创建基础文件结构
2. 实现配置和数据加载器
3. 逐个实现各章节的绘图代码
4. 测试和验证
5. 生成完整图表集

**需要我现在开始实施吗？**
