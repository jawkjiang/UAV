# 论文可视化生成说明

## ✅ 已实现的核心图表（可直接运行）

### 图表列表

#### **必选图表（5个）**
- ✅ **Fig 2**: DR@Δt曲线对比图 - `fig2_dr_dt_curves.svg`
- ✅ **Fig 3**: 传统 vs 时间感知指标对比 - `fig3_traditional_vs_timeaware.svg`
- ✅ **Fig 7**: 多模型性能雷达图 - `fig7_radar_chart.svg`
- ✅ **Fig 8**: ADD-MTBFA散点图 - `fig8_add_mtbfa_scatter.svg`

#### **强烈推荐图表（3个）**
- ✅ **Fig 9**: 场景适应性热力图 - `fig9_scenario_heatmap.svg`
- ✅ **Fig 10**: 按攻击类型分组柱状图 - `fig10_per_attack_performance.svg`
- ✅ **Fig 13**: 复杂度vs性能权衡 - `fig13_complexity_performance.svg`

#### **核心表格（4个）**
- ✅ **Table 1**: 指标定义对比 - `table1_metric_definitions.csv`
- ✅ **Table 2**: 数据集统计 - `table2_dataset_statistics.csv`
- ✅ **Table 4**: 整体性能对比（核心） - `table4_overall_performance.csv`
- ✅ **Table 5**: 场景适应性评估 - `table5_scenario_suitability.csv`

---

## 📝 使用方法

### 1. 运行脚本
```bash
cd c:\Users\Jawk\PycharmProjects\UAV
python paper_visualizations.py
```

### 2. 输出位置
```
UAV/
├── paper_figures/
│   ├── svg/                      # SVG格式（论文投稿用，300 DPI）
│   │   ├── fig2_dr_dt_curves.svg
│   │   ├── fig3_traditional_vs_timeaware.svg
│   │   ├── fig7_radar_chart.svg
│   │   ├── fig8_add_mtbfa_scatter.svg
│   │   ├── fig9_scenario_heatmap.svg
│   │   ├── fig10_per_attack_performance.svg
│   │   └── fig13_complexity_performance.svg
│   │
│   ├── csv/                      # CSV表格
│   │   ├── table1_metric_definitions.csv
│   │   ├── table2_dataset_statistics.csv
│   │   ├── table4_overall_performance.csv
│   │   └── table5_scenario_suitability.csv
│   │
│   └── *.png                     # PNG预览（方便查看）
```

---

## ⚠️ 需要补充数据的图表

### **Fig 1: 时间感知指标概念示意图** 
**需要的数据**:
- 一个真实的攻击检测案例（完整时序数据）
- 包含：真实攻击时间、检测时间点、预测概率序列
- 建议从测试集中选择一个典型攻击实例

**获取方式**:
```python
# 需要在step6中保存完整的预测序列
# 修改 step6/main.py，在评估时保存：
# - 每个窗口的时间戳
# - 每个窗口的预测概率
# - 攻击开始和结束时间

# 示例代码：
predictions_detail = {
    'timestamps': timestamps,
    'y_true': y_true,
    'y_pred_prob': y_pred_prob,  # 概率值
    'flight_ids': flight_ids,
    'attack_info': attack_segments_info
}
np.savez('output/predictions_detail.npz', **predictions_detail)
```

---

### **Fig 4: TimeGAN架构与数据流程图**
**需要的数据**:
- 这是架构图，不需要实验数据
- 可以手绘或使用绘图工具（如draw.io、PowerPoint）
- 或者使用matplotlib手动绘制流程框图

**建议**: 
- 使用专业绘图工具更美观
- 如需matplotlib实现，我可以提供代码模板

---

### **Fig 6: GPS攻击类型示例轨迹**
**需要的数据**:
- 6种攻击的原始轨迹数据（真实GPS坐标）
- 正常轨迹 vs 攻击后轨迹的对比

**获取方式**:
```python
# 从step5的数据中提取
# 需要运行：
import pandas as pd

test_df = pd.read_csv('step5_timegan/output/test_with_attacks.csv')
attack_info = pd.read_csv('step5_timegan/output/test_attack_info.csv')

# 对每种攻击类型，选择一个代表性flight
for attack_type in ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']:
    # 找到该攻击类型的一个flight
    attack_flight = attack_info[attack_info['attack_type'] == attack_type].iloc[0]
    flight_id = attack_flight['flight']
    
    # 提取轨迹数据
    flight_data = test_df[test_df['flight'] == flight_id]
    
    # 保存 position_x, position_y, position_z, time
    # 以及 attack_start_time, attack_end_time
```

我可以帮你写一个数据提取脚本。

---

### **Fig 11: 检测延迟分布箱线图**
**需要的数据**:
- 每个攻击实例的检测延迟（不仅是平均值）
- 当前只有overall_metrics.csv中的ADD平均值

**获取方式**:
```python
# step6中已经有 detailed_delays.csv
# 检查该文件是否包含每个攻击的延迟值
df = pd.read_csv('step6_newMetrcisWithTimegan/output/time_aware_metrics/detailed_delays.csv')
print(df.head())

# 如果有，可以直接使用
# 如果没有，需要在step6/evaluation.py中保存详细延迟
```

我可以检查一下这个文件。

---

### **Fig 12: 时序检测可视化（真实案例）**
**需要的数据**: 
- 与Fig 1类似，但需要7个模型的预测对比
- 一个完整的flight的时序数据 + 所有模型的预测

**获取方式**:
同Fig 1，需要保存详细的预测序列。

---

### **Fig 15: TimeGAN数据扩充效果对比**
**需要的数据**:
- Baseline（无TimeGAN）的实验结果
- 不同扩充倍数的对比实验

**状态**: 
- 这是消融实验，如果没有运行baseline实验，可以跳过
- 或者从step3b的结果中获取baseline数据

---

## 🔧 补充数据脚本（我可以帮你写）

我可以创建以下辅助脚本：

### 1. **提取攻击轨迹数据**
```python
# extract_attack_trajectories.py
# 从test_with_attacks.csv中提取6种攻击的代表性轨迹
```

### 2. **保存详细预测序列**
```python
# save_detailed_predictions.py
# 修改step6/main.py，保存完整的预测序列用于时序可视化
```

### 3. **检测延迟详细分析**
```python
# analyze_delay_distribution.py
# 从detailed_delays.csv生成箱线图数据
```

---

## 📊 图表质量说明

### SVG格式优势
- ✅ **矢量格式**: 无限放大不失真
- ✅ **高DPI**: 300 DPI，符合多数期刊要求
- ✅ **可编辑**: 可在Illustrator/Inkscape中进一步编辑
- ✅ **小文件**: 比PNG更小

### 样式特点
- ✅ **学术风格**: 使用提供的sci_*.mplstyle样式
- ✅ **色盲友好**: 7种模型使用不同颜色+标记
- ✅ **黑白打印**: 即使打印成黑白也能区分
- ✅ **字体**: 支持中文（如果需要）

---

## 🎯 优先级建议

### **论文必须包含的图表** (优先运行)
1. **Fig 2** - DR@Δt曲线（核心方法论）
2. **Fig 7** - 雷达图（多维对比）
3. **Fig 8** - ADD-MTBFA散点图（Trade-off）
4. **Table 4** - 整体性能表（数据支撑）

### **强烈建议包含**
5. **Fig 3** - 传统vs时间感知对比（证明必要性）
6. **Fig 9** - 场景适应性（实际价值）

### **可选补充**
7. Fig 10, Fig 13（深入分析）
8. Fig 1, Fig 6（如有时间补充数据）

---

## 💡 下一步行动

### Option 1: 立即运行现有图表
```bash
python paper_visualizations.py
```
查看已生成的7个图表和4个表格，评估是否满足论文需求。

### Option 2: 补充数据后完善
1. 我帮你写数据提取脚本
2. 运行脚本获取缺失数据
3. 添加Fig 1, Fig 6, Fig 11, Fig 12
4. 完整的图表集

### Option 3: 分批进行
1. 先运行现有图表，写论文初稿
2. 在审稿/修改阶段补充更多图表

---

## 📧 需要我帮助的部分

请告诉我：

1. **是否立即运行脚本**？
   - 我可以帮你在PowerShell中执行

2. **是否需要补充数据脚本**？
   - 攻击轨迹提取
   - 详细预测序列保存
   - 延迟分布分析

3. **是否需要调整图表样式**？
   - 颜色、字体、尺寸
   - 子图布局

4. **是否需要额外的图表**？
   - 混淆矩阵（Fig 14）
   - 其他自定义可视化

---

## 🚀 快速开始（推荐）

```bash
# 步骤1: 运行可视化脚本
cd c:\Users\Jawk\PycharmProjects\UAV
python paper_visualizations.py

# 步骤2: 查看生成的图表
# 打开 paper_figures/ 文件夹

# 步骤3: 根据需要补充数据
# 我会提供具体的数据提取脚本
```
