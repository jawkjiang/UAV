# 论文实验补充需求 - 总结报告

## 📊 快速诊断结果

### ✅ 已有的核心数据（可直接使用）

根据对现有数据的分析，您的实验数据**基本完整**，大部分论文章节所需的数据都已经有了！

#### 1. 传统指标 vs 时间感知指标 ✅
- **传统指标**：Precision/Recall/F1（step5数据）
- **时间感知指标**：DR@5s/ADD/MTBFA（step6数据）
- **Precision-MTBFA悖论**：已提取（见下方）

#### 2. DR@Δt曲线数据 ✅
- 9个时间点的完整数据：0.5s, 1s, 1.5s, 2s, 3s, 5s, 10s, 15s, 30s
- 所有7个模型的曲线

#### 3. 分攻击类型性能 ⚠️（数据有但粒度不够细）
- 已有4类合并后的攻击类型数据
- 需要确认是否需要6类细分

#### 4. 计算效率分析 ✅
- 推理延迟、内存占用、FLOPs（step7数据）

---

## 🎯 关键发现：Precision-MTBFA悖论（已确认）

### 数据汇总

| Model | Precision | Rank | MTBFA (h) | Rank | FP Windows | FP Events | 聚合率 |
|-------|-----------|------|-----------|------|------------|-----------|--------|
| **TCN** | 0.9816 | **#1** | 0.29 | **#1** | 51 | 9 | 0.18 |
| **CNN** | 0.9408 | **#2** | 0.06 | **#6** | 186 | 45 | 0.24 |
| CNN_LSTM | 0.9393 | #3 | 0.10 | #5 | 194 | 27 | 0.14 |
| LSTM | 0.9268 | #4 | 0.12 | #3 | 238 | 21 | 0.09 |
| **BiLSTM** | 0.9161 | **#5** | 0.12 | **#2** | 281 | 22 | 0.08 |
| GRU | 0.9149 | #6 | 0.12 | #4 | 286 | 22 | 0.08 |
| **Transformer** | 0.8802 | **#7** | 0.01 | **#7** | 400 | 198 | 0.49 |

### 悖论案例（论文可用）

**案例1：CNN vs BiLSTM**
- **CNN**: Precision=0.9408 (排名#2), MTBFA=0.06h (排名#6)
  - 186个FP窗口 → 45个FP事件（聚合率0.24）
  - **解释**：FP窗口较少但分散，形成更多独立事件
  
- **BiLSTM**: Precision=0.9161 (排名#5), MTBFA=0.12h (排名#2)
  - 281个FP窗口 → 22个FP事件（聚合率0.08）
  - **解释**：FP窗口更多但集中，聚合成更少事件

**案例2：Transformer的极端情况**
- 400个FP窗口 → 198个FP事件（聚合率0.49）
- Precision最低，但FP窗口几乎全部分散
- MTBFA也最低（0.01h = 0.6分钟）

---

## ❌ 确定缺失的实验

### 🔴 必须补充（2项）

#### 1. **误报事件持续时长分布** ⭐⭐⭐
**重要性**：高  
**难度**：中  
**时间**：1-2小时

**缺失内容**：
- 每个FP事件的持续时间（duration）
- 分布统计（<5s, 5-10s, >10s）
- 直方图可视化

**为什么重要**：
- 解释为什么有些模型聚合率高（FP事件持续时间长）
- 评估误报对运行的实际影响

**解决方案**：
```python
# 需要修改 step6_newMetrcisWithTimegan/time_aware_metrics.py
# 在calculate_mtbfa函数中记录FP事件的start_time和end_time
# 输出每个FP事件的duration
```

**实验步骤**：
1. 修改`time_aware_metrics.py`，添加FP持续时长记录
2. 重新运行`step6/main.py`
3. 生成分布统计和可视化

---

#### 2. **核心可视化图表** ⭐⭐⭐
**重要性**：高  
**难度**：低  
**时间**：1-2小时

**缺失内容**（5个核心图表）：
1. **Precision vs MTBFA散点图**（展示悖论）
2. **DR@Δt曲线图**（7个模型对比）
3. **ADD vs DR@5s散点图**（展示相关性）
4. **Pareto前沿图**（DR@5s vs MTBFA权衡）
5. **误报持续时长分布直方图**（需先完成实验1）

**解决方案**：
- 完善`step6/visualizations.py`或新建`paper_analysis/generate_figures.py`
- 数据已有，直接绘图即可

---

### 🟡 建议补充（3项，可选）

#### 3. **细分攻击类型分析**
**重要性**：中  
**难度**：低-中  
**时间**：30分钟-2小时

**现状**：
- 当前有4类：step, delay, takeover, drift
- 大纲要求6类：step, drift_ramp, drift_sigmoid, delay, takeover_step, takeover_ramp

**需要确认**：
1. 数据中是否有细分标注？
2. 论文是否必须要6类细分？

**建议**：
- 检查`step5/output/test_with_attacks.csv`中的attack_type列
- 如果有细分：修改evaluation代码保留原始类型
- 如果没有：用4类也足够，大纲可调整

---

#### 4. **相关性分析**
**重要性**：中  
**难度**：低  
**时间**：30分钟

**缺失内容**：
- Recall vs DR@5s相关系数（大纲提到r=0.742）
- Precision vs MTBFA相关系数

**解决方案**：
```python
from scipy.stats import pearsonr
r, p_value = pearsonr(recall_values, dr5s_values)
print(f"Correlation: r={r:.3f}, p={p_value:.4f}")
```

---

#### 5. **100小时运行场景分析**
**重要性**：低  
**难度**：低  
**时间**：15分钟

**缺失内容**：
- 具体案例：100小时连续运行的预期误报次数

**解决方案**：
```python
scenario_hours = 100
for model, mtbfa in mtbfa_values.items():
    expected_fa = scenario_hours / mtbfa
    print(f"{model}: {expected_fa:.0f} false alarms in {scenario_hours}h")
```

结果预览：
- Transformer: 7692次误报/100h
- CNN: 1667次/100h
- TCN: 345次/100h

---

## 🚀 推荐实验计划

### 方案A：最小可行方案（3小时）

**目标**：获得论文必需的核心数据和图表

**步骤**：
1. ✅ **提取FP窗口数**（已完成，30分钟）
   - 运行`paper_analysis/quick_fp_analysis.py`
   - 已生成Precision-MTBFA对比表

2. 🔲 **计算FP持续时长**（1.5小时）
   - 修改`time_aware_metrics.py`
   - 重新运行`step6/main.py`
   - 生成分布统计

3. 🔲 **生成5个核心图表**（1小时）
   - Precision vs MTBFA散点图
   - DR@Δt曲线图
   - ADD vs DR@5s散点图
   - Pareto前沿图
   - FP持续时长直方图

**输出**：
- ✅ 所有关键表格数据
- ✅ 所有核心可视化
- ✅ 论文B、C、D、E、I章节完整

---

### 方案B：完整方案（5小时）

**额外补充**：
4. 🔲 **细分攻击类型**（1小时）
   - 检查数据
   - 如需要，重新生成per-attack表格

5. 🔲 **相关性分析**（30分钟）
   - 计算Recall-DR@5s相关系数
   - 计算Precision-MTBFA相关系数

6. 🔲 **100小时场景分析**（30分钟）
   - 生成运行负担评估表格

**输出**：
- ✅ 论文所有章节完整
- ✅ 可选分析全部完成

---

## 📝 立即可用的数据

### 已经可以直接写进论文的发现

1. **Precision-MTBFA悖论**：
   - CNN: Precision排名#2，MTBFA排名#6（悖论案例）
   - BiLSTM: Precision排名#5，MTBFA排名#2（悖论案例）
   - 原因：窗口聚合率差异（0.08 vs 0.24）

2. **DR@Δt关键阈值**：
   - DR@1s: Transformer最快（0.282）
   - DR@5s: 大部分模型达到87% (CNN/LSTM/BiLSTM/GRU/CNN-LSTM)
   - DR@15s: 所有模型接近100%

3. **模型推荐**（基于现有数据）：
   - **安全关键**：GRU（DR@5s=0.874, ADD=2.08s）
   - **低误报**：TCN（MTBFA=0.29h，但DR@5s较低0.631）
   - **平衡**：CNN-LSTM（DR@5s=0.874, MTBFA=0.10h）

4. **计算效率**：
   - 最快：GRU（0.56ms latency, 1765 windows/s）
   - 最小：TCN（77K params, 0.31MB）
   - 最慢：Transformer（1.45ms, 只有703 windows/s）

---

## 🎯 总结

### 好消息 ✅
1. **80%的数据已经有了**！
2. **关键发现已经可以写**（Precision-MTBFA悖论）
3. **核心表格数据完整**

### 需要补充 🔲
1. **FP事件持续时长**（唯一缺失的核心数据）
2. **可视化图表**（数据有，需要绘图）
3. **可选分析**（增强说服力，非必需）

### 建议
- **优先执行方案A**（3小时）→ 论文核心内容完整
- **时间充裕执行方案B**（5小时）→ 论文内容完美

---

## 📂 下一步行动

### 立即可做（无需实验）

1. 使用已提取的数据开始写论文B节
2. 绘制Precision vs MTBFA散点图
3. 绘制DR@Δt曲线图
4. 计算相关性系数

### 需要实验（1-2小时）

1. 修改`time_aware_metrics.py`记录FP持续时长
2. 重新运行`step6/main.py`
3. 生成FP持续时长分布图

### 可选补充（1-2小时）

1. 检查攻击类型细分
2. 完善所有可视化
3. 生成100小时场景分析
