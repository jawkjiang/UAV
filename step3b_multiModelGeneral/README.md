# Step 3b: Multi-Model Training with Mixed Attack Types

这是对step3的改进版本，采用**混合攻击训练策略**来提高模型的泛化能力。

## 核心改进

### Step3 的问题
- **训练策略**: 每个模型针对单一攻击类型训练
- **实验规模**: 8种攻击 × 7种模型 = 56个独立模型
- **泛化问题**: 每个模型只见过一种攻击，泛化能力差
- **部署困难**: 实际应用中攻击类型未知，难以选择合适的模型

### Step3b 的方案
- **训练策略**: 每个模型在混合多种攻击的数据集上训练
- **实验规模**: 7种模型（每个见过所有攻击类型）
- **泛化能力**: 强！每个模型都能处理多种攻击
- **部署简单**: 选择综合性能最好的模型即可

---

## 实验设计

### 攻击类型（6种）

排除了replay攻击（因为在step3中所有模型表现都很差）

1. **step** - 阶跃攻击
2. **drift_ramp** - 斜坡漂移
3. **drift_sigmoid** - 平滑漂移
4. **delay** - 延迟攻击
5. **takeover_step** - 阶跃接管
6. **takeover_ramp** - 斜坡接管

### 模型架构（7种）

1. **CNN** - 1D卷积神经网络
2. **LSTM** - 长短期记忆网络
3. **BiLSTM** - 双向LSTM
4. **GRU** - 门控循环单元
5. **CNN-LSTM** - 混合架构
6. **TCN** - 时序卷积网络
7. **Transformer** - 基于注意力机制

### 混合注入策略

**航班级混合**：每个航班随机分配1种攻击类型

```
训练集 (70%航班):
  总攻击率: 50%
  ├── 正常航班: 35个 (50%)
  └── 攻击航班: 35个 (50%)
      ├── step: ~6个航班
      ├── drift_ramp: ~6个航班
      ├── drift_sigmoid: ~6个航班
      ├── delay: ~6个航班
      ├── takeover_step: ~6个航班
      └── takeover_ramp: ~5个航班

验证集 (15%航班):
  总攻击率: 10%
  └── 混合6种攻击

测试集 (15%航班):
  总攻击率: 10%
  └── 混合6种攻击
```

---

## 文件结构

```
step3b_multiModelGeneral/
├── config.py                      # 配置文件
├── mixed_attack_injector.py       # 混合攻击注入器
├── multi_attack_injector.py       # 基础攻击注入器（从step3复制）
├── main.py                        # 主训练脚本
├── compare_models.py              # 结果分析和可视化
├── model.py                       # 模型定义（从step3复制）
├── data_loader.py                 # 数据加载（从step3复制）
├── labeling.py                    # 标签生成（从step3复制）
├── feature_engineering.py         # 特征工程（从step3复制）
├── window_creation.py             # 窗口创建（从step3复制）
├── training.py                    # 训练逻辑（从step3复制）
├── evaluation.py                  # 评估逻辑（从step3复制）
├── README.md                      # 本文档
└── output/
    ├── cnn/
    │   ├── best_model.pth
    │   ├── test_metrics.json             # 整体性能
    │   ├── per_attack_metrics.json       # 各攻击类型性能
    │   ├── training_history.json
    │   ├── normalization_stats.json
    │   ├── train_attack_info.csv
    │   ├── val_attack_info.csv
    │   └── test_attack_info.csv
    ├── lstm/
    ├── bilstm/
    ├── gru/
    ├── cnn_lstm/
    ├── tcn/
    ├── transformer/
    ├── flight_splits.json
    ├── overall_comparison.csv            # 模型整体对比
    ├── per_attack_auc.csv                # 各攻击AUC矩阵
    ├── per_attack_f1.csv                 # 各攻击F1矩阵
    ├── generalization_analysis.csv       # 泛化能力分析
    ├── ANALYSIS_REPORT.md                # 详细分析报告
    └── visualizations/
        ├── heatmap_auc_roc.png           # 模型×攻击热力图（AUC）
        ├── heatmap_f1_score.png          # 模型×攻击热力图（F1）
        ├── generalization_boxplot.png    # 泛化能力箱线图
        ├── radar_charts.png              # 各模型6维雷达图
        └── model_ranking.png             # 模型排名
```

---

## 使用方法

### 1. 训练所有模型

```bash
cd step3b_multiModelGeneral
python main.py
```

**预计时间**: 6-10小时（取决于GPU性能）

**输出**: 
- 每个模型的训练进度
- 每个模型的整体性能和各攻击类型性能
- 所有结果保存到 `output/` 目录

### 2. 分析结果

```bash
python compare_models.py
```

**生成内容**:
- 对比表格（CSV格式）
- 可视化图表（PNG格式）
- 详细分析报告（ANALYSIS_REPORT.md）

### 3. 查看结果

```bash
# 查看整体对比
cat output/overall_comparison.csv

# 查看详细报告
cat output/ANALYSIS_REPORT.md

# 查看可视化
# Windows资源管理器打开: output/visualizations/
```

---

## 评估维度

### 1. 整体性能
- **AUC-ROC**: 整体分类能力
- **F1 Score**: 精确率和召回率的平衡
- **Precision**: 检测准确率
- **Recall**: 攻击召回率

### 2. 按攻击类型分析
每个模型对6种攻击类型分别评估：
- 哪个模型对特定攻击最有效？
- 哪种攻击最难检测？

### 3. 泛化能力分析
- **Mean AUC**: 平均性能
- **Std AUC**: 性能方差（越小越好，说明泛化能力强）
- **Min/Max AUC**: 最差和最佳攻击性能

---

## 关键配置

### 数据划分
```python
TRAIN_RATIO = 0.70    # 70%航班训练
VAL_RATIO = 0.15      # 15%航班验证
TEST_RATIO = 0.15     # 15%航班测试
```

### 攻击率
```python
TRAIN_ATTACK_RATIO = 0.50   # 训练集50%航班攻击
VAL_ATTACK_RATIO = 0.10     # 验证集10%航班攻击
TEST_ATTACK_RATIO = 0.10    # 测试集10%航班攻击
```

### 窗口参数
```python
WINDOW_SIZE = 50      # 每个窗口50个采样点
STEP_SIZE = 5         # 滑动步长5
```

### 训练参数
```python
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
```

---

## 预期优势

### 1. 更强的泛化能力
每个模型见过所有6种攻击类型，能够识别未知攻击模式的特征。

### 2. 更合理的模型选择
基于综合性能（所有攻击类型的平均）选择最佳模型，而不是针对单一攻击。

### 3. 更少的模型数量
只需维护7个模型权重，而不是56个。

### 4. 更真实的应用场景
实际部署中攻击类型未知，需要通用检测器。

### 5. 更可解释的指标体系
可以分析：
- 哪个模型泛化能力最强？
- 哪种攻击最难检测？
- 不同模型的优劣势在哪里？

---

## 与Step3的对比

| 维度 | Step3 | Step3b |
|-----|-------|--------|
| 训练策略 | 单一攻击 | 混合攻击 |
| 模型数量 | 56个 | 7个 |
| 泛化能力 | 弱 | 强 |
| 部署复杂度 | 高 | 低 |
| 可解释性 | 难 | 易 |
| 实际应用 | 不现实 | 可行 |

---

## 技术细节

### 混合注入实现

`MixedAttackInjector` 继承自 `MultiAttackInjector`，核心方法：

```python
def inject_mixed_attacks_to_dataset(self, df, attack_ratio, attack_types):
    """
    1. 随机选择 attack_ratio × 总航班数 个航班
    2. 为每个被攻击航班均匀分配攻击类型
    3. 对每个航班注入对应的攻击
    4. 返回混合攻击后的数据集
    """
```

### 按攻击类型评估

测试阶段，除了整体评估外，还针对每种攻击类型单独评估：

```python
for attack_type in ATTACK_TYPES:
    # 筛选该攻击类型的测试样本
    attack_samples = filter_by_attack_type(test_set, attack_type)
    # 单独计算指标
    metrics = evaluate(model, attack_samples)
    per_attack_metrics[attack_type] = metrics
```

---

## 故障排除

### GPU内存不足
```python
# 在 config.py 中降低批次大小
BATCH_SIZE = 32  # 或更小
```

### 训练时间过长
```python
# 在 config.py 中减少最大轮数
MAX_EPOCHS = 50

# 或者启用跳过已训练模型
SKIP_EXISTING = True
```

### 某个攻击类型样本不足
检查 `output/{model}/test_attack_info.csv`，确认各攻击类型的分布。

---

## 下一步工作

1. **超参数优化**: 尝试不同的学习率、批次大小、窗口大小
2. **集成学习**: 结合多个模型的预测结果
3. **特征选择**: 分析哪些特征对检测最重要
4. **实时检测**: 将最佳模型部署到实时检测系统

---

## 参考

- Step3 Multi-Model Multi-Attack 实验
- 混合训练策略文献
- GPS欺骗攻击检测综述

---

**作者**: UAV GPS Spoofing Detection Project  
**日期**: 2026-01-07  
**版本**: 1.0
