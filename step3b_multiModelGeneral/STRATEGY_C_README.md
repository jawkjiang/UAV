# Strategy C: Conservative Hybrid Implementation

## 概述

实现了**Strategy C（Conservative Hybrid）**航班复用策略，解决了原始测试集攻击类型覆盖不全的问题。

## 核心特点

### 数据策略

1. **训练集/验证集**：使用航班复用
   - 每个航班生成7个版本（6种攻击 + 1个正常）
   - 训练集：~102航班 → ~714样本
   - 验证集：~51航班 → ~357样本

2. **测试集**：不复用，分层采样
   - ~56个独立航班
   - 每种攻击类型：8个独立航班
   - 正常航班：8个
   - **关键**：保持航班级独立性，可验证泛化能力

### 优势

✅ **训练数据充足**：714训练样本 vs 原来146航班  
✅ **测试集完整**：所有6种攻击类型都有充分测试  
✅ **评估可信**：测试集航班完全独立于训练集  
✅ **统计显著**：每种攻击8个独立样本

## 使用方法

### 方式1：直接运行（推荐）

```bash
python run_strategyC.py
```

这会：
1. 清理旧的output目录
2. 使用分层采样划分数据
3. 对训练/验证集进行航班复用扩充
4. 训练所有7个模型
5. 生成完整的评估报告

### 方式2：手动运行

```bash
# 删除旧结果
rm -r output

# 运行主程序
python main_strategyC.py
```

## 文件结构

```
step3b_multiModelGeneral/
├── config.py                      # 配置（已更新为Strategy C）
├── stratified_split.py           # 分层采样和航班复用工具
├── flight_reuse_injector.py      # 航班复用注入器
├── main_strategyC.py             # 主训练脚本（Strategy C）
├── run_strategyC.py              # 启动脚本
├── FLIGHT_REUSE_ANALYSIS.md      # 详细技术分析
├── compare_strategies.py         # 策略对比工具
└── output/                        # 输出目录
    ├── flight_splits.json        # 数据划分信息
    ├── overall_summary.csv       # 总体性能摘要
    ├── cnn/                      # 各模型的详细结果
    ├── lstm/
    ├── bilstm/
    ├── gru/
    ├── cnn_lstm/
    ├── tcn/
    └── transformer/
```

## 配置说明

### config.py中的关键参数

```python
# 数据分割比例
TRAIN_RATIO = 0.50  # 50%用于训练
VAL_RATIO = 0.25    # 25%用于验证
TEST_RATIO = 0.25   # 25%用于测试

# 航班复用策略
USE_FLIGHT_REUSE = True              # 启用航班复用
REUSE_TEST_SET = False               # 测试集不复用
MIN_TEST_FLIGHTS_PER_ATTACK = 8     # 每种攻击最少8个测试航班

# 攻击类型
ATTACK_TYPES = [
    'step', 'drift_ramp', 'drift_sigmoid',
    'delay', 'takeover_step', 'takeover_ramp'
]
```

## 输出说明

### 1. flight_splits.json

```json
{
  "train": [1, 3, 4, ...],           // 训练航班ID
  "val": [2, 5, 8, ...],             // 验证航班ID
  "test": [6, 9, 12, ...],           // 测试航班ID
  "test_allocation": {                // 测试集攻击分配
    "step": [6, 9, ...],
    "drift_ramp": [12, 15, ...],
    ...
  },
  "train_samples": 714,               // 复用后的样本数
  "val_samples": 357,
  "strategy": "Conservative Hybrid (C)"
}
```

### 2. per_attack_metrics.json

每个模型目录下都有此文件，包含每种攻击的详细性能：

```json
{
  "step": {
    "auc_roc": 0.9875,
    "f1_score": 0.9654,
    "n_samples": 234,      // 该攻击的样本数
    "n_positive": 123
  },
  "drift_ramp": { ... },
  ...
}
```

**与原版对比**：现在所有6种攻击都有指标，而不是只有3种！

### 3. test_attack_info.csv

```csv
flight,attacked,attack_type,attack_start_time,...
6,True,step,45.2,...
9,True,step,52.1,...
12,True,drift_ramp,38.5,...
...
```

每种攻击至少有8个航班。

## 预期结果

### 数据分布

| 数据集 | 航班数 | 样本数 | 说明 |
|-------|-------|--------|------|
| 训练集 | 102 | ~714 | 每航班×7版本 |
| 验证集 | 51 | ~357 | 每航班×7版本 |
| 测试集 | 56 | ~56 | 不复用 |

### 测试集组成

- step: 8航班
- drift_ramp: 8航班  
- drift_sigmoid: 8航班
- delay: 8航班 ✅（原来0个）
- takeover_step: 8航班 ✅（原来0个）
- takeover_ramp: 8航班 ✅（原来0个）
- none: 8航班

## 评估指标解读

### 整体性能（overall_summary.csv）

- **ROC-AUC**: 整体区分能力
- **F1 Score**: 精确率和召回率的平衡
- **Precision**: 预测为攻击的准确率
- **Recall**: 检测到攻击的比例

### 每种攻击性能（per_attack_metrics.json）

独立评估每种攻击的检测能力，现在可以回答：
- ✅ 模型对delay攻击的检测效果如何？
- ✅ 模型对takeover系列攻击是否鲁棒？
- ✅ 哪种攻击最难检测？

## 与原版对比

| 指标 | 原版 | Strategy C |
|-----|------|-----------|
| 训练样本 | 146航班 | 714样本 |
| 测试航班 | 32航班 | 56航班 |
| 测试覆盖 | 3/6攻击类型 | 6/6攻击类型 ✅ |
| 每种攻击测试样本 | 0-1航班 | 8航班 ✅ |
| 统计显著性 | 不足 | 充分 ✅ |
| 航班独立性 | ✅ | ✅ |

## 注意事项

### ⚠️ 训练集复用的影响

**可能的风险**：
- 模型可能记住特定航班的正常飞行模式
- 需要使用强正则化（dropout=0.5, weight_decay）

**缓解措施**：
- 已在config中设置较高dropout
- 窗口平衡策略避免过度关注特定模式
- 特征工程使用相对特征而非绝对值

### ✅ 测试集完全独立

- 测试集航班与训练集完全不重叠
- 可以验证对"新航班"的泛化能力
- 性能指标可信度高

## 下一步分析

训练完成后：

```bash
# 1. 查看总体摘要
cat output/overall_summary.csv

# 2. 分析详细结果
python compare_models.py

# 3. 与原始结果对比
python compare_with_step3.py

# 4. 检查是否有过拟合
python analyze_results.py
```

## 技术文档

详细的技术分析和其他策略对比，参见：
- [FLIGHT_REUSE_ANALYSIS.md](FLIGHT_REUSE_ANALYSIS.md) - 完整的策略分析
- [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) - 原始结果分析
- [compare_strategies.py](compare_strategies.py) - 策略对比工具

## FAQ

**Q: 为什么训练集可以复用但测试集不能？**

A: 训练集复用是为了增加数据量，帮助模型学习。但测试集必须保持独立性，才能真实评估模型对"新航班"的泛化能力。

**Q: 714个训练样本会不会导致过拟合？**

A: 虽然样本来自102个底层航班，但通过正则化、dropout和窗口平衡可以缓解。最重要的是，测试集完全独立，可以真实反映泛化能力。

**Q: 如何判断结果是否可信？**

A: 检查测试集的per_attack_metrics.json，每种攻击都应该有8个独立航班的结果。如果所有攻击都有充分测试且性能合理（不是全1.0），就是可信的。

**Q: 与step3的结果如何对比？**

A: Step3是每种攻击单独训练56个模型，step3b是混合攻击训练7个模型。Strategy C确保step3b也能充分测试所有攻击类型。
