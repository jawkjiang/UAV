# 时间感知评估框架 - 实施指南

## 问题摘要

当前评估存在3个致命错误导致结果不可信：

1. **使用错误的预测数据**：使用了概率值而不是二值化预测
2. **时间戳不准确**：简单的0, 0.05, 0.1...序列无法对应真实攻击时间（95.8s, 165.5s等）
3. **缺失元数据**：test_predictions.npz缺少flight_ids、window_metadata等关键信息

**结果**：检测延迟全部为0.002s，DR@Δt不随Δt变化

## 解决方案概览

已提供完整的修复代码，包括两个版本：

### 版本1：完整方案（推荐）

**修改文件**：
- `step3b_multiModelGeneral/main.py` - 生成完整元数据
- `step4b_newMetrics/data_loader_with_metadata.py` - 新数据加载器
- `step4b_newMetrics/time_aware_metrics.py` - 添加v2函数
- `step4b_newMetrics/main_v2.py` - 新主程序

**优点**：
- 使用真实的attack_start_time计算延迟
- 完整的窗口元数据支持
- 准确的时间感知指标

**缺点**：
- 需要重新运行step3b生成新数据（可能需要几分钟到几小时）

### 版本2：快速修复方案

**仅修改**：
- `step4b_newMetrics/data_loader_simple.py` - 修复probabilities使用

**优点**：
- 不需要重新运行step3b
- 立即可用

**缺点**：
- 时间戳仍然是估算的
- 但至少使用了正确的二值化预测

## 实施步骤

### 方案A：完整实施（推荐）

#### 步骤1：检查修改是否已应用

```bash
# 检查step3b的main.py是否已修改
cd step3b_multiModelGeneral
grep -A 5 "window_metadata" main.py
# 应该看到新增的window_metadata生成代码

# 检查step4b的新文件是否存在
cd ../step4b_newMetrics
ls -la data_loader_with_metadata.py main_v2.py
```

#### 步骤2：重新运行step3b（仅对一个模型测试）

```bash
cd step3b_multiModelGeneral

# 备份现有输出
cp -r output output_backup_$(date +%Y%m%d)

# 选择一个模型重新评估（最快方式）
# 编辑main.py，临时修改MODEL_TYPES只包含一个模型
python main.py

# 检查生成的新文件
ls output/tcn/
# 应该看到：
# - test_predictions.npz (更大了，包含更多元数据)
# - window_metadata.csv (新文件)
# - test_attack_info.csv (已存在)
```

#### 步骤3：运行新的评估

```bash
cd ../step4b_newMetrics

# 测试新的数据加载器
python data_loader_with_metadata.py

# 运行完整评估
python main_v2.py
```

#### 步骤4：对比结果

```bash
# 对比旧结果和新结果
python -c "
import pandas as pd

old = pd.read_csv('output/time_aware_metrics/overall_metrics.csv')
new = pd.read_csv('output/time_aware_metrics/overall_metrics_v2.csv')

print('旧版本结果:')
print(old[['model', 'dr@5s', 'add', 'mtbfa']].head())

print('\\n新版本结果:')
print(new[['model', 'dr@5s', 'add', 'mtbfa']].head())
"
```

**预期差异**：
- 旧版：ADD = 0.002s, DR@1s = DR@30s
- 新版：ADD = 1-5s, DR随Δt递增

### 方案B：快速修复（如果时间紧迫）

#### 修改data_loader_simple.py

```python
# 在load_model_data_simple()函数中
# 第35-42行左右，修改：

# 原来（错误）
if 'labels' in data:
    y_true = data['labels']
    y_pred = data['predictions']  # ❌ 这是概率！
    y_prob = data.get('probabilities', None)

# 修改为（正确）
if 'labels' in data:
    y_true = data['labels']
    y_prob = data['probabilities']  # ✅ 获取概率
    y_pred = (y_prob > 0.5).astype(int)  # ✅ 二值化
```

#### 重新运行评估

```bash
python main.py
```

这至少能修复"使用概率而不是二值化预测"的问题，结果会更接近真实。

## 验证检查清单

### ✅ 修改正确性检查

1. **step3b修改**：
   ```bash
   # 检查npz文件大小是否增加
   ls -lh step3b_multiModelGeneral/output/tcn/test_predictions.npz
   # 新版应该 > 100KB（包含更多元数据）
   
   # 检查是否生成window_metadata.csv
   head step3b_multiModelGeneral/output/tcn/window_metadata.csv
   ```

2. **step4b修改**：
   ```bash
   # 测试数据加载
   cd step4b_newMetrics
   python data_loader_with_metadata.py
   
   # 应该看到：
   # ✓ Loading window_metadata.csv
   # Attack segments with timing: 20 (或其他非0数字)
   ```

### ✅ 结果合理性检查

运行评估后，检查结果：

```python
import pandas as pd
import numpy as np

# 加载结果
overall = pd.read_csv('output/time_aware_metrics/overall_metrics_v2.csv')
delays = pd.read_csv('output/time_aware_metrics/detailed_delays_v2.csv')

# 检查1：ADD应该 > 0.1s（窗口粒度至少0.05s）
print("ADD range:", overall['add'].min(), "-", overall['add'].max())
assert overall['add'].min() > 0.05, "❌ ADD太小！"

# 检查2：DR应该随Δt递增
for model in overall['model'].unique():
    model_data = overall[overall['model'] == model].iloc[0]
    dr_values = [model_data[f'dr@{dt}s'] for dt in [1, 2, 5, 10, 15, 30]]
    assert dr_values == sorted(dr_values), f"❌ {model}: DR未递增！"
    print(f"✓ {model}: DR递增正常")

# 检查3：延迟分布合理
delays_valid = delays[delays['detected'] == True]['delay']
print(f"延迟分布: mean={delays_valid.mean():.3f}s, "
      f"median={delays_valid.median():.3f}s, "
      f"range=[{delays_valid.min():.3f}s, {delays_valid.max():.3f}s]")
```

**合理范围**：
- ADD: 0.5s - 10s
- DR@1s: 10-40%
- DR@30s: 60-95%
- 延迟范围: 0-20s

## 常见问题

### Q1: 重新运行step3b需要多久？

A: 取决于：
- **仅评估**（模型已训练）：5-10分钟
- **重新训练**：1-3小时（7个模型）
- **建议**：先测试1个模型（如TCN）

### Q2: window_metadata.csv不存在怎么办？

A: 两种选择：
1. 重新运行step3b（推荐）
2. 使用data_loader_simple_v2.py的回退逻辑（次优）

### Q3: 结果仍然异常怎么办？

A: 运行诊断：

```bash
cd step4b_newMetrics
python diagnose_issue.py > diagnosis.txt
cat diagnosis.txt
```

检查：
- probabilities是否被正确二值化
- timestamps是否有合理的范围
- attack_segments_info是否包含真实的attack_start_time

### Q4: 可以只修改step4b吗？

A: 可以，使用方案B（快速修复）。至少能修复probabilities使用错误，结果会有改善。

## 文件清单

### 已创建/修改的文件

**step3b_multiModelGeneral/**
- ✅ `main.py` - 修改了np.savez部分，添加window_metadata生成

**step4b_newMetrics/**
- ✅ `SOLUTION_COMPLETE.md` - 完整的问题分析和解决方案文档
- ✅ `diagnose_issue.py` - 诊断脚本
- ✅ `data_loader_with_metadata.py` - 新的数据加载器（完整版）
- ✅ `time_aware_metrics.py` - 添加了v2函数
- ✅ `main_v2.py` - 新的主程序
- ✅ `IMPLEMENTATION_GUIDE.md` - 本文件

### 需要生成的文件（运行step3b后）

**step3b_multiModelGeneral/output/{model}/**
- `test_predictions.npz` - 更新版（包含更多元数据）
- `window_metadata.csv` - 新文件（窗口级元数据）
- `test_attack_info.csv` - 已存在（攻击信息）

## 下一步

1. **立即可做**：
   - 查看 `SOLUTION_COMPLETE.md` 了解完整的技术细节
   - 运行 `diagnose_issue.py` 确认问题
   - 决定使用完整方案还是快速修复

2. **推荐行动**：
   - 使用完整方案（修改step3b + step4b）
   - 先测试1个模型（TCN）验证正确性
   - 确认结果合理后，对所有模型重新评估

3. **验证成功**：
   - ADD应该在0.5-10s范围
   - DR@Δt应该随Δt递增
   - 延迟分布应该有合理的spread

## 联系与支持

如有问题，检查：
1. `SOLUTION_COMPLETE.md` - 完整技术文档
2. `diagnose_issue.py` - 诊断工具
3. `main_v2.py` 的输出日志

关键修复点记住三个：
1. ✅ 使用 `(probabilities > 0.5).astype(int)`
2. ✅ 使用 `attack_start_time` 而不是 `first_window_time`
3. ✅ 保存和加载完整的窗口元数据
