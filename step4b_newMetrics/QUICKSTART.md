# 快速开始 - 5分钟修复指南

## 你的问题
- 检测延迟 0.002s（不合理）
- DR@Δt 不随时间变化（异常）

## 根本原因
1. ❌ 使用概率值而非二值化预测
2. ❌ 时间戳错误（0, 0.05, 0.1... vs 真实95.8s, 165.5s...）
3. ❌ 缺失元数据

## 立即修复（2个选择）

### 选择A：完整修复（推荐，需30分钟）

```bash
# 1. 重新运行step3b生成新数据
cd step3b_multiModelGeneral
python main.py  # 已修改，会生成window_metadata.csv

# 2. 运行新评估
cd ../step4b_newMetrics
python main_v2.py

# 3. 查看结果
cat output/time_aware_metrics/evaluation_summary_v2.md
```

**预期结果**：ADD = 1-5s，DR递增

### 选择B：快速修复（5分钟，次优）

修改 `step4b_newMetrics/data_loader_simple.py` 第38行：

```python
# 原来
y_pred = data['predictions']  # ❌

# 改为
y_prob = data['probabilities']
y_pred = (y_prob > 0.5).astype(int)  # ✅
```

然后：
```bash
cd step4b_newMetrics
python main.py  # 使用原main.py
```

## 已准备的文件

✅ `COMPLETE_CONTEXT_SUMMARY.md` - 完整context（**先看这个**）
✅ `SOLUTION_COMPLETE.md` - 详细技术方案
✅ `IMPLEMENTATION_GUIDE.md` - 分步指南
✅ `main_v2.py` - 新评估程序
✅ `data_loader_with_metadata.py` - 新数据加载器
✅ step3b的main.py已修改 - 生成window_metadata.csv

## 验证成功

```python
import pandas as pd
df = pd.read_csv('output/time_aware_metrics/overall_metrics_v2.csv')
print(df[['model', 'dr@1s', 'dr@5s', 'dr@30s', 'add']])

# 应该看到：
# - ADD: 0.5-10秒（不是0.002）
# - dr@1s < dr@5s < dr@30s（递增）
```

## 下一步

**现在就做**：
1. 阅读 `COMPLETE_CONTEXT_SUMMARY.md`（5分钟）
2. 选择方案A或B
3. 执行并验证

**需要帮助**：
- 详细步骤 → `IMPLEMENTATION_GUIDE.md`
- 技术细节 → `SOLUTION_COMPLETE.md`
- 诊断问题 → 运行 `diagnose_issue.py`

## 关键修复记住3点
1. ✅ `y_pred = (probabilities > 0.5).astype(int)`
2. ✅ 使用 `attack_start_time`（真实）不是窗口时间
3. ✅ 加载 `window_metadata.csv`
