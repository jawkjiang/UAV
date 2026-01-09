# 解决测试集覆盖不足问题的方案

## 问题诊断

### 原始 main.py 的问题
- **症状**：所有模型指标都接近1（AUC ≈ 1.0），缺乏区分度
- **根本原因**：测试集中缺失某些攻击类型
  - 使用 `inject_mixed_attacks_to_dataset()` 时，如果 `TEST_ATTACK_RATIO < 1.0`
  - 可能导致某些攻击类型完全没有被测试
  - 模型看起来"完美"，但实际上只是测试不充分

### main_strategyC.py 的问题
- **症状**：PR-AUC 降到 ≈ 0.1
- **根本原因**：Flight Reuse 导致严重的数据泄漏
  - 训练集中同一个飞行轨迹重复7次（6种攻击 + 1个正常）
  - 模型学习到的是**飞行本身的特征**，而非**攻击的特征**
  - 测试集使用全新飞行时，模型无法泛化

---

## 解决方案：改进的 main.py（推荐）

### 核心改进

#### 1. 新增方法：`inject_mixed_attacks_with_guarantee()`

**位置**：`mixed_attack_injector.py`

**特性**：
- ✓ 保证所有攻击类型都被覆盖
- ✓ 每种攻击类型至少有 `min_per_attack` 个样本
- ✓ 无需 flight reuse，保持数据独立性
- ✓ 避免数据泄漏

**工作原理**：
```python
# Phase 1: 保证最小覆盖
for each attack_type:
    分配 min_per_attack 个飞行

# Phase 2: 分配剩余攻击
remaining_attacks = n_to_attack - (len(attack_types) × min_per_attack)
均匀分配到各个攻击类型
```

#### 2. 配置调整

**文件**：`config.py`

```python
# 调整攻击比例，确保测试集有足够样本
TRAIN_ATTACK_RATIO = 0.60   # 60%
VAL_ATTACK_RATIO = 0.50     # 50%
TEST_ATTACK_RATIO = 0.70    # 70% (更高，确保覆盖所有攻击)

# 每种攻击类型的最小样本数
MIN_TEST_FLIGHTS_PER_ATTACK = 3  # 至少3个飞行/攻击类型
```

#### 3. main.py 修改

```python
# 测试集使用保证覆盖的方法
test_df, test_attack_info = injector.inject_mixed_attacks_with_guarantee(
    test_df,
    attack_ratio=config.TEST_ATTACK_RATIO,
    attack_types=config.ATTACK_TYPES,
    min_per_attack=config.MIN_TEST_FLIGHTS_PER_ATTACK
)
```

---

## 验证工具

### 使用方法

```bash
python verify_test_coverage.py
```

### 检查内容

1. **攻击类型覆盖**
   - 确认测试集包含所有6种攻击类型
   - 显示每种攻击的样本数和百分比

2. **飞行独立性**
   - 验证 train/val/test 之间没有飞行重叠
   - 防止数据泄漏

3. **对比分析**
   - 比较 main.py vs main_strategyC.py
   - 诊断指标虚高/虚低的原因

---

## 预期结果

### 改进后的 main.py

#### 测试集特征
- ✓ 所有6种攻击类型都存在
- ✓ 每种攻击至少3个样本
- ✓ 无 flight reuse，真实反映泛化能力
- ✓ 无数据泄漏

#### 性能指标
- **更真实的区分度**：不同模型会显示出差异
- **AUC 可能降低**：从 ~1.0 降到 0.85-0.95（正常现象）
- **这是好事**：说明测试更全面，能真正区分模型优劣

### 示例输出

```
[GUARANTEED MIXED ATTACK INJECTION]
Total flights: 40
Target attack ratio: 70.0%
Flights to attack: 28
Attack types: 6
Min per attack: 3

Planned attack distribution (GUARANTEED):
  step                :   5 flights (12.50%)
  drift_ramp          :   5 flights (12.50%)
  drift_sigmoid       :   5 flights (12.50%)
  delay               :   5 flights (12.50%)
  takeover_step       :   4 flights (10.00%)
  takeover_ramp       :   4 flights (10.00%)

✓ All 6 attack types covered!
```

---

## 为什么不推荐 Flight Reuse

### 问题

1. **数据泄漏**
   ```
   Flight 123 → 7个版本 (6攻击 + 1正常)
   所有版本共享相同的基础轨迹
   模型记住轨迹，而非学习攻击特征
   ```

2. **过拟合到飞行ID**
   - 训练集：看过 Flight 1-100 的各种版本
   - 测试集：全新的 Flight 101-120
   - 结果：模型在新飞行上失败（PR-AUC ≈ 0.1）

3. **误导性验证集**
   - Val set 也使用 reuse，看起来效果好
   - 但这是假象，因为基础轨迹相同
   - 真实测试集揭示真相

### 正确的做法

- **训练集**：每个飞行只出现1次，随机分配攻击
- **验证集**：每个飞行只出现1次，随机分配攻击
- **测试集**：使用 `inject_mixed_attacks_with_guarantee()`，确保覆盖所有攻击

---

## 使用指南

### 1. 更新代码
已完成以下修改：
- ✓ `mixed_attack_injector.py`：添加 `inject_mixed_attacks_with_guarantee()`
- ✓ `main.py`：测试集使用新方法
- ✓ `config.py`：调整参数
- ✓ `verify_test_coverage.py`：新增验证工具

### 2. 运行训练
```bash
python main.py
```

### 3. 验证结果
```bash
python verify_test_coverage.py
```

### 4. 期待的变化
- **测试集**：所有6种攻击类型都存在
- **指标**：可能从 ~1.0 降到 0.85-0.95
- **区分度**：不同模型开始显示差异
- **可信度**：结果更可靠，真实反映模型性能

---

## 总结

| 方法 | 测试覆盖 | 数据泄漏 | 指标 | 可信度 |
|------|---------|---------|------|--------|
| **原始 main.py** | ✗ 不完整 | ✓ 无 | ~1.0（虚高） | 低 |
| **main_strategyC** | ✓ 完整 | ✗ 严重 | ~0.1（虚低） | 低 |
| **改进 main.py** | ✓ 完整 | ✓ 无 | 0.85-0.95 | **高** |

**推荐使用改进后的 main.py**，这才是正确的评估方式。
