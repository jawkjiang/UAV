# 修复数据泄漏的推荐方案

## 核心原则

1. **训练集和测试集flights完全分离**
2. **TimeGAN只在训练flights上训练**
3. **测试集足够大，能充分评估所有攻击类型**

## 推荐方案：不等比例策略性注入

### 数据集划分（第一步，最重要）

```
原始数据: 209 flights

划分策略:
├─ Train: 125 flights (60%)
│  └─ 用于TimeGAN训练和模型训练
│
├─ Val: 24 flights (11%)
│  └─ 用于模型超参数调优
│
└─ Test: 60 flights (29%)  ← 加大测试集
   └─ 专门用于充分评估攻击检测能力
```

**为什么加大测试集？**
- 需要注入60个攻击（6种×10个）
- 60个测试flights刚好可以每个注入1个攻击
- 或者部分flights注入攻击，部分保持正常（测试误报）

### TimeGAN扩充（第二步）

```
只在训练集:
├─ 输入: 125个原始train flights
├─ TimeGAN学习: 仅基于这125个flights
└─ 生成: ~1000个合成flights

最终训练集:
└─ 125原始 + 1000合成 = 1125 flights
```

**关键：TimeGAN完全不接触测试集的60个flights**

### 攻击注入策略（第三步）

#### 训练集注入（模拟真实低比例）
```python
train_attack_config = {
    'total_flights': 1125,
    'attack_ratio': 0.08,  # 8%
    'attack_count': 90,     # 90个攻击flights
    
    # 随机分布6种攻击
    'attack_distribution': {
        'step': 15,
        'ramp': 15,
        'takeover_step': 15,
        'takeover_ramp': 15,
        'delay': 15,
        'drift': 15
    }
}
```

#### 测试集注入（策略性充分评估）
```python
test_attack_config = {
    'total_flights': 60,
    
    # 方式1: 均匀注入（推荐）
    'attack_flights': 48,  # 80%注入攻击
    'normal_flights': 12,  # 20%保持正常（测试误报）
    
    # 每种攻击充分覆盖
    'attack_distribution': {
        'step': 8,
        'ramp': 8,
        'takeover_step': 8,
        'takeover_ramp': 8,
        'delay': 8,
        'drift': 8
    },
    
    # 每种攻击测试多种参数
    'parameter_variation': True,
    'instances_per_attack': 8  # 每种8个实例，不同参数
}
```

**或者方式2：专用攻击测试flights**
```python
# 从60个测试flights中
test_normal_flights = 12   # 测试误报率
test_attack_flights = 48   # 测试检测率

# 每种攻击8个独立flights
# 确保充分评估每种攻击的DR、ADD
```

### 数据流程图

```
原始209 flights
       │
       ├──────────────────────────────┐
       │                              │
   划分(第一步)                        │
       │                              │
       ▼                              ▼
 Train:125  Val:24  Test:60     保留完整性
       │      │       │
       │      │       └─────► 不参与TimeGAN
       │      │
       ▼      │
  TimeGAN训练  │
       │      │
       ▼      │
  生成1000    │
       │      │
       ▼      ▼
 合并: 1125   24
       │      │
       ▼      ▼
  注入攻击(8%) 注入攻击(少量)
  90个攻击     2-3个攻击
       │      │
       ▼      ▼
  最终训练集  最终验证集
       │      │
       └──────┴──────► 训练7个模型
                           │
                           ▼
                      测试集评估
                      (60 flights)
                      ├─ 48攻击flights
                      │  (每种攻击8个)
                      └─ 12正常flights
```

## 优势分析

### ✅ 解决数据泄漏
- TimeGAN只在125个train flights上训练
- 测试集60个flights完全独立
- 无任何信息泄漏

### ✅ 测试集充分
- 每种攻击8个实例 → 可靠的DR/ADD统计
- 12个正常flights → 可靠的MTBFA估计
- 总共60个flights → 统计显著性足够

### ✅ 训练集真实
- 8%攻击比例 → 模拟真实低比例场景
- TimeGAN扩充 → 解决正常样本不足问题
- 1125个flights → 足够训练深度模型

### ✅ 可发表
- 方法论正确，无泄漏
- 测试充分，每种攻击都有评估
- 训练/测试分离清晰

## 预期结果变化

### 修复前（有泄漏）
```
所有模型DR@1s = 83.3% (10/12)
所有模型DR@2s = 91.7% (11/12)
所有模型DR@5s = 100% (12/12)
97.9%窗口预测一致
```

### 修复后（无泄漏）
```
可能的结果：
- 模型DR会有差异（不再完全相同）
- 整体性能可能下降（更真实）
- 不同攻击类型的DR会有差异
- 体现模型架构的真实优势

示例：
LSTM    DR@1s=75% DR@2s=85% DR@5s=95%
CNN     DR@1s=70% DR@2s=82% DR@5s=93%
TCN     DR@1s=78% DR@2s=88% DR@5s=96%
...
```

## 实施步骤

### 第一阶段：数据重组
```bash
# 1. 先划分原始flights
python split_original_data.py
  输出: train_flights.txt (125个)
       val_flights.txt (24个)
       test_flights.txt (60个)

# 2. 提取训练flights数据
python extract_train_data.py
  输入: train_flights.txt
  输出: train_normal_flights.csv

# 3. 在训练数据上TimeGAN
python train_timegan_on_train_only.py
  输入: train_normal_flights.csv
  输出: timegan_model.pt
       synthetic_train_flights.csv (1000个)
```

### 第二阶段：攻击注入
```bash
# 4. 训练集注入（低比例）
python inject_attacks_train.py
  输入: train (125原始 + 1000合成)
  输出: train_with_attacks.csv (90个攻击)

# 5. 测试集注入（策略性）
python inject_attacks_test_strategic.py
  输入: test_flights.txt (60个)
  配置: 每种攻击8个实例
  输出: test_with_attacks.csv (48个攻击 + 12正常)
```

### 第三阶段：训练和评估
```bash
# 6. 训练模型
python train_all_models.py

# 7. 评估
python evaluate_with_time_aware_metrics.py
```

## 关键检查点

### 验证无泄漏
```python
# 运行检查
python verify_no_leakage.py

预期输出:
✓ TimeGAN训练flights ∩ 测试flights = ∅
✓ 训练集flights ∩ 测试flights = ∅
✓ 合成flights编号 ∩ 原始flights编号 = ∅
```

### 验证测试充分性
```python
# 检查测试集
python check_test_coverage.py

预期输出:
✓ 每种攻击类型 >= 8个实例
✓ 正常flights >= 10个
✓ 攻击参数变化充分
```

## 对比实验（可选）

修复后可以做对比：
1. **有TimeGAN vs 无TimeGAN**
   - 证明TimeGAN的价值
   
2. **不同扩充比例**
   - 1:1, 1:5, 1:10的对比
   
3. **不同攻击注入比例**
   - 5%, 10%, 15%的影响

## 总结

这个方案：
- ✅ **彻底解决数据泄漏**
- ✅ **测试集足够充分**（每种攻击8个）
- ✅ **训练集真实低比例**（8%攻击）
- ✅ **TimeGAN有效扩充**（1000合成flights）
- ✅ **方法论正确可发表**

关键是**调整测试集大小**从20%提升到29%，确保能容纳60个攻击实例。
