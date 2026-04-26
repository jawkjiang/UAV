# Step5 TimeGAN 数据泄漏修复方案

## 执行日期
2026-01-09

## 问题总结

### 当前问题
1. **数据泄漏**：TimeGAN在全部209个flights上训练，包括了40个后来用作测试的flights (15.9%泄漏)
2. **测试无效**：所有模型97.9%窗口预测一致，DR@Δt完全相同
3. **流程错误**：TimeGAN训练 → 数据划分（应该反过来）

### 根本原因
```python
# 错误流程
prepare_normal_data.py → 提取全部209个flights
train_timegan.py → TimeGAN学习全部209个flights ❌
merge_and_split.py → 然后才划分train/val/test
```

## 修复方案（务实版）

### 核心思路
1. **先划分，再TimeGAN** - 彻底反转流程顺序
2. **独立TimeGAN** - 每个split训练独立的TimeGAN模型
3. **分层扩充** - Train扩充最多，Test扩充保守
4. **平衡注入** - 确保测试集既有充分攻击又有充分正常

### 数据流程图

```
原始flights.csv (209 flights)
          ↓
    【第一步：先划分】
          ↓
    ┌─────┴─────┬─────────┐
    ↓           ↓         ↓
Train: 115   Val: 30   Test: 64
    ↓           ↓         ↓
【第二步：独立TimeGAN训练】
    ↓           ↓         ↓
TimeGAN-1   TimeGAN-2  TimeGAN-3
(只见115)   (只见30)   (只见64)
    ↓           ↓         ↓
【第三步：分别扩充】
    ↓           ↓         ↓
+1000       +120       +200
synthetic   synthetic  synthetic
    ↓           ↓         ↓
= 1115      = 150      = 264
    ↓           ↓         ↓
【第四步：分层注入攻击】
    ↓           ↓         ↓
89攻击      22攻击     53攻击
(8%)        (15%)      (20%)
            ↓          211正常
    ↓           ↓         ↓
最终数据集 ───────────────┘
    ↓
【第五步：训练和评估】
```

## 详细实施计划

### 阶段1：新建文件（5个核心脚本）

#### 文件1：`split_original_first.py`
**功能**：第一步，从原始数据中划分train/val/test
```python
输入：../data/src/flights.csv
输出：
  - output/split_flights/train_flights.txt (115个flight IDs)
  - output/split_flights/val_flights.txt (30个flight IDs)
  - output/split_flights/test_flights.txt (64个flight IDs)
  - output/split_flights/train_normal.csv (完整数据)
  - output/split_flights/val_normal.csv
  - output/split_flights/test_normal.csv
  - output/split_info.json (记录划分详情)
```

**关键特性**：
- 使用固定随机种子确保可重复
- 保存flight ID列表便于追踪
- 验证无重叠

#### 文件2：`train_independent_timegans.py`
**功能**：第二步，为每个split独立训练TimeGAN
```python
输入：
  - output/split_flights/train_normal.csv
  - output/split_flights/val_normal.csv  
  - output/split_flights/test_normal.csv
  
输出：
  - timegan_models/timegan_train.pt
  - timegan_models/timegan_val.pt
  - timegan_models/timegan_test.pt
  - timegan_models/norm_params_train.npz
  - timegan_models/norm_params_val.npz
  - timegan_models/norm_params_test.npz
```

**关键特性**：
- 3个完全独立的TimeGAN模型
- 每个只在各自数据上训练
- 保存各自的归一化参数

#### 文件3：`generate_per_split.py`
**功能**：第三步，为每个split生成合成数据
```python
输入：
  - timegan_models/timegan_train.pt
  - timegan_models/timegan_val.pt
  - timegan_models/timegan_test.pt
  
配置：
  - train扩充比例: 9倍 (生成1000个)
  - val扩充比例: 4倍 (生成120个)
  - test扩充比例: 3倍 (生成200个)
  
输出：
  - output/augmented/train_augmented.csv (115原始 + 1000合成)
  - output/augmented/val_augmented.csv (30原始 + 120合成)
  - output/augmented/test_augmented.csv (64原始 + 200合成)
```

**关键特性**：
- 合成flight ID使用新范围（避免冲突）
- Train: 1000-1999, Val: 2000-2119, Test: 2200-2399
- 保持原始和合成的可追溯性

#### 文件4：`inject_attacks_stratified.py`
**功能**：第四步，分层注入攻击
```python
输入：
  - output/augmented/train_augmented.csv
  - output/augmented/val_augmented.csv
  - output/augmented/test_augmented.csv
  
配置：
  train_attack_ratio: 0.08 (8%)
  val_attack_ratio: 0.15 (15%)
  test_attack_ratio: 0.20 (20%)
  
输出：
  - output/train_with_attacks.csv
  - output/val_with_attacks.csv
  - output/test_with_attacks.csv
  - output/train_attack_info.csv
  - output/val_attack_info.csv
  - output/test_attack_info.csv
  - output/attack_injection_summary.json
```

**关键特性**：
- 每种攻击类型均匀分布
- Test确保每种攻击8-9个实例
- 记录详细的注入信息

#### 文件5：`run_fixed_pipeline.py`
**功能**：完整流程控制脚本
```python
功能：
  - 按顺序执行上述4个步骤
  - 验证每步输出
  - 检查数据泄漏
  - 生成完整报告
  
输出：
  - output/pipeline_execution_report.json
  - output/data_leakage_check.txt
```

### 阶段2：修改配置文件

#### 修改：`config.py`
```python
添加：
  # 数据划分比例（先划分策略）
  SPLIT_RATIOS = {
      'train': 0.55,  # 115 flights
      'val': 0.14,    # 30 flights
      'test': 0.31    # 64 flights
  }
  
  # TimeGAN扩充倍数
  AUGMENTATION_MULTIPLIERS = {
      'train': 9,   # 生成1000个
      'val': 4,     # 生成120个
      'test': 3     # 生成200个
  }
  
  # 分层攻击注入比例
  ATTACK_INJECTION_RATIOS = {
      'train': 0.08,   # 8%
      'val': 0.15,     # 15%
      'test': 0.20     # 20%
  }
  
  # 测试集攻击分布（确保充分）
  TEST_ATTACKS_PER_TYPE = 8  # 每种攻击至少8个
  
  # 合成flight ID范围（避免冲突）
  SYNTHETIC_FLIGHT_ID_RANGES = {
      'train': (1000, 1999),
      'val': (2000, 2199),
      'test': (2200, 2399)
  }
```

### 阶段3：验证和测试

#### 新文件：`verify_fix.py`
```python
功能：
  1. 验证数据划分无重叠
  2. 验证TimeGAN训练数据独立
  3. 验证测试集攻击覆盖
  4. 验证MTBFA测试充分性
  
检查项：
  ✓ Train ∩ Test = ∅
  ✓ Train-TimeGAN只见过train flights
  ✓ Test-TimeGAN只见过test flights
  ✓ Test每种攻击 >= 8个实例
  ✓ Test正常flights >= 200个
  ✓ 合成flight ID无冲突
```

## 执行计划

### 阶段A：准备（备份当前数据）
```bash
# 1. 备份当前output
mv output output_backup_20260109

# 2. 创建新output目录
mkdir output
mkdir output/split_flights
mkdir output/augmented
```

### 阶段B：创建新脚本（5个文件）
```
1. split_original_first.py         ✓ 创建
2. train_independent_timegans.py   ✓ 创建
3. generate_per_split.py           ✓ 创建
4. inject_attacks_stratified.py    ✓ 创建
5. run_fixed_pipeline.py           ✓ 创建
6. verify_fix.py                   ✓ 创建
```

### 阶段C：修改配置
```
1. config.py                       ✓ 添加新配置
```

### 阶段D：执行修复
```bash
# 运行完整流程
python run_fixed_pipeline.py

预期输出：
  [1/4] Splitting original data... ✓
  [2/4] Training independent TimeGANs... ✓
  [3/4] Generating synthetic flights... ✓
  [4/4] Injecting attacks... ✓
  
  Verification:
    ✓ No data leakage detected
    ✓ Test coverage sufficient
    ✓ MTBFA testable
    
  Results saved to output/
```

### 阶段E：验证修复
```bash
# 运行验证
python verify_fix.py

预期输出：
  [Checking data splits]
    ✓ Train: 115 flights
    ✓ Val: 30 flights
    ✓ Test: 64 flights
    ✓ No overlap between splits
  
  [Checking TimeGAN independence]
    ✓ Train-TimeGAN: trained on 115 train flights only
    ✓ Val-TimeGAN: trained on 30 val flights only
    ✓ Test-TimeGAN: trained on 64 test flights only
    ✓ No cross-contamination
  
  [Checking test coverage]
    ✓ Test total: 264 flights
    ✓ Test attacks: 53 flights (20%)
    ✓ Test normal: 211 flights (80%)
    ✓ Attacks per type: 8-9 instances each
    ✓ MTBFA estimation feasible
  
  [Checking synthetic IDs]
    ✓ No ID conflicts
    ✓ Train synthetic: 1000-1999
    ✓ Val synthetic: 2000-2199
    ✓ Test synthetic: 2200-2399
  
  ✅ All checks passed! Data leakage fixed.
```

### 阶段F：重新训练模型
```bash
# 在新数据上训练
python main.py --all

预期变化：
  - 模型DR@Δt不再完全相同
  - 出现架构差异
  - 整体性能可能略降（更真实）
```

## 关键改进点

### 1. 流程顺序反转
```
之前：提取数据 → TimeGAN → 划分 ❌
现在：划分 → 独立TimeGAN → 扩充 ✓
```

### 2. 独立TimeGAN
```
之前：1个TimeGAN见全部数据 ❌
现在：3个TimeGAN各自独立 ✓
```

### 3. 测试集设计
```
之前：12个攻击 + 240正常 (MTBFA太高)
现在：53个攻击 + 211正常 (平衡) ✓
```

### 4. 可追溯性
```
之前：难以追踪flight来源
现在：
  - 原始flights: 保留原始ID
  - 合成flights: 分段ID范围
  - 每个文件记录来源
```

## 预期效果

### 修复后的数据分布
```
Train: 1115 flights
  - 原始: 115 (10.3%)
  - 合成: 1000 (89.7%)
  - 攻击: 89 (8.0%)
  - 正常: 1026 (92.0%)

Val: 150 flights
  - 原始: 30 (20.0%)
  - 合成: 120 (80.0%)
  - 攻击: 22 (14.7%)
  - 正常: 128 (85.3%)

Test: 264 flights ← 关键改进
  - 原始: 64 (24.2%)
  - 合成: 200 (75.8%)
  - 攻击: 53 (20.1%) ← 每种8-9个
  - 正常: 211 (79.9%) ← MTBFA可测
```

### 预期性能变化
```
当前（有泄漏）：
  所有模型DR@1s = 83.3%
  所有模型DR@2s = 91.7%
  所有模型DR@5s = 100.0%
  MTBFA差异小

修复后（无泄漏）：
  模型DR@1s = 70-80% (有差异)
  模型DR@2s = 80-90% (体现架构优势)
  模型DR@5s = 90-98% (不再100%)
  MTBFA差异显著
  
  示例：
  TCN:        DR@1s=78%, MTBFA=1.5h
  LSTM:       DR@1s=74%, MTBFA=1.2h
  Transformer:DR@1s=72%, MTBFA=0.8h
```

## 风险和缓解

### 风险1：TimeGAN训练时间
- **风险**：需要训练3个TimeGAN，时间增加
- **缓解**：可以并行训练，或减少epochs

### 风险2：Test集性能下降
- **风险**：修复后性能可能降低10-15%
- **缓解**：这是正常的，反映真实泛化能力

### 风险3：实验时间
- **风险**：完整流程可能需要数小时
- **缓解**：提供中间检查点，可以分步执行

## 文档和透明度

### 生成的文档
```
output/
  ├── SPLIT_DETAILS.md          # 数据划分详情
  ├── TIMEGAN_TRAINING.md       # TimeGAN训练记录
  ├── AUGMENTATION_REPORT.md    # 数据扩充报告
  ├── ATTACK_INJECTION.md       # 攻击注入详情
  └── LEAKAGE_CHECK.md          # 数据泄漏检查
```

### 论文使用
```
Methodology部分：
  "We employ a stratified augmentation approach to address 
   data scarcity while maintaining test set independence..."
  
Limitation部分：
  "While test set augmentation may introduce bias, we mitigate
   this through independent TimeGAN training..."
```

## 确认清单

请确认以下要点后，我将开始实施：

- [ ] 同意先划分再TimeGAN的流程
- [ ] 同意训练3个独立TimeGAN
- [ ] 同意测试集扩充到264 flights (20%攻击 + 80%正常)
- [ ] 同意备份当前output到output_backup_20260109
- [ ] 同意完整重新生成数据集
- [ ] 理解性能可能下降但更真实
- [ ] 同意在论文limitation中说明

## 下一步

确认后，我将：
1. 创建6个新Python文件
2. 修改config.py
3. 执行run_fixed_pipeline.py
4. 运行verify_fix.py验证
5. 生成完整报告

预计总时间：1-2小时（取决于TimeGAN训练）

---

**请确认是否开始实施？**
