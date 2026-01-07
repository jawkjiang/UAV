# GPS欺骗检测多模型对比方案

## 任务特点分析

### 数据特征
- **输入**: 滑动窗口 [50 samples × 13 features]
- **采样率**: 10 Hz (每个窗口覆盖5秒)
- **特征类型**: 
  - 时序特征: position, velocity, acceleration (随时间变化)
  - 一致性残差: residual_pv, residual_va (检测运动学异常)
- **任务类型**: 二分类 (正常/攻击)
- **关键挑战**: 
  - 时序依赖性强
  - 需要捕捉短时异常模式 (尖峰、突变)
  - 需要捕捉长时持续异常 (漂移、偏离)

### 当前基线模型
- **1D-CNN**: 卷积提取局部时序特征 + 全局池化
- **性能**: AUC-ROC 0.64-1.00 (取决于攻击类型)

---

## 推荐模型列表

### ✅ **1. LSTM (Long Short-Term Memory) - 强烈推荐**

**架构概述:**
```python
Input: [batch, 50, 13]
  ↓
LSTM layers (hidden_size=128, num_layers=2-3)
  ↓
最后时刻hidden state [batch, 128]
  ↓
FC layers: 128 → 64 → 1
  ↓
Sigmoid → [batch, 1]
```

**优势:**
- ✅ **专为时序建模设计** - 天然适合捕捉时间依赖
- ✅ **长短期记忆能力** - 既能检测瞬时异常(Step)，又能检测持续异常(Drift)
- ✅ **自动特征提取** - 学习时序模式，不需要手工设计
- ✅ **对Delay攻击敏感** - LSTM能记住过去的运动模式，检测时序错位
- ✅ **成熟稳定** - 在时序异常检测任务中广泛验证

**预期表现:**
- Delay/Drift: 可能 **优于** CNN (更强的时序建模)
- Step: 与CNN相当 (尖峰特征明显)
- Replay: 可能 **略优于** CNN (更好的序列记忆)

**实现复杂度:** ⭐⭐ (简单)

---

### ✅ **2. Bidirectional LSTM (BiLSTM) - 强烈推荐**

**架构概述:**
```python
Input: [batch, 50, 13]
  ↓
BiLSTM (forward + backward, hidden_size=64, num_layers=2)
  ↓
Concat forward & backward: [batch, 128]
  ↓
FC: 128 → 64 → 1 → Sigmoid
```

**优势:**
- ✅ **双向上下文** - 同时看过去和未来，更好理解局部异常
- ✅ **拼接点检测增强** - 对Replay攻击的拼接点更敏感
- ✅ **鲁棒性更强** - 不受时序方向影响

**预期表现:**
- Replay: 可能 **显著优于** LSTM/CNN (拼接点前后都有异常)
- 其他攻击: 与LSTM相当或略优

**实现复杂度:** ⭐⭐ (简单，只需设置 bidirectional=True)

---

### ✅ **3. GRU (Gated Recurrent Unit) - 推荐**

**架构概述:**
```python
与LSTM类似，但门控机制更简单
Input: [batch, 50, 13]
  ↓
GRU (hidden_size=128, num_layers=2-3)
  ↓
FC → Sigmoid
```

**优势:**
- ✅ **计算效率高** - 比LSTM参数少30%，训练快
- ✅ **性能接近LSTM** - 在很多任务上与LSTM相当
- ✅ **不易过拟合** - 参数少，正则化效果更好

**预期表现:**
- 与LSTM接近，可能在小数据集上更好

**实现复杂度:** ⭐⭐ (简单)

---

### ✅ **4. 1D-CNN + LSTM (Hybrid) - 强烈推荐**

**架构概述:**
```python
Input: [batch, 50, 13]
  ↓
1D Conv layers (提取局部特征)
  ↓  [batch, seq_len', 128]
LSTM (建模时序依赖)
  ↓  [batch, 128]
FC → Sigmoid
```

**优势:**
- ✅ **结合CNN和LSTM优势** 
  - CNN: 快速提取局部模式（尖峰、突变）
  - LSTM: 建模全局时序依赖
- ✅ **层次化特征** - 先局部再全局
- ✅ **性能潜力最大** - 在很多时序任务中SOTA

**预期表现:**
- **最佳综合性能** - 所有攻击类型都可能表现最好
- Step: CNN部分捕捉尖峰
- Delay/Drift: LSTM部分捕捉时序错位
- Replay: 两者结合检测拼接

**实现复杂度:** ⭐⭐⭐ (中等)

---

### ✅ **5. Temporal Convolutional Network (TCN) - 推荐**

**架构概述:**
```python
Input: [batch, 50, 13]
  ↓
多层膨胀卷积 (dilation=1,2,4,8,...)
  ↓
大感受野 (覆盖整个窗口)
  ↓
Global Pool → FC → Sigmoid
```

**优势:**
- ✅ **并行计算** - 比LSTM训练快很多
- ✅ **大感受野** - 膨胀卷积覆盖长时序
- ✅ **避免梯度消失** - 比LSTM更稳定

**当前已有实现** - 你的代码中已经有 `TemporalConvNet`

**预期表现:**
- 与LSTM接近，但训练更快
- 可能在长时依赖上略弱于LSTM

**实现复杂度:** ⭐⭐ (已实现)

---

### ⚠️ **6. Transformer - 有条件推荐**

**架构概述:**
```python
Input: [batch, 50, 13]
  ↓
Positional Encoding (添加位置信息)
  ↓
Multi-Head Self-Attention (num_heads=4-8, num_layers=2-4)
  ↓
Feed-Forward Networks
  ↓
Global Pool → FC → Sigmoid
```

**Transformer 是否适合该任务？**

#### ✅ **优势:**
1. **强大的时序建模** - Self-attention捕捉全局依赖
2. **并行训练** - 比LSTM快
3. **灵活的注意力机制** - 可能发现CNN/LSTM忽略的模式
4. **可解释性** - Attention权重可视化哪些时刻重要

#### ❌ **劣势 (针对你的任务):**
1. **数据量需求大** ⚠️
   - Transformer通常需要大规模数据
   - 你的数据集: ~300航班 → 可能不够
   - 风险: 过拟合，性能可能不如LSTM

2. **序列较短** ⚠️
   - 窗口只有50个样本
   - Transformer在长序列(>100)上优势更明显
   - 短序列上LSTM/CNN可能更高效

3. **计算成本高** ⚠️
   - 参数量大，训练慢
   - 对于50长度序列可能过度设计

4. **位置编码的影响** ⚠️
   - 时序信号中绝对位置意义不大
   - 相对位置更重要 (Delay的时序错位)

#### 💡 **建议:**
- **可以尝试，但不作为主力模型**
- 使用 **轻量级Transformer**:
  - num_layers = 2-3 (不要太深)
  - num_heads = 4 (不要太多)
  - hidden_dim = 64-128 (不要太大)
- 与LSTM对比:
  - 如果性能相当 → 选LSTM (更简单)
  - 如果显著更好 → 采用Transformer

**预期表现:**
- **可能不如LSTM** (数据量限制)
- **可能略优于纯CNN** (全局attention)
- **训练时间最长**

**实现复杂度:** ⭐⭐⭐⭐ (较复杂)

---

### ❌ **7. 不推荐的模型**

**ResNet / DenseNet (2D架构改1D):**
- ❌ 设计用于图像，对1D时序过度复杂
- ❌ 深度残差在短序列上意义不大

**Autoencoder (无监督):**
- ❌ 你已有标签数据，监督学习更直接
- ❌ 重构误差不如直接分类稳定

**SVM / Random Forest:**
- ❌ 需要手工特征工程
- ❌ 无法充分利用时序结构

---

## 推荐实验方案

### 🎯 **核心对比组 (必做)**

| 模型 | 优先级 | 预期性能 | 训练时间 | 实现难度 |
|-----|--------|---------|---------|---------|
| **1D-CNN** | ⭐⭐⭐⭐⭐ | 基线 | 快 | 已完成 |
| **LSTM** | ⭐⭐⭐⭐⭐ | 最优 | 中 | 简单 |
| **BiLSTM** | ⭐⭐⭐⭐⭐ | 最优 | 中 | 简单 |
| **CNN-LSTM** | ⭐⭐⭐⭐ | 最优+ | 中 | 中等 |
| **GRU** | ⭐⭐⭐ | 优秀 | 快 | 简单 |

### 📊 **扩展对比组 (可选)**

| 模型 | 优先级 | 目的 |
|-----|--------|------|
| **TCN** | ⭐⭐⭐ | 已实现，对比膨胀卷积效果 |
| **Transformer** | ⭐⭐ | 探索attention机制潜力 |
| **Attention-LSTM** | ⭐⭐ | LSTM + Attention权重 |

---

## 统一实验框架

### 保持一致的部分

```python
# 1. 数据处理 (完全一致)
- load_flights_data()
- split_flights()
- compute_delta_t()
- convert_to_local_coordinates()
- inject_attacks()
- generate_labels()
- compute_features()
- normalize_features()
- create_windows()
- balance_windows()

# 2. 训练设置 (完全一致)
- WINDOW_SIZE = 50
- STEP_SIZE = 5
- BATCH_SIZE = 64
- LEARNING_RATE = 0.001
- NUM_EPOCHS = 50
- EARLY_STOPPING_PATIENCE = 10
- Loss: BCELoss with pos_weight
- Optimizer: Adam

# 3. 评估指标 (完全一致)
- AUC-ROC
- AUC-PR
- F1, Precision, Recall
- Per-flight metrics
- Confusion matrix
```

### 仅修改的部分

```python
# model.py 中添加新模型类

def create_model(model_type, n_features, window_size, dropout):
    if model_type == 'cnn':
        return GPSSpoofingDetector(...)
    elif model_type == 'lstm':
        return LSTMDetector(...)
    elif model_type == 'bilstm':
        return BiLSTMDetector(...)
    elif model_type == 'gru':
        return GRUDetector(...)
    elif model_type == 'cnn_lstm':
        return CNNLSTMDetector(...)
    elif model_type == 'tcn':
        return TemporalConvNet(...)
    elif model_type == 'transformer':
        return TransformerDetector(...)
    else:
        raise ValueError(f"Unknown model: {model_type}")
```

### 运行脚本示例

```python
# main.py 中
models_to_test = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm']

for model_type in models_to_test:
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*80}")
    
    model = create_model(
        model_type=model_type,
        n_features=len(feature_cols),
        window_size=config.WINDOW_SIZE,
        dropout=0.3
    )
    
    # 相同的训练流程
    history = train_model(model, train_loader, val_loader, ...)
    
    # 相同的评估流程
    metrics = evaluate_model(model, test_loader, ...)
    
    # 保存到 output/{attack_type}/{model_type}/
```

---

## 预期实验结果

### 各模型在不同攻击上的预期表现

| 攻击类型 | CNN | LSTM | BiLSTM | CNN-LSTM | Transformer |
|---------|-----|------|--------|----------|-------------|
| **Step** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Delay** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Drift** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Takeover** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Replay** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

**关键洞察:**
- **Replay是区分度最大的攻击** - BiLSTM/CNN-LSTM可能显著优于CNN
- **其他攻击都较易检测** - 模型差异可能不大
- **综合性能**: CNN-LSTM ≥ BiLSTM > LSTM > CNN ≈ GRU
- **训练效率**: GRU > CNN > TCN > LSTM > BiLSTM > CNN-LSTM > Transformer

---

## 实施步骤建议

### Phase 1: 基础对比 (1-2天)
1. ✅ 实现 LSTM
2. ✅ 实现 BiLSTM
3. ✅ 实现 GRU
4. 🔧 在单个攻击(如step)上快速验证
5. 📊 对比 CNN vs LSTM vs BiLSTM vs GRU

### Phase 2: 高级模型 (2-3天)
6. ✅ 实现 CNN-LSTM hybrid
7. ✅ 实现 Transformer (可选)
8. 🔧 在所有攻击类型上完整测试
9. 📊 生成完整对比报告

### Phase 3: 分析优化 (1-2天)
10. 📈 超参数调优 (hidden_size, num_layers, dropout)
11. 📊 可视化attention/hidden states (理解模型)
12. 📝 撰写论文分析部分

---

## 论文可以论证的点

1. **模型选择的系统性对比**
   - 不同架构对不同攻击类型的检测能力
   - CNN擅长局部模式，LSTM擅长时序依赖

2. **Replay攻击的特殊性**
   - 双向模型(BiLSTM)对拼接点更敏感
   - 混合模型(CNN-LSTM)综合性能最优

3. **轻量级vs复杂模型权衡**
   - Transformer性能 vs 计算成本
   - GRU vs LSTM: 效率与性能平衡

4. **实际部署建议**
   - 资源受限: GRU/CNN
   - 性能优先: CNN-LSTM/BiLSTM
   - 实时检测: CNN/TCN (并行快)

---

## 总结建议

### ✅ **最终推荐方案**

**必须实现 (核心对比):**
1. **LSTM** - 时序建模标准方法
2. **BiLSTM** - Replay检测增强
3. **CNN-LSTM** - 最佳综合性能

**可选实现 (完整性):**
4. **GRU** - 效率对比
5. **TCN** - 已有实现，直接测试
6. **Transformer** - 探索性研究

**不建议:**
- 复杂的2D架构改编
- 无监督方法

### 🎯 **Transformer 使用建议**

- ✅ **可以尝试** - 作为探索性工作
- ⚠️ **预期**: 可能不如LSTM (数据量限制)
- 💡 **价值**: 论文可以讨论"为何Transformer不如LSTM"
  - 数据量需求
  - 序列长度影响
  - 归纳偏置的重要性

### 📊 **预期最终结论**

"对于UAV GPS欺骗检测任务:
- **CNN-LSTM** 综合性能最优
- **BiLSTM** 在Replay检测上最佳
- **GRU** 效率与性能平衡最好
- **Transformer** 因数据量限制未能发挥优势"

这将是一个非常完整和有说服力的实验对比！
