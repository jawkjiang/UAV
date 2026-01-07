# Step 3: Multi-Model Multi-Attack GPS Spoofing Detection

完整实现了多个深度学习模型在多种攻击类型上的交叉训练和验证系统。

## 项目概述

**实验矩阵**: 8种攻击类型 × 7种模型架构 = **56个独立实验**

### 攻击类型 (8种)

1. **step** - 阶跃攻击
2. **drift_ramp** - 斜坡漂移攻击
3. **drift_sigmoid** - 平滑漂移攻击
4. **delay** - 延迟攻击
5. **replay_same_hard** - 同航班硬拼接重放攻击
6. **replay_other_soft** - 跨航班软拼接重放攻击
7. **takeover_step** - 阶跃接管攻击
8. **takeover_ramp** - 斜坡接管攻击

### 模型架构 (7种)

1. **CNN** - 1D卷积神经网络 (基线模型)
2. **LSTM** - 长短期记忆网络
3. **BiLSTM** - 双向LSTM
4. **GRU** - 门控循环单元
5. **CNN-LSTM** - 混合架构
6. **TCN** - 时序卷积网络
7. **Transformer** - 基于注意力机制的Transformer

## 文件结构

```
step3_multiModel/
├── config_step3.py          # 配置文件 (从step2复制)
├── multi_attack_injector.py # 攻击注入器 (从step2复制)
├── model.py                 # 所有模型架构实现
├── main.py                  # 实验运行器 (56个实验)
├── compare_models.py        # 结果分析和可视化
├── test_models.py           # 模型接口验证
├── data_loader.py           # 数据加载 (从step1复制)
├── labeling.py              # 标签生成 (从step1复制)
├── feature_engineering.py   # 特征工程 (从step1复制)
├── window_creation.py       # 窗口创建 (从step1复制)
├── training.py              # 训练逻辑 (从step1复制)
├── evaluation.py            # 评估逻辑 (从step1复制)
├── README.md                # 本文档
└── output/                  # 实验结果输出
    ├── {attack_type}/
    │   └── {model_type}/
    │       ├── best_model.pth
    │       ├── test_metrics.json
    │       ├── training_history.json
    │       └── test_predictions.npz
    ├── overall_comparison.csv
    ├── comparison_auc_roc.csv
    ├── comparison_f1_score.csv
    ├── model_rankings.csv
    ├── attack_difficulty.csv
    ├── ANALYSIS_REPORT.md
    └── visualizations/
        ├── heatmap_auc_roc.png
        ├── heatmap_f1_score.png
        ├── boxplot_models.png
        ├── boxplot_attacks.png
        ├── bar_model_performance.png
        └── scatter_precision_recall.png
```

## 使用方法

### 1. 验证模型接口

首先验证所有模型是否符合规范：

```bash
python test_models.py
```

预期输出：
```
✓ All models passed interface validation!
```

### 2. 运行完整实验矩阵

运行所有56个实验（需要8-12小时，建议使用GPU）：

```bash
python main.py
```

进度追踪：
- 每个实验会显示当前进度 (X/56)
- 显示训练过程和评估指标
- 自动保存中间结果
- 支持断点续传（通过`SKIP_EXISTING`配置）

### 3. 分析结果

生成详细的对比分析和可视化：

```bash
python compare_models.py
```

输出：
- CSV格式的对比表格
- 热力图可视化
- 箱线图分布
- Markdown格式的分析报告

## 模型架构详情

### 1. CNN (Baseline)
```
参数量: 263,873
Input [batch, 50, 13]
  ↓ Conv1D layers
  ↓ Global pooling
  ↓ FC layers
Output [batch, 1]
```

### 2. LSTM
```
参数量: 215,873
Input [batch, 50, 13]
  ↓ LSTM (hidden=128, layers=2)
  ↓ Last hidden state
  ↓ FC layers
Output [batch, 1]
```

### 3. BiLSTM
```
参数量: 150,337
Input [batch, 50, 13]
  ↓ BiLSTM (hidden=64×2, layers=2)
  ↓ Concat forward & backward
  ↓ FC layers
Output [batch, 1]
```

### 4. GRU
```
参数量: 164,545
Input [batch, 50, 13]
  ↓ GRU (hidden=128, layers=2)
  ↓ Last hidden state
  ↓ FC layers
Output [batch, 1]
```

### 5. CNN-LSTM
```
参数量: 301,953
Input [batch, 50, 13]
  ↓ 1D Conv feature extraction
  ↓ LSTM temporal modeling
  ↓ FC layers
Output [batch, 1]
```

### 6. TCN
```
参数量: 77,313
Input [batch, 50, 13]
  ↓ Dilated convolutions
  ↓ Global pooling
  ↓ FC layers
Output [batch, 1]
```

### 7. Transformer
```
参数量: 103,489
Input [batch, 50, 13]
  ↓ Input projection
  ↓ Positional encoding
  ↓ Transformer encoder (heads=4, layers=3)
  ↓ Global avg pooling
  ↓ FC layers
Output [batch, 1]
```

## 关键配置

所有实验使用相同的配置（从`config_step3.py`）：

```python
WINDOW_SIZE = 50
STEP_SIZE = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
CONSISTENCY_MODES = ['pos_vel_acc']
```

## 评估指标

每个实验计算以下指标：

- **AUC-ROC**: 接收者操作特征曲线下面积
- **AUC-PR**: 精确率-召回率曲线下面积
- **F1 Score**: F1分数（准确率和召回率的调和平均）
- **Precision**: 精确率
- **Recall**: 召回率

## 预期结果

完成所有实验后，你将获得：

1. **性能对比矩阵**: 每种模型在每种攻击上的表现
2. **模型排名**: 按平均性能排序的模型列表
3. **攻击难度分析**: 识别最难检测的攻击类型
4. **最优配置**: 针对每种攻击的最佳模型
5. **可视化图表**: 热力图、箱线图、散点图等

## 时间估算

- **模型接口验证**: < 1分钟
- **单个实验**: 10-20分钟 (GPU) / 30-60分钟 (CPU)
- **全部56个实验**: 8-12小时 (GPU) / 24-48小时 (CPU)
- **结果分析**: < 5分钟

**建议**: 使用GPU并启用`SKIP_EXISTING`以支持断点续传

## 故障排除

### 内存不足
- 减小 `BATCH_SIZE`
- 使用更小的模型
- 一次只运行部分实验

### CUDA内存错误
```python
# 在main.py中添加
torch.cuda.empty_cache()
```

### 跳过已完成的实验
```python
# 在config_step3.py中设置
SKIP_EXISTING = True
```

## 扩展实验

要添加新的模型或攻击类型：

1. **添加新模型**:
   - 在`model.py`中定义新的模型类
   - 在`create_model()`中添加注册
   - 在`test_models.py`中添加测试

2. **添加新攻击**:
   - 在`ATTACK_TYPES`列表中添加
   - 在`ATTACK_PARAMS`中定义参数
   - 在`ATTACK_TYPE_MAP`中映射类型

## 引用

如果使用此代码，请引用原始规范文档：
- `STEP3_IMPLEMENTATION_SPEC.md`

## 许可

遵循项目主许可证。
