# Step5: TimeGAN数据增强实验

## 目标
通过TimeGAN扩充正常飞行数据，降低训练集攻击比例，改善模型的FPR（误报率）

## 核心思路
1. **问题**: Step3b中训练集攻击比例过高（60%），导致模型过度学习攻击模式，FPR居高不下
2. **方案**: 使用TimeGAN生成大量合成正常飞行，增加正常样本的数量和多样性
3. **效果**: 降低训练集攻击比例至15%，使模型学习更真实的先验分布

## 数据流程

```
原始数据提取
    ↓
prepare_normal_data.py: 提取正常飞行
    ↓
train_timegan.py: 训练TimeGAN学习正常飞行模式
    ↓
generate_synthetic_flights.py: 生成5倍合成正常飞行
    ↓
merge_and_split.py: 合并原始+合成，划分Train/Val/Test
    ↓
inject_attacks.py: 注入低比例攻击(15%/10%/5%)
    ↓
main.py: 特征工程 + 窗口创建 + 模型训练
    ↓
compare_with_baseline.py: 与Step3b对比
```

## 快速开始

### 1. 准备数据
```bash
# 提取正常飞行
python prepare_normal_data.py

# 训练TimeGAN（约5-10分钟）
python train_timegan.py

# 生成合成飞行
python generate_synthetic_flights.py

# 合并和划分
python merge_and_split.py

# 注入攻击
python inject_attacks.py
```

### 2. 训练模型
```bash
# 训练LSTM
python main.py --model lstm

# 训练GRU
python main.py --model gru

# 训练CNN-LSTM
python main.py --model cnn_lstm
```

### 3. 对比分析
```bash
python compare_with_baseline.py --model lstm
```

## 配置参数

在 `config.py` 中调整：

- `TIMEGAN_EXPANSION_FACTOR = 5`: 扩充倍数（5倍正常数据）
- `TRAIN_ATTACK_RATIO = 0.15`: 训练集攻击比例（15%）
- `VAL_ATTACK_RATIO = 0.10`: 验证集攻击比例（10%）
- `TEST_ATTACK_RATIO = 0.05`: 测试集攻击比例（5%）

## 预期改进

| 指标 | Step3b | Step5 | 改进 |
|------|--------|-------|------|
| 训练攻击比例 | 40% | 12% | ↓70% |
| 正常样本数 | ~100 | ~600 | ↑500% |
| FPR | 高 | 低 | ✅ |
| Recall | 高 | 中-高 | ✅ |
| F1 | 中 | 高 | ✅ |

## 输出文件

- `output/normal_flights.csv`: 原始正常飞行
- `output/synthetic_normal_flights.csv`: 合成正常飞行
- `output/train_data.npz`: 训练窗口
- `output/best_model_lstm.pth`: 训练好的模型
- `output/test_metrics_lstm.json`: 测试指标
- `output/comparison_report.json`: 与baseline对比报告

## 技术细节

### TimeGAN架构
- Embedder: 学习数据表示
- Recovery: 重构原始数据
- Generator: 从噪声生成隐藏表示
- Supervisor: 学习时序动态
- Discriminator: 判别真假数据

### 三阶段训练
1. 自编码器（100 epochs）: 学习数据压缩
2. Supervisor（100 epochs）: 学习时序规律
3. GAN（200 epochs）: 对抗训练生成

## 依赖
- PyTorch
- NumPy, Pandas
- Matplotlib
- step3b_multiModelGeneral（复用代码）

## 作者
UAV GPS Spoofing Detection Project
