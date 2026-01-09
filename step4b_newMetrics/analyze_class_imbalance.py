"""
分析当前数据集的类别分布和不平衡问题
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 80)
print("数据集类别分布分析")
print("=" * 80)

# 加载训练集和测试集的数据
train_file = '../step3b_multiModelGeneral/output/cnn/train_val_data.npz'
test_file = '../step3b_multiModelGeneral/output/cnn/test_predictions.npz'

print("\n【1. 训练集/验证集分析】")
try:
    train_data = np.load(train_file)
    print(f"训练集文件: {train_file}")
    print(f"可用keys: {list(train_data.keys())}")
except:
    print("训练集数据文件不存在，尝试其他方法...")

print("\n【2. 测试集分析】")
test_data = np.load(test_file)
y_true = test_data['labels']
y_pred = (test_data['probabilities'] > 0.5).astype(int)
y_prob = test_data['probabilities']
flight_ids = test_data['flight_ids']

print(f"\n测试集窗口数: {len(y_true)}")
print(f"  正常窗口: {(y_true == 0).sum()} ({100*(y_true == 0).sum()/len(y_true):.1f}%)")
print(f"  攻击窗口: {(y_true == 1).sum()} ({100*(y_true == 1).sum()/len(y_true):.1f}%)")
print(f"\n类别比例: 正常:攻击 = {(y_true == 0).sum()}:{(y_true == 1).sum()} ≈ {(y_true == 0).sum()/(y_true == 1).sum():.1f}:1")

print("\n【3. 预测分布分析】")
print(f"预测为攻击: {(y_pred == 1).sum()} ({100*(y_pred == 1).sum()/len(y_pred):.1f}%)")
print(f"预测为正常: {(y_pred == 0).sum()} ({100*(y_pred == 0).sum()/len(y_pred):.1f}%)")

print("\n【4. 混淆矩阵】")
TP = ((y_true == 1) & (y_pred == 1)).sum()
FP = ((y_true == 0) & (y_pred == 1)).sum()
TN = ((y_true == 0) & (y_pred == 0)).sum()
FN = ((y_true == 1) & (y_pred == 0)).sum()

print(f"\n                    预测")
print(f"                攻击      正常")
print(f"    攻击        {TP:4d}      {FN:4d}   (实际攻击: {TP+FN})")
print(f"真实")
print(f"    正常        {FP:4d}      {TN:4d}   (实际正常: {FP+TN})")
print(f"             (预测攻击:{TP+FP})  (预测正常:{FN+TN})")

print("\n【5. 性能指标】")
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall / Sensitivity
FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0

print(f"TPR (True Positive Rate / Recall):  {TPR:.4f} = {TPR*100:.2f}%")
print(f"FPR (False Positive Rate):          {FPR:.4f} = {FPR*100:.2f}%")
print(f"Precision (精确率):                  {Precision:.4f} = {Precision*100:.2f}%")
print(f"F1 Score:                            {F1:.4f}")

print("\n【6. 概率分布分析】")
print(f"\n正常样本的预测概率分布:")
normal_probs = y_prob[y_true == 0]
print(f"  Mean: {normal_probs.mean():.4f}")
print(f"  Median: {np.median(normal_probs):.4f}")
print(f"  Std: {normal_probs.std():.4f}")
print(f"  >0.5的比例: {(normal_probs > 0.5).sum()}/{len(normal_probs)} = {100*(normal_probs > 0.5).sum()/len(normal_probs):.2f}%")
print(f"  >0.7的比例: {(normal_probs > 0.7).sum()}/{len(normal_probs)} = {100*(normal_probs > 0.7).sum()/len(normal_probs):.2f}%")

print(f"\n攻击样本的预测概率分布:")
attack_probs = y_prob[y_true == 1]
print(f"  Mean: {attack_probs.mean():.4f}")
print(f"  Median: {np.median(attack_probs):.4f}")
print(f"  Std: {attack_probs.std():.4f}")
print(f"  >0.5的比例: {(attack_probs > 0.5).sum()}/{len(attack_probs)} = {100*(attack_probs > 0.5).sum()/len(attack_probs):.2f}%")
print(f"  >0.9的比例: {(attack_probs > 0.9).sum()}/{len(attack_probs)} = {100*(attack_probs > 0.9).sum()/len(attack_probs):.2f}%")

print("\n【7. 阈值调整模拟】")
print(f"\n不同阈值下的TPR和FPR:")
print(f"{'阈值':>6} | {'TPR':>7} | {'FPR':>7} | {'Precision':>10} | {'F1':>7} | {'误报数':>6}")
print("-" * 65)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred_t = (y_prob > threshold).astype(int)
    TP_t = ((y_true == 1) & (y_pred_t == 1)).sum()
    FP_t = ((y_true == 0) & (y_pred_t == 1)).sum()
    TN_t = ((y_true == 0) & (y_pred_t == 0)).sum()
    FN_t = ((y_true == 1) & (y_pred_t == 0)).sum()
    
    TPR_t = TP_t / (TP_t + FN_t) if (TP_t + FN_t) > 0 else 0
    FPR_t = FP_t / (FP_t + TN_t) if (FP_t + TN_t) > 0 else 0
    Prec_t = TP_t / (TP_t + FP_t) if (TP_t + FP_t) > 0 else 0
    F1_t = 2 * Prec_t * TPR_t / (Prec_t + TPR_t) if (Prec_t + TPR_t) > 0 else 0
    
    print(f"{threshold:6.1f} | {TPR_t:7.3f} | {FPR_t:7.3f} | {Prec_t:10.3f} | {F1_t:7.3f} | {FP_t:6d}")

print("\n【8. 类别不平衡影响分析】")
imbalance_ratio = (y_true == 0).sum() / (y_true == 1).sum()
print(f"\n当前类别比例: {imbalance_ratio:.1f}:1 (正常:攻击)")
print(f"\n类别不平衡的影响:")
print(f"  - 模型倾向于预测多数类（正常）")
print(f"  - 但当前FPR={FPR:.4f}，说明模型倾向于预测攻击")
print(f"  - 这可能是因为:")
print(f"    1. 训练集中攻击比例更高")
print(f"    2. 损失函数对攻击类别的权重更高")
print(f"    3. 模型过拟合到攻击特征")

# 检查训练时的攻击比例（从README或config推断）
print("\n从配置文件推断训练集分布:")
print("  训练集: 60% attacked (推测)")
print("  验证集: 50% attacked (推测)")
print("  测试集: 70% attacked (推测)")
print("  实际测试集: {:.1f}% attacked".format(100*(y_true==1).sum()/len(y_true)))

print("\n" + "=" * 80)
