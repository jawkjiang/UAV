"""
完整分析：为什么Transformer的Precision高但MTBFA低
"""
import json
import pandas as pd
import numpy as np

print("=" * 100)
print("为什么Transformer的Precision不低，但MTBFA很低（经常误报）？".center(100))
print("=" * 100)

# 加载数据
df_time = pd.read_csv("step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv")

models = ['transformer', 'tcn', 'gru', 'lstm']
all_data = {}

for model in models:
    with open(f'step5_timegan/output/test_metrics_{model}.json', 'r') as f:
        all_data[model] = json.load(f)

print("\n【第一部分：窗口级别的Precision分析】")
print("-" * 100)
print(f"{'模型':<12} {'TP':<10} {'FP':<10} {'TN':<10} {'FN':<10} {'Precision':<12} {'Recall':<12} {'总窗口':<10}")
print("-" * 100)

for model in models:
    data = all_data[model]
    tp = data['true_positives']
    fp = data['false_positives']
    tn = data['true_negatives']
    fn = data['false_negatives']
    precision = data['precision']
    recall = data['recall']
    total = tp + fp + tn + fn
    
    print(f"{model.upper():<12} {tp:<10.0f} {fp:<10.0f} {tn:<10.0f} {fn:<10.0f} "
          f"{precision:<12.4f} {recall:<12.4f} {total:<10.0f}")

print("-" * 100)
print("\n解读：")
print("  • TP (True Positive): 正确检测为攻击的窗口数")
print("  • FP (False Positive): 误报为攻击的窗口数（正常窗口被错误标记为攻击）")
print("  • TN (True Negative): 正确识别为正常的窗口数")
print("  • FN (False Negative): 漏报的窗口数（攻击窗口未被检测到）")
print("  • Precision = TP / (TP + FP) = 在所有预测为攻击的窗口中，真正是攻击的比例")
print()
print("  Transformer有107个假阳性窗口，但总共有2024+107=2131个被预测为攻击的窗口")
print("  所以Precision = 2024/2131 = 0.9498 (94.98%)，看起来很高！")

print("\n\n【第二部分：飞行级别的MTBFA分析】")
print("-" * 100)

time_data = {}
for _, row in df_time.iterrows():
    time_data[row['model']] = row

print(f"{'模型':<12} {'误报次数':<12} {'检测攻击数':<12} {'总攻击数':<12} {'MTBFA(小时)':<15} {'MTBFA(分钟)':<15}")
print("-" * 100)

for model in models:
    row = time_data[model]
    mtbfa_minutes = row['MTBFA'] * 60
    
    print(f"{model.upper():<12} {row['n_false_alarms']:<12.0f} "
          f"{row['n_detected']:<12.0f} {row['n_attacks']:<12.0f} "
          f"{row['MTBFA']:<15.4f} {mtbfa_minutes:<15.1f}")

print("-" * 100)
print("\n解读：")
print("  • 误报次数: 在正常飞行中产生的误报次数（飞行级别，不是窗口级别）")
print("  • MTBFA = 正常飞行总时长 / 误报次数")
print("  • Transformer产生了18次误报，平均每13.1分钟误报一次")
print("  • GRU/LSTM只产生了11次误报，平均每21.5分钟误报一次")

print("\n\n【第三部分：关键矛盾解析】")
print("=" * 100)

print("""
问题：为什么Transformer的Precision高达0.9498，但MTBFA却只有13.1分钟？

答案：**Precision和MTBFA衡量的是不同维度的性能！**

┌─────────────────────────────────────────────────────────────────────────────┐
│ 维度 1: 窗口级别 (Window-Level) - Precision衡量的维度                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  测试集总共有约20,000个窗口：                                                 │
│  ├─ 正常窗口: 18,080 + 107 = 18,187个 (约90%)                               │
│  └─ 攻击窗口: 2,024 + 91 = 2,115个 (约10%)                                  │
│                                                                             │
│  Transformer的预测：                                                         │
│  ├─ 预测为攻击: 2,131个窗口                                                  │
│  │   ├─ 其中正确(TP): 2,024个 ✓                                             │
│  │   └─ 其中错误(FP): 107个 ✗                                               │
│  └─ Precision = 2,024 / 2,131 = 94.98% 👍                                  │
│                                                                             │
│  结论：在窗口级别，Transformer的Precision确实很高！                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 维度 2: 飞行级别 (Flight-Level) - MTBFA衡量的维度                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  测试集有正常飞行若干个，总时长约: 18次 × 0.219小时 ≈ 3.94小时               │
│                                                                             │
│  这107个FP窗口分布在正常飞行中：                                              │
│  ├─ 聚合为18次独立的误报事件（连续的FP窗口算一次误报）                         │
│  └─ 平均每 3.94小时/18次 = 0.219小时 = 13.1分钟 误报一次                     │
│                                                                             │
│  实际部署场景：                                                               │
│  ├─ 飞行员在正常飞行时，每13分钟就会收到一次虚假警报                           │
│  └─ 这会导致"警报疲劳"，飞行员可能会忽略真实警报 ⚠️                          │
│                                                                             │
│  结论：在实际应用中，Transformer的误报频率太高！                              │
└─────────────────────────────────────────────────────────────────────────────┘

关键洞察：

1. **窗口数量 vs 误报事件**
   - 107个FP窗口 ≠ 107次误报事件
   - 连续的FP窗口会聚合为一次误报事件
   - 但107个窗口分散在~2000个预测窗口中，占比只有5%（高Precision）

2. **数据集不平衡的影响**
   - 正常窗口约18,000个，攻击窗口约2,000个（9:1）
   - 即使FP率很低(107/18,187=0.59%)，在时间维度上仍然频繁
   - 因为正常飞行时间远多于攻击时间

3. **为什么GRU/LSTM更好？**
   - GRU产生11次误报 vs Transformer的18次
   - MTBFA: 21.5分钟 vs 13.1分钟（提升64%）
   - Precision虽然略低(0.933 vs 0.950)，但实际误报少得多

4. **为什么TCN更差？**
   - TCN的Precision最高(0.955)，但MTBFA最低(13.1分钟)
   - TCN漏检了很多攻击(Recall=0.792)，但仍产生18次误报
   - 这说明TCN在正常飞行期间过于敏感

┌─────────────────────────────────────────────────────────────────────────────┐
│ 总结：传统Precision/Recall为什么不够用？                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  传统指标的局限性：                                                            │
│  ✗ 忽略了时序特性：窗口之间的时间关系                                          │
│  ✗ 忽略了实际影响：误报对用户的实际影响                                        │
│  ✗ 忽略了数据不平衡：正常时间 >> 攻击时间                                     │
│                                                                             │
│  时间感知指标的优势：                                                          │
│  ✓ MTBFA: 衡量误报的实际频率（用户体验）                                       │
│  ✓ DR@Δt: 衡量检测速度（安全关键）                                            │
│  ✓ ADD: 衡量平均检测延迟（响应时间）                                          │
│                                                                             │
│  推荐模型：GRU                                                                │
│  ├─ DR@10s = 100% (最快检测)                                                │
│  ├─ MTBFA = 21.5分钟 (误报少)                                               │
│  ├─ ADD = 0.542秒 (快速响应)                                                │
│  └─ F1 = 0.948 (整体性能优秀)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 100)
print("分析完成！".center(100))
print("=" * 100)
