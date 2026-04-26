"""
分析Precision与MTBFA的关系
解释为什么Transformer的Precision不低，但MTBFA很低
"""
import pandas as pd
import numpy as np
import json

print("=" * 80)
print("Precision vs MTBFA 矛盾分析")
print("=" * 80)

# 1. 加载时间感知指标
df_time = pd.read_csv("step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv")

# 2. 加载传统指标
traditional_metrics = {}
for model in ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']:
    with open(f'step5_timegan/output/test_metrics_{model}.json', 'r') as f:
        traditional_metrics[model] = json.load(f)

# 3. 合并数据
print("\n【核心指标对比】")
print("-" * 80)
print(f"{'模型':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MTBFA(h)':<12} {'误报次数':<12}")
print("-" * 80)

for _, row in df_time.iterrows():
    model = row['model']
    trad = traditional_metrics[model]
    
    print(f"{model.upper():<12} "
          f"{trad['precision']:<12.4f} "
          f"{trad['recall']:<12.4f} "
          f"{trad['f1']:<12.4f} "
          f"{row['MTBFA']:<12.4f} "
          f"{row['n_false_alarms']:<12.0f}")

print("-" * 80)

# 4. 重点分析Transformer和TCN（MTBFA最低的两个）
print("\n\n【深入分析：Transformer vs TCN】")
print("-" * 80)

trans_row = df_time[df_time['model'] == 'transformer'].iloc[0]
tcn_row = df_time[df_time['model'] == 'tcn'].iloc[0]

trans_trad = traditional_metrics['transformer']
tcn_trad = traditional_metrics['tcn']

print("\nTransformer:")
print(f"  Precision = {trans_trad['precision']:.4f} (在预测为攻击的样本中，94.98%是真攻击)")
print(f"  Recall    = {trans_trad['recall']:.4f} (检测到95.70%的真实攻击)")
print(f"  MTBFA     = {trans_row['MTBFA']:.4f}小时 = {trans_row['MTBFA']*60:.1f}分钟")
print(f"  误报次数  = {trans_row['n_false_alarms']:.0f}次")
print(f"  检测次数  = {trans_row['n_detected']:.0f}次（真阳性）")

print("\nTCN:")
print(f"  Precision = {tcn_trad['precision']:.4f} (在预测为攻击的样本中，95.50%是真攻击)")
print(f"  Recall    = {tcn_trad['recall']:.4f} (仅检测到79.24%的真实攻击)")
print(f"  MTBFA     = {tcn_row['MTBFA']:.4f}小时 = {tcn_row['MTBFA']*60:.1f}分钟")
print(f"  误报次数  = {tcn_row['n_false_alarms']:.0f}次")
print(f"  检测次数  = {tcn_row['n_detected']:.0f}次（真阳性）")

# 5. 关键发现
print("\n\n【关键发现：Precision ≠ 低误报】")
print("=" * 80)

print("""
1. **Precision的定义**：
   Precision = TP / (TP + FP)
   - 在所有"预测为攻击"的样本中，真正是攻击的比例
   - Transformer预测了很多次攻击（包括真阳性和假阳性）
   - 虽然假阳性只占5%，但绝对数量仍然很高

2. **MTBFA的定义**：
   MTBFA = 正常飞行总时长 / 误报总次数
   - 专门衡量在正常飞行期间的误报频率
   - 与测试集的攻击/正常比例无关
   - 直接反映实际部署时的用户体验

3. **为什么会出现"矛盾"**？
   
   场景A: Transformer在测试集上
   - 测试集有52个攻击样本
   - Transformer检测到50个真攻击（TP=50，高Recall）
   - 同时产生18次误报（FP=18）
   - Precision = 50/(50+18) = 0.735... 等等，这不对！
""")

# 6. 重新计算验证
print("\n【重新计算验证】")
print("-" * 80)

print("\n从test_metrics_*.json加载的传统指标是在整个测试集上的：")
print("  - 测试集包含攻击样本和正常样本")
print("  - TP, TN, FP, FN是在窗口级别计算的，不是飞行级别")
print()

# 加载详细的混淆矩阵
for model in ['transformer', 'tcn']:
    with open(f'step5_timegan/output/test_metrics_{model}.json', 'r') as f:
        data = json.load(f)
    
    print(f"\n{model.upper()}的详细数据：")
    if 'confusion_matrix' in data:
        cm = data['confusion_matrix']
        tn = cm['TN']
        fp = cm['FP']
        fn = cm['FN']
        tp = cm['TP']
        
        total_windows = tn + fp + fn + tp
        
        print(f"  总窗口数: {total_windows}")
        print(f"  TP (真阳性): {tp}")
        print(f"  TN (真阴性): {tn}")
        print(f"  FP (假阳性): {fp}")
        print(f"  FN (假阴性): {fn}")
        print(f"  Precision = {tp}/{tp+fp} = {tp/(tp+fp):.4f}")
        print(f"  在{total_windows}个窗口中有{fp}个假阳性窗口")
        print(f"  假阳性率 = {fp}/{tn+fp} = {fp/(tn+fp):.4f}")

print("\n\n【结论】")
print("=" * 80)
print("""
关键点：**Precision是窗口级别的，MTBFA是时间级别的**

1. Transformer的高Precision (0.9498)意味着：
   - 在所有被标记为"攻击窗口"的窗口中，94.98%确实是攻击
   - 但这不代表误报少，只代表误报占总预测的比例小

2. Transformer的低MTBFA (0.219小时 ≈ 13分钟)意味着：
   - 在正常飞行期间，平均每13分钟就会误报一次
   - 这对实际应用来说频率太高，用户体验很差

3. 为什么TCN的MTBFA也很低但Precision更高？
   - TCN有更高的Precision (0.9550) 但更低的Recall (0.7924)
   - TCN更"保守"，只在非常确定时才报警
   - 但它仍然产生了18次误报（与Transformer相同）
   - 所以MTBFA也是0.219小时

4. 最佳模型应该是什么？
   - 看LSTM/BiLSTM: Precision=0.93, MTBFA=0.358小时 ≈ 21分钟
   - 看GRU: Precision=0.933, MTBFA=0.358小时，DR@10s=1.0
   - GRU在各方面都更平衡！

**教训**：传统的Precision/Recall不足以评估时序异常检测系统，
         必须引入MTBFA这样的时间维度指标！
""")

print("=" * 80)
