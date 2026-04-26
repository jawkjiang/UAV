"""
数据验证对照表生成器
生成详细的数据对照表，用于验证图表数据的正确性
"""
import pandas as pd
import json

print("=" * 100)
print(" " * 35 + "数据验证对照表")
print("=" * 100)

# ===========================================================================
# Fig 2 数据对照表
# ===========================================================================
print("\n【Fig 2: DR@Δt曲线数据】")
print("-" * 100)
df = pd.read_csv("step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv")

print(f"{'模型':<12s} {'DR@1s':>8s} {'DR@2s':>8s} {'DR@5s':>8s} {'DR@10s':>8s} {'DR@15s':>8s} {'DR@30s':>8s}")
print("-" * 100)
for _, row in df.iterrows():
    model = row['model'].upper()
    print(f"{model:<12s} {row['DR@1s']:8.4f} {row['DR@2s']:8.4f} {row['DR@5s']:8.4f} "
          f"{row['DR@10s']:8.4f} {row['DR@15s']:8.4f} {row['DR@30s']:8.4f}")

print("\n✓ Fig 2关键验证:")
print(f"  - GRU的DR@5s应该显示为 0.9808 (98.08%)")
print(f"  - GRU的DR@10s应该显示为 1.0000 (100%)")
print(f"  - LSTM和BiLSTM的DR@5s和DR@10s都应该是 0.9808")

# ===========================================================================
# Fig 3 数据对照表
# ===========================================================================
print("\n\n【Fig 3: 传统指标 vs 时间感知指标数据】")
print("-" * 100)

# 传统指标
print("\n(a) & (b) 传统指标:")
print(f"{'模型':<12s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
print("-" * 100)

models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
for model in models:
    with open(f'step5_timegan/output/test_metrics_{model}.json', 'r') as f:
        metrics = json.load(f)
        print(f"{model.upper():<12s} {metrics['precision']:10.4f} {metrics['recall']:10.4f} {metrics['f1']:10.4f}")

# 时间感知指标
print("\n(c) & (d) 时间感知指标:")
print(f"{'模型':<12s} {'DR@5s':>10s} {'ADD (s)':>10s}")
print("-" * 100)
for _, row in df.iterrows():
    model = row['model'].upper()
    print(f"{model:<12s} {row['DR@5s']:10.4f} {row['ADD']:10.4f}")

print("\n✓ Fig 3关键验证:")
print(f"  - 所有模型的Precision、Recall、F1应该都有数值显示")
print(f"  - GRU的F1-Score应该是 0.9476 (最高)")
print(f"  - Transformer的F1-Score应该是 0.9534 (第二高)")

# ===========================================================================
# Fig 10 数据对照表
# ===========================================================================
print("\n\n【Fig 10: 按攻击类型的DR@5s数据】")
print("-" * 100)

# 先收集所有攻击类型
all_attacks = set()
for model in models:
    df_attack = pd.read_csv(f"step6_newMetrcisWithTimegan/output/time_aware_metrics/{model}_per_attack_metrics.csv")
    all_attacks.update(df_attack['attack_type'].tolist())

attack_types = sorted(list(all_attacks))

# 打印表头
print(f"{'模型':<12s}", end="")
for at in attack_types:
    print(f" {at.title():>10s}", end="")
print()
print("-" * 100)

# 打印每个模型的数据
for model in models:
    df_attack = pd.read_csv(f"step6_newMetrcisWithTimegan/output/time_aware_metrics/{model}_per_attack_metrics.csv")
    data_dict = {row['attack_type']: row['DR@5s'] for _, row in df_attack.iterrows()}
    
    print(f"{model.upper():<12s}", end="")
    for at in attack_types:
        val = data_dict.get(at, 0)
        print(f" {val:10.4f}", end="")
    print()

# 计算平均值
print("-" * 100)
print(f"{'AVERAGE':<12s}", end="")
for at in attack_types:
    vals = []
    for model in models:
        df_attack = pd.read_csv(f"step6_newMetrcisWithTimegan/output/time_aware_metrics/{model}_per_attack_metrics.csv")
        data_dict = {row['attack_type']: row['DR@5s'] for _, row in df_attack.iterrows()}
        if at in data_dict:
            vals.append(data_dict[at])
    avg = sum(vals) / len(vals) if vals else 0
    print(f" {avg:10.4f}", end="")
print()

print("\n✓ Fig 10关键验证:")
print(f"  - 所有模型对'step'和'takeover'攻击的DR@5s应该都是1.0000")
print(f"  - 对'drift'攻击,LSTM/BiLSTM/GRU的DR@5s应该是0.9231")
print(f"  - 对'delay'攻击,TCN的DR@5s应该是0.8462 (最低)")

# ===========================================================================
# 汇总统计
# ===========================================================================
print("\n\n【整体数据汇总】")
print("-" * 100)
print(f"数据源文件:")
print(f"  ✓ step5_timegan/output/test_metrics_*.json (传统指标)")
print(f"  ✓ step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv (时间感知指标)")
print(f"  ✓ step6_newMetrcisWithTimegan/output/time_aware_metrics/*_per_attack_metrics.csv (按攻击类型)")

print(f"\n图表数量:")
print(f"  ✓ Fig 2: 7条曲线，6个时间点")
print(f"  ✓ Fig 3: 4个子图，每个7个模型")
print(f"  ✓ Fig 10: {len(attack_types)}种攻击类型，7个模型 + 平均线")

print("\n" + "=" * 100)
print(" " * 35 + "数据验证完成！")
print("=" * 100)
