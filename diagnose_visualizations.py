"""
诊断可视化数据加载问题
"""
import pandas as pd
import json
import numpy as np

print("=" * 80)
print("VISUALIZATION DATA DIAGNOSIS")
print("=" * 80)

# 1. 检查Fig2数据源（DR@Δt曲线）
print("\n1. Fig2 DR@Δt数据检查")
print("-" * 80)
df = pd.read_csv("step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv")
print("可用的列:", df.columns.tolist())
print("\nGRU的DR@Δt数据:")
gru_row = df[df['model'] == 'gru'].iloc[0]
print(f"  DR@1s: {gru_row['DR@1s']}")
print(f"  DR@2s: {gru_row['DR@2s']}")
print(f"  DR@5s: {gru_row['DR@5s']}")
print(f"  DR@10s: {gru_row['DR@10s']}")
print(f"  DR@15s: {gru_row['DR@15s']}")
print(f"  DR@30s: {gru_row['DR@30s']}")

# 2. 检查Fig3数据源（传统指标）
print("\n\n2. Fig3 传统指标数据检查")
print("-" * 80)
models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
traditional_metrics = {}

for model in models:
    try:
        # 尝试从step5_timegan加载
        with open(f'step5_timegan/output/test_metrics_{model}.json', 'r') as f:
            metrics = json.load(f)
            traditional_metrics[model] = {
                'precision': metrics.get('precision', None),
                'recall': metrics.get('recall', None),
                'f1': metrics.get('f1', None)
            }
            print(f"✓ {model.upper()}: P={metrics.get('precision', 'N/A'):.4f}, "
                  f"R={metrics.get('recall', 'N/A'):.4f}, F1={metrics.get('f1', 'N/A'):.4f}")
    except Exception as e:
        print(f"✗ {model.upper()}: 加载失败 - {e}")
        traditional_metrics[model] = {'precision': None, 'recall': None, 'f1': None}

# 3. 检查Fig10数据源（按攻击类型）
print("\n\n3. Fig10 按攻击类型数据检查")
print("-" * 80)
for model in models:
    try:
        df = pd.read_csv(f"step6_newMetrcisWithTimegan/output/time_aware_metrics/{model}_per_attack_metrics.csv")
        print(f"\n✓ {model.upper()}:")
        print(f"  可用列: {df.columns.tolist()}")
        print(f"  攻击类型: {df['attack_type'].tolist()}")
        print(f"  DR@5s值: {df['DR@5s'].tolist()}")
    except Exception as e:
        print(f"✗ {model.upper()}: 加载失败 - {e}")

# 4. 检查comprehensive_report数据（Fig13）
print("\n\n4. Fig13 综合报告数据检查")
print("-" * 80)
try:
    df = pd.read_csv("step7_performance_analysis/output/comprehensive_report.csv")
    print("可用列:", df.columns.tolist())
    print("\n前3行数据:")
    print(df.head(3))
except Exception as e:
    print(f"✗ 加载失败: {e}")

print("\n" + "=" * 80)
print("诊断完成！")
print("=" * 80)
