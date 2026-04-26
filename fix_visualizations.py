"""
重新生成有问题的图表，并打印详细调试信息
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 配置路径
STEP6_METRICS = "step6_newMetrcisWithTimegan/output/time_aware_metrics"
STEP5_METRICS = "step5_timegan/output"
OUTPUT_DIR = "paper_figures"
STYLES_DIR = "styles"

# 模型颜色
MODEL_COLORS = {
    'cnn': '#1f77b4',
    'lstm': '#ff7f0e',
    'bilstm': '#2ca02c',
    'gru': '#d62728',
    'cnn_lstm': '#9467bd',
    'tcn': '#8c564b',
    'transformer': '#e377c2'
}

MODEL_NAMES = {
    'cnn': 'CNN',
    'lstm': 'LSTM',
    'bilstm': 'BiLSTM',
    'gru': 'GRU',
    'cnn_lstm': 'CNN-LSTM',
    'tcn': 'TCN',
    'transformer': 'Transformer'
}

print("=" * 80)
print("重新生成有问题的图表")
print("=" * 80)

# ===========================================================================
# FIG 2: DR@Δt曲线 - 检查GRU数据
# ===========================================================================
print("\n1. Fig2: DR@Δt曲线重新生成")
print("-" * 80)

df = pd.read_csv(f"{STEP6_METRICS}/overall_metrics.csv")
print(f"✓ 加载了 {len(df)} 个模型的数据")

plt.style.use(f'{STYLES_DIR}/sci_large_12px.mplstyle')
fig, ax = plt.subplots(figsize=(7, 4.2))

delta_t_cols = ['DR@1s', 'DR@2s', 'DR@5s', 'DR@10s', 'DR@15s', 'DR@30s']
delta_t_values = [1, 2, 5, 10, 15, 30]

for _, row in df.iterrows():
    model = row['model']
    dr_values = [row[col] for col in delta_t_cols]
    
    print(f"  {MODEL_NAMES[model]:12s}: DR@5s={row['DR@5s']:.4f}, DR@10s={row['DR@10s']:.4f}")
    
    ax.plot(delta_t_values, dr_values, 
            marker='o', linewidth=2, markersize=6,
            color=MODEL_COLORS[model], 
            label=MODEL_NAMES[model],
            alpha=0.9)

ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Safety Threshold')
ax.axvline(x=5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Target Time')

ax.set_xlabel('Time Threshold Δt (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Detection Rate DR@Δt', fontsize=12, fontweight='bold')
ax.set_title('Detection Rate vs Time Threshold', fontsize=12, fontweight='bold', pad=15)

ax.set_xlim(0, 32)
ax.set_ylim(0.75, 1.02)
ax.set_xticks(delta_t_values)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', framealpha=0.9, fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_dr_dt_curves_FIXED.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Fig2 已保存（带FIXED后缀）")

# ===========================================================================
# FIG 3: 传统指标 vs 时间感知指标
# ===========================================================================
print("\n\n2. Fig3: 传统指标 vs 时间感知指标")
print("-" * 80)

# 加载传统指标
traditional_metrics = {}
for model in ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']:
    with open(f'{STEP5_METRICS}/test_metrics_{model}.json', 'r') as f:
        metrics = json.load(f)
        traditional_metrics[model] = {
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0)
        }
        print(f"  {MODEL_NAMES[model]:12s}: F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

plt.style.use(f'{STYLES_DIR}/sci_large_12px.mplstyle')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = df['model'].tolist()
x_pos = np.arange(len(models))
width = 0.6

# 子图1: Precision & Recall
ax = axes[0, 0]
precision_vals = [traditional_metrics[m]['precision'] for m in models]
recall_vals = [traditional_metrics[m]['recall'] for m in models]

x_offset = np.arange(len(models))
ax.bar(x_offset - width/4, precision_vals, width/2, label='Precision', 
       color='#3498db', alpha=0.8, edgecolor='black')
ax.bar(x_offset + width/4, recall_vals, width/2, label='Recall', 
       color='#e74c3c', alpha=0.8, edgecolor='black')

ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('(a) Precision & Recall', fontsize=11, fontweight='bold')
ax.set_xticks(x_offset)
ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=45, ha='right')
ax.set_ylim(0.75, 1.0)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 子图2: F1-Score
ax = axes[0, 1]
f1_vals = [traditional_metrics[m]['f1'] for m in models]
bars = ax.bar(x_pos, f1_vals, width, color='#9b59b6', alpha=0.8, edgecolor='black')

for i, (bar, val) in enumerate(zip(bars, f1_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
           f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax.set_title('(b) F1-Score', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=45, ha='right')
ax.set_ylim(0.75, 1.0)
ax.grid(axis='y', alpha=0.3)

# 子图3: DR@5s
ax = axes[1, 0]
dr5s_vals = df['DR@5s'].tolist()
bars = ax.bar(x_pos, dr5s_vals, width, color='#27ae60', alpha=0.8, edgecolor='black')

for i, (bar, val) in enumerate(zip(bars, dr5s_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
           f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Target: 0.95')
ax.set_ylabel('Detection Rate', fontsize=11, fontweight='bold')
ax.set_title('(c) DR@5s', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=45, ha='right')
ax.set_ylim(0.75, 1.02)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 子图4: ADD
ax = axes[1, 1]
add_vals = df['ADD'].tolist()
bars = ax.bar(x_pos, add_vals, width, color='#f39c12', alpha=0.8, edgecolor='black')

for i, (bar, val) in enumerate(zip(bars, add_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
           f'{val:.2f}s', ha='center', va='bottom', fontsize=8)

ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Target: <0.5s')
ax.set_ylabel('Delay (seconds)', fontsize=11, fontweight='bold')
ax.set_title('(d) ADD', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=45, ha='right')
ax.set_ylim(0, 0.7)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Comparison: Traditional vs Time-Aware Metrics', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_traditional_vs_timeaware_FIXED.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Fig3 已保存（带FIXED后缀）")

# ===========================================================================
# FIG 10: 按攻击类型的性能
# ===========================================================================
print("\n\n3. Fig10: 按攻击类型的性能")
print("-" * 80)

plt.style.use(f'{STYLES_DIR}/sci_large_12px.mplstyle')

# 收集所有模型的per-attack数据
all_data = {}
attack_types = []

for model in models:
    df_attack = pd.read_csv(f"{STEP6_METRICS}/{model}_per_attack_metrics.csv")
    all_data[model] = {}
    
    print(f"  {MODEL_NAMES[model]:12s}:", end=" ")
    for _, row in df_attack.iterrows():
        attack = row['attack_type']
        dr5s = row['DR@5s']
        all_data[model][attack] = dr5s
        
        if attack not in attack_types:
            attack_types.append(attack)
        
        print(f"{attack}={dr5s:.3f}", end=", ")
    print()

print(f"\n  攻击类型: {attack_types}")

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(attack_types))
width = 0.12

for i, model in enumerate(models):
    values = [all_data[model].get(at, 0) for at in attack_types]
    offset = (i - len(models)/2) * width
    ax.bar(x + offset, values, width, label=MODEL_NAMES[model],
          color=MODEL_COLORS[model], alpha=0.8, edgecolor='black', linewidth=0.5)

# 平均线
avg_line = [np.mean([all_data[m].get(at, 0) for m in models if at in all_data[m]]) 
            for at in attack_types]
ax.plot(x, avg_line, 'k--', linewidth=2, label='Average', alpha=0.6)

ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Detection Rate @ 5s', fontsize=12, fontweight='bold')
ax.set_title('Performance across Attack Types', 
            fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([at.replace('_', ' ').title() for at in attack_types], fontsize=10)
ax.set_ylim(0.75, 1.05)
ax.legend(loc='lower right', fontsize=9, ncol=4)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig10_per_attack_performance_FIXED.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Fig10 已保存（带FIXED后缀）")

print("\n" + "=" * 80)
print("✓ 所有修正版图表已生成！请检查 *_FIXED.png 文件")
print("=" * 80)
