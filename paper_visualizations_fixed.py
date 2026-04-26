"""
论文核心图表生成脚本 - 完整修复版
生成高质量学术论文图表（SVG格式，300 DPI）
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import json

# 配置路径
STEP5_METRICS = "step5_timegan/output"  # 传统指标
STEP6_METRICS = "step6_newMetrcisWithTimegan/output/time_aware_metrics"  # 时间感知指标
STEP7_METRICS = "step7_performance_analysis/output"  # 综合报告
STYLES_DIR = "styles"
OUTPUT_DIR = "paper_figures"

# 创建输出目录
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(f"{OUTPUT_DIR}/svg").mkdir(exist_ok=True)
Path(f"{OUTPUT_DIR}/csv").mkdir(exist_ok=True)

# 模型颜色配置（色盲友好）
MODEL_COLORS = {
    'cnn': '#1f77b4',        # 蓝色
    'lstm': '#ff7f0e',       # 橙色
    'bilstm': '#2ca02c',     # 绿色
    'gru': '#d62728',        # 红色
    'cnn_lstm': '#9467bd',   # 紫色
    'tcn': '#8c564b',        # 棕色
    'transformer': '#e377c2' # 粉色
}

MODEL_NAMES_DISPLAY = {
    'cnn': 'CNN',
    'lstm': 'LSTM',
    'bilstm': 'BiLSTM',
    'gru': 'GRU',
    'cnn_lstm': 'CNN-LSTM',
    'tcn': 'TCN',
    'transformer': 'Transformer'
}

# 语义配色
SEMANTIC_COLORS = {
    'attack': '#FF6B6B',      # 红色（攻击）
    'normal': '#51CF66',      # 绿色（正常）
    'detection': '#339AF0',   # 蓝色（检测）
    'false_alarm': '#FFA94D', # 橙色（误报）
    'ideal': '#20C997',       # 青绿（理想）
    'warning': '#FFD43B'      # 黄色（警告）
}


def load_data():
    """加载所有需要的数据"""
    data = {}
    
    # Step6时间感知指标
    data['overall'] = pd.read_csv(f"{STEP6_METRICS}/overall_metrics.csv")
    
    # Step7综合报告
    data['comprehensive'] = pd.read_csv(f"{STEP7_METRICS}/comprehensive_report.csv")
    
    # 按攻击类型的指标
    per_attack = {}
    for model in ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']:
        try:
            per_attack[model] = pd.read_csv(f"{STEP6_METRICS}/{model}_per_attack_metrics.csv")
        except:
            pass
    data['per_attack'] = per_attack
    
    # 加载传统指标
    traditional = {}
    for model in ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']:
        try:
            with open(f'{STEP5_METRICS}/test_metrics_{model}.json', 'r') as f:
                traditional[model] = json.load(f)
        except Exception as e:
            print(f"警告: 无法加载 {model} 的传统指标: {e}")
            traditional[model] = {'precision': 0, 'recall': 0, 'f1': 0}
    data['traditional'] = traditional
    
    return data


# ===========================================================================
# Fig 2: 多模型DR@Δt曲线对比 ⭐⭐⭐⭐⭐
# ===========================================================================
def fig2_dr_dt_curves(data):
    """DR@Δt曲线对比图"""
    plt.style.use(f'{STYLES_DIR}/sci_large_12px.mplstyle')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    
    df = data['overall']
    delta_t_cols = ['DR@1s', 'DR@2s', 'DR@5s', 'DR@10s', 'DR@15s', 'DR@30s']
    delta_t_values = [1, 2, 5, 10, 15, 30]
    
    # 绘制每个模型的曲线
    for _, row in df.iterrows():
        model = row['model']
        dr_values = [row[col] for col in delta_t_cols]
        
        ax.plot(delta_t_values, dr_values, 
                marker='o', linewidth=2, markersize=6,
                color=MODEL_COLORS[model], 
                label=MODEL_NAMES_DISPLAY[model],
                alpha=0.9)
    
    # 关键阈值线
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Safety-Critical Threshold')
    ax.axvline(x=5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Target Response Time')
    
    # 样式设置
    ax.set_xlabel('Time Threshold Δt (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Rate DR@Δt', fontsize=12, fontweight='bold')
    ax.set_title('Detection Rate vs Time Threshold for Different Models', fontsize=12, fontweight='bold', pad=15)
    
    ax.set_xlim(0, 32)
    ax.set_ylim(0.75, 1.02)
    ax.set_xticks(delta_t_values)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/svg/fig2_dr_dt_curves.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig2_dr_dt_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 2: DR@Δt curves saved")


# ===========================================================================
# Fig 3: 传统指标 vs 时间感知指标对比 ⭐⭐⭐⭐⭐
# ===========================================================================
def fig3_traditional_vs_timeaware(data):
    """传统指标与时间感知指标对比"""
    plt.style.use(f'{STYLES_DIR}/sci_large_12px.mplstyle')
    
    df_time = data['overall']
    traditional_metrics = data['traditional']
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = df_time['model'].tolist()
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
    ax.set_title('(a) Traditional Metrics: Precision & Recall', fontsize=11, fontweight='bold')
    ax.set_xticks(x_offset)
    ax.set_xticklabels([MODEL_NAMES_DISPLAY[m] for m in models], rotation=45, ha='right')
    ax.set_ylim(0.75, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 子图2: F1-Score
    ax = axes[0, 1]
    f1_vals = [traditional_metrics[m]['f1'] for m in models]
    bars = ax.bar(x_pos, f1_vals, width, color='#9b59b6', alpha=0.8, edgecolor='black')
    
    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, f1_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('(b) Traditional Metric: F1-Score', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_NAMES_DISPLAY[m] for m in models], rotation=45, ha='right')
    ax.set_ylim(0.75, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # 子图3: DR@5s
    ax = axes[1, 0]
    dr5s_vals = df_time['DR@5s'].tolist()
    bars = ax.bar(x_pos, dr5s_vals, width, color='#27ae60', alpha=0.8, edgecolor='black')
    
    for i, (bar, val) in enumerate(zip(bars, dr5s_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Target: 0.95')
    ax.set_ylabel('Detection Rate', fontsize=11, fontweight='bold')
    ax.set_title('(c) Time-Aware Metric: DR@5s', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_NAMES_DISPLAY[m] for m in models], rotation=45, ha='right')
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 子图4: ADD
    ax = axes[1, 1]
    add_vals = df_time['ADD'].tolist()
    bars = ax.bar(x_pos, add_vals, width, color='#f39c12', alpha=0.8, edgecolor='black')
    
    for i, (bar, val) in enumerate(zip(bars, add_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Target: <0.5s')
    ax.set_ylabel('Delay (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('(d) Time-Aware Metric: ADD', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_NAMES_DISPLAY[m] for m in models], rotation=45, ha='right')
    ax.set_ylim(0, 0.7)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comparison: Traditional vs Time-Aware Metrics', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/svg/fig3_traditional_vs_timeaware.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig3_traditional_vs_timeaware.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 3: Traditional vs Time-aware comparison saved")


# ===========================================================================
# Fig 10: 按攻击类型的检测性能分组柱状图 ⭐⭐⭐⭐
# ===========================================================================
def fig10_per_attack_performance(data):
    """按攻击类型的DR@5s性能"""
    plt.style.use(f'{STYLES_DIR}/sci_large_12px.mplstyle')
    
    per_attack_data = data['per_attack']
    models = list(per_attack_data.keys())
    
    # 收集攻击类型和数据
    attack_types = []
    all_data = {}
    
    for model, df in per_attack_data.items():
        if df is not None and len(df) > 0:
            all_data[model] = {}
            for _, row in df.iterrows():
                attack = row['attack_type']
                if attack not in attack_types:
                    attack_types.append(attack)
                all_data[model][attack] = row['DR@5s']
    
    # 准备绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(attack_types))
    width = 0.12
    
    # 为每个模型绘制柱状图
    for i, model in enumerate(models):
        if model in all_data:
            values = [all_data[model].get(at, 0) for at in attack_types]
            offset = (i - len(models)/2) * width
            ax.bar(x + offset, values, width, label=MODEL_NAMES_DISPLAY[model],
                  color=MODEL_COLORS[model], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 平均线
    avg_line = [np.mean([all_data[m].get(at, 0) for m in all_data.keys() if at in all_data[m]]) 
                for at in attack_types]
    ax.plot(x, avg_line, 'k--', linewidth=2, label='Average', alpha=0.6)
    
    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Rate @ 5s', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance across Different Attack Types', 
                fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([at.replace('_', ' ').title() for at in attack_types], fontsize=10)
    ax.set_ylim(0.75, 1.05)
    ax.legend(loc='lower right', fontsize=9, ncol=4)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/svg/fig10_per_attack_performance.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig10_per_attack_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 10: Per-attack performance saved")


# ===========================================================================
# 主执行函数
# ===========================================================================
def main():
    print("=" * 80)
    print("论文核心图表生成 - 完整修复版")
    print("=" * 80)
    print("\n加载数据...")
    data = load_data()
    print("✓ 数据加载完成\n")
    
    print("生成图表...")
    print("-" * 80)
    
    fig2_dr_dt_curves(data)
    fig3_traditional_vs_timeaware(data)
    fig10_per_attack_performance(data)
    
    print("-" * 80)
    print("\n✓ 所有图表已生成!")
    print(f"  - SVG文件: {OUTPUT_DIR}/svg/")
    print(f"  - PNG文件: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
