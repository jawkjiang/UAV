"""
额外可视化：Fig 11 检测延迟分布箱线图
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

STYLES_DIR = "styles"
OUTPUT_DIR = "paper_figures"

MODEL_COLORS = {
    'cnn': '#1f77b4',
    'lstm': '#ff7f0e',
    'bilstm': '#2ca02c',
    'gru': '#d62728',
    'cnn_lstm': '#9467bd',
    'tcn': '#8c564b',
    'transformer': '#e377c2'
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

def fig11_delay_boxplot():
    """检测延迟分布箱线图"""
    plt.style.use(f'{STYLES_DIR}/sci_large_12px.mplstyle')
    
    # 加载数据
    df = pd.read_csv('step6_newMetrcisWithTimegan/output/time_aware_metrics/detailed_delays.csv')
    
    # 只保留检测成功的记录
    df = df[df['detected'] == True].copy()
    
    # 准备数据
    models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
    data_to_plot = []
    labels = []
    colors = []
    
    for model in models:
        model_data = df[df['model'] == model]['delay'].values
        if len(model_data) > 0:
            data_to_plot.append(model_data)
            labels.append(MODEL_NAMES_DISPLAY[model])
            colors.append(MODEL_COLORS[model])
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制箱线图
    bp = ax.boxplot(data_to_plot, 
                     labels=labels,
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=6, 
                                   markeredgecolor='darkred', markeredgewidth=1.5),
                     medianprops=dict(color='black', linewidth=2),
                     boxprops=dict(linewidth=1.5),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    # 为每个箱子填充颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # 添加统计信息
    for i, (model_data, label) in enumerate(zip(data_to_plot, labels), 1):
        median = np.median(model_data)
        mean = np.mean(model_data)
        # 在箱子上方标注中位数
        ax.text(i, median + 0.1, f'{median:.2f}s', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 样式设置
    ax.set_ylabel('Detection Delay (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Detection Delays across Models\n(Boxplot with Mean ◆ and Median —)', 
                fontsize=12, fontweight='bold', pad=15)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(-0.2, max([max(d) for d in data_to_plot]) * 1.15)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='Mean',
              markerfacecolor='red', markersize=8, markeredgecolor='darkred'),
        Line2D([0], [0], color='black', linewidth=2, label='Median')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/svg/fig11_delay_boxplot.svg', dpi=300, format='svg', bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/fig11_delay_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fig 11: Delay distribution boxplot saved")


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "ADDITIONAL VISUALIZATION: Fig 11")
    print("="*70)
    fig11_delay_boxplot()
    print("\n✅ Completed!")
    print(f"📁 Output: {OUTPUT_DIR}/svg/fig11_delay_boxplot.svg\n")
