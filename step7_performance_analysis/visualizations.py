"""
可视化模块
生成各种对比图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import config


def plot_latency_comparison(df: pd.DataFrame, save_path: str):
    """延迟对比图（条形图+误差线）"""
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    models = df['model'].values
    mean = df['latency_mean_ms'].values
    p95 = df['latency_p95_ms'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, mean, width, label='Mean', alpha=0.8)
    ax.bar(x + width/2, p95, width, label='P95', alpha=0.8)
    
    ax.axhline(y=100, color='r', linestyle='--', label='Real-time threshold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_latency(df: pd.DataFrame, save_path: str):
    """准确性 vs 延迟散点图"""
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    for _, row in df.iterrows():
        ax.scatter(
            row['latency_p95_ms'], 
            row['dr_5s']*100,
            s=row['params_millions']*50,  # 大小代表参数量
            alpha=0.6,
            label=row['model'].upper(),
            color=config.MODEL_COLORS.get(row['model'], '#333333')
        )
        
        ax.annotate(
            row['model'].upper(),
            (row['latency_p95_ms'], row['dr_5s']*100),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    ax.set_xlabel('P95 Latency (ms)')
    ax.set_ylabel('DR@5s (%)')
    ax.set_title('Accuracy vs Latency Trade-off\n(bubble size = parameters)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_params_vs_performance(df: pd.DataFrame, save_path: str):
    """参数量 vs 性能"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.FIGSIZE_WIDE)
    
    # 参数量 vs DR@5s
    ax1.scatter(df['params_millions'], df['dr_5s']*100, s=100, alpha=0.6)
    for _, row in df.iterrows():
        ax1.annotate(
            row['model'].upper(),
            (row['params_millions'], row['dr_5s']*100),
            fontsize=8
        )
    ax1.set_xlabel('Parameters (Millions)')
    ax1.set_ylabel('DR@5s (%)')
    ax1.set_title('Parameters vs Detection Rate')
    ax1.grid(True, alpha=0.3)
    
    # 参数量 vs 延迟
    ax2.scatter(df['params_millions'], df['latency_p95_ms'], s=100, alpha=0.6)
    for _, row in df.iterrows():
        ax2.annotate(
            row['model'].upper(),
            (row['params_millions'], row['latency_p95_ms']),
            fontsize=8
        )
    ax2.set_xlabel('Parameters (Millions)')
    ax2.set_ylabel('P95 Latency (ms)')
    ax2.set_title('Parameters vs Latency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_chart(df: pd.DataFrame, save_path: str):
    """雷达图（多维度对比）"""
    # 选择top 3模型
    top_models = df.nsmallest(3, 'rank')
    
    categories = ['DR@5s', 'Speed', 'Lightness', 'MTBFA']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE, subplot_kw=dict(projection='polar'))
    
    for _, row in top_models.iterrows():
        values = [
            row['dr_5s'],
            row['norm_latency'],
            row['norm_params'],
            min(row['mtbfa_hours'] / 2.0, 1.0)  # 归一化MTBFA
        ]
        values += values[:1]
        
        ax.plot(
            angles, values,
            'o-', linewidth=2,
            label=row['model'].upper(),
            color=config.MODEL_COLORS.get(row['model'], '#333333')
        )
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Dimensional Model Comparison\n(Top 3 Models)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_deployment_matrix(df: pd.DataFrame, save_path: str):
    """部署场景适配矩阵"""
    scenarios = list(config.DEPLOYMENT_SCENARIOS.keys())
    models = df['model'].values
    
    # 构建适配矩阵
    matrix = np.zeros((len(models), len(scenarios)))
    
    for i, model_name in enumerate(models):
        row = df[df['model'] == model_name].iloc[0]
        
        for j, scenario_name in enumerate(scenarios):
            constraints = config.DEPLOYMENT_SCENARIOS[scenario_name]['constraints']
            
            # 检查是否满足所有约束
            meets = (
                row['latency_p95_ms'] <= constraints['max_latency_ms'] and
                row['params_millions'] <= constraints['max_params_m'] and
                row['memory_mb'] <= constraints['max_memory_mb'] and
                row['dr_5s'] >= constraints['min_dr_5s']
            )
            
            matrix[i, j] = 1 if meets else 0
    
    fig, ax = plt.subplots(figsize=config.FIGSIZE_SINGLE)
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        xticklabels=[s.replace('_', ' ').title() for s in scenarios],
        yticklabels=[m.upper() for m in models],
        cbar_kws={'label': 'Suitable (1=Yes, 0=No)'},
        ax=ax
    )
    
    ax.set_title('Model-Scenario Compatibility Matrix')
    ax.set_xlabel('Deployment Scenario')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_visualizations(df: pd.DataFrame):
    """生成所有可视化"""
    print("\n" + "="*80)
    print("Generating Visualizations...")
    print("="*80)
    
    viz_dir = config.VISUALIZATION_DIR
    
    plots = [
        ('latency_comparison.png', plot_latency_comparison),
        ('accuracy_vs_latency.png', plot_accuracy_vs_latency),
        ('params_vs_performance.png', plot_params_vs_performance),
        ('radar_chart.png', plot_radar_chart),
        ('deployment_matrix.png', plot_deployment_matrix)
    ]
    
    for filename, plot_func in plots:
        save_path = os.path.join(viz_dir, filename)
        try:
            plot_func(df, save_path)
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
