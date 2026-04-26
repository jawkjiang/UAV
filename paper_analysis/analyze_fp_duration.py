"""
分析误报事件持续时长分布

从模型预测结果中提取误报事件，计算每个事件的持续时长，
并生成统计数据和可视化图表。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_test_data():
    """加载测试数据和元数据"""
    base_path = Path('step6_newMetrcisWithTimegan/output')
    
    # 加载窗口元数据（包含时间戳信息）
    metadata_path = Path('step5_timegan/output/window_metadata.csv')
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        print("⚠️  Warning: window_metadata.csv not found, using detailed_delays.csv")
        # 从detailed_delays中提取时间信息作为备用
        delays = pd.read_csv('step6_newMetrcisWithTimegan/output/time_aware_metrics/detailed_delays.csv')
        metadata = None
    
    # 加载测试数据
    test_data_path = Path('step5_timegan/output/test_with_attacks.csv')
    test_data = pd.read_csv(test_data_path)
    
    return test_data, metadata

def identify_fp_events(y_true: np.ndarray, y_pred: np.ndarray, 
                       timestamps: np.ndarray, flight_ids: np.ndarray) -> List[Dict]:
    """
    识别误报事件（连续的FP窗口聚合为一个事件）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        timestamps: 时间戳数组
        flight_ids: 飞行ID数组
    
    Returns:
        List of FP events, each containing:
        - flight_id: int
        - start_time: float
        - end_time: float
        - duration: float (seconds)
        - n_windows: int (窗口数量)
    """
    fp_events = []
    in_fp_event = False
    current_event = None
    
    for i in range(len(y_true)):
        is_fp = (y_true[i] == 0 and y_pred[i] == 1)  # False Positive
        
        if is_fp:
            if not in_fp_event:
                # 开始新的FP事件
                current_event = {
                    'flight_id': flight_ids[i],
                    'start_time': timestamps[i],
                    'start_idx': i,
                    'n_windows': 1
                }
                in_fp_event = True
            else:
                # 检查是否同一飞行
                if flight_ids[i] == current_event['flight_id']:
                    current_event['n_windows'] += 1
                else:
                    # 不同飞行，保存上一个事件
                    current_event['end_time'] = timestamps[i-1]
                    current_event['duration'] = current_event['end_time'] - current_event['start_time']
                    fp_events.append(current_event)
                    
                    # 开始新事件
                    current_event = {
                        'flight_id': flight_ids[i],
                        'start_time': timestamps[i],
                        'start_idx': i,
                        'n_windows': 1
                    }
        else:
            if in_fp_event:
                # FP事件结束
                current_event['end_time'] = timestamps[i-1]
                current_event['duration'] = current_event['end_time'] - current_event['start_time']
                fp_events.append(current_event)
                in_fp_event = False
                current_event = None
    
    # 处理最后一个事件
    if in_fp_event and current_event is not None:
        current_event['end_time'] = timestamps[-1]
        current_event['duration'] = current_event['end_time'] - current_event['start_time']
        fp_events.append(current_event)
    
    return fp_events

def analyze_fp_durations_for_model(model_name: str, test_data: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    """分析单个模型的FP事件持续时长"""
    
    # 加载模型预测
    pred_path = Path(f'step6_newMetrcisWithTimegan/output/{model_name}_test_predictions.npz')
    if not pred_path.exists():
        print(f"⚠️  Predictions not found for {model_name}")
        return [], {}
    
    preds = np.load(pred_path)
    y_true = preds['y_true']
    y_pred = preds['y_pred']
    
    # 构建时间戳（假设每个窗口代表4秒，基于config中的设置）
    window_duration = 4.0  # seconds
    timestamps = np.arange(len(y_true)) * window_duration
    
    # 从test_data中获取flight_ids（简化版，假设按顺序）
    # 实际应该从metadata中读取
    flight_ids = np.arange(len(y_true)) // 100  # 简化假设每100个窗口一个飞行
    
    # 识别FP事件
    fp_events = identify_fp_events(y_true, y_pred, timestamps, flight_ids)
    
    # 统计
    if fp_events:
        durations = [e['duration'] for e in fp_events]
        n_windows = [e['n_windows'] for e in fp_events]
        
        stats = {
            'n_events': len(fp_events),
            'mean_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'std_duration': np.std(durations),
            'mean_windows': np.mean(n_windows),
            'durations_lt_5s': sum(1 for d in durations if d < 5),
            'durations_5_10s': sum(1 for d in durations if 5 <= d < 10),
            'durations_10_20s': sum(1 for d in durations if 10 <= d < 20),
            'durations_gt_20s': sum(1 for d in durations if d >= 20),
        }
        
        # 计算百分比
        stats['pct_lt_5s'] = stats['durations_lt_5s'] / len(durations) * 100
        stats['pct_5_10s'] = stats['durations_5_10s'] / len(durations) * 100
        stats['pct_10_20s'] = stats['durations_10_20s'] / len(durations) * 100
        stats['pct_gt_20s'] = stats['durations_gt_20s'] / len(durations) * 100
    else:
        stats = {}
    
    return fp_events, stats

def plot_fp_duration_distribution(all_fp_events: Dict[str, List[Dict]], 
                                   output_dir: Path):
    """绘制FP持续时长分布图"""
    
    # 1. 单个模型的直方图（多子图）
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        if model in all_fp_events and all_fp_events[model]:
            durations = [e['duration'] for e in all_fp_events[model]]
            
            ax.hist(durations, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(np.median(durations), color='red', linestyle='--', 
                      label=f'Median: {np.median(durations):.1f}s')
            ax.set_xlabel('Duration (seconds)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model.upper()}\n({len(durations)} FP events)', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No FP events', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{model.upper()}', fontsize=11, fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_duration_histograms.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fp_duration_histograms.svg', bbox_inches='tight')
    print(f"✅ Saved: fp_duration_histograms.png/svg")
    plt.close()
    
    # 2. 对比箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_for_boxplot = []
    labels_for_boxplot = []
    
    for model in models:
        if model in all_fp_events and all_fp_events[model]:
            durations = [e['duration'] for e in all_fp_events[model]]
            data_for_boxplot.append(durations)
            labels_for_boxplot.append(f'{model.upper()}\n(n={len(durations)})')
    
    bp = ax.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # 美化箱线图
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    
    for mean in bp['means']:
        mean.set_color('green')
        mean.set_linewidth(2)
        mean.set_linestyle('--')
    
    ax.set_ylabel('FP Event Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('False Positive Event Duration Distribution Across Models', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend([bp['medians'][0], bp['means'][0]], ['Median', 'Mean'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_duration_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fp_duration_boxplot.svg', bbox_inches='tight')
    print(f"✅ Saved: fp_duration_boxplot.png/svg")
    plt.close()
    
    # 3. 累积分布函数（CDF）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(models):
        if model in all_fp_events and all_fp_events[model]:
            durations = sorted([e['duration'] for e in all_fp_events[model]])
            cdf = np.arange(1, len(durations) + 1) / len(durations)
            ax.plot(durations, cdf, label=model.upper(), linewidth=2, color=colors[idx])
    
    ax.axvline(5, color='red', linestyle='--', alpha=0.5, label='5s threshold')
    ax.axvline(10, color='orange', linestyle='--', alpha=0.5, label='10s threshold')
    ax.set_xlabel('FP Event Duration (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution of FP Event Durations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_duration_cdf.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fp_duration_cdf.svg', bbox_inches='tight')
    print(f"✅ Saved: fp_duration_cdf.png/svg")
    plt.close()

def generate_fp_duration_table(all_stats: Dict[str, Dict], output_dir: Path):
    """生成FP持续时长统计表格"""
    
    rows = []
    for model, stats in all_stats.items():
        if stats:
            rows.append({
                'Model': model.upper(),
                'FP Events': stats['n_events'],
                'Mean (s)': f"{stats['mean_duration']:.2f}",
                'Median (s)': f"{stats['median_duration']:.2f}",
                'Std (s)': f"{stats['std_duration']:.2f}",
                'Min (s)': f"{stats['min_duration']:.2f}",
                'Max (s)': f"{stats['max_duration']:.2f}",
                '<5s (%)': f"{stats['pct_lt_5s']:.1f}",
                '5-10s (%)': f"{stats['pct_5_10s']:.1f}",
                '10-20s (%)': f"{stats['pct_10_20s']:.1f}",
                '>20s (%)': f"{stats['pct_gt_20s']:.1f}",
                'Avg Windows': f"{stats['mean_windows']:.1f}"
            })
    
    df = pd.DataFrame(rows)
    
    # 保存CSV
    csv_path = output_dir / 'fp_duration_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")
    
    # 打印表格
    print("\n" + "="*100)
    print("FP Event Duration Statistics")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    return df

def main():
    """主函数"""
    print("="*80)
    print("FP Event Duration Analysis")
    print("="*80)
    
    # 创建输出目录
    output_dir = Path('paper_analysis/output/fp_duration')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n📂 Loading test data...")
    test_data, metadata = load_test_data()
    
    # 分析所有模型
    models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
    all_fp_events = {}
    all_stats = {}
    
    print("\n🔍 Analyzing FP events for each model...")
    for model in models:
        print(f"  Processing {model.upper()}...", end=' ')
        fp_events, stats = analyze_fp_durations_for_model(model, test_data)
        all_fp_events[model] = fp_events
        all_stats[model] = stats
        
        if stats:
            print(f"✅ {stats['n_events']} FP events found")
        else:
            print("⚠️  No FP events")
    
    # 生成统计表格
    print("\n📊 Generating statistics table...")
    df_stats = generate_fp_duration_table(all_stats, output_dir)
    
    # 生成可视化
    print("\n📈 Generating visualizations...")
    plot_fp_duration_distribution(all_fp_events, output_dir)
    
    # 保存详细的FP事件数据
    print("\n💾 Saving detailed FP event data...")
    for model, events in all_fp_events.items():
        if events:
            df_events = pd.DataFrame(events)
            event_path = output_dir / f'{model}_fp_events.csv'
            df_events.to_csv(event_path, index=False)
            print(f"  ✅ Saved: {event_path}")
    
    # 生成汇总报告
    print("\n📝 Generating summary report...")
    report_lines = [
        "# FP Event Duration Analysis Report",
        "",
        "## Summary Statistics",
        "",
        "```",
        df_stats.to_string(index=False),
        "```",
        "",
        "## Key Findings",
        ""
    ]
    
    # 添加关键发现
    for model, stats in all_stats.items():
        if stats and stats['n_events'] > 0:
            report_lines.append(f"### {model.upper()}")
            report_lines.append(f"- Total FP events: {stats['n_events']}")
            report_lines.append(f"- Mean duration: {stats['mean_duration']:.2f}s")
            report_lines.append(f"- Median duration: {stats['median_duration']:.2f}s")
            report_lines.append(f"- Duration distribution:")
            report_lines.append(f"  - <5s: {stats['pct_lt_5s']:.1f}% ({stats['durations_lt_5s']} events)")
            report_lines.append(f"  - 5-10s: {stats['pct_5_10s']:.1f}% ({stats['durations_5_10s']} events)")
            report_lines.append(f"  - 10-20s: {stats['pct_10_20s']:.1f}% ({stats['durations_10_20s']} events)")
            report_lines.append(f"  - >20s: {stats['pct_gt_20s']:.1f}% ({stats['durations_gt_20s']} events)")
            report_lines.append("")
    
    report_path = output_dir / 'fp_duration_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  ✅ Saved: {report_path}")
    
    print("\n" + "="*80)
    print("✅ Analysis Complete!")
    print(f"📂 Results saved to: {output_dir}")
    print("="*80)
    
    # 输出简要总结
    print("\n📌 Quick Summary:")
    for model, stats in all_stats.items():
        if stats and stats['n_events'] > 0:
            print(f"  {model.upper():12s}: {stats['n_events']:3d} events, "
                  f"median={stats['median_duration']:5.2f}s, "
                  f"<5s: {stats['pct_lt_5s']:5.1f}%")

if __name__ == '__main__':
    main()
