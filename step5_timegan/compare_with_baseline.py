"""
对比Step5(TimeGAN增强)与Step3b(Baseline)的性能
"""
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(step_name, model_type='lstm'):
    """加载指标"""
    if step_name == 'step3b':
        metrics_path = Path('../step3b_multiModelGeneral/output') / f'test_metrics_{model_type}.json'
    else:  # step5
        metrics_path = Path('./output') / f'test_metrics_{model_type}.json'
    
    if not metrics_path.exists():
        print(f"警告: 未找到 {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_data_stats(step_name):
    """加载数据集统计"""
    if step_name == 'step3b':
        base_path = Path('../step3b_multiModelGeneral/output')
    else:
        base_path = Path('./output')
    
    stats = {}
    
    # 加载窗口数据
    for split in ['train', 'val', 'test']:
        data_path = base_path / f'{split}_data.npz'
        if data_path.exists():
            data = np.load(data_path)
            stats[f'{split}_windows'] = len(data['y'])
            stats[f'{split}_attack_ratio'] = data['y'].mean()
    
    # 加载攻击信息
    for split in ['train', 'val', 'test']:
        info_path = base_path / f'{split}_attack_info.csv'
        if info_path.exists():
            df = pd.read_csv(info_path)
            stats[f'{split}_flights_attacked'] = df['flight_id'].nunique()
    
    return stats


def compare_metrics(model_type='lstm'):
    """对比两个实验的指标"""
    print("=" * 80)
    print(" " * 20 + "Step3b vs Step5 性能对比")
    print("=" * 80)
    print(f"模型类型: {model_type}")
    
    # 加载指标
    step3b_metrics = load_metrics('step3b', model_type)
    step5_metrics = load_metrics('step5', model_type)
    
    if step3b_metrics is None or step5_metrics is None:
        print("\n错误: 无法加载指标文件")
        return
    
    # 加载数据统计
    step3b_stats = load_data_stats('step3b')
    step5_stats = load_data_stats('step5')
    
    # 打印对比
    print("\n" + "=" * 80)
    print("数据集统计对比")
    print("=" * 80)
    print(f"\n{'指标':<30} {'Step3b (Baseline)':<20} {'Step5 (TimeGAN)':<20} {'变化':<15}")
    print("-" * 85)
    
    # 窗口数
    for split in ['train', 'val', 'test']:
        key = f'{split}_windows'
        if key in step3b_stats and key in step5_stats:
            s3b = step3b_stats[key]
            s5 = step5_stats[key]
            change = f"+{(s5/s3b-1)*100:.1f}%" if s5 > s3b else f"{(s5/s3b-1)*100:.1f}%"
            print(f"{split.capitalize()}窗口数{'':<30} {s3b:<20,} {s5:<20,} {change:<15}")
    
    print()
    
    # 攻击比例
    for split in ['train', 'val', 'test']:
        key = f'{split}_attack_ratio'
        if key in step3b_stats and key in step5_stats:
            s3b = step3b_stats[key] * 100
            s5 = step5_stats[key] * 100
            change = f"{s5-s3b:+.1f}pp"
            print(f"{split.capitalize()}攻击比例{'':<30} {s3b:<20.2f}% {s5:<20.2f}% {change:<15}")
    
    # 模型性能对比
    print("\n" + "=" * 80)
    print("模型性能对比")
    print("=" * 80)
    print(f"\n{'指标':<30} {'Step3b':<20} {'Step5':<20} {'变化':<15}")
    print("-" * 85)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'auc']
    
    for metric in metrics_to_compare:
        if metric in step3b_metrics and metric in step5_metrics:
            s3b = step3b_metrics[metric]
            s5 = step5_metrics[metric]
            
            if isinstance(s3b, (int, float)) and isinstance(s5, (int, float)):
                if metric == 'fpr':
                    # FPR越低越好
                    change = f"{(s5-s3b)*100:.2f}pp ({'✓' if s5 < s3b else '✗'})"
                else:
                    # 其他指标越高越好
                    change = f"{(s5-s3b)*100:.2f}pp ({'✓' if s5 > s3b else '✗'})"
                
                print(f"{metric.upper():<30} {s3b:<20.4f} {s5:<20.4f} {change:<15}")
    
    # 关键改进总结
    print("\n" + "=" * 80)
    print("关键改进总结")
    print("=" * 80)
    
    improvements = []
    
    # FPR改进
    if 'fpr' in step3b_metrics and 'fpr' in step5_metrics:
        fpr_reduction = (step3b_metrics['fpr'] - step5_metrics['fpr']) / step3b_metrics['fpr'] * 100
        improvements.append(f"✓ FPR降低: {fpr_reduction:.1f}%")
    
    # Recall变化
    if 'recall' in step3b_metrics and 'recall' in step5_metrics:
        recall_change = (step5_metrics['recall'] - step3b_metrics['recall']) / step3b_metrics['recall'] * 100
        if recall_change > -10:  # Recall下降不超过10%
            improvements.append(f"✓ Recall保持: {recall_change:+.1f}%")
        else:
            improvements.append(f"✗ Recall下降: {recall_change:.1f}%")
    
    # F1改进
    if 'f1' in step3b_metrics and 'f1' in step5_metrics:
        f1_change = (step5_metrics['f1'] - step3b_metrics['f1']) / step3b_metrics['f1'] * 100
        if f1_change > 0:
            improvements.append(f"✓ F1提升: {f1_change:.1f}%")
    
    print()
    for imp in improvements:
        print(f"  {imp}")
    
    # 保存对比报告
    report = {
        'step3b_metrics': step3b_metrics,
        'step5_metrics': step5_metrics,
        'step3b_stats': step3b_stats,
        'step5_stats': step5_stats,
        'improvements': improvements
    }
    
    report_path = Path('./output') / 'comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ 对比报告已保存: {report_path}")
    
    # 绘制对比图
    plot_comparison(step3b_metrics, step5_metrics, model_type)


def plot_comparison(step3b_metrics, step5_metrics, model_type):
    """绘制性能对比图"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    step3b_values = [step3b_metrics.get(m, 0) for m in metrics]
    step5_values = [step5_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, step3b_values, width, label='Step3b (Baseline)', alpha=0.8)
    ax.bar(x + width/2, step5_values, width, label='Step5 (TimeGAN)', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title(f'Step3b vs Step5 Performance Comparison ({model_type.upper()})')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path('./output') / f'comparison_{model_type}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='对比Step3b和Step5')
    parser.add_argument('--model', type=str, default='lstm',
                      choices=['lstm', 'gru', 'cnn_lstm', 'transformer'],
                      help='模型类型')
    
    args = parser.parse_args()
    
    compare_metrics(model_type=args.model)
