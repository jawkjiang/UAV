"""
Model Comparison and Analysis Tool

Generates detailed comparison tables and visualizations for all experiments.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_all_results(output_dir='./output'):
    """
    Load all experiment results from the output directory.
    
    Returns:
        DataFrame with all results
    """
    results = []
    
    output_path = Path(output_dir)
    
    # Iterate through all attack type directories
    for attack_dir in output_path.iterdir():
        if not attack_dir.is_dir() or attack_dir.name == '__pycache__':
            continue
        
        attack_type = attack_dir.name
        
        # Iterate through all model directories
        for model_dir in attack_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == '__pycache__':
                continue
            
            model_type = model_dir.name
            
            # Load metrics
            metrics_path = model_dir / 'test_metrics.json'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Load training history
                history_path = model_dir / 'training_history.json'
                training_time = None
                best_epoch = None
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    best_epoch = history.get('best_epoch', None)
                
                results.append({
                    'attack_type': attack_type,
                    'model_type': model_type,
                    'auc_roc': metrics['roc_auc'],
                    'auc_pr': metrics['pr_auc'],
                    'f1_score': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'best_epoch': best_epoch
                })
    
    return pd.DataFrame(results)


def create_comparison_tables(df, output_dir='./output'):
    """Create detailed comparison tables."""
    
    print("\n" + "="*80)
    print("Creating Comparison Tables")
    print("="*80)
    
    # 1. Pivot table: Attack x Model (AUC-ROC)
    pivot_auc = df.pivot(index='attack_type', columns='model_type', values='auc_roc')
    pivot_auc = pivot_auc.round(4)
    pivot_auc['mean'] = pivot_auc.mean(axis=1)
    pivot_auc.loc['mean'] = pivot_auc.mean(axis=0)
    
    pivot_auc.to_csv(os.path.join(output_dir, 'comparison_auc_roc.csv'))
    print("\nAUC-ROC Comparison (Attack x Model):")
    print(pivot_auc.to_string())
    
    # 2. Pivot table: Attack x Model (F1 Score)
    pivot_f1 = df.pivot(index='attack_type', columns='model_type', values='f1_score')
    pivot_f1 = pivot_f1.round(4)
    pivot_f1['mean'] = pivot_f1.mean(axis=1)
    pivot_f1.loc['mean'] = pivot_f1.mean(axis=0)
    
    pivot_f1.to_csv(os.path.join(output_dir, 'comparison_f1_score.csv'))
    print("\nF1 Score Comparison (Attack x Model):")
    print(pivot_f1.to_string())
    
    # 3. Model rankings
    model_rankings = df.groupby('model_type').agg({
        'auc_roc': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max']
    }).round(4)
    model_rankings = model_rankings.sort_values(('auc_roc', 'mean'), ascending=False)
    
    model_rankings.to_csv(os.path.join(output_dir, 'model_rankings.csv'))
    print("\nModel Rankings:")
    print(model_rankings.to_string())
    
    # 4. Attack difficulty rankings
    attack_rankings = df.groupby('attack_type').agg({
        'auc_roc': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max']
    }).round(4)
    attack_rankings = attack_rankings.sort_values(('auc_roc', 'mean'), ascending=True)
    
    attack_rankings.to_csv(os.path.join(output_dir, 'attack_difficulty.csv'))
    print("\nAttack Difficulty (sorted by detection difficulty):")
    print(attack_rankings.to_string())
    
    # 5. Best model for each attack
    best_models = df.loc[df.groupby('attack_type')['auc_roc'].idxmax()]
    best_models = best_models[['attack_type', 'model_type', 'auc_roc', 'f1_score']]
    best_models = best_models.sort_values('auc_roc', ascending=False)
    
    best_models.to_csv(os.path.join(output_dir, 'best_model_per_attack.csv'), index=False)
    print("\nBest Model for Each Attack:")
    print(best_models.to_string(index=False))
    
    return pivot_auc, pivot_f1, model_rankings, attack_rankings


def create_visualizations(df, output_dir='./output'):
    """Create visualization plots."""
    
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Heatmap: AUC-ROC
    plt.figure(figsize=(14, 8))
    pivot_auc = df.pivot(index='attack_type', columns='model_type', values='auc_roc')
    sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                cbar_kws={'label': 'AUC-ROC'})
    plt.title('AUC-ROC Comparison: Models vs Attack Types', fontsize=16, fontweight='bold')
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Attack Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'heatmap_auc_roc.png'), dpi=300)
    print(f"Saved: heatmap_auc_roc.png")
    plt.close()
    
    # 2. Heatmap: F1 Score
    plt.figure(figsize=(14, 8))
    pivot_f1 = df.pivot(index='attack_type', columns='model_type', values='f1_score')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                cbar_kws={'label': 'F1 Score'})
    plt.title('F1 Score Comparison: Models vs Attack Types', fontsize=16, fontweight='bold')
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Attack Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'heatmap_f1_score.png'), dpi=300)
    print(f"Saved: heatmap_f1_score.png")
    plt.close()
    
    # 3. Box plot: Model performance distribution
    plt.figure(figsize=(14, 6))
    df_sorted = df.sort_values('model_type')
    sns.boxplot(data=df_sorted, x='model_type', y='auc_roc', palette='Set2')
    plt.title('AUC-ROC Distribution by Model Type', fontsize=16, fontweight='bold')
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('AUC-ROC', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'boxplot_models.png'), dpi=300)
    print(f"Saved: boxplot_models.png")
    plt.close()
    
    # 4. Box plot: Attack difficulty distribution
    plt.figure(figsize=(14, 6))
    df_sorted = df.sort_values('attack_type')
    sns.boxplot(data=df_sorted, x='attack_type', y='auc_roc', palette='Set3')
    plt.title('AUC-ROC Distribution by Attack Type', fontsize=16, fontweight='bold')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('AUC-ROC', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'boxplot_attacks.png'), dpi=300)
    print(f"Saved: boxplot_attacks.png")
    plt.close()
    
    # 5. Bar chart: Average performance by model
    plt.figure(figsize=(12, 6))
    model_avg = df.groupby('model_type')[['auc_roc', 'f1_score']].mean().sort_values('auc_roc', ascending=False)
    x = np.arange(len(model_avg))
    width = 0.35
    
    plt.bar(x - width/2, model_avg['auc_roc'], width, label='AUC-ROC', alpha=0.8)
    plt.bar(x + width/2, model_avg['f1_score'], width, label='F1 Score', alpha=0.8)
    
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Average Performance by Model Type', fontsize=16, fontweight='bold')
    plt.xticks(x, model_avg.index, rotation=45)
    plt.legend()
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'bar_model_performance.png'), dpi=300)
    print(f"Saved: bar_model_performance.png")
    plt.close()
    
    # 6. Scatter plot: Precision vs Recall
    plt.figure(figsize=(12, 8))
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model]
        plt.scatter(model_data['recall'], model_data['precision'], 
                   label=model, s=100, alpha=0.7)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision vs Recall by Model Type', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 1.05)
    plt.ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'scatter_precision_recall.png'), dpi=300)
    print(f"Saved: scatter_precision_recall.png")
    plt.close()
    
    print(f"\nAll visualizations saved to: {viz_dir}")


def generate_markdown_report(df, output_dir='./output'):
    """Generate a comprehensive markdown report."""
    
    print("\n" + "="*80)
    print("Generating Markdown Report")
    print("="*80)
    
    report = []
    report.append("# Multi-Model Multi-Attack GPS Spoofing Detection")
    report.append("\n## Experiment Overview\n")
    report.append(f"- Total experiments: {len(df)}")
    report.append(f"- Attack types tested: {df['attack_type'].nunique()}")
    report.append(f"- Model architectures tested: {df['model_type'].nunique()}")
    
    # Best overall
    best_overall = df.loc[df['auc_roc'].idxmax()]
    report.append("\n## Best Overall Configuration\n")
    report.append(f"- **Attack Type:** {best_overall['attack_type']}")
    report.append(f"- **Model Type:** {best_overall['model_type']}")
    report.append(f"- **AUC-ROC:** {best_overall['auc_roc']:.4f}")
    report.append(f"- **F1 Score:** {best_overall['f1_score']:.4f}")
    
    # Model rankings
    report.append("\n## Model Rankings\n")
    model_avg = df.groupby('model_type')[['auc_roc', 'f1_score']].mean().sort_values('auc_roc', ascending=False)
    report.append("| Rank | Model | Avg AUC-ROC | Avg F1 Score |")
    report.append("|------|-------|-------------|--------------|")
    for i, (model, row) in enumerate(model_avg.iterrows(), 1):
        report.append(f"| {i} | {model} | {row['auc_roc']:.4f} | {row['f1_score']:.4f} |")
    
    # Attack difficulty
    report.append("\n## Attack Detection Difficulty\n")
    report.append("(Sorted by detection difficulty - lower AUC-ROC = harder to detect)\n")
    attack_avg = df.groupby('attack_type')[['auc_roc', 'f1_score']].mean().sort_values('auc_roc', ascending=True)
    report.append("| Rank | Attack Type | Avg AUC-ROC | Avg F1 Score |")
    report.append("|------|-------------|-------------|--------------|")
    for i, (attack, row) in enumerate(attack_avg.iterrows(), 1):
        report.append(f"| {i} | {attack} | {row['auc_roc']:.4f} | {row['f1_score']:.4f} |")
    
    # Best model per attack
    report.append("\n## Best Model for Each Attack Type\n")
    report.append("| Attack Type | Best Model | AUC-ROC | F1 Score |")
    report.append("|-------------|------------|---------|----------|")
    for attack in df['attack_type'].unique():
        attack_data = df[df['attack_type'] == attack]
        best = attack_data.loc[attack_data['auc_roc'].idxmax()]
        report.append(f"| {attack} | {best['model_type']} | {best['auc_roc']:.4f} | {best['f1_score']:.4f} |")
    
    # Visualizations
    report.append("\n## Visualizations\n")
    report.append("See the `visualizations/` directory for detailed plots:")
    report.append("- `heatmap_auc_roc.png` - AUC-ROC heatmap")
    report.append("- `heatmap_f1_score.png` - F1 Score heatmap")
    report.append("- `boxplot_models.png` - Model performance distributions")
    report.append("- `boxplot_attacks.png` - Attack difficulty distributions")
    report.append("- `bar_model_performance.png` - Average model performance")
    report.append("- `scatter_precision_recall.png` - Precision-Recall tradeoffs")
    
    # Save report
    report_text = '\n'.join(report)
    report_path = os.path.join(output_dir, 'ANALYSIS_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nMarkdown report saved to: {report_path}")
    
    return report_text


def main():
    """Main execution."""
    
    print("="*80)
    print("Model Comparison and Analysis")
    print("="*80)
    
    output_dir = './output'
    
    # Load results
    print("\nLoading results...")
    df = load_all_results(output_dir)
    
    if len(df) == 0:
        print("ERROR: No results found. Please run main.py first.")
        return
    
    print(f"Loaded {len(df)} experiment results")
    
    # Create comparison tables
    create_comparison_tables(df, output_dir)
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Generate markdown report
    report = generate_markdown_report(df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    print(f"\nResults saved to: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
