"""
Model Comparison and Analysis for Mixed Attack Training

Analyzes and visualizes the performance of all models trained on mixed attacks.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config


def load_all_results(output_dir):
    """Load test metrics and per-attack metrics for all models."""
    
    results = []
    
    for model_type in config.MODEL_TYPES:
        model_dir = os.path.join(output_dir, model_type)
        
        # Load overall metrics
        metrics_path = os.path.join(model_dir, 'test_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"Warning: Metrics not found for {model_type}")
            continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load per-attack metrics
        per_attack_path = os.path.join(model_dir, 'per_attack_metrics.json')
        if os.path.exists(per_attack_path):
            with open(per_attack_path, 'r') as f:
                per_attack_metrics = json.load(f)
        else:
            per_attack_metrics = {}
        
        results.append({
            'model': model_type,
            'overall_metrics': metrics,
            'per_attack_metrics': per_attack_metrics
        })
    
    return results


def create_overall_comparison(results):
    """Create overall comparison table."""
    
    rows = []
    for result in results:
        model = result['model']
        metrics = result['overall_metrics']
        
        rows.append({
            'Model': model,
            'AUC-ROC': metrics['auc_roc'],
            'F1 Score': metrics['f1_score'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Accuracy': metrics.get('accuracy', np.nan)
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('AUC-ROC', ascending=False)
    
    return df


def create_per_attack_comparison(results):
    """Create per-attack comparison matrix."""
    
    # Create matrix: models × attacks
    models = [r['model'] for r in results]
    attacks = config.ATTACK_TYPES
    
    # AUC-ROC matrix
    auc_matrix = []
    f1_matrix = []
    
    for result in results:
        model = result['model']
        per_attack = result['per_attack_metrics']
        
        auc_row = []
        f1_row = []
        
        for attack in attacks:
            if attack in per_attack:
                auc_row.append(per_attack[attack]['auc_roc'])
                f1_row.append(per_attack[attack]['f1_score'])
            else:
                auc_row.append(np.nan)
                f1_row.append(np.nan)
        
        auc_matrix.append(auc_row)
        f1_matrix.append(f1_row)
    
    auc_df = pd.DataFrame(auc_matrix, index=models, columns=attacks)
    f1_df = pd.DataFrame(f1_matrix, index=models, columns=attacks)
    
    return auc_df, f1_df


def compute_generalization_metrics(results):
    """Compute generalization ability for each model."""
    
    rows = []
    
    for result in results:
        model = result['model']
        per_attack = result['per_attack_metrics']
        
        # Collect AUC-ROC for all attacks
        aucs = []
        f1s = []
        
        for attack in config.ATTACK_TYPES:
            if attack in per_attack:
                aucs.append(per_attack[attack]['auc_roc'])
                f1s.append(per_attack[attack]['f1_score'])
        
        if len(aucs) > 0:
            rows.append({
                'Model': model,
                'Mean_AUC': np.mean(aucs),
                'Std_AUC': np.std(aucs),
                'Min_AUC': np.min(aucs),
                'Max_AUC': np.max(aucs),
                'Mean_F1': np.mean(f1s),
                'Std_F1': np.std(f1s),
                'Min_F1': np.min(f1s),
                'Max_F1': np.max(f1s),
                'N_Attacks': len(aucs)
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Std_AUC')  # Lower std = better generalization
    
    return df


def plot_heatmap(df, title, filename, output_dir):
    """Plot heatmap for model × attack matrix."""
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'Score'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_generalization_boxplot(results, output_dir):
    """Plot boxplot showing generalization ability."""
    
    data = []
    
    for result in results:
        model = result['model']
        per_attack = result['per_attack_metrics']
        
        for attack in config.ATTACK_TYPES:
            if attack in per_attack:
                data.append({
                    'Model': model,
                    'Attack': attack,
                    'AUC-ROC': per_attack[attack]['auc_roc'],
                    'F1 Score': per_attack[attack]['f1_score']
                })
    
    df = pd.DataFrame(data)
    
    # Plot AUC-ROC
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # AUC-ROC boxplot
    df_pivot = df.pivot(index='Attack', columns='Model', values='AUC-ROC')
    df_pivot.plot(kind='box', ax=axes[0])
    axes[0].set_title('Model Generalization - AUC-ROC Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('AUC-ROC', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.5, 1.05])
    
    # F1 Score boxplot
    df_pivot = df.pivot(index='Attack', columns='Model', values='F1 Score')
    df_pivot.plot(kind='box', ax=axes[1])
    axes[1].set_title('Model Generalization - F1 Score Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generalization_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: generalization_boxplot.png")


def plot_radar_chart(results, output_dir):
    """Plot radar chart for each model showing per-attack performance."""
    
    attacks = config.ATTACK_TYPES
    n_attacks = len(attacks)
    
    angles = np.linspace(0, 2 * np.pi, n_attacks, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        model = result['model']
        per_attack = result['per_attack_metrics']
        
        # Collect AUC values
        values = []
        for attack in attacks:
            if attack in per_attack:
                values.append(per_attack[attack]['auc_roc'])
            else:
                values.append(0.0)
        
        values += values[:1]  # Close the plot
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([a.replace('_', '\n') for a in attacks], fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(model.upper(), fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Per-Attack Performance Radar Charts (AUC-ROC)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_charts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: radar_charts.png")


def plot_model_ranking(overall_df, output_dir):
    """Plot bar chart of model rankings."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # AUC-ROC ranking
    df_sorted = overall_df.sort_values('AUC-ROC', ascending=True)
    df_sorted.plot(x='Model', y='AUC-ROC', kind='barh', ax=axes[0], 
                   color='skyblue', legend=False)
    axes[0].set_title('Model Ranking by AUC-ROC', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('AUC-ROC', fontsize=12)
    axes[0].set_ylabel('Model', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_xlim([0.5, 1.0])
    
    # F1 Score ranking
    df_sorted = overall_df.sort_values('F1 Score', ascending=True)
    df_sorted.plot(x='Model', y='F1 Score', kind='barh', ax=axes[1], 
                   color='lightcoral', legend=False)
    axes[1].set_title('Model Ranking by F1 Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('F1 Score', fontsize=12)
    axes[1].set_ylabel('Model', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xlim([0.0, 1.0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: model_ranking.png")


def generate_markdown_report(overall_df, per_attack_auc, per_attack_f1, 
                             generalization_df, output_dir):
    """Generate comprehensive markdown report."""
    
    report = []
    report.append("# Multi-Model Training Results - Mixed Attack Dataset")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append(f"- **Total Models Trained**: {len(overall_df)}")
    report.append(f"- **Attack Types Included**: {len(config.ATTACK_TYPES)}")
    report.append(f"- **Training Strategy**: Mixed attack injection (each model sees all attack types)")
    report.append("")
    report.append("### Attack Types")
    for i, attack in enumerate(config.ATTACK_TYPES, 1):
        report.append(f"{i}. `{attack}`")
    report.append("")
    
    report.append("## Overall Model Performance")
    report.append("")
    report.append(overall_df.to_markdown(index=False))
    report.append("")
    
    report.append("## Best Model")
    report.append("")
    best_model = overall_df.iloc[0]
    report.append(f"- **Model**: {best_model['Model']}")
    report.append(f"- **AUC-ROC**: {best_model['AUC-ROC']:.4f}")
    report.append(f"- **F1 Score**: {best_model['F1 Score']:.4f}")
    report.append(f"- **Precision**: {best_model['Precision']:.4f}")
    report.append(f"- **Recall**: {best_model['Recall']:.4f}")
    report.append("")
    
    report.append("## Generalization Analysis")
    report.append("")
    report.append("Models ranked by consistency (lower Std_AUC = better generalization):")
    report.append("")
    report.append(generalization_df.to_markdown(index=False))
    report.append("")
    
    report.append("## Per-Attack Performance (AUC-ROC)")
    report.append("")
    report.append(per_attack_auc.to_markdown())
    report.append("")
    
    report.append("## Per-Attack Performance (F1 Score)")
    report.append("")
    report.append(per_attack_f1.to_markdown())
    report.append("")
    
    report.append("## Attack Difficulty Analysis")
    report.append("")
    attack_difficulty = []
    for attack in config.ATTACK_TYPES:
        if attack in per_attack_auc.columns:
            mean_auc = per_attack_auc[attack].mean()
            std_auc = per_attack_auc[attack].std()
            attack_difficulty.append({
                'Attack': attack,
                'Mean_AUC': mean_auc,
                'Std_AUC': std_auc
            })
    
    difficulty_df = pd.DataFrame(attack_difficulty)
    difficulty_df = difficulty_df.sort_values('Mean_AUC')
    report.append(difficulty_df.to_markdown(index=False))
    report.append("")
    report.append("*(Lower Mean_AUC = harder to detect)*")
    report.append("")
    
    report.append("## Visualizations")
    report.append("")
    report.append("- `heatmap_auc_roc.png` - AUC-ROC heatmap (Model × Attack)")
    report.append("- `heatmap_f1_score.png` - F1 Score heatmap (Model × Attack)")
    report.append("- `generalization_boxplot.png` - Generalization ability comparison")
    report.append("- `radar_charts.png` - Per-attack performance radar charts")
    report.append("- `model_ranking.png` - Overall model ranking")
    report.append("")
    
    report.append("## Key Findings")
    report.append("")
    report.append("### Best Generalization")
    best_gen = generalization_df.iloc[0]
    report.append(f"- **{best_gen['Model']}** has the most consistent performance across attacks")
    report.append(f"  - Mean AUC: {best_gen['Mean_AUC']:.4f}")
    report.append(f"  - Std AUC: {best_gen['Std_AUC']:.4f}")
    report.append("")
    
    report.append("### Hardest Attack to Detect")
    easiest_attack = difficulty_df.iloc[-1]
    hardest_attack = difficulty_df.iloc[0]
    report.append(f"- **Hardest**: `{hardest_attack['Attack']}` (Mean AUC: {hardest_attack['Mean_AUC']:.4f})")
    report.append(f"- **Easiest**: `{easiest_attack['Attack']}` (Mean AUC: {easiest_attack['Mean_AUC']:.4f})")
    report.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'ANALYSIS_REPORT.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Saved: ANALYSIS_REPORT.md")


def main():
    """Main analysis workflow."""
    
    print("="*80)
    print("Multi-Model Analysis - Mixed Attack Training")
    print("="*80)
    
    output_dir = config.OUTPUT_DIR
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load results
    print("\nLoading results...")
    results = load_all_results(output_dir)
    print(f"Loaded results for {len(results)} models")
    
    # Create comparison tables
    print("\nCreating comparison tables...")
    overall_df = create_overall_comparison(results)
    per_attack_auc, per_attack_f1 = create_per_attack_comparison(results)
    generalization_df = compute_generalization_metrics(results)
    
    # Save tables
    overall_df.to_csv(os.path.join(output_dir, 'overall_comparison.csv'), index=False)
    per_attack_auc.to_csv(os.path.join(output_dir, 'per_attack_auc.csv'))
    per_attack_f1.to_csv(os.path.join(output_dir, 'per_attack_f1.csv'))
    generalization_df.to_csv(os.path.join(output_dir, 'generalization_analysis.csv'), index=False)
    
    print("\nOverall Performance:")
    print(overall_df.to_string(index=False))
    
    print("\nGeneralization Analysis:")
    print(generalization_df.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_heatmap(per_attack_auc, 'Model × Attack Performance (AUC-ROC)', 
                'heatmap_auc_roc.png', viz_dir)
    plot_heatmap(per_attack_f1, 'Model × Attack Performance (F1 Score)', 
                'heatmap_f1_score.png', viz_dir)
    plot_generalization_boxplot(results, viz_dir)
    plot_radar_chart(results, viz_dir)
    plot_model_ranking(overall_df, viz_dir)
    
    # Generate report
    print("\nGenerating markdown report...")
    generate_markdown_report(overall_df, per_attack_auc, per_attack_f1, 
                            generalization_df, output_dir)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"\nView report: {os.path.join(output_dir, 'ANALYSIS_REPORT.md')}")


if __name__ == '__main__':
    main()
