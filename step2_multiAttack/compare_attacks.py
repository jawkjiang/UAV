"""
Attack Comparison and Analysis Script

Provides detailed analysis and visualization comparing all attack types:
- Performance metrics comparison
- Attack characteristics analysis
- Detection difficulty ranking
- Visualization of results
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def load_experiment_results(output_dir: str) -> pd.DataFrame:
    """
    Load all experiment results from output directory.
    
    Returns:
        DataFrame with comprehensive results for all attack types
    """
    results = []
    
    # Automatically discover all attack type directories
    # Look for subdirectories that contain test_metrics.json
    attack_types = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                metrics_path = os.path.join(item_path, 'test_metrics.json')
                if os.path.exists(metrics_path):
                    attack_types.append(item)
    
    if not attack_types:
        print(f"Warning: No experiment results found in {output_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(attack_types)} experiment result(s): {', '.join(sorted(attack_types))}")
    
    for attack_type in attack_types:
        attack_dir = os.path.join(output_dir, attack_type)
        
        if not os.path.exists(attack_dir):
            print(f"Warning: {attack_type} results not found, skipping...")
            continue
        
        # Load metrics
        metrics_path = os.path.join(attack_dir, 'test_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"Warning: {attack_type} metrics not found, skipping...")
            continue
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load attack info
        attack_info_path = os.path.join(attack_dir, 'test_attack_info.csv')
        if os.path.exists(attack_info_path):
            attack_info_df = pd.read_csv(attack_info_path)
            n_attacked = attack_info_df['attacked'].sum()
            n_total = len(attack_info_df)
            attack_rate = n_attacked / n_total if n_total > 0 else 0
        else:
            attack_rate = None
        
        # Compile result
        result = {
            'attack_type': attack_type,
            'auc_roc': metrics.get('roc_auc', metrics.get('auc_roc', 0)),
            'auc_pr': metrics.get('pr_auc', metrics.get('auc_pr', 0)),
            'f1_score': metrics.get('f1', metrics.get('f1_score', 0)),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'accuracy': metrics.get('accuracy', 0),
            'attack_rate': attack_rate
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("ATTACK TYPE COMPARISON - DETECTION PERFORMANCE")
    print("="*100)
    
    # Sort by AUC-ROC descending (easiest to detect first)
    df_sorted = df.sort_values('auc_roc', ascending=False).copy()
    
    # Format columns
    display_df = df_sorted.copy()
    display_df['auc_roc'] = display_df['auc_roc'].apply(lambda x: f"{x:.4f}")
    display_df['auc_pr'] = display_df['auc_pr'].apply(lambda x: f"{x:.4f}")
    display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}")
    display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.4f}")
    display_df['recall'] = display_df['recall'].apply(lambda x: f"{x:.4f}")
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.4f}")
    
    if 'attack_rate' in display_df.columns:
        display_df['attack_rate'] = display_df['attack_rate'].apply(
            lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
        )
    
    print(display_df.to_string(index=False))
    print("="*100)


def analyze_detection_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and rank attack types by detection difficulty.
    
    Lower AUC-ROC = harder to detect
    """
    df_ranked = df.sort_values('auc_roc', ascending=True).copy()
    df_ranked['difficulty_rank'] = range(1, len(df_ranked) + 1)
    
    # Categorize difficulty
    def categorize_difficulty(auc_roc):
        if auc_roc >= 0.95:
            return 'Very Easy'
        elif auc_roc >= 0.90:
            return 'Easy'
        elif auc_roc >= 0.80:
            return 'Moderate'
        elif auc_roc >= 0.70:
            return 'Hard'
        else:
            return 'Very Hard'
    
    df_ranked['difficulty_category'] = df_ranked['auc_roc'].apply(categorize_difficulty)
    
    return df_ranked


def plot_performance_comparison(df: pd.DataFrame, output_dir: str):
    """Create comparison visualizations."""
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Bar chart of AUC-ROC
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('auc_roc', ascending=True)
    
    colors = ['red' if x < 0.8 else 'orange' if x < 0.9 else 'green' 
              for x in df_sorted['auc_roc']]
    
    plt.barh(df_sorted['attack_type'], df_sorted['auc_roc'], color=colors, alpha=0.7)
    plt.xlabel('AUC-ROC', fontsize=12)
    plt.ylabel('Attack Type', fontsize=12)
    plt.title('Detection Performance by Attack Type (AUC-ROC)', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.05)
    
    # Add value labels
    for i, v in enumerate(df_sorted['auc_roc']):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_auc_roc.png'), dpi=300)
    print(f"✓ Saved: {os.path.join(output_dir, 'comparison_auc_roc.png')}")
    plt.close()
    
    # 2. Heatmap of all metrics
    plt.figure(figsize=(10, 8))
    
    metrics_cols = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall', 'accuracy']
    heatmap_data = df.set_index('attack_type')[metrics_cols]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    plt.title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Attack Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_heatmap.png'), dpi=300)
    print(f"✓ Saved: {os.path.join(output_dir, 'comparison_heatmap.png')}")
    plt.close()
    
    # 3. Radar chart comparing top metrics
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    metrics = ['auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for _, row in df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['attack_type'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper().replace('_', ' ') for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Performance Metrics Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_radar.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'comparison_radar.png')}")
    plt.close()


def generate_analysis_report(df: pd.DataFrame, output_dir: str):
    """Generate comprehensive analysis report in markdown."""
    
    df_ranked = analyze_detection_difficulty(df)
    
    report = []
    report.append("# Multi-Attack GPS Spoofing Detection - Analysis Report\n")
    report.append("## Executive Summary\n")
    report.append(f"- **Total Attack Types Evaluated:** {len(df)}\n")
    report.append(f"- **Best Detection Performance:** {df.loc[df['auc_roc'].idxmax(), 'attack_type']} "
                 f"(AUC-ROC: {df['auc_roc'].max():.4f})\n")
    report.append(f"- **Worst Detection Performance:** {df.loc[df['auc_roc'].idxmin(), 'attack_type']} "
                 f"(AUC-ROC: {df['auc_roc'].min():.4f})\n")
    report.append(f"- **Average AUC-ROC:** {df['auc_roc'].mean():.4f}\n")
    report.append("\n---\n")
    
    report.append("## Detection Difficulty Ranking\n")
    report.append("| Rank | Attack Type | AUC-ROC | Difficulty |\n")
    report.append("|------|-------------|---------|------------|\n")
    
    for _, row in df_ranked.iterrows():
        report.append(f"| {row['difficulty_rank']} | {row['attack_type']} | "
                     f"{row['auc_roc']:.4f} | {row['difficulty_category']} |\n")
    
    report.append("\n---\n")
    
    report.append("## Detailed Performance Metrics\n")
    report.append("| Attack Type | AUC-ROC | AUC-PR | F1 | Precision | Recall | Accuracy |\n")
    report.append("|-------------|---------|--------|----|-----------| -------|----------|\n")
    
    for _, row in df.iterrows():
        report.append(f"| {row['attack_type']} | {row['auc_roc']:.4f} | "
                     f"{row['auc_pr']:.4f} | {row['f1_score']:.4f} | "
                     f"{row['precision']:.4f} | {row['recall']:.4f} | "
                     f"{row['accuracy']:.4f} |\n")
    
    report.append("\n---\n")
    
    report.append("## Attack Type Characteristics\n\n")
    
    report.append("### A. Drift Spoofing\n")
    report.append("- **drift**: Ramp profile - linear position offset accumulation\n")
    report.append("- **drift_sigmoid**: Sigmoid profile - smooth S-curve offset accumulation\n\n")
    
    report.append("### B. Delay/Replay\n")
    report.append("- **delay**: Fixed time delay - positions delayed by constant time offset\n")
    report.append("- **replay_same**: Replay from same flight earlier segment (hard stitch)\n")
    report.append("- **replay_other_soft**: Replay from other flight (soft stitch with blending)\n\n")
    
    report.append("### C. Consistent Takeover\n")
    report.append("- **takeover_step**: Step offset with first-order tracking dynamics\n")
    report.append("- **takeover_ramp**: Ramp offset with first-order tracking dynamics\n\n")
    
    report.append("---\n")
    
    report.append("## Key Insights\n\n")
    
    # Find hardest to detect
    hardest = df.loc[df['auc_roc'].idxmin()]
    report.append(f"1. **Most Challenging Attack:** {hardest['attack_type']}\n")
    report.append(f"   - This attack achieves the lowest AUC-ROC ({hardest['auc_roc']:.4f}), "
                 "indicating it is the most difficult to detect.\n")
    report.append(f"   - Detection rate (recall): {hardest['recall']:.4f}\n\n")
    
    # Find easiest to detect
    easiest = df.loc[df['auc_roc'].idxmax()]
    report.append(f"2. **Easiest to Detect:** {easiest['attack_type']}\n")
    report.append(f"   - High AUC-ROC ({easiest['auc_roc']:.4f}) suggests strong detectability.\n")
    report.append(f"   - Detection rate (recall): {easiest['recall']:.4f}\n\n")
    
    # Precision vs Recall tradeoff
    report.append("3. **Precision-Recall Tradeoff:**\n")
    for _, row in df.iterrows():
        if row['precision'] > 0.95 and row['recall'] < 0.7:
            report.append(f"   - {row['attack_type']}: High precision ({row['precision']:.4f}) "
                         f"but lower recall ({row['recall']:.4f}) - conservative detector\n")
        elif row['recall'] > 0.95 and row['precision'] < 0.7:
            report.append(f"   - {row['attack_type']}: High recall ({row['recall']:.4f}) "
                         f"but lower precision ({row['precision']:.4f}) - aggressive detector\n")
    
    report.append("\n---\n")
    report.append("## Recommendations\n\n")
    report.append("1. **Defense Priority:** Focus on improving detection of the hardest attack types\n")
    report.append("2. **Model Tuning:** Consider ensemble methods combining detectors trained on different attack types\n")
    report.append("3. **Feature Engineering:** Attacks with lower AUC may benefit from attack-specific features\n")
    report.append("4. **Threshold Tuning:** Balance precision-recall based on operational requirements\n\n")
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"✓ Saved analysis report: {report_path}")


def main():
    """Main analysis pipeline."""
    
    print("="*80)
    print("Multi-Attack Comparison and Analysis")
    print("="*80)
    
    output_dir = './output'
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run main.py first to generate results.")
        return
    
    # Load results
    print("\nLoading experiment results...")
    df = load_experiment_results(output_dir)
    
    if df.empty:
        print("Error: No results found. Please run main.py first.")
        return
    
    print(f"✓ Loaded results for {len(df)} attack types")
    
    # Print comparison table
    print_comparison_table(df)
    
    # Analyze difficulty
    print("\n" + "="*80)
    print("DETECTION DIFFICULTY ANALYSIS")
    print("="*80)
    
    df_ranked = analyze_detection_difficulty(df)
    
    print("\nRanking (1 = Hardest to Detect):")
    for _, row in df_ranked.iterrows():
        print(f"  {row['difficulty_rank']}. {row['attack_type']:<25} "
              f"AUC-ROC: {row['auc_roc']:.4f}  [{row['difficulty_category']}]")
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_performance_comparison(df, output_dir)
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING ANALYSIS REPORT")
    print("="*80 + "\n")
    
    generate_analysis_report(df, output_dir)
    
    # Save comparison CSV
    comparison_path = os.path.join(output_dir, 'attack_comparison_detailed.csv')
    df_ranked.to_csv(comparison_path, index=False)
    print(f"✓ Saved detailed comparison: {comparison_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
