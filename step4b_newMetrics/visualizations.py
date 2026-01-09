"""visualizations.py

Generate visualizations for time-aware evaluation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

import config


def plot_dr_vs_delay(all_metrics: Dict[str, Dict], save_path: str = None):
    """
    Plot Detection Rate vs Time Delay curves for all models.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=config.FIGSIZE_SINGLE)
    
    for model_name, metrics in all_metrics.items():
        dr_values = [metrics['DR'][dt] for dt in config.DELTA_T_VALUES]
        
        plt.plot(config.DELTA_T_VALUES, dr_values, 
                marker='o', linewidth=2, markersize=6,
                color=config.MODEL_COLORS.get(model_name, 'gray'),
                label=model_name.upper())
    
    plt.xlabel('Time Threshold Δt (seconds)', fontsize=12)
    plt.ylabel('Detection Rate (DR@Δt)', fontsize=12)
    plt.title('Detection Rate vs Time Delay', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dr_vs_mtbfa(all_metrics: Dict[str, Dict], save_path: str = None):
    """
    Plot Detection Rate vs MTBFA trade-off scatter plot.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
        save_path: Path to save the figure
    """
    plt.figure(figsize=config.FIGSIZE_SINGLE)
    
    # Extract data
    mtbfa_values = []
    dr_5s_values = []
    model_names = []
    
    for model_name, metrics in all_metrics.items():
        # Skip if MTBFA is inf
        if np.isinf(metrics['MTBFA']):
            mtbfa = 100  # Cap at 100h for visualization
        else:
            mtbfa = metrics['MTBFA']
        
        mtbfa_values.append(mtbfa)
        dr_5s_values.append(metrics['DR'][5])
        model_names.append(model_name)
    
    # Create scatter plot
    for i, model_name in enumerate(model_names):
        plt.scatter(mtbfa_values[i], dr_5s_values[i], 
                   s=200, alpha=0.7,
                   color=config.MODEL_COLORS.get(model_name, 'gray'),
                   edgecolors='black', linewidths=1.5,
                   label=model_name.upper())
        
        # Add text label
        plt.annotate(model_name.upper(), 
                    (mtbfa_values[i], dr_5s_values[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)
    
    plt.xlabel('Mean Time Between False Alarms (hours)', fontsize=12)
    plt.ylabel('Detection Rate @ 5s', fontsize=12)
    plt.title('DR@5s vs MTBFA Trade-off', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_delay_distribution(all_metrics: Dict[str, Dict], save_path: str = None):
    """
    Plot detection delay distribution as box plots.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
        save_path: Path to save the figure
    """
    plt.figure(figsize=config.FIGSIZE_SINGLE)
    
    # Prepare data for boxplot
    delay_data = []
    model_labels = []
    
    for model_name, metrics in all_metrics.items():
        # Filter out None values (undetected attacks)
        detected_delays = [d for d in metrics['delays'] if d is not None]
        if detected_delays:
            delay_data.append(detected_delays)
            model_labels.append(model_name.upper())
    
    # Create box plot
    bp = plt.boxplot(delay_data, labels=model_labels, patch_artist=True,
                     showmeans=True, meanline=False,
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', 
                                   markeredgecolor='green', markersize=6))
    
    # Color boxes
    for patch, model_name in zip(bp['boxes'], model_labels):
        color = config.MODEL_COLORS.get(model_name.lower(), 'lightblue')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.ylabel('Detection Delay (seconds)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title('Detection Delay Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_attack_heatmap(all_metrics: Dict[str, Dict], save_path: str = None):
    """
    Plot heatmap of DR@5s for each model-attack combination.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
        save_path: Path to save the figure
    """
    # Build data matrix
    models = sorted(all_metrics.keys())
    attack_types = sorted(config.ATTACK_TYPES)
    
    data_matrix = np.zeros((len(models), len(attack_types)))
    
    for i, model_name in enumerate(models):
        for j, attack_type in enumerate(attack_types):
            if attack_type in all_metrics[model_name]['per_attack']:
                data_matrix[i, j] = all_metrics[model_name]['per_attack'][attack_type]['DR@5s']
            else:
                data_matrix[i, j] = np.nan
    
    # Create heatmap
    plt.figure(figsize=config.FIGSIZE_HEATMAP)
    
    sns.heatmap(data_matrix, 
                annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[a.upper() for a in attack_types],
                yticklabels=[m.upper() for m in models],
                cbar_kws={'label': 'DR@5s'},
                linewidths=1, linecolor='gray')
    
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Detection Rate @ 5s by Model and Attack Type', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_mtbfa_comparison(all_metrics: Dict[str, Dict], save_path: str = None):
    """
    Plot MTBFA comparison as horizontal bar chart.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
        save_path: Path to save the figure
    """
    plt.figure(figsize=config.FIGSIZE_SINGLE)
    
    # Extract data
    models = []
    mtbfa_values = []
    
    for model_name, metrics in all_metrics.items():
        models.append(model_name.upper())
        # Cap infinite values at 200h for visualization
        mtbfa = min(metrics['MTBFA'], 200) if not np.isinf(metrics['MTBFA']) else 200
        mtbfa_values.append(mtbfa)
    
    # Sort by MTBFA
    sorted_indices = np.argsort(mtbfa_values)[::-1]
    models = [models[i] for i in sorted_indices]
    mtbfa_values = [mtbfa_values[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    colors = [config.MODEL_COLORS.get(m.lower(), 'gray') for m in models]
    plt.barh(models, mtbfa_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for i, (model, value) in enumerate(zip(models, mtbfa_values)):
        if value >= 200:
            label = '>200h'
        else:
            label = f'{value:.1f}h'
        plt.text(value + 2, i, label, va='center', fontsize=10)
    
    plt.xlabel('Mean Time Between False Alarms (hours)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('MTBFA Comparison Across Models', fontsize=14, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_visualizations(all_metrics: Dict[str, Dict]):
    """
    Generate all visualization plots.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
    """
    print("\nGenerating visualizations...")
    
    config.ensure_output_dir()
    
    # 1. DR vs Delay
    plot_dr_vs_delay(
        all_metrics,
        save_path=os.path.join(config.OUTPUT_DIR, config.DR_VS_DELAY_PNG)
    )
    
    # 2. DR vs MTBFA
    plot_dr_vs_mtbfa(
        all_metrics,
        save_path=os.path.join(config.OUTPUT_DIR, config.DR_VS_MTBFA_PNG)
    )
    
    # 3. Delay Distribution
    plot_delay_distribution(
        all_metrics,
        save_path=os.path.join(config.OUTPUT_DIR, config.DELAY_DISTRIBUTION_PNG)
    )
    
    # 4. Per-Attack Heatmap
    plot_per_attack_heatmap(
        all_metrics,
        save_path=os.path.join(config.OUTPUT_DIR, config.PER_ATTACK_HEATMAP_PNG)
    )
    
    # 5. MTBFA Comparison
    plot_mtbfa_comparison(
        all_metrics,
        save_path=os.path.join(config.OUTPUT_DIR, config.MTBFA_COMPARISON_PNG)
    )
    
    print("All visualizations generated!")


if __name__ == "__main__":
    # Test with dummy data
    print("Testing visualization module...")
    
    # Create dummy metrics
    dummy_metrics = {
        'cnn': {
            'DR': {1: 0.65, 2: 0.78, 5: 0.92, 10: 0.96, 15: 0.98, 30: 0.99},
            'ADD': 3.2,
            'MTBFA': 18.5,
            'delays': [1.2, 2.5, 3.1, 0.8, 4.2, None, 1.5],
            'per_attack': {
                'step': {'DR@5s': 0.95},
                'drift_ramp': {'DR@5s': 0.87},
                'delay': {'DR@5s': 0.91}
            }
        },
        'lstm': {
            'DR': {1: 0.72, 2: 0.85, 5: 0.95, 10: 0.98, 15: 0.99, 30: 1.00},
            'ADD': 2.8,
            'MTBFA': 12.3,
            'delays': [0.9, 1.8, 2.2, 1.1, 3.5, 0.6, 1.9],
            'per_attack': {
                'step': {'DR@5s': 0.97},
                'drift_ramp': {'DR@5s': 0.92},
                'delay': {'DR@5s': 0.94}
            }
        }
    }
    
    # Generate test plots
    generate_all_visualizations(dummy_metrics)
    print("\nTest visualizations created!")
