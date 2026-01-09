"""scenario_analysis.py

Analyze which models perform best under different deployment scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple
import os

import config


def evaluate_scenario(model_name: str, metrics: Dict, scenario: Dict) -> Tuple[bool, float, str]:
    """
    Evaluate if a model meets scenario requirements and compute score.
    
    Args:
        model_name: Name of the model
        metrics: Model's evaluation metrics
        scenario: Scenario specification with constraints and weights
    
    Returns:
        Tuple of (meets_requirements, score, reason)
    """
    constraints = scenario['constraints']
    weights = scenario['weights']
    
    # Check each constraint
    failures = []
    
    for key, threshold in constraints.items():
        if key == 'DR@5s':
            value = metrics['DR'][5]
            if value < threshold:
                failures.append(f"DR@5s={value:.3f} < {threshold}")
        
        elif key == 'DR@10s':
            value = metrics['DR'][10]
            if value < threshold:
                failures.append(f"DR@10s={value:.3f} < {threshold}")
        
        elif key == 'ADD':
            value = metrics['ADD']
            if value is None or value > threshold:
                failures.append(f"ADD={value if value is not None else 'N/A'} > {threshold}")
        
        elif key == 'MTBFA':
            value = metrics['MTBFA']
            if np.isinf(value):
                value = 1000  # Treat infinite as very high
            if value < threshold:
                failures.append(f"MTBFA={value:.1f}h < {threshold}h")
    
    meets_requirements = len(failures) == 0
    
    # Compute weighted score (normalized)
    score = 0.0
    
    if 'DR@5s' in weights:
        score += weights['DR@5s'] * metrics['DR'][5]
    
    if 'DR@10s' in weights:
        score += weights['DR@10s'] * metrics['DR'][10]
    
    if 'ADD' in weights:
        # Normalize ADD: lower is better, cap at 30s
        add_val = metrics['ADD'] if metrics['ADD'] is not None else 30
        add_normalized = 1.0 - min(add_val, 30) / 30
        score += weights['ADD'] * add_normalized
    
    if 'MTBFA' in weights:
        # Normalize MTBFA: higher is better, cap at 100h
        mtbfa = metrics['MTBFA'] if not np.isinf(metrics['MTBFA']) else 100
        mtbfa_normalized = min(mtbfa, 100) / 100
        score += weights['MTBFA'] * mtbfa_normalized
    
    reason = '; '.join(failures) if failures else 'All constraints met'
    
    return meets_requirements, score, reason


def analyze_all_scenarios(all_metrics: Dict[str, Dict]) -> Dict:
    """
    Analyze all models under all scenarios.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    for scenario in config.SCENARIOS:
        scenario_name = scenario['name']
        results[scenario_name] = {
            'scenario': scenario,
            'models': {}
        }
        
        for model_name, metrics in all_metrics.items():
            meets, score, reason = evaluate_scenario(model_name, metrics, scenario)
            
            results[scenario_name]['models'][model_name] = {
                'meets_requirements': meets,
                'score': score,
                'reason': reason
            }
        
        # Find best model for this scenario
        best_model = max(
            results[scenario_name]['models'].items(),
            key=lambda x: x[1]['score']
        )
        results[scenario_name]['best_model'] = best_model[0]
        results[scenario_name]['best_score'] = best_model[1]['score']
    
    return results


def generate_scenario_report(all_metrics: Dict[str, Dict], 
                             analysis: Dict,
                             save_path: str = None) -> str:
    """
    Generate a text report for scenario analysis.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
        analysis: Scenario analysis results
        save_path: Path to save the report
    
    Returns:
        Report text
    """
    lines = []
    lines.append("="*80)
    lines.append("SCENARIO ANALYSIS REPORT")
    lines.append("Time-Aware Evaluation of GPS Spoofing Detection Models")
    lines.append("="*80)
    lines.append("")
    
    for scenario in config.SCENARIOS:
        scenario_name = scenario['name']
        scenario_results = analysis[scenario_name]
        
        lines.append("-" * 80)
        lines.append(f"SCENARIO: {scenario_name}")
        lines.append(f"Description: {scenario['description']}")
        lines.append("-" * 80)
        
        lines.append("\nRequirements:")
        for key, value in scenario['constraints'].items():
            if 'DR@' in key:
                lines.append(f"  - {key} >= {value:.2f}")
            elif key == 'ADD':
                lines.append(f"  - {key} <= {value:.1f}s")
            elif key == 'MTBFA':
                lines.append(f"  - {key} >= {value:.1f}h")
        
        lines.append("\nModel Performance:")
        
        # Sort models by score
        sorted_models = sorted(
            scenario_results['models'].items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        for rank, (model_name, model_results) in enumerate(sorted_models, 1):
            metrics = all_metrics[model_name]
            
            status = "✓ PASS" if model_results['meets_requirements'] else "✗ FAIL"
            lines.append(f"\n  {rank}. {model_name.upper()} - Score: {model_results['score']:.3f} [{status}]")
            
            add_str = f"{metrics['ADD']:.2f}s" if metrics['ADD'] is not None else "N/A"
            mtbfa_str = f"{metrics['MTBFA']:.1f}h" if not np.isinf(metrics['MTBFA']) else "inf"
            
            lines.append(f"     ADD: {add_str} | "
                        f"MTBFA: {mtbfa_str} | "
                        f"DR@5s: {metrics['DR'][5]:.3f} | "
                        f"DR@10s: {metrics['DR'][10]:.3f}")
            
            if not model_results['meets_requirements']:
                lines.append(f"     Failures: {model_results['reason']}")
        
        best = scenario_results['best_model']
        lines.append(f"\n  ⭐ RECOMMENDED: {best.upper()}")
        lines.append("")
    
    lines.append("="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)
    
    report_text = '\n'.join(lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"  Saved scenario analysis: {save_path}")
    
    return report_text


def generate_evaluation_summary(all_metrics: Dict[str, Dict],
                                analysis: Dict,
                                save_path: str = None) -> str:
    """
    Generate a markdown summary of the evaluation.
    
    Args:
        all_metrics: Dictionary mapping model_name -> evaluation_metrics
        analysis: Scenario analysis results
        save_path: Path to save the summary
    
    Returns:
        Summary markdown text
    """
    lines = []
    lines.append("# Time-Aware Evaluation Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Models Evaluated**: {len(all_metrics)}")
    lines.append(f"- **Attack Types**: {', '.join(config.ATTACK_TYPES)}")
    lines.append(f"- **Time Thresholds**: {config.DELTA_T_VALUES}")
    lines.append("")
    
    lines.append("## Overall Performance")
    lines.append("")
    lines.append("| Model | ADD (s) | MTBFA (h) | DR@1s | DR@5s | DR@10s | False Alarms |")
    lines.append("|-------|---------|-----------|-------|-------|--------|--------------|")
    
    for model_name in sorted(all_metrics.keys()):
        m = all_metrics[model_name]
        add_str = f"{m['ADD']:.2f}" if m['ADD'] is not None else "N/A"
        mtbfa_str = f"{m['MTBFA']:.1f}" if not np.isinf(m['MTBFA']) else "inf"
        
        lines.append(f"| {model_name.upper()} | "
                    f"{add_str} | "
                    f"{mtbfa_str} | "
                    f"{m['DR'][1]:.3f} | "
                    f"{m['DR'][5]:.3f} | "
                    f"{m['DR'][10]:.3f} | "
                    f"{m['n_false_alarms']} |")
    
    lines.append("")
    lines.append("## Scenario Recommendations")
    lines.append("")
    
    for scenario in config.SCENARIOS:
        scenario_name = scenario['name']
        best_model = analysis[scenario_name]['best_model']
        
        lines.append(f"### {scenario_name}")
        lines.append(f"**Recommended Model**: {best_model.upper()}")
        lines.append(f"- {scenario['description']}")
        lines.append("")
    
    lines.append("## Key Findings")
    lines.append("")
    
    # Best ADD
    models_with_add = {k: v for k, v in all_metrics.items() if v['ADD'] is not None}
    if models_with_add:
        best_add_model = min(models_with_add.items(), key=lambda x: x[1]['ADD'])
        lines.append(f"- **Fastest Detection**: {best_add_model[0].upper()} "
                    f"(ADD={best_add_model[1]['ADD']:.2f}s)")
    
    # Best MTBFA
    best_mtbfa_model = max(all_metrics.items(), 
                          key=lambda x: x[1]['MTBFA'] if not np.isinf(x[1]['MTBFA']) else 0)
    mtbfa_str = f"{best_mtbfa_model[1]['MTBFA']:.1f}h" if not np.isinf(best_mtbfa_model[1]['MTBFA']) else "inf"
    lines.append(f"- **Lowest False Alarm Rate**: {best_mtbfa_model[0].upper()} "
                f"(MTBFA={mtbfa_str})")
    
    # Best DR@5s
    best_dr5_model = max(all_metrics.items(), key=lambda x: x[1]['DR'][5])
    lines.append(f"- **Highest DR@5s**: {best_dr5_model[0].upper()} "
                f"(DR@5s={best_dr5_model[1]['DR'][5]:.3f})")
    
    lines.append("")
    
    summary_text = '\n'.join(lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"  Saved evaluation summary: {save_path}")
    
    return summary_text


if __name__ == "__main__":
    # Test with dummy data
    print("Testing scenario analysis...")
    
    dummy_metrics = {
        'cnn': {
            'DR': {1: 0.65, 2: 0.78, 5: 0.92, 10: 0.96, 15: 0.98, 30: 0.99},
            'ADD': 3.2,
            'MTBFA': 18.5,
            'n_false_alarms': 12
        },
        'lstm': {
            'DR': {1: 0.72, 2: 0.85, 5: 0.95, 10: 0.98, 15: 0.99, 30: 1.00},
            'ADD': 2.8,
            'MTBFA': 12.3,
            'n_false_alarms': 18
        }
    }
    
    analysis = analyze_all_scenarios(dummy_metrics)
    
    print("\nScenario Analysis Results:")
    for scenario_name, results in analysis.items():
        print(f"\n{scenario_name}:")
        print(f"  Best model: {results['best_model']}")
        print(f"  Best score: {results['best_score']:.3f}")
    
    # Generate reports
    report = generate_scenario_report(dummy_metrics, analysis)
    print("\n" + report)
    
    summary = generate_evaluation_summary(dummy_metrics, analysis)
    print("\n" + summary)
