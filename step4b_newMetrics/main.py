"""main.py

Main script to run complete time-aware evaluation pipeline.
"""

import os
import sys

import config
import evaluation
import visualizations
import scenario_analysis


def main():
    """Main evaluation pipeline."""
    
    print("\n" + "="*80)
    print("TIME-AWARE EVALUATION FRAMEWORK")
    print("GPS Spoofing Detection - Step 4b")
    print("="*80)
    
    # Ensure output directory exists
    config.ensure_output_dir()
    
    print("\nConfiguration:")
    print(f"  Models: {config.MODELS}")
    print(f"  Delta-t values: {config.DELTA_T_VALUES}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    print(f"  Step3b source: {config.STEP3B_OUTPUT_DIR}")
    
    # Step 1: Evaluate all models
    print("\n" + "-"*80)
    print("STEP 1: Computing Time-Aware Metrics")
    print("-"*80)
    
    all_metrics = evaluation.evaluate_all_models(save_results=True)
    
    if not all_metrics:
        print("ERROR: No models were successfully evaluated!")
        return 1
    
    # Step 2: Generate visualizations
    print("\n" + "-"*80)
    print("STEP 2: Generating Visualizations")
    print("-"*80)
    
    visualizations.generate_all_visualizations(all_metrics)
    
    # Step 3: Scenario analysis
    print("\n" + "-"*80)
    print("STEP 3: Scenario Analysis")
    print("-"*80)
    
    analysis = scenario_analysis.analyze_all_scenarios(all_metrics)
    
    # Generate scenario report
    scenario_report_path = os.path.join(
        config.OUTPUT_DIR, 
        config.SCENARIO_ANALYSIS_TXT
    )
    scenario_analysis.generate_scenario_report(
        all_metrics, analysis, save_path=scenario_report_path
    )
    
    # Generate evaluation summary
    summary_path = os.path.join(
        config.OUTPUT_DIR,
        config.EVALUATION_SUMMARY_MD
    )
    scenario_analysis.generate_evaluation_summary(
        all_metrics, analysis, save_path=summary_path
    )
    
    # Step 4: Print final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    print("\nGenerated Files:")
    print(f"  📊 Metrics: {os.path.join(config.OUTPUT_DIR, config.OVERALL_METRICS_CSV)}")
    print(f"  📈 Visualizations:")
    for viz_file in [config.DR_VS_DELAY_PNG, config.DR_VS_MTBFA_PNG,
                     config.DELAY_DISTRIBUTION_PNG, config.PER_ATTACK_HEATMAP_PNG,
                     config.MTBFA_COMPARISON_PNG]:
        print(f"      - {viz_file}")
    print(f"  📝 Reports:")
    print(f"      - {config.SCENARIO_ANALYSIS_TXT}")
    print(f"      - {config.EVALUATION_SUMMARY_MD}")
    
    print("\nKey Results:")
    
    # Best performers (handle None ADD values)
    models_with_add = {k: v for k, v in all_metrics.items() if v['ADD'] is not None}
    if models_with_add:
        best_add = min(models_with_add.items(), key=lambda x: x[1]['ADD'])
        print(f"  🚀 Fastest Detection: {best_add[0].upper()} (ADD={best_add[1]['ADD']:.2f}s)")
    
    best_mtbfa = max(all_metrics.items(), 
                     key=lambda x: x[1]['MTBFA'] if not float('inf') == x[1]['MTBFA'] else 0)
    mtbfa_val = best_mtbfa[1]['MTBFA']
    mtbfa_str = f"{mtbfa_val:.1f}h" if mtbfa_val < 1000 else ">100h"
    print(f"  ✅ Lowest False Alarms: {best_mtbfa[0].upper()} (MTBFA={mtbfa_str})")
    
    best_dr5 = max(all_metrics.items(), key=lambda x: x[1]['DR'][5])
    print(f"  ⚡ Highest DR@5s: {best_dr5[0].upper()} (DR@5s={best_dr5[1]['DR'][5]:.3f})")
    
    print("\nScenario Recommendations:")
    for scenario in config.SCENARIOS:
        best_model = analysis[scenario['name']]['best_model']
        print(f"  {scenario['name']}: {best_model.upper()}")
    
    print("\n" + "="*80)
    print(f"📁 All results saved to: {config.OUTPUT_DIR}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
