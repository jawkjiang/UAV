"""
Step6 Main Script: TimeGAN模型 + 时间感知评估

加载Step5训练的所有模型，使用Step4b的时间感知指标进行评估
"""
import os
import sys
import argparse

import config
import data_loader
import evaluation
import scenario_analysis
import visualizations


def main(use_cache: bool = True):
    """
    主评估流程
    
    Args:
        use_cache: 是否使用缓存的预测结果
    """
    print("\n" + "="*80)
    print(" " * 20 + "STEP6: TimeGAN + Time-Aware Evaluation")
    print("="*80)
    
    config.ensure_output_dir()
    
    print("\nConfiguration:")
    print(f"  Models: {config.MODEL_NAMES}")
    print(f"  Delta-t values: {config.DELTA_T_VALUES}")
    print(f"  Step5 source: {config.STEP5_OUTPUT_DIR}")
    print(f"  Output directory: {config.OUTPUT_DIR}")
    
    # ========================================================================
    # Step 1: 加载测试数据
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Loading Test Data from Step5")
    print("-"*80)
    
    test_df, attack_info = data_loader.load_test_data()
    
    X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info = \
        data_loader.prepare_test_windows(test_df, attack_info)
    
    print(f"\nTest Data Summary:")
    print(f"  Windows: {len(X_test):,}")
    print(f"  Features: {X_test.shape[2]}")
    print(f"  Positive windows: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    print(f"  Attack segments: {len(attack_segments_info)}")
    
    # ========================================================================
    # Step 2: 加载所有模型并生成预测
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Loading Models and Generating Predictions")
    print("-"*80)
    
    all_data = data_loader.load_all_models_predictions(
        X_test, y_test, timestamps, flight_ids, attack_types, attack_segments_info,
        use_cache=use_cache
    )
    
    if not all_data:
        print("ERROR: No models were successfully loaded!")
        return 1
    
    print(f"\nSuccessfully loaded {len(all_data)} models:")
    for model_name in all_data.keys():
        print(f"  ✓ {model_name}")
    
    # ========================================================================
    # Step 3: 计算时间感知指标
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Computing Time-Aware Metrics")
    print("-"*80)
    
    all_metrics = evaluation.evaluate_all_models(all_data, save_results=True)
    
    if not all_metrics:
        print("ERROR: No models were successfully evaluated!")
        return 1
    
    # ========================================================================
    # Step 4: 生成可视化
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Generating Visualizations")
    print("-"*80)
    
    try:
        visualizations.generate_all_visualizations(all_metrics)
    except Exception as e:
        print(f"WARNING: Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Step 5: 场景分析
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 5: Scenario Analysis")
    print("-"*80)
    
    try:
        analysis = scenario_analysis.analyze_all_scenarios(all_metrics)
        
        # 生成场景报告
        scenario_report_path = os.path.join(
            config.TIME_AWARE_OUTPUT,
            config.SCENARIO_ANALYSIS_TXT
        )
        scenario_analysis.generate_scenario_report(
            all_metrics, analysis, save_path=scenario_report_path
        )
        
        # 生成评估总结
        summary_path = os.path.join(
            config.TIME_AWARE_OUTPUT,
            config.EVALUATION_SUMMARY_MD
        )
        scenario_analysis.generate_evaluation_summary(
            all_metrics, analysis, save_path=summary_path
        )
    except Exception as e:
        print(f"WARNING: Scenario analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print(f"\n  Metrics (CSV):")
    print(f"    • {os.path.join(config.TIME_AWARE_OUTPUT, config.OVERALL_METRICS_CSV)}")
    print(f"    • {os.path.join(config.TIME_AWARE_OUTPUT, config.DETAILED_DELAYS_CSV)}")
    for model_name in all_metrics.keys():
        per_attack_file = config.PER_ATTACK_METRICS_CSV_TEMPLATE.format(model=model_name)
        print(f"    • {os.path.join(config.TIME_AWARE_OUTPUT, per_attack_file)}")
    
    print(f"\n  Visualizations (PNG):")
    viz_files = [
        config.DR_VS_DELAY_PNG,
        config.DR_VS_MTBFA_PNG,
        config.DELAY_DISTRIBUTION_PNG,
        config.PER_ATTACK_HEATMAP_PNG,
        config.MTBFA_COMPARISON_PNG
    ]
    for viz_file in viz_files:
        viz_path = os.path.join(config.VISUALIZATION_OUTPUT, viz_file)
        if os.path.exists(viz_path):
            print(f"    • {viz_path}")
    
    print(f"\n  Reports:")
    print(f"    • {os.path.join(config.TIME_AWARE_OUTPUT, config.SCENARIO_ANALYSIS_TXT)}")
    print(f"    • {os.path.join(config.TIME_AWARE_OUTPUT, config.EVALUATION_SUMMARY_MD)}")
    
    # 关键结果
    print("\n🏆 Key Results:")
    
    # Best ADD (fastest detection)
    models_with_add = {k: v for k, v in all_metrics.items() if v['ADD'] is not None}
    if models_with_add:
        best_add = min(models_with_add.items(), key=lambda x: x[1]['ADD'])
        print(f"  🚀 Fastest Detection (ADD): {best_add[0].upper()} - {best_add[1]['ADD']:.2f}s")
    
    # Best MTBFA (lowest false alarm rate)
    finite_mtbfa = {k: v for k, v in all_metrics.items() if not np.isinf(v['MTBFA'])}
    if finite_mtbfa:
        best_mtbfa = max(finite_mtbfa.items(), key=lambda x: x[1]['MTBFA'])
        print(f"  ⏱️  Lowest False Alarms (MTBFA): {best_mtbfa[0].upper()} - {best_mtbfa[1]['MTBFA']:.1f}h")
    
    # Best DR@5s
    best_dr5 = max(all_metrics.items(), key=lambda x: x[1]['DR'][5])
    print(f"  🎯 Best DR@5s: {best_dr5[0].upper()} - {best_dr5[1]['DR'][5]:.3f}")
    
    # Overall detection rate
    for model_name, metrics in all_metrics.items():
        det_rate = metrics['n_detected'] / max(metrics['n_attacks'], 1)
        print(f"  • {model_name.upper()}: {det_rate*100:.1f}% detected ({metrics['n_detected']}/{metrics['n_attacks']})")
    
    print("\n" + "="*80)
    print("✅ Step6 evaluation completed successfully!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step6: TimeGAN + Time-Aware Evaluation')
    parser.add_argument('--no-cache', action='store_true',
                       help='Force regenerate predictions (ignore cache)')
    
    args = parser.parse_args()
    
    import numpy as np  # For final summary
    
    exit_code = main(use_cache=not args.no_cache)
    sys.exit(exit_code)
