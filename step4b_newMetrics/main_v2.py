"""
Time-Aware Evaluation - Main Pipeline (v2)

使用完整元数据的准确评估版本
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import config
from data_loader_with_metadata import load_model_data_with_metadata
from time_aware_metrics import (
    identify_attack_segments_with_timing,
    calculate_detection_delay_v2,
    calculate_dr_at_delta_t,
    calculate_average_detection_delay,
    calculate_mtbfa,
    calculate_per_attack_metrics
)
from visualizations import (
    plot_dr_vs_delay,
    plot_dr_vs_mtbfa,
    plot_delay_distribution,
    plot_per_attack_heatmap,
    plot_mtbfa_comparison
)
from scenario_analysis import analyze_all_scenarios, generate_evaluation_summary


def evaluate_single_model_v2(model_name: str):
    """
    评估单个模型（使用完整元数据）
    
    Args:
        model_name: 模型名称
    
    Returns:
        评估结果字典
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Model: {model_name.upper()} (v2 with metadata)")
    print(f"{'='*70}")
    
    # 1. 加载数据（使用新的加载器）
    print(f"\n[1] Loading data with complete metadata...")
    try:
        model_data = load_model_data_with_metadata(model_name)
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        print(f"  Skipping {model_name}")
        return None
    
    y_true = model_data['y_true']
    y_pred = model_data['y_pred']  # 已经二值化
    y_prob = model_data['y_prob']
    timestamps = model_data['timestamps']
    flight_ids = model_data['flight_ids']
    attack_types = model_data['attack_types']
    attack_segments_info = model_data['attack_segments_info']
    
    # 2. 识别攻击片段（使用真实攻击时间）
    print(f"\n[2] Identifying attack segments with timing...")
    attack_segments = identify_attack_segments_with_timing(
        y_true=y_true,
        timestamps=timestamps,
        flight_ids=flight_ids,
        attack_types=attack_types,
        attack_segments_info=attack_segments_info
    )
    
    if len(attack_segments) == 0:
        print(f"  ⚠ No attack segments found!")
        return None
    
    print(f"  Found {len(attack_segments)} attack segments")
    print(f"  Time range: {timestamps.min():.2f}s - {timestamps.max():.2f}s")
    
    # 3. 计算检测延迟（使用真实攻击时间）
    print(f"\n[3] Calculating detection delays...")
    delays, detected = calculate_detection_delay_v2(
        y_pred=y_pred,
        timestamps=timestamps,
        attack_segments=attack_segments
    )
    
    n_detected = sum(detected)
    n_total = len(detected)
    print(f"  Detected: {n_detected}/{n_total} ({n_detected/n_total:.1%})")
    
    valid_delays = [d for d in delays if d is not None]
    if valid_delays:
        print(f"  Delay stats: mean={np.mean(valid_delays):.3f}s, "
              f"median={np.median(valid_delays):.3f}s, "
              f"range=[{min(valid_delays):.3f}s, {max(valid_delays):.3f}s]")
    
    # 4. 计算时间感知指标
    print(f"\n[4] Computing time-aware metrics...")
    
    # DR@Δt
    dr_dict = calculate_dr_at_delta_t(delays, config.DELTA_T_VALUES)
    print(f"  DR@Δt:")
    for dt in config.DELTA_T_VALUES:
        print(f"    DR@{dt}s = {dr_dict[dt]:.3f}")
    
    # ADD
    add = calculate_average_detection_delay(delays)
    add_str = f"{add:.3f}s" if add is not None else "N/A"
    print(f"  ADD = {add_str}")
    
    # MTBFA
    mtbfa, false_alarm_count, total_normal_hours = calculate_mtbfa(
        y_pred=y_pred,
        y_true=y_true,
        timestamps=timestamps,
        flight_ids=flight_ids
    )
    mtbfa_str = f"{mtbfa:.2f}h" if mtbfa != float('inf') else "inf (no false alarms)"
    print(f"  MTBFA = {mtbfa_str} (FAs: {false_alarm_count}, Normal time: {total_normal_hours:.1f}h)")
    
    # 5. Per-attack-type分析
    print(f"\n[5] Computing per-attack-type metrics...")
    per_attack_dict = calculate_per_attack_metrics(
        y_pred=y_pred,
        timestamps=timestamps,
        attack_segments=attack_segments,
        delta_t_values=config.DELTA_T_VALUES
    )
    
    print(f"  Results by attack type:")
    for attack_type in config.ATTACK_TYPES:
        if attack_type in per_attack_dict:
            metrics = per_attack_dict[attack_type]
            add_str = f"{metrics['ADD']:.3f}s" if metrics['ADD'] is not None else "N/A"
            print(f"    {attack_type:15s}: "
                  f"DR@5s={metrics['DR@5s']:.3f}, "
                  f"ADD={add_str}")
    
    # 6. 构建结果
    results = {
        'model': model_name,
        'n_windows': len(y_true),
        'n_attacks': n_total,
        'n_detected': n_detected,
        'dr_dict': dr_dict,
        'add': add,
        'mtbfa': mtbfa,
        'per_attack': per_attack_dict,
        'delays': delays,
        'detected': detected,
        'attack_segments': attack_segments
    }
    
    return results


def main():
    """主评估流程"""
    print("="*80)
    print("Time-Aware GPS Spoofing Detection Evaluation (v2)")
    print(f"Using complete metadata from step3b")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确保输出目录存在
    config.ensure_output_dir()
    output_dir = config.OUTPUT_DIR
    
    # 评估所有模型
    all_results = {}
    
    for model_name in config.MODEL_NAMES:
        results = evaluate_single_model_v2(model_name)
        if results is not None:
            all_results[model_name] = results
    
    if len(all_results) == 0:
        print("\n✗ No models were successfully evaluated!")
        return
    
    print(f"\n{'='*80}")
    print(f"Successfully evaluated {len(all_results)} models")
    print(f"{'='*80}")
    
    # 保存overall metrics
    print(f"\n[6] Saving evaluation results...")
    
    overall_metrics = []
    for model_name, results in all_results.items():
        row = {
            'model': model_name,
            'n_attacks': results['n_attacks'],
            'n_detected': results['n_detected'],
            'detection_rate': results['n_detected'] / results['n_attacks'] if results['n_attacks'] > 0 else 0,
            'add': results['add'],
            'mtbfa': results['mtbfa']
        }
        # 添加DR@Δt
        for dt in config.DELTA_T_VALUES:
            row[f'dr@{dt}s'] = results['dr_dict'][dt]
        
        overall_metrics.append(row)
    
    overall_df = pd.DataFrame(overall_metrics)
    overall_df.to_csv(os.path.join(output_dir, 'overall_metrics_v2.csv'), index=False)
    print(f"  ✓ Saved overall_metrics_v2.csv")
    
    # 保存per-attack metrics
    for model_name, results in all_results.items():
        per_attack_rows = []
        for attack_type, metrics in results['per_attack'].items():
            row = {'attack_type': attack_type, **metrics}
            per_attack_rows.append(row)
        
        if per_attack_rows:
            per_attack_df = pd.DataFrame(per_attack_rows)
            per_attack_df.to_csv(
                os.path.join(output_dir, f'{model_name}_per_attack_metrics_v2.csv'),
                index=False
            )
    
    print(f"  ✓ Saved per-attack metrics for {len(all_results)} models")
    
    # 保存detailed delays
    delay_rows = []
    for model_name, results in all_results.items():
        for i, seg in enumerate(results['attack_segments']):
            delay = results['delays'][i]
            detected = results['detected'][i]
            delay_rows.append({
                'model': model_name,
                'attack_id': i,
                'flight_id': seg['flight_id'],
                'attack_type': seg['attack_type'],
                't_attack': seg['t_attack'],
                't_first_window': seg['t_first_window'],
                'delay': delay,
                'detected': detected
            })
    
    if delay_rows:
        delay_df = pd.DataFrame(delay_rows)
        delay_df.to_csv(os.path.join(output_dir, 'detailed_delays_v2.csv'), index=False)
        print(f"  ✓ Saved detailed_delays_v2.csv")
    
    # 生成可视化
    print(f"\n[7] Generating visualizations...")
    
    try:
        plot_dr_vs_delay(all_results, os.path.join(output_dir, 'dr_vs_delay_v2.png'))
        plot_dr_vs_mtbfa(all_results, os.path.join(output_dir, 'dr_vs_mtbfa_v2.png'))
        plot_delay_distribution(all_results, os.path.join(output_dir, 'delay_distribution_v2.png'))
        plot_per_attack_heatmap(all_results, os.path.join(output_dir, 'per_attack_heatmap_v2.png'))
        plot_mtbfa_comparison(all_results, os.path.join(output_dir, 'mtbfa_comparison_v2.png'))
        print(f"  ✓ Generated 5 visualization plots")
    except Exception as e:
        print(f"  ✗ Error generating visualizations: {e}")
    
    # 场景分析
    print(f"\n[8] Performing scenario analysis...")
    
    try:
        scenario_results = analyze_all_scenarios(all_results)
        generate_evaluation_summary(all_results, scenario_results, 
                                    os.path.join(output_dir, 'evaluation_summary_v2.md'))
        print(f"  ✓ Generated evaluation_summary_v2.md")
    except Exception as e:
        print(f"  ✗ Error in scenario analysis: {e}")
    
    # 打印最终总结
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nModel Performance (sorted by DR@5s):")
    overall_df_sorted = overall_df.sort_values('dr@5s', ascending=False)
    
    for _, row in overall_df_sorted.iterrows():
        print(f"\n{row['model'].upper()}:")
        print(f"  Detection Rate: {row['detection_rate']:.1%} ({row['n_detected']}/{row['n_attacks']})")
        print(f"  DR@5s:  {row['dr@5s']:.3f}")
        print(f"  ADD:    {row['add']:.3f}s" if row['add'] is not None and not np.isnan(row['add']) else "  ADD:    N/A")
        print(f"  MTBFA:  {row['mtbfa']:.2f}h" if row['mtbfa'] != float('inf') else "  MTBFA:  inf (no false alarms)")
    
    # 推荐最佳模型
    if len(overall_df_sorted) > 0:
        best_model = overall_df_sorted.iloc[0]
        print(f"\n{'='*80}")
        print(f"RECOMMENDED MODEL: {best_model['model'].upper()}")
        print(f"  • Highest DR@5s: {best_model['dr@5s']:.1%}")
        if best_model['add'] is not None and not np.isnan(best_model['add']):
            print(f"  • Average Detection Delay: {best_model['add']:.3f}s")
        print(f"={'='*80}")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    print("✓ Evaluation Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
