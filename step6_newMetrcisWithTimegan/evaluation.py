"""
Evaluation Module for Step6
使用时间感知指标评估所有模型
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import os

import config
import time_aware_metrics as tam


def evaluate_single_model(model_name: str, data: Dict[str, np.ndarray]) -> Dict:
    """
    评估单个模型使用时间感知指标
    
    Args:
        model_name: 模型名称
        data: 包含y_true, y_pred, timestamps, flight_ids, attack_types, attack_segments_info的字典
    
    Returns:
        评估指标字典
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*60}")
    
    # 提取数据
    y_true = data['y_true']
    y_pred = data['y_pred']
    timestamps = data['timestamps']
    flight_ids = data['flight_ids']
    attack_types = data['attack_types']
    attack_segments_info = data.get('attack_segments_info', [])
    
    # 使用time_aware_metrics的统一计算函数
    metrics = tam.compute_all_metrics(
        y_true, y_pred, timestamps, flight_ids, attack_types,
        attack_segments_info=attack_segments_info
    )
    
    # 打印摘要
    print(f"\nOverall Metrics:")
    print(f"  Total attacks: {metrics['n_attacks']}")
    print(f"  Detected: {metrics['n_detected']} ({metrics['n_detected']/max(metrics['n_attacks'],1)*100:.1f}%)")
    add_str = f"{metrics['ADD']:.3f}s" if metrics['ADD'] is not None else "N/A"
    print(f"  ADD: {add_str}")
    mtbfa_str = f"{metrics['MTBFA']:.1f}h" if not np.isinf(metrics['MTBFA']) else "inf"
    print(f"  MTBFA: {mtbfa_str}")
    print(f"  False alarms: {metrics['n_false_alarms']}")
    
    print(f"\nDetection Rate at different thresholds:")
    for dt in [1, 2, 5, 10]:
        if dt in metrics['DR']:
            dr = metrics['DR'][dt]
            print(f"  DR@{dt}s: {dr:.3f}")
    
    return metrics


def evaluate_all_models(all_data: Dict[str, Dict], save_results: bool = True) -> Dict[str, Dict]:
    """
    评估所有模型
    
    Args:
        all_data: 所有模型的数据字典
        save_results: 是否保存结果
    
    Returns:
        所有模型的评估指标
    """
    print("\n" + "="*70)
    print("TIME-AWARE EVALUATION - ALL MODELS")
    print("="*70)
    
    all_metrics = {}
    
    for model_name, data in all_data.items():
        try:
            metrics = evaluate_single_model(model_name, data)
            all_metrics[model_name] = metrics
        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if save_results:
        print("\nSaving results...")
        save_evaluation_results(all_metrics)
    
    print("\n" + "="*70)
    print(f"Evaluation complete! Evaluated {len(all_metrics)} models.")
    print("="*70)
    
    return all_metrics


def save_evaluation_results(all_metrics: Dict[str, Dict]):
    """
    保存评估结果到CSV文件
    """
    config.ensure_output_dir()
    
    # 1. Overall metrics CSV
    overall_rows = []
    for model_name, metrics in all_metrics.items():
        row = {
            'model': model_name,
            'n_attacks': metrics['n_attacks'],
            'n_detected': metrics['n_detected'],
            'detection_rate': metrics['n_detected'] / max(metrics['n_attacks'], 1),
            'ADD': metrics['ADD'],
            'MTBFA': metrics['MTBFA'],
            'n_false_alarms': metrics['n_false_alarms']
        }
        
        # Add DR@Δt columns
        for dt in config.DELTA_T_VALUES:
            row[f'DR@{dt}s'] = metrics['DR'][dt]
        
        overall_rows.append(row)
    
    overall_df = pd.DataFrame(overall_rows)
    overall_csv = os.path.join(config.TIME_AWARE_OUTPUT, config.OVERALL_METRICS_CSV)
    overall_df.to_csv(overall_csv, index=False)
    print(f"  ✓ Saved overall metrics: {overall_csv}")
    
    # 2. Per-attack metrics for each model
    for model_name, metrics in all_metrics.items():
        per_attack_rows = []
        
        for attack_type, attack_metrics in metrics['per_attack'].items():
            row = {
                'model': model_name,
                'attack_type': attack_type,
                'n_instances': attack_metrics['n_instances'],
                'ADD': attack_metrics['ADD'],
                'DR@5s': attack_metrics['DR@5s'],
                'DR@10s': attack_metrics['DR@10s']
            }
            per_attack_rows.append(row)
        
        if per_attack_rows:
            per_attack_df = pd.DataFrame(per_attack_rows)
            per_attack_csv = os.path.join(
                config.TIME_AWARE_OUTPUT,
                config.PER_ATTACK_METRICS_CSV_TEMPLATE.format(model=model_name)
            )
            per_attack_df.to_csv(per_attack_csv, index=False)
            print(f"  ✓ Saved per-attack metrics for {model_name}")
    
    # 3. Detailed delays CSV
    detailed_rows = []
    for model_name, metrics in all_metrics.items():
        for seg, delay in zip(metrics['segments'], metrics['delays']):
            row = {
                'model': model_name,
                'flight_id': seg['flight_id'],
                'attack_type': seg['attack_type'],
                'start_time': seg.get('t_attack', seg.get('t_start', 0)),
                'delay': delay if delay is not None else np.nan,
                'detected': delay is not None
            }
            detailed_rows.append(row)
    
    if detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv = os.path.join(config.TIME_AWARE_OUTPUT, config.DETAILED_DELAYS_CSV)
        detailed_df.to_csv(detailed_csv, index=False)
        print(f"  ✓ Saved detailed delays: {detailed_csv}")


if __name__ == '__main__':
    print("Testing evaluation module...")
    # This would require actual data to test
    print("Import data_loader first to test")
