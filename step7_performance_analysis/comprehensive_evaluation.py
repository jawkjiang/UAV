"""
综合评估模块
整合：
1. Step6的检测性能指标（DR@5s, ADD, MTBFA等）
2. Step7的推理性能指标
3. Step7的模型复杂度指标
生成统一的评估报告
"""

import pandas as pd
import os
import config


def load_step6_metrics() -> pd.DataFrame:
    """加载Step6的评估指标"""
    metrics_path = os.path.join(
        config.STEP6_OUTPUT, 'overall_metrics.csv'
    )
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Step6 metrics not found: {metrics_path}")
    
    df = pd.read_csv(metrics_path)
    return df


def integrate_all_metrics(
    step6_metrics: pd.DataFrame,
    inference_perf: dict,
    model_complexity: dict
) -> pd.DataFrame:
    """
    整合所有指标
    Returns:
        综合评估DataFrame
    """
    rows = []
    
    for model_name in config.MODEL_NAMES:
        if model_name not in inference_perf:
            continue
        
        # Step6检测指标
        s6_row = step6_metrics[step6_metrics['model'] == model_name].iloc[0]
        
        # 推理性能（优先GPU，无GPU则CPU）
        device = 'cuda' if 'cuda' in inference_perf[model_name] else 'cpu'
        inf_perf = inference_perf[model_name][device]
        
        # 模型复杂度
        complexity = model_complexity.get(model_name, {})
        
        row = {
            'model': model_name,
            
            # 检测性能
            'dr_1s': s6_row.get('DR@1s', 0),
            'dr_2s': s6_row.get('DR@2s', 0),
            'dr_5s': s6_row.get('DR@5s', 0),
            'add_seconds': s6_row.get('ADD', 0),
            'mtbfa_hours': s6_row.get('MTBFA', 0),
            'total_detections': s6_row.get('detected', 0),
            'total_attacks': s6_row.get('total_attacks', 0),
            
            # 推理性能
            'device': device,
            'latency_mean_ms': inf_perf['latency']['mean_ms'],
            'latency_p95_ms': inf_perf['latency']['p95_ms'],
            'latency_p99_ms': inf_perf['latency']['p99_ms'],
            'throughput_windows_per_sec': inf_perf['throughput']['batch_1']['throughput_samples_per_sec'],
            'meets_real_time': inf_perf['real_time_capability']['meets_real_time'],
            
            # 模型复杂度
            'total_params': complexity.get('parameters', {}).get('total', 0),
            'params_millions': complexity.get('parameters', {}).get('total', 0) / 1e6,
            'model_size_mb': complexity.get('model_size_mb', 0),
            'flops': complexity.get('flops', 0),
            'flops_millions': complexity.get('flops', 0) / 1e6,
            'memory_mb': complexity.get('memory', {}).get('total_per_sample_mb', 0)
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 计算综合得分
    df = calculate_composite_scores(df)
    
    return df


def calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合得分
    考虑：准确性、速度、复杂度的平衡
    """
    # 归一化（0-1）
    df['norm_dr_5s'] = df['dr_5s']
    df['norm_latency'] = 1 - (df['latency_p95_ms'] / df['latency_p95_ms'].max())
    df['norm_params'] = 1 - (df['params_millions'] / df['params_millions'].max())
    
    # 综合得分（可调权重）
    df['composite_score'] = (
        0.5 * df['norm_dr_5s'] +        # 50% 检测性能
        0.3 * df['norm_latency'] +      # 30% 推理速度
        0.2 * df['norm_params']         # 20% 模型轻量
    )
    
    # 排名
    df['rank'] = df['composite_score'].rank(ascending=False)
    
    return df
