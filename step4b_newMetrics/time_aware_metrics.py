"""
Time-Aware Metrics for GPS Spoofing Detection

This module implements time-aware performance metrics:
- Detection Rate within Δt (DR@Δt)
- Average Detection Delay (ADD)
- Mean Time Between False Alarms (MTBFA)

v2: 支持使用真实攻击时间的准确计算
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def identify_attack_segments_with_timing(y_true: np.ndarray,
                                        timestamps: np.ndarray,
                                        flight_ids: np.ndarray,
                                        attack_types: np.ndarray,
                                        attack_segments_info: list) -> List[Dict]:
    """
    识别攻击片段，使用真实的攻击开始时间
    
    这个版本使用attack_segments_info中提供的真实attack_start_time，
    而不是简单地用第一个攻击窗口的时间戳。
    
    Args:
        y_true: 真实标签数组 (0=normal, 1=attack)
        timestamps: 每个窗口的时间戳 (秒)
        flight_ids: 飞行ID数组
        attack_types: 攻击类型数组
        attack_segments_info: 包含真实attack_start_time的列表
            每项包含: {'flight_id', 'attack_type', 'attack_start_time'}
    
    Returns:
        List of attack segments, each containing:
        - flight_id: int
        - attack_type: str
        - t_attack: float (真实的攻击开始时间)
        - t_first_window: float (第一个攻击窗口的时间)
        - t_last_window: float (最后一个攻击窗口的时间)
        - indices: List[int] (攻击窗口的索引)
        - n_windows: int (窗口数量)
    """
    segments = []
    
    # 创建攻击信息字典便于查找
    attack_info_dict = {item['flight_id']: item for item in attack_segments_info}
    
    # 为每个飞行识别攻击片段
    for flight_id in np.unique(flight_ids):
        flight_mask = flight_ids == flight_id
        flight_indices = np.where(flight_mask)[0]
        
        # 找到该飞行的攻击窗口
        attack_mask = (y_true[flight_mask] == 1)
        attack_indices_local = np.where(attack_mask)[0]
        
        if len(attack_indices_local) == 0:
            continue  # 该飞行没有攻击
        
        # 获取该飞行的真实攻击信息
        if flight_id in attack_info_dict:
            attack_info = attack_info_dict[flight_id]
            t_attack_real = attack_info['attack_start_time']
            attack_type_real = attack_info['attack_type']
        else:
            # 回退：使用第一个攻击窗口的时间
            first_attack_idx = flight_indices[attack_indices_local[0]]
            t_attack_real = timestamps[first_attack_idx]
            attack_type_real = attack_types[first_attack_idx]
            logger.warning(f"Flight {flight_id}: No attack_start_time found, using first window time")
        
        # 获取攻击窗口的全局索引
        attack_indices_global = flight_indices[attack_indices_local]
        
        # 分割连续片段（处理可能的间隙）
        splits = np.where(np.diff(attack_indices_global) > 1)[0] + 1
        continuous_segments = np.split(attack_indices_global, splits)
        
        # 为每个连续片段创建记录
        for seg_indices in continuous_segments:
            if len(seg_indices) == 0:
                continue
            
            segments.append({
                'flight_id': int(flight_id),
                'attack_type': attack_type_real,
                't_attack': float(t_attack_real),  # 真实攻击开始时间
                't_first_window': float(timestamps[seg_indices[0]]),
                't_last_window': float(timestamps[seg_indices[-1]]),
                'indices': seg_indices.tolist(),
                'n_windows': len(seg_indices)
            })
    
    logger.info(f"Identified {len(segments)} attack segments with timing info")
    return segments


def calculate_detection_delay_v2(y_pred: np.ndarray,
                                 timestamps: np.ndarray,
                                 attack_segments: List[Dict]) -> Tuple[List[Optional[float]], List[bool]]:
    """
    计算检测延迟（版本2：使用真实攻击时间）
    
    延迟 = 首次检测时间 - 真实攻击开始时间
    
    Args:
        y_pred: 预测标签 (0/1 binary)
        timestamps: 窗口时间戳
        attack_segments: 攻击片段（包含t_attack - 真实攻击时间）
    
    Returns:
        delays: 检测延迟列表 (秒), None表示未检测到
        detected: 是否被检测标志列表
    """
    delays = []
    detected_flags = []
    
    for seg in attack_segments:
        t_attack = seg['t_attack']  # 真实攻击开始时间
        attack_indices = seg['indices']
        
        # 在攻击窗口中查找首次检测
        t_detect = None
        for idx in attack_indices:
            if y_pred[idx] == 1:
                t_detect = timestamps[idx]
                break
        
        if t_detect is not None:
            # 延迟 = 检测时间 - 攻击时间
            delay = t_detect - t_attack
            
            # 处理负延迟（窗口中心可能早于攻击开始时间）
            # 这是正常的，因为窗口有持续时间，窗口中心可能在攻击开始之前
            # 但如果窗口被标记为攻击，说明窗口内有攻击采样点
            if delay < 0:
                delay = 0.0  # 认为立即检测（窗口内检测）
            
            delays.append(delay)
            detected_flags.append(True)
        else:
            delays.append(None)
            detected_flags.append(False)
    
    # 统计信息
    valid_delays = [d for d in delays if d is not None]
    detection_rate = sum(detected_flags) / len(detected_flags) if detected_flags else 0
    
    if valid_delays:
        avg_delay = np.mean(valid_delays)
        min_delay = min(valid_delays)
        max_delay = max(valid_delays)
        logger.info(f"Detection rate: {detection_rate:.2%}")
        logger.info(f"Average delay: {avg_delay:.3f}s")
        logger.info(f"Delay range: [{min_delay:.3f}s, {max_delay:.3f}s]")
    else:
        logger.info(f"Detection rate: {detection_rate:.2%}, No detections")
    
    return delays, detected_flags


def identify_attack_segments(y_true: np.ndarray, 
                            flight_ids: np.ndarray,
                            attack_types: np.ndarray,
                            timestamps: np.ndarray) -> List[Dict]:
    """
    识别所有攻击片段
    
    Args:
        y_true: 真实标签数组 (0=normal, 1=attack)
        flight_ids: 飞行ID数组
        attack_types: 攻击类型数组
        timestamps: 时间戳数组 (秒)
    
    Returns:
        List of attack segments, each containing:
        - flight_id: int
        - attack_type: str
        - t_start: float (attack start time)
        - t_end: float (attack end time)
        - indices: List[int] (indices in the global array)
    """
    segments = []
    in_attack = False
    current_segment = None
    
    for i in range(len(y_true)):
        if y_true[i] == 1:  # Attack point
            if not in_attack:
                # Start of new attack segment
                current_segment = {
                    'flight_id': flight_ids[i],
                    'attack_type': attack_types[i],
                    't_start': timestamps[i],
                    'indices': [i]
                }
                in_attack = True
            else:
                # Check if same flight and attack type
                if (flight_ids[i] == current_segment['flight_id'] and 
                    attack_types[i] == current_segment['attack_type']):
                    current_segment['indices'].append(i)
                else:
                    # Different flight/attack, save previous and start new
                    current_segment['t_end'] = timestamps[current_segment['indices'][-1]]
                    segments.append(current_segment)
                    
                    current_segment = {
                        'flight_id': flight_ids[i],
                        'attack_type': attack_types[i],
                        't_start': timestamps[i],
                        'indices': [i]
                    }
        else:  # Normal point
            if in_attack:
                # End of attack segment
                current_segment['t_end'] = timestamps[current_segment['indices'][-1]]
                segments.append(current_segment)
                in_attack = False
                current_segment = None
    
    # Handle case where last point is attack
    if in_attack and current_segment is not None:
        current_segment['t_end'] = timestamps[current_segment['indices'][-1]]
        segments.append(current_segment)
    
    logger.info(f"Identified {len(segments)} attack segments")
    return segments


def calculate_detection_delay(y_pred: np.ndarray,
                             timestamps: np.ndarray,
                             attack_segments: List[Dict]) -> Tuple[List[Optional[float]], List[bool]]:
    """
    计算每个攻击片段的检测延迟
    
    Args:
        y_pred: 模型预测数组 (0/1 binary)
        timestamps: 时间戳数组 (秒)
        attack_segments: 攻击片段列表
    
    Returns:
        delays: 检测延迟列表 (秒), None表示未检测到
        detected: 是否被检测标志列表
    """
    delays = []
    detected_flags = []
    
    for seg in attack_segments:
        t_attack = seg['t_start']
        attack_indices = seg['indices']
        
        # 查找首次检测时间
        t_detect = None
        for idx in attack_indices:
            if y_pred[idx] == 1:
                t_detect = timestamps[idx]
                break
        
        if t_detect is not None:
            delay = t_detect - t_attack
            delays.append(delay)
            detected_flags.append(True)
        else:
            delays.append(None)
            detected_flags.append(False)
    
    # Log statistics
    valid_delays = [d for d in delays if d is not None]
    detection_rate = sum(detected_flags) / len(detected_flags) if detected_flags else 0
    avg_delay = np.mean(valid_delays) if valid_delays else None
    
    logger.info(f"Detection rate: {detection_rate:.2%}, Average delay: {avg_delay:.2f}s" 
                if avg_delay is not None else f"Detection rate: {detection_rate:.2%}, No detections")
    
    return delays, detected_flags


def calculate_dr_at_delta_t(delays: List[Optional[float]], 
                           delta_t_values: List[float]) -> Dict[float, float]:
    """
    计算不同Δt下的检测率
    
    Args:
        delays: 检测延迟列表 (None表示未检测)
        delta_t_values: Δt阈值列表 (如[1,2,5,10,15,30])
    
    Returns:
        dr_dict: {Δt: DR@Δt}
    """
    total_attacks = len(delays)
    dr_dict = {}
    
    if total_attacks == 0:
        logger.warning("No attacks found for DR@Δt calculation")
        return {dt: 0.0 for dt in delta_t_values}
    
    for delta_t in delta_t_values:
        detected_within = sum(
            1 for d in delays 
            if d is not None and d <= delta_t
        )
        dr_dict[delta_t] = detected_within / total_attacks
    
    # Verify monotonic increase
    prev_dr = 0.0
    for dt in sorted(delta_t_values):
        current_dr = dr_dict[dt]
        if current_dr < prev_dr:
            logger.warning(f"DR@Δt not monotonic: DR@{dt}s={current_dr:.3f} < previous={prev_dr:.3f}")
        prev_dr = current_dr
    
    return dr_dict


def calculate_average_detection_delay(delays: List[Optional[float]]) -> Optional[float]:
    """
    计算平均检测延迟 (只计算被检测到的攻击)
    
    Args:
        delays: 检测延迟列表 (None表示未检测)
    
    Returns:
        Average detection delay in seconds, or None if no detections
    """
    valid_delays = [d for d in delays if d is not None]
    
    if not valid_delays:
        logger.warning("No detections found, ADD is undefined")
        return None
    
    add = np.mean(valid_delays)
    logger.info(f"ADD = {add:.2f}s (n_detected={len(valid_delays)}/{len(delays)})")
    
    return add


def calculate_mtbfa(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   timestamps: np.ndarray,
                   flight_ids: np.ndarray) -> Tuple[float, int, float]:
    """
    计算平均误报时间间隔（按飞行计算）
    
    MTBFA = 总正常飞行时间 / 误报事件数
    
    Args:
        y_true: 真实标签数组 (0=normal, 1=attack)
        y_pred: 模型预测数组 (0/1 binary)
        timestamps: 时间戳数组 (秒)
        flight_ids: 飞行ID数组
    
    Returns:
        mtbfa_hours: MTBFA in hours
        false_alarm_count: Number of false alarm events
        total_normal_hours: Total normal flight time in hours
    """
    # 按飞行分组计算（修复：避免窗口交错导致的段切割问题）
    unique_flights = np.unique(flight_ids)
    
    total_normal_time = 0.0  # seconds
    total_false_alarms = 0
    
    # 估算窗口步长对应的时间（假设步长=5，采样率从timestamps推断）
    # 相邻窗口的平均时间间隔
    time_per_window = np.median(np.diff(timestamps[flight_ids == unique_flights[0]][:100]))
    if time_per_window <= 0 or np.isnan(time_per_window):
        time_per_window = 0.05  # 默认0.05秒（步长5@100Hz）
    
    for flight_id in unique_flights:
        # 获取此飞行的所有窗口
        flight_mask = (flight_ids == flight_id)
        flight_y_true = y_true[flight_mask]
        flight_y_pred = y_pred[flight_mask]
        
        # 统计正常窗口数
        n_normal_windows = (flight_y_true == 0).sum()
        
        # 估算此飞行的正常时间（正常窗口数 × 窗口间隔）
        normal_time = n_normal_windows * time_per_window
        total_normal_time += normal_time
        
        # 统计此飞行的误报事件（连续误报算作一次）
        in_false_alarm = False
        for i in range(len(flight_y_true)):
            if flight_y_true[i] == 0 and flight_y_pred[i] == 1:  # False Positive
                if not in_false_alarm:
                    total_false_alarms += 1
                    in_false_alarm = True
            elif flight_y_true[i] == 0:  # Normal correctly predicted
                in_false_alarm = False
            else:  # Attack window
                in_false_alarm = False
    
    # Convert to hours
    total_normal_hours = total_normal_time / 3600
    
    if total_false_alarms > 0:
        mtbfa_hours = total_normal_hours / total_false_alarms
    else:
        mtbfa_hours = float('inf')
    
    logger.info(f"MTBFA = {mtbfa_hours:.4f}h = {mtbfa_hours*60:.2f}min (FAs={total_false_alarms}, Normal time={total_normal_hours:.2f}h)")
    
    return mtbfa_hours, total_false_alarms, total_normal_hours


def calculate_per_attack_metrics(y_pred: np.ndarray,
                                timestamps: np.ndarray,
                                attack_segments: List[Dict],
                                delta_t_values: List[float]) -> Dict[str, Dict]:
    """
    计算每种攻击类型的性能指标
    
    Args:
        y_pred: 模型预测数组
        timestamps: 时间戳数组
        attack_segments: 攻击片段列表
        delta_t_values: Δt阈值列表
    
    Returns:
        per_attack_metrics: {attack_type: {metrics}}
    """
    # Group segments by attack type
    attack_groups = {}
    for seg in attack_segments:
        att_type = seg['attack_type']
        if att_type not in attack_groups:
            attack_groups[att_type] = []
        attack_groups[att_type].append(seg)
    
    per_attack_metrics = {}
    
    for att_type, segs in attack_groups.items():
        # Calculate delays for this attack type (use v2 with real attack times)
        delays, detected = calculate_detection_delay_v2(y_pred, timestamps, segs)
        
        # Calculate metrics
        dr_dict = calculate_dr_at_delta_t(delays, delta_t_values)
        add = calculate_average_detection_delay(delays)
        detection_rate = sum(detected) / len(detected) if detected else 0
        
        per_attack_metrics[att_type] = {
            'n_instances': len(segs),
            'detection_rate': detection_rate,
            'ADD': add,
            **{f'DR@{dt}s': dr for dt, dr in dr_dict.items()}
        }
    
    return per_attack_metrics


def get_delay_statistics(delays: List[Optional[float]]) -> Dict[str, float]:
    """
    计算检测延迟的统计信息
    
    Args:
        delays: 检测延迟列表
    
    Returns:
        Statistics dictionary (mean, median, std, min, max, percentiles)
    """
    valid_delays = [d for d in delays if d is not None]
    
    if not valid_delays:
        return {
            'mean': None,
            'median': None,
            'std': None,
            'min': None,
            'max': None,
            'p25': None,
            'p75': None,
            'p90': None,
            'p95': None
        }
    
    return {
        'mean': np.mean(valid_delays),
        'median': np.median(valid_delays),
        'std': np.std(valid_delays),
        'min': np.min(valid_delays),
        'max': np.max(valid_delays),
        'p25': np.percentile(valid_delays, 25),
        'p75': np.percentile(valid_delays, 75),
        'p90': np.percentile(valid_delays, 90),
        'p95': np.percentile(valid_delays, 95)
    }


def compute_all_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       timestamps: np.ndarray,
                       flight_ids: np.ndarray,
                       attack_types: np.ndarray) -> Dict:
    """
    Compute all time-aware metrics for a model.
    
    Args:
        y_true: Ground truth labels, shape (N,)
        y_pred: Model predictions, shape (N,)
        timestamps: Time in seconds, shape (N,)
        flight_ids: Flight IDs, shape (N,)
        attack_types: Attack type labels, shape (N,)
    
    Returns:
        Dictionary with all metrics
    """
    # Identify attack segments
    segments = identify_attack_segments(y_true, flight_ids, attack_types, timestamps)
    
    # Calculate detection delays
    delays, detected = calculate_detection_delay(y_pred, timestamps, segments)
    
    # Calculate DR@Δt (Detection Rate at different time thresholds)
    import config
    dr_dict = calculate_dr_at_delta_t(delays, config.DELTA_T_VALUES)
    
    # Calculate ADD (Average Detection Delay)
    add = calculate_average_detection_delay(delays)
    
    # Calculate MTBFA (Mean Time Between False Alarms)
    mtbfa, n_fa, total_normal_time = calculate_mtbfa(y_true, y_pred, timestamps, flight_ids)
    
    # Per-attack-type metrics
    per_attack = calculate_per_attack_metrics(y_pred, timestamps, segments, config.DELTA_T_VALUES)
    
    return {
        'n_attacks': len(segments),
        'n_detected': sum(detected),
        'detection_rate': sum(detected) / len(segments) if segments else 0.0,
        'ADD': add,
        'MTBFA': mtbfa,
        'n_false_alarms': n_fa,
        'DR': dr_dict,
        'per_attack': per_attack,
        'delays': delays,
        'segments': segments
    }


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing time-aware metrics...")
    
    # Create synthetic data: 1000 samples, 10-second flight, 100 Hz
    n = 1000
    timestamps = np.linspace(0, 10, n)
    flight_ids = np.zeros(n, dtype=int)
    
    # Attack from t=5s to t=8s
    y_true = np.zeros(n, dtype=int)
    attack_start_idx = 500
    attack_end_idx = 800
    y_true[attack_start_idx:attack_end_idx] = 1
    
    attack_types = np.array(['normal'] * n, dtype=object)
    attack_types[attack_start_idx:attack_end_idx] = 'step'
    
    # Model detects at t=5.2s (delay = 0.2s)
    y_pred = np.zeros(n, dtype=int)
    y_pred[520:attack_end_idx] = 1
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, timestamps, flight_ids, attack_types)
    
    print(f"\nResults:")
    print(f"  Total attacks: {metrics['n_attacks']}")
    print(f"  Detected: {metrics['n_detected']}")
    print(f"  ADD: {metrics['ADD']:.3f}s")
    print(f"  DR@1s: {metrics['DR'][1]:.3f}")
    print(f"  DR@5s: {metrics['DR'][5]:.3f}")
    print(f"  MTBFA: {metrics['MTBFA']:.1f}h")
    print(f"  False alarms: {metrics['n_false_alarms']}")

