"""
电力系统无人机巡检场景 - 指标计算模块
Power Grid UAV Inspection - Metrics Calculation Module

实现5套指标体系：
1. 基于代价的安全-效率权衡
2. 基于时效性的实时检测能力
3. 基于可靠性的运行质量
4. 基于风险的关键任务保障
5. 基于资源约束的边缘部署适配性
"""

import os
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import ast
import warnings
warnings.filterwarnings('ignore')


class PowerGridMetrics:
    """电力巡检场景指标计算器"""
    
    def __init__(self, output_dir='../step3_multiModel/output'):
        self.output_dir = output_dir
        self.window_size = 50
        self.step_size = 5
        self.sampling_interval = 0.5  # seconds
        
        # 模型参数数量
        self.model_params = {
            'cnn': 263873,
            'lstm': 215873,
            'bilstm': 150337,
            'gru': 164545,
            'cnn_lstm': 301953,
            'tcn': 77313,
            'transformer': 103489
        }
        
        # 攻击严重度权重
        self.attack_weights = {
            'replay_same_hard': 5,
            'replay_other_soft': 4,
            'delay': 3,
            'drift_ramp': 2,
            'drift_sigmoid': 2,
            'step': 1,
            'takeover_step': 1,
            'takeover_ramp': 1
        }
        
        # 电力巡检风险模型
        from config_step4 import POWER_GRID_RISK_MODEL, MAGNITUDE_BINS
        self.risk_model = POWER_GRID_RISK_MODEL
        self.magnitude_bins = MAGNITUDE_BINS
    
    def load_experiment_results(self) -> List[Dict]:
        """加载所有实验结果"""
        print("加载实验数据...")
        results = []
        
        # 加载overall_comparison.csv
        overall_path = os.path.join(self.output_dir, 'overall_comparison.csv')
        overall_df = pd.read_csv(overall_path)
        
        # 遍历所有实验
        for idx, row in overall_df.iterrows():
            attack_type = row['attack_type']
            model_type = row['model_type']
            
            exp_dir = os.path.join(self.output_dir, attack_type, model_type)
            
            try:
                # 加载详细指标
                metrics_path = os.path.join(exp_dir, 'test_metrics.json')
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # 加载预测结果
                predictions_path = os.path.join(exp_dir, 'test_predictions.npz')
                predictions_data = np.load(predictions_path)
                
                # 加载攻击信息
                attack_info_path = os.path.join(exp_dir, 'test_attack_info.csv')
                attack_info = pd.read_csv(attack_info_path)
                
                results.append({
                    'attack_type': attack_type,
                    'model_type': model_type,
                    'metrics': metrics,
                    'predictions': predictions_data['predictions'],
                    'targets': predictions_data['targets'],
                    'flight_ids': predictions_data['flight_ids'],
                    'attack_info': attack_info
                })
                
            except Exception as e:
                print(f"警告: 加载 {attack_type}/{model_type} 失败: {e}")
                continue
        
        print(f"成功加载 {len(results)} 个实验结果")
        return results
    
    @staticmethod
    def get_confusion_matrix(precision: float, recall: float, 
                            total_positive: int, total_negative: int) -> Tuple[int, int, int, int]:
        """从precision和recall反推confusion matrix"""
        if recall == 0:
            TP = 0
            FN = total_positive
            FP = 0
            TN = total_negative
        elif precision == 0:
            TP = 0
            FP = max(1, total_negative // 2)  # 避免除零
            FN = total_positive
            TN = total_negative - FP
        else:
            TP = recall * total_positive
            FP = TP * (1/precision - 1)
            FN = total_positive - TP
            TN = total_negative - FP
        
        return int(max(0, TN)), int(max(0, FP)), int(max(0, FN)), int(max(0, TP))
    
    @staticmethod
    def extract_risk_factors(attack_info_row):
        """
        从attack_info.csv的attack_params字段提取风险因子
        
        Args:
            attack_info_row: attack_info DataFrame的一行数据
            
        Returns:
            Dict或None: {
                'magnitude': float,  # 米
                'duration': float,   # 秒
                'attack_type': str,
                'profile': str,      # step/ramp/sigmoid等
            }
        """
        attack_type = attack_info_row['attack_type']
        
        if attack_type == 'none' or not attack_info_row['attacked']:
            return None
        
        params_str = str(attack_info_row['attack_params'])
        
        try:
            # 解析字符串为字典（处理np.float64和np.int64）
            # 方法：使用正则表达式替换np类型
            params_str_cleaned = re.sub(r'np\.(float64|int64)\(([-\d.]+)\)', r'\2', params_str)
            params = ast.literal_eval(params_str_cleaned)
        except Exception as e:
            print(f"警告: 解析attack_params失败: {e}")
            print(f"  原始字符串: {params_str[:100]}...")
            return None
        
        magnitude = 0.0
        duration = 0.0
        profile = 'unknown'
        
        if attack_type == 'step':
            magnitude = params.get('magnitude', 0.0)
            duration = params.get('attack_duration', 3.0)
            profile = 'step'
        
        elif attack_type in ['drift_ramp', 'drift_sigmoid']:
            magnitude = params.get('M', 0.0)
            duration = params.get('T_drift', 0.0)
            profile = params.get('profile', 'ramp')
        
        elif attack_type == 'delay':
            delay_sec = params.get('delay_seconds', 0.0)
            magnitude = delay_sec * 5.0  # 延迟秒数×典型速度估算偏移
            # 延迟攻击持续到飞行结束，但用合理的平均飞行时长估算
            # 假设平均飞行200秒，延迟从开始后持续
            duration = min(200.0, 200.0 - 20.0)  # 平均飞行时长 - 起始缓冲
            profile = 'delay'
        
        elif attack_type in ['replay_same_hard', 'replay_other_soft']:
            # Replay攻击的偏移幅度难以直接量化，但通常较大
            # 使用segment_duration作为参考，假设越长偏移越大
            segment_dur = params.get('segment_duration', 10.0)
            magnitude = 15.0 + segment_dur * 0.5  # 基础15m + 时长相关
            duration = segment_dur
            profile = 'replay'
        
        elif attack_type in ['takeover_step', 'takeover_ramp']:
            # Takeover使用M而非magnitude字段
            magnitude = params.get('M', 0.0)
            # Takeover的持续时间从T_takeover或推算
            if attack_type == 'takeover_step':
                # Step类型的takeover，使用alpha混合时长估算
                duration = params.get('T_takeover', 10.0)
            else:  # takeover_ramp
                duration = params.get('T_takeover', 10.0)
            profile = params.get('offset_profile', 'step')
        
        return {
            'magnitude': float(magnitude),
            'duration': float(duration),
            'attack_type': attack_type,
            'profile': profile
        }
    
    def calculate_attack_risk_score(self, magnitude, duration, attack_type, profile):
        """
        计算单个攻击的风险分数
        
        ARS = α×CR + β×MFR + γ×CLR + δ×SAR
        
        Args:
            magnitude: 攻击幅度（米）
            duration: 持续时间（秒）
            attack_type: 攻击类型
            profile: 攻击轮廓
            
        Returns:
            float: 攻击风险分数 [0, 100+]
        """
        model = self.risk_model
        
        # A. 碰撞风险
        cr_thresholds = model['collision_risk']['thresholds']
        cr_weights = model['collision_risk']['weights']
        if magnitude < cr_thresholds[0]:
            cr = cr_weights[0]
        elif magnitude < cr_thresholds[1]:
            cr = cr_weights[1]
        elif magnitude < cr_thresholds[2]:
            cr = cr_weights[2]
        else:
            cr = cr_weights[3]
        cr = cr * magnitude  # 乘以实际偏移量
        
        # B. 任务失效风险
        mfr_thresholds = model['mission_failure_risk']['thresholds']
        mfr_weights = model['mission_failure_risk']['weights']
        if duration < mfr_thresholds[0]:
            mfr = mfr_weights[0]
        elif duration < mfr_thresholds[1]:
            mfr = mfr_weights[1]
        else:
            mfr = mfr_weights[2]
        mfr = mfr * duration  # 乘以实际时长
        
        # C. 操作失控风险
        clr = model['control_loss_risk']['type_weights'].get(profile, 5)
        
        # D. 隐蔽累积风险
        sar = model['stealth_accumulation_risk']['type_weights'].get(attack_type, 5)
        
        # 加权求和
        α = model['collision_risk']['coefficient']
        β = model['mission_failure_risk']['coefficient']
        γ = model['control_loss_risk']['coefficient']
        δ = model['stealth_accumulation_risk']['coefficient']
        
        ars = α * cr + β * mfr + γ * clr + δ * sar
        
        return ars
    
    def calculate_cost_sensitive_metrics(self, results: List[Dict], 
                                         C_FP: float = 1, C_FN: float = 50) -> pd.DataFrame:
        """
        指标体系1: 基于代价的安全-效率权衡
        
        Args:
            results: 实验结果列表
            C_FP: 误报代价 (默认1)
            C_FN: 漏报代价 (默认50)
        
        Returns:
            DataFrame containing cost-sensitive metrics
        """
        print(f"\n计算指标体系1: 代价权衡 (C_FP={C_FP}, C_FN={C_FN})...")
        metrics = []
        
        for exp in results:
            precision = exp['metrics']['precision']
            recall = exp['metrics']['recall']
            
            # 1.1 Weighted F-Score
            f_scores = {}
            for beta in [0.5, 1.0, 2.0, 5.0]:
                if precision + recall > 0:
                    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
                else:
                    f_beta = 0
                f_scores[f'F_{beta}'] = f_beta
            
            # 1.2 Expected Operational Cost (EOC)
            total_samples = len(exp['targets'])
            total_positive = int(exp['targets'].sum())
            total_negative = total_samples - total_positive
            
            TN, FP, FN, TP = self.get_confusion_matrix(
                precision, recall, total_positive, total_negative
            )
            
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
            
            EOC = C_FP * FPR + C_FN * FNR
            
            # 1.3 Cost-Benefit Score (CBS)
            attack_probability = 0.01
            attack_damage = 1000
            false_alarm_cost = 1
            normal_operation_frequency = 1
            
            benefit = recall * attack_probability * attack_damage
            cost = FPR * normal_operation_frequency * false_alarm_cost
            CBS = benefit - cost
            
            metrics.append({
                'attack_type': exp['attack_type'],
                'model_type': exp['model_type'],
                **f_scores,
                'EOC': EOC,
                'CBS': CBS,
                'FPR': FPR,
                'FNR': FNR,
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN
            })
        
        return pd.DataFrame(metrics)
    
    def calculate_detection_delay_metrics(self, results: List[Dict], 
                                         threshold: float = 0.5) -> pd.DataFrame:
        """
        指标体系2: 基于时效性的实时检测能力
        
        注意：由于当前数据格式限制，使用简化的计算方法
        
        Args:
            results: 实验结果列表
            threshold: 检测阈值 (默认0.5)
        
        Returns:
            DataFrame containing detection delay metrics
        """
        print(f"\n计算指标体系2: 时效性 (threshold={threshold})...")
        print("  ⚠️ 使用简化方法（数据格式限制）")
        
        metrics = []
        
        for exp in results:
            predictions = exp['predictions']
            targets = exp['targets']
            attack_info = exp['attack_info']
            
            # 统计攻击样本数
            num_attacks = len(attack_info[attack_info['attacked'] == True])
            
            # 简化的延迟估算：
            # 找到所有攻击样本的位置（target=1）
            attack_indices = np.where(targets == 1)[0]
            
            if len(attack_indices) > 0:
                # 对于每个连续的攻击段，估算首次检测延迟
                delays = []
                i = 0
                while i < len(attack_indices):
                    # 找到连续段的开始
                    segment_start = attack_indices[i]
                    segment_end = segment_start
                    
                    # 找到连续段的结束
                    while i + 1 < len(attack_indices) and attack_indices[i + 1] == attack_indices[i] + 1:
                        i += 1
                        segment_end = attack_indices[i]
                    
                    # 在这个段内找首次检测点
                    segment_predictions = predictions[segment_start:segment_end + 1]
                    detected_indices = np.where(segment_predictions > threshold)[0]
                    
                    if len(detected_indices) > 0:
                        first_detection = detected_indices[0]
                    else:
                        first_detection = len(segment_predictions)
                    
                    delay_seconds = first_detection * self.step_size * self.sampling_interval
                    delays.append(delay_seconds)
                    
                    i += 1
                
                # 2.1 Mean Time to Detection (MTTD)
                MTTD = np.mean(delays) if len(delays) > 0 else float('inf')
                
                # 2.2 Detection Delay Distribution
                P50 = np.percentile(delays, 50) if len(delays) > 0 else float('inf')
                P90 = np.percentile(delays, 90) if len(delays) > 0 else float('inf')
                P95 = np.percentile(delays, 95) if len(delays) > 0 else float('inf')
                P99 = np.percentile(delays, 99) if len(delays) > 0 else float('inf')
                
                # 2.3 Rapid Response Rate (RRR@k)
                k_values = [5, 10, 20]  # windows
                RRR_dict = {}
                for k in k_values:
                    threshold_seconds = k * self.step_size * self.sampling_interval
                    RRR_k = sum(d <= threshold_seconds for d in delays) / len(delays) if len(delays) > 0 else 0
                    RRR_dict[f'RRR@{k}'] = RRR_k
            else:
                MTTD = P50 = P90 = P95 = P99 = float('inf')
                RRR_dict = {f'RRR@{k}': 0 for k in [5, 10, 20]}
                delays = []
            
            metrics.append({
                'attack_type': exp['attack_type'],
                'model_type': exp['model_type'],
                'MTTD': MTTD,
                'P50_delay': P50,
                'P90_delay': P90,
                'P95_delay': P95,
                'P99_delay': P99,
                **RRR_dict,
                'num_segments': len(delays)
            })
        
        return pd.DataFrame(metrics)
    
    def calculate_reliability_metrics(self, results: List[Dict], 
                                      target_tpr: float = 0.99) -> pd.DataFrame:
        """
        指标体系3: 基于可靠性的运行质量
        
        Args:
            results: 实验结果列表
            target_tpr: 目标TPR (默认0.99)
        
        Returns:
            DataFrame containing reliability metrics
        """
        print(f"\n计算指标体系3: 可靠性 (target_TPR={target_tpr})...")
        metrics = []
        
        for exp in results:
            precision = exp['metrics']['precision']
            recall = exp['metrics']['recall']
            
            total_samples = len(exp['targets'])
            total_positive = int(exp['targets'].sum())
            total_negative = total_samples - total_positive
            
            TN, FP, FN, TP = self.get_confusion_matrix(
                precision, recall, total_positive, total_negative
            )
            
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            
            # 3.1 False Alarm Rate @ Operating Point
            # 当前recall即为TPR
            current_TPR = recall
            if current_TPR >= target_tpr:
                FAR_99 = FPR
            else:
                # 估算需要降低阈值，会增加FPR
                # 简化估算: 线性外推
                extrapolation_factor = target_tpr / max(current_TPR, 0.01)
                FAR_99 = min(FPR * extrapolation_factor, 1.0)
            
            # 3.2 Mean Time Between False Alarms (MTBFA)
            samples_per_flight = 3600  # 30分钟 * 60秒 * 2采样/秒
            attack_probability = 0.01
            normal_samples = int((1 - attack_probability) * total_samples)
            
            if FPR > 0:
                FP_count = FPR * normal_samples
                MTBFA_samples = normal_samples / max(FP_count, 1)
                MTBFA_flights = MTBFA_samples / samples_per_flight
            else:
                MTBFA_flights = float('inf')
            
            # 3.3 Availability Score
            inspection_time = 10  # 每次误报需10分钟检查
            flight_duration = 30   # 每次飞行30分钟
            downtime_ratio = FPR * (inspection_time / flight_duration)
            Availability = max(0, 1 - downtime_ratio)
            
            metrics.append({
                'attack_type': exp['attack_type'],
                'model_type': exp['model_type'],
                'FAR@99TPR': FAR_99,
                'MTBFA_flights': MTBFA_flights,
                'Availability': Availability,
                'current_TPR': current_TPR
            })
        
        # 3.4 Robustness Coefficient (跨攻击类型的稳定性)
        # 需要按模型汇总
        df = pd.DataFrame(metrics)
        
        return df
    
    def calculate_robustness_coefficient(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """计算鲁棒性系数 (跨攻击类型的AUC稳定性)"""
        print("\n计算鲁棒性系数...")
        
        # 加载AUC数据
        overall_path = os.path.join(self.output_dir, 'overall_comparison.csv')
        overall_df = pd.read_csv(overall_path)
        
        robustness_metrics = []
        
        for model in overall_df['model_type'].unique():
            model_data = overall_df[overall_df['model_type'] == model]
            auc_values = model_data['auc_roc'].values
            
            mean_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
            
            # 变异系数的倒数
            CV = std_auc / mean_auc if mean_auc > 0 else 1
            Robustness = 1 - CV
            
            robustness_metrics.append({
                'model_type': model,
                'mean_AUC': mean_auc,
                'std_AUC': std_auc,
                'CV': CV,
                'Robustness': Robustness
            })
        
        return pd.DataFrame(robustness_metrics)
    
    def calculate_risk_weighted_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        指标体系4: 基于风险的关键任务保障
        
        Args:
            results: 实验结果列表
        
        Returns:
            DataFrame containing risk-weighted metrics
        """
        print("\n计算指标体系4: 风险加权...")
        
        # 按模型汇总
        model_metrics = {}
        
        for exp in results:
            model = exp['model_type']
            attack = exp['attack_type']
            recall = exp['metrics']['recall']
            f1 = exp['metrics']['f1']
            precision = exp['metrics']['precision']
            
            if model not in model_metrics:
                model_metrics[model] = {
                    'weighted_recalls': [],
                    'weights': [],
                    'f1_scores': [],
                    'critical_stats': {'TP': 0, 'FN': 0},
                    'attack_recalls': {}
                }
            
            weight = self.attack_weights.get(attack, 1)
            model_metrics[model]['weighted_recalls'].append(recall * weight)
            model_metrics[model]['weights'].append(weight)
            model_metrics[model]['f1_scores'].append(f1)
            model_metrics[model]['attack_recalls'][attack] = recall
            
            # Critical Miss Rate (只统计重放攻击)
            if attack in ['replay_same_hard', 'replay_other_soft']:
                total_samples = len(exp['targets'])
                total_positive = int(exp['targets'].sum())
                
                TN, FP, FN, TP = self.get_confusion_matrix(
                    precision, recall, total_positive, total_samples - total_positive
                )
                model_metrics[model]['critical_stats']['TP'] += TP
                model_metrics[model]['critical_stats']['FN'] += FN
        
        # 计算汇总指标
        metrics = []
        for model, data in model_metrics.items():
            # 4.1 Risk-Weighted Detection Rate (RWDR)
            RWDR = sum(data['weighted_recalls']) / sum(data['weights'])
            
            # 4.2 Critical Miss Rate (CMR)
            TP = data['critical_stats']['TP']
            FN = data['critical_stats']['FN']
            CMR = FN / (TP + FN) if (TP + FN) > 0 else 1.0
            
            # 4.4 Worst-Case Performance (WCP)
            WCP = min(data['f1_scores'])
            
            # 最佳情况性能
            BCP = max(data['f1_scores'])
            
            # 平均性能
            AVG_F1 = np.mean(data['f1_scores'])
            
            metrics.append({
                'model_type': model,
                'RWDR': RWDR,
                'CMR': CMR,
                'WCP': WCP,
                'BCP': BCP,
                'AVG_F1': AVG_F1,
                'performance_gap': BCP - WCP
            })
        
        return pd.DataFrame(metrics)
    
    def calculate_advanced_risk_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        计算风险导向的创新指标
        包含: RWMR, MSDR, DS-FNR, CTDR
        
        Args:
            results: 实验结果列表
            
        Returns:
            DataFrame containing advanced risk metrics
        """
        print("\n计算风险导向创新指标 (RWMR, MSDR, DS-FNR, CTDR)...")
        
        metrics = []
        
        for exp in results:
            attack_type = exp['attack_type']
            model_type = exp['model_type']
            attack_info = exp['attack_info']
            predictions = exp['predictions']
            targets = exp['targets']
            flight_ids = exp['flight_ids']
            
            # 提取飞行级别的风险因子
            flight_risk_factors = {}  # flight -> risk_factors
            
            for idx, row in attack_info.iterrows():
                factors = self.extract_risk_factors(row)
                flight_id = row['flight']
                
                if factors is not None:
                    score = self.calculate_attack_risk_score(
                        factors['magnitude'],
                        factors['duration'],
                        factors['attack_type'],
                        factors['profile']
                    )
                    flight_risk_factors[flight_id] = {
                        'score': score,
                        'magnitude': factors['magnitude'],
                        'duration': factors['duration']
                    }
                else:
                    flight_risk_factors[flight_id] = {
                        'score': 0.0,
                        'magnitude': 0.0,
                        'duration': 0.0
                    }
            
            # 将风险因子映射到窗口级别
            n_samples = len(targets)
            risk_scores = np.zeros(n_samples)
            magnitudes = np.zeros(n_samples)
            durations = np.zeros(n_samples)
            
            for i in range(n_samples):
                flight_id = flight_ids[i]
                if flight_id in flight_risk_factors:
                    risk_scores[i] = flight_risk_factors[flight_id]['score']
                    magnitudes[i] = flight_risk_factors[flight_id]['magnitude']
                    durations[i] = flight_risk_factors[flight_id]['duration']
            
            # 转换为numpy数组
            y_true = targets
            y_pred = (predictions > 0.5).astype(int)
            
            # 计算指标1: RWMR (Risk-Weighted Miss Rate)
            attack_mask = (y_true == 1)
            fn_mask = (y_true == 1) & (y_pred == 0)
            
            total_attack_risk = risk_scores[attack_mask].sum()
            missed_attack_risk = risk_scores[fn_mask].sum()
            
            RWMR = missed_attack_risk / total_attack_risk if total_attack_risk > 0 else 0.0
            
            # 计算指标2: MSDR (Magnitude-Stratified Detection Rate)
            MSDR_low = MSDR_mid = MSDR_high = None
            
            for bin_name, (low, high) in self.magnitude_bins.items():
                bin_mask = (magnitudes >= low) & (magnitudes < high) & attack_mask
                
                if bin_mask.sum() > 0:
                    tp = ((y_pred == 1) & bin_mask).sum()
                    fn = ((y_pred == 0) & bin_mask).sum()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    if bin_name == 'low':
                        MSDR_low = tpr
                    elif bin_name == 'mid':
                        MSDR_mid = tpr
                    elif bin_name == 'high':
                        MSDR_high = tpr
            
            # 计算指标3: DS-FNR (Duration-Sensitive False Negative Rate)
            total_attack_duration = durations[attack_mask].sum()
            missed_attack_duration = durations[fn_mask].sum()
            
            DS_FNR = missed_attack_duration / total_attack_duration if total_attack_duration > 0 else 0.0
            
            # 计算指标4: CTDR (Critical Threshold Detection Rate)
            critical_mag_threshold = self.risk_model['critical_thresholds']['magnitude']
            critical_dur_threshold = self.risk_model['critical_thresholds']['duration']
            
            critical_mask = (y_true == 1) & (
                (magnitudes > critical_mag_threshold) | 
                (durations > critical_dur_threshold)
            )
            
            if critical_mask.sum() > 0:
                critical_detected = ((y_pred == 1) & critical_mask).sum()
                critical_total = critical_mask.sum()
                CTDR = critical_detected / critical_total
            else:
                CTDR = None
            
            # 计算传统FNR用于对比
            traditional_FNR = fn_mask.sum() / attack_mask.sum() if attack_mask.sum() > 0 else 0.0
            
            # 计算平均风险分数
            avg_risk_score = risk_scores[attack_mask].mean() if attack_mask.sum() > 0 else 0.0
            
            metrics.append({
                'attack_type': attack_type,
                'model_type': model_type,
                'RWMR': RWMR,
                'MSDR_low': MSDR_low,
                'MSDR_mid': MSDR_mid,
                'MSDR_high': MSDR_high,
                'DS_FNR': DS_FNR,
                'CTDR': CTDR,
                'traditional_FNR': traditional_FNR,
                'avg_risk_score': avg_risk_score,
                'total_attacks': int(attack_mask.sum()),
                'missed_attacks': int(fn_mask.sum()),
                'critical_attacks': int(critical_mask.sum()) if CTDR is not None else 0
            })
        
        return pd.DataFrame(metrics)
    
    def calculate_edge_deployment_metrics(self, results: List[Dict]) -> pd.DataFrame:
        """
        指标体系5: 基于资源约束的边缘部署适配性
        
        Args:
            results: 实验结果列表
        
        Returns:
            DataFrame containing edge deployment metrics
        """
        print("\n计算指标体系5: 边缘部署适配性...")
        
        # 加载AUC数据
        overall_path = os.path.join(self.output_dir, 'overall_comparison.csv')
        overall_df = pd.read_csv(overall_path)
        
        # 按模型汇总AUC
        model_auc = overall_df.groupby('model_type')['auc_roc'].mean().to_dict()
        
        metrics = []
        
        for model, params in self.model_params.items():
            if model not in model_auc:
                continue
            
            auc = model_auc[model]
            
            # 5.1 & 5.3 Model Size
            model_size_MB = params * 4 / (1024**2)  # float32
            
            # 5.3 Memory Footprint Score
            max_params = max(self.model_params.values())
            min_params = min(self.model_params.values())
            Memory_Score = 1 - (params - min_params) / (max_params - min_params)
            
            # 5.1 Performance-Efficiency Ratio (PER)
            # 简化版: 使用模型大小作为资源消耗指标
            PER = auc / (model_size_MB ** 0.5)
            
            # 5.4 Energy-Performance Trade-off
            # 估算: 参数越多，能耗越高
            Energy_Score = auc / (params / 1e6)
            
            metrics.append({
                'model_type': model,
                'model_params': params,
                'model_size_MB': model_size_MB,
                'mean_AUC': auc,
                'Memory_Score': Memory_Score,
                'PER': PER,
                'Energy_Score': Energy_Score
            })
        
        return pd.DataFrame(metrics)
    
    def generate_comprehensive_summary(self, all_metrics: Dict[str, pd.DataFrame], 
                                      output_dir: str):
        """生成综合分析报告"""
        print("\n生成综合分析报告...")
        
        report_lines = [
            "# 电力系统无人机巡检场景 - 指标体系分析报告",
            "",
            "## 📋 概述",
            "",
            f"- **分析日期**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **实验数量**: {len(all_metrics['cost_sensitive']) if 'cost_sensitive' in all_metrics else 0}",
            f"- **模型数量**: 7 (CNN, LSTM, BiLSTM, GRU, CNN-LSTM, TCN, Transformer)",
            f"- **攻击类型**: 8",
            "",
            "---",
            "",
            "## 📊 5套指标体系对比",
            "",
        ]
        
        # 指标体系说明
        metric_systems = [
            ("1. 代价权衡指标", "Cost-Sensitive Safety-Efficiency Trade-off", 
             "关注误报和漏报的代价差异，适用于不对称风险场景"),
            ("2. 时效性指标", "Real-time Detection Capability", 
             "关注检测延迟和响应速度，适用于实时性要求高的场景"),
            ("3. 可靠性指标", "Operational Reliability", 
             "关注系统稳定性和误报控制，适用于长期运行场景"),
            ("4. 风险加权指标", "Risk-Based Mission Assurance", 
             "关注高危攻击的检测能力，适用于关键任务保障"),
            ("5. 边缘部署指标", "Edge Deployment Suitability", 
             "关注资源消耗和性能平衡，适用于资源受限环境")
        ]
        
        for name_cn, name_en, description in metric_systems:
            report_lines.extend([
                f"### {name_cn}",
                f"**{name_en}**",
                "",
                description,
                ""
            ])
        
        report_lines.extend([
            "---",
            "",
            "## 🎯 关键发现",
            "",
        ])
        
        # 分析各指标体系的区分度
        if 'cost_sensitive' in all_metrics:
            df = all_metrics['cost_sensitive']
            eoc_by_model = df.groupby('model_type')['EOC'].mean()
            best_model_cost = eoc_by_model.idxmin()
            worst_model_cost = eoc_by_model.idxmax()
            
            report_lines.extend([
                "### 指标体系1: 代价权衡",
                "",
                f"- **最佳模型**: {best_model_cost} (EOC = {eoc_by_model[best_model_cost]:.4f})",
                f"- **最差模型**: {worst_model_cost} (EOC = {eoc_by_model[worst_model_cost]:.4f})",
                f"- **区分度**: CV = {eoc_by_model.std() / eoc_by_model.mean():.4f}",
                ""
            ])
        
        if 'detection_delay' in all_metrics:
            df = all_metrics['detection_delay']
            mttd_by_model = df.groupby('model_type')['MTTD'].mean()
            best_model_delay = mttd_by_model.idxmin()
            
            report_lines.extend([
                "### 指标体系2: 时效性",
                "",
                f"- **最快检测**: {best_model_delay} (MTTD = {mttd_by_model[best_model_delay]:.2f}s)",
                f"- **平均检测延迟**: {mttd_by_model.mean():.2f}s",
                ""
            ])
        
        if 'risk_weighted' in all_metrics:
            df = all_metrics['risk_weighted']
            best_model_risk = df.loc[df['RWDR'].idxmax(), 'model_type']
            best_rwdr = df['RWDR'].max()
            
            report_lines.extend([
                "### 指标体系4: 风险加权",
                "",
                f"- **最佳模型**: {best_model_risk} (RWDR = {best_rwdr:.4f})",
                f"- **关键任务漏报率**: 详见CSV文件",
                ""
            ])
        
        if 'edge_deployment' in all_metrics:
            df = all_metrics['edge_deployment']
            best_model_edge = df.loc[df['PER'].idxmax(), 'model_type']
            best_per = df['PER'].max()
            
            report_lines.extend([
                "### 指标体系5: 边缘部署",
                "",
                f"- **最佳性能效率比**: {best_model_edge} (PER = {best_per:.4f})",
                f"- **最轻量模型**: {df.loc[df['model_size_MB'].idxmin(), 'model_type']} "
                f"({df['model_size_MB'].min():.2f} MB)",
                ""
            ])
        
        report_lines.extend([
            "---",
            "",
            "## 💡 电力巡检场景建议",
            "",
            "### 推荐指标体系",
            "",
            "1. **首选**: 指标体系4 (风险加权) - 重视重放攻击检测",
            "2. **次选**: 指标体系2 (时效性) - 确保快速响应",
            "3. **参考**: 指标体系1 (代价权衡) - 平衡误报和漏报",
            "",
            "### 模型选择建议",
            "",
            "根据不同约束条件的推荐：",
            "",
            "- **性能优先**: 选择RWDR最高的模型",
            "- **实时性优先**: 选择MTTD最低的模型",
            "- **资源受限**: 选择PER最高的轻量级模型",
            "- **平衡选择**: 综合考虑多个指标的排名",
            "",
            "---",
            "",
            "## 📈 详细数据",
            "",
            "所有详细指标数据已保存在以下CSV文件中：",
            "",
            "- `metrics1_cost_sensitive.csv` - 代价权衡指标",
            "- `metrics2_detection_delay.csv` - 时效性指标",
            "- `metrics3_reliability.csv` - 可靠性指标",
            "- `metrics3_robustness.csv` - 鲁棒性系数",
            "- `metrics4_risk_weighted.csv` - 风险加权指标",
            "- `metrics5_edge_deployment.csv` - 边缘部署指标",
            "- `comprehensive_rankings.csv` - 综合排名",
            "",
            "可视化图表保存在 `visualizations/` 目录下。",
            ""
        ])
        
        # 保存报告
        report_path = os.path.join(output_dir, 'POWER_GRID_REPORT.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 综合报告已保存: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("电力系统无人机巡检场景 - 指标体系计算")
    print("Power Grid UAV Inspection - Metrics Calculation")
    print("="*80)
    
    # 初始化计算器
    calculator = PowerGridMetrics(output_dir='../step3_multiModel/output')
    
    # 创建输出目录
    output_dir = './output/power_grid_analysis'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # 加载数据
    results = calculator.load_experiment_results()
    
    if len(results) == 0:
        print("错误: 未能加载任何实验结果!")
        return
    
    # 存储所有指标
    all_metrics = {}
    
    # 计算各指标体系
    print("\n" + "="*80)
    print("开始计算各项指标...")
    print("="*80)
    
    # 指标体系1: 代价权衡
    metrics1 = calculator.calculate_cost_sensitive_metrics(results, C_FP=1, C_FN=50)
    metrics1.to_csv(os.path.join(output_dir, 'metrics1_cost_sensitive.csv'), index=False)
    all_metrics['cost_sensitive'] = metrics1
    print(f"✓ 指标体系1完成: {len(metrics1)} 条记录")
    
    # 指标体系2: 时效性
    metrics2 = calculator.calculate_detection_delay_metrics(results, threshold=0.5)
    metrics2.to_csv(os.path.join(output_dir, 'metrics2_detection_delay.csv'), index=False)
    all_metrics['detection_delay'] = metrics2
    print(f"✓ 指标体系2完成: {len(metrics2)} 条记录")
    
    # 指标体系3: 可靠性
    metrics3 = calculator.calculate_reliability_metrics(results, target_tpr=0.99)
    metrics3.to_csv(os.path.join(output_dir, 'metrics3_reliability.csv'), index=False)
    all_metrics['reliability'] = metrics3
    print(f"✓ 指标体系3完成: {len(metrics3)} 条记录")
    
    # 指标体系3附加: 鲁棒性系数
    metrics3_robust = calculator.calculate_robustness_coefficient(metrics3)
    metrics3_robust.to_csv(os.path.join(output_dir, 'metrics3_robustness.csv'), index=False)
    all_metrics['robustness'] = metrics3_robust
    print(f"✓ 鲁棒性系数完成: {len(metrics3_robust)} 条记录")
    
    # 指标体系4: 风险加权
    metrics4 = calculator.calculate_risk_weighted_metrics(results)
    metrics4.to_csv(os.path.join(output_dir, 'metrics4_risk_weighted.csv'), index=False)
    all_metrics['risk_weighted'] = metrics4
    print(f"✓ 指标体系4完成: {len(metrics4)} 条记录")
    
    # 指标体系5: 边缘部署
    metrics5 = calculator.calculate_edge_deployment_metrics(results)
    metrics5.to_csv(os.path.join(output_dir, 'metrics5_edge_deployment.csv'), index=False)
    all_metrics['edge_deployment'] = metrics5
    print(f"✓ 指标体系5完成: {len(metrics5)} 条记录")
    
    # 指标体系6: 风险导向创新指标 (新增)
    metrics6 = calculator.calculate_advanced_risk_metrics(results)
    metrics6.to_csv(os.path.join(output_dir, 'metrics6_advanced_risk.csv'), index=False)
    all_metrics['advanced_risk'] = metrics6
    print(f"✓ 指标体系6完成 (风险导向): {len(metrics6)} 条记录")
    
    # 生成综合报告
    calculator.generate_comprehensive_summary(all_metrics, output_dir)
    
    print("\n" + "="*80)
    print(f"✓ 所有指标计算完成!")
    print(f"✓ 结果保存在: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
