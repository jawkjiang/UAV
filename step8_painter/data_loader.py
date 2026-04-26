"""
Data Loader - 统一的数据加载接口
从各个step的输出中加载数据
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from config import (
    STEP5_OUTPUT, STEP6_OUTPUT, STEP7_OUTPUT, 
    PAPER_ANALYSIS_OUTPUT, MODELS
)

logger = logging.getLogger(__name__)


class PaperDataLoader:
    """论文数据加载器"""
    
    def __init__(self):
        self.step5_output = STEP5_OUTPUT
        self.step6_output = STEP6_OUTPUT
        self.step7_output = STEP7_OUTPUT
        self.paper_output = PAPER_ANALYSIS_OUTPUT
        
    def load_traditional_metrics(self) -> pd.DataFrame:
        """
        加载传统指标（Precision/Recall/F1）
        来源：step5_timegan/output/test_metrics_*.json
        """
        data = []
        
        for model in MODELS:
            json_path = self.step5_output / f'test_metrics_{model}.json'
            if not json_path.exists():
                logger.warning(f"Missing: {json_path}")
                continue
                
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                
            data.append({
                'model': model,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'pr_auc': metrics['pr_auc'],
                'roc_auc': metrics['roc_auc'],
                'tp': int(metrics['true_positives']),
                'fp': int(metrics['false_positives']),
                'tn': int(metrics['true_negatives']),
                'fn': int(metrics['false_negatives'])
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded traditional metrics for {len(df)} models")
        return df
    
    def load_time_aware_metrics(self) -> pd.DataFrame:
        """
        加载时间感知指标（DR@Δt/ADD/MTBFA）
        来源：step6_newMetrcisWithTimegan/output/time_aware_metrics/overall_metrics.csv
        """
        csv_path = self.step6_output / 'time_aware_metrics' / 'overall_metrics.csv'
        
        if not csv_path.exists():
            logger.error(f"Missing: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded time-aware metrics for {len(df)} models")
        return df
    
    def load_fp_window_event_comparison(self) -> pd.DataFrame:
        """
        加载FP窗口vs事件对比数据
        来源：paper_analysis/output/tables/precision_mtbfa_comparison.csv
        """
        csv_path = self.paper_output / 'tables' / 'precision_mtbfa_comparison.csv'
        
        if not csv_path.exists():
            logger.warning(f"Missing: {csv_path}, generating from raw data...")
            return self._generate_fp_comparison()
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded FP comparison data for {len(df)} models")
        return df
    
    def _generate_fp_comparison(self) -> pd.DataFrame:
        """从原始数据生成FP对比"""
        trad = self.load_traditional_metrics()
        time_aware = self.load_time_aware_metrics()
        
        merged = trad.merge(time_aware[['model', 'n_false_alarms', 'MTBFA']], on='model')
        merged['fp_windows'] = merged['fp']
        merged['fp_events'] = merged['n_false_alarms']
        merged['aggregation_ratio'] = merged['fp_events'] / merged['fp_windows']
        
        return merged
    
    def load_fp_duration_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载FP持续时长数据
        来源：paper_analysis/output/fp_duration/{model}_fp_events.csv
        """
        fp_data = {}
        
        for model in MODELS:
            csv_path = self.paper_output / 'fp_duration' / f'{model}_fp_events.csv'
            if csv_path.exists():
                fp_data[model] = pd.read_csv(csv_path)
            else:
                logger.warning(f"Missing FP duration data for {model}")
        
        logger.info(f"Loaded FP duration data for {len(fp_data)} models")
        return fp_data
    
    def load_fp_duration_statistics(self) -> pd.DataFrame:
        """
        加载FP持续时长统计
        来源：paper_analysis/output/fp_duration/fp_duration_statistics.csv
        """
        csv_path = self.paper_output / 'fp_duration' / 'fp_duration_statistics.csv'
        
        if not csv_path.exists():
            logger.error(f"Missing: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded FP duration statistics for {len(df)} models")
        return df
    
    def load_per_attack_metrics(self) -> Dict[str, Dict[str, Dict]]:
        """
        加载分攻击类型指标
        来源：step6_newMetrcisWithTimegan/output/time_aware_metrics/{model}_per_attack_metrics.csv
        返回格式：{model: {attack_type: {metric: value}}}
        """
        per_attack_data = {}
        
        for model in MODELS:
            csv_path = self.step6_output / 'time_aware_metrics' / f'{model}_per_attack_metrics.csv'
            if not csv_path.exists():
                logger.warning(f"Missing per-attack data for {model}")
                continue
                
            df = pd.read_csv(csv_path)
            model_data = {}
            
            for _, row in df.iterrows():
                attack_type = row['attack_type']
                
                model_data[attack_type] = {
                    'DR@5s': row.get('DR@5s', 0),
                    'ADD': row.get('ADD', 0),
                    'DR@10s': row.get('DR@10s', 0),
                    'n_instances': row.get('n_instances', 0)
                }
            
            per_attack_data[model] = model_data
        
        logger.info(f"Loaded per-attack metrics for {len(per_attack_data)} models")
        return per_attack_data
    
    def load_efficiency_metrics(self) -> pd.DataFrame:
        """
        加载计算效率指标
        来源：step7_performance_analysis/output/comprehensive_report.csv
        """
        csv_path = self.step7_output / 'comprehensive_report.csv'
        
        if not csv_path.exists():
            logger.warning(f"Missing: {csv_path}, trying alternative sources...")
            return self._load_efficiency_from_separate_files()
        
        df = pd.read_csv(csv_path)
        
        # 统一列名：将实际的列名映射到期望的列名
        column_mapping = {
            'latency_mean_ms': 'inference_time_ms',
            'total_params': 'parameters',
            'params_millions': 'parameters',  # 如果没有total_params就用这个
            'model_size_mb': 'memory_mb',
            'flops': 'flops'
        }
        
        # 应用列名映射（只映射存在的列）
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # 确保parameters列存在（优先使用total_params，否则从params_millions计算）
        if 'parameters' not in df.columns:
            if 'total_params' in df.columns:
                df['parameters'] = df['total_params']
            elif 'params_millions' in df.columns:
                df['parameters'] = df['params_millions'] * 1e6
        
        # 添加训练时间（如果没有就设为0，因为这个数据可能不在comprehensive_report中）
        if 'training_time_minutes' not in df.columns:
            df['training_time_minutes'] = 0
        
        logger.info(f"Loaded efficiency metrics for {len(df)} models")
        return df
    
    def _load_efficiency_from_separate_files(self) -> pd.DataFrame:
        """从分离的文件加载效率指标"""
        perf_path = self.step7_output / 'performance_metrics.csv'
        complexity_path = self.step7_output / 'model_complexity.csv'
        
        if not perf_path.exists() or not complexity_path.exists():
            logger.error("Missing efficiency data files")
            return pd.DataFrame()
        
        perf = pd.read_csv(perf_path)
        complexity = pd.read_csv(complexity_path)
        
        # 合并数据
        merged = perf.merge(complexity, on='model')
        return merged
    
    def load_detailed_delays(self) -> pd.DataFrame:
        """
        加载详细的检测延迟数据
        来源：step6_newMetrcisWithTimegan/output/time_aware_metrics/detailed_delays.csv
        """
        csv_path = self.step6_output / 'time_aware_metrics' / 'detailed_delays.csv'
        
        if not csv_path.exists():
            logger.error(f"Missing: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} detailed delay records")
        return df
    
    def get_dr_at_thresholds(self, thresholds: List[float] = None) -> pd.DataFrame:
        """
        获取不同时间阈值下的DR
        
        Args:
            thresholds: 时间阈值列表，默认[0.5, 1, 2, 5, 10, 15, 30]
        
        Returns:
            DataFrame with columns: model, threshold, dr
        """
        if thresholds is None:
            thresholds = [0.5, 1, 1.5, 2, 3, 5, 10, 15, 30]
        
        time_aware = self.load_time_aware_metrics()
        
        data = []
        for _, row in time_aware.iterrows():
            model = row['model']
            for t in thresholds:
                col_name = f'DR@{t}s'
                if col_name in row:
                    data.append({
                        'model': model,
                        'threshold': t,
                        'dr': row[col_name]
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Extracted DR@Δt for {len(thresholds)} thresholds")
        return df
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """一次性加载所有数据"""
        logger.info("Loading all data...")
        
        all_data = {
            'traditional': self.load_traditional_metrics(),
            'time_aware': self.load_time_aware_metrics(),
            'fp_comparison': self.load_fp_window_event_comparison(),
            'fp_duration_stats': self.load_fp_duration_statistics(),
            'fp_duration_events': self.load_fp_duration_data(),
            'per_attack': self.load_per_attack_metrics(),
            'efficiency': self.load_efficiency_metrics(),
            'detailed_delays': self.load_detailed_delays(),
            'dr_thresholds': self.get_dr_at_thresholds()
        }
        
        logger.info("All data loaded successfully")
        return all_data
    
    def validate_data(self) -> Dict[str, bool]:
        """验证数据完整性"""
        validation = {}
        
        validation['traditional'] = (self.step5_output / 'test_metrics_cnn.json').exists()
        validation['time_aware'] = (self.step6_output / 'time_aware_metrics' / 'overall_metrics.csv').exists()
        validation['fp_duration'] = (self.paper_output / 'fp_duration' / 'fp_duration_statistics.csv').exists()
        validation['efficiency'] = (self.step7_output / 'comprehensive_report.csv').exists()
        
        logger.info(f"Data validation: {validation}")
        return validation


# ============================================================================
# 便捷函数
# ============================================================================
def load_data() -> PaperDataLoader:
    """创建数据加载器实例"""
    return PaperDataLoader()


if __name__ == '__main__':
    # 测试数据加载
    logging.basicConfig(level=logging.INFO)
    
    loader = PaperDataLoader()
    
    print("\n" + "="*80)
    print("Data Validation")
    print("="*80)
    validation = loader.validate_data()
    for key, status in validation.items():
        status_str = "✅" if status else "❌"
        print(f"{status_str} {key}")
    
    print("\n" + "="*80)
    print("Loading Sample Data")
    print("="*80)
    
    # 加载传统指标
    trad = loader.load_traditional_metrics()
    print(f"\n📊 Traditional Metrics ({len(trad)} models):")
    print(trad[['model', 'precision', 'recall', 'f1']].to_string(index=False))
    
    # 加载时间感知指标
    time_aware = loader.load_time_aware_metrics()
    print(f"\n⏱️  Time-Aware Metrics ({len(time_aware)} models):")
    print(time_aware[['model', 'ADD', 'MTBFA', 'DR@5s']].to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ Data Loader Test Complete")
    print("="*80)


# 为了兼容性，创建别名
DataLoader = PaperDataLoader
