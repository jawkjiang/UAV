"""
Statistics - 统计分析工具
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_correlation(x: np.ndarray, y: np.ndarray, 
                         method: str = 'pearson') -> Tuple[float, float]:
    """
    计算相关系数
    
    Args:
        x, y: 数据数组
        method: 'pearson', 'spearman', 或 'kendall'
    
    Returns:
        (correlation, p_value)
    """
    if method == 'pearson':
        r, p = stats.pearsonr(x, y)
    elif method == 'spearman':
        r, p = stats.spearmanr(x, y)
    elif method == 'kendall':
        r, p = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return r, p


def perform_ttest(group1: np.ndarray, group2: np.ndarray, 
                 paired: bool = False) -> Tuple[float, float]:
    """
    执行t检验
    
    Args:
        group1, group2: 两组数据
        paired: 是否配对t检验
    
    Returns:
        (t_statistic, p_value)
    """
    if paired:
        t, p = stats.ttest_rel(group1, group2)
    else:
        t, p = stats.ttest_ind(group1, group2)
    
    return t, p


def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    计算Cohen's d效应量
    
    Args:
        group1, group2: 两组数据
    
    Returns:
        Cohen's d
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    return d


def normalize_scores(scores: np.ndarray, 
                     higher_is_better: bool = True) -> np.ndarray:
    """
    归一化分数到[0, 1]
    
    Args:
        scores: 分数数组
        higher_is_better: 是否越高越好
    
    Returns:
        归一化后的分数
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.ones_like(scores)
    
    normalized = (scores - min_score) / (max_score - min_score)
    
    if not higher_is_better:
        normalized = 1 - normalized
    
    return normalized


def calculate_composite_score(metrics_dict: Dict[str, np.ndarray],
                              weights: Dict[str, float],
                              higher_is_better: Dict[str, bool]) -> np.ndarray:
    """
    计算综合评分
    
    Args:
        metrics_dict: 指标字典 {metric_name: values}
        weights: 权重字典 {metric_name: weight}
        higher_is_better: 方向字典 {metric_name: bool}
    
    Returns:
        综合评分数组
    """
    # 归一化所有指标
    normalized_metrics = {}
    for metric, values in metrics_dict.items():
        is_higher_better = higher_is_better.get(metric, True)
        normalized_metrics[metric] = normalize_scores(values, is_higher_better)
    
    # 计算加权和
    composite = np.zeros(len(next(iter(metrics_dict.values()))))
    total_weight = sum(weights.values())
    
    for metric, weight in weights.items():
        if metric in normalized_metrics:
            composite += normalized_metrics[metric] * (weight / total_weight)
    
    return composite


def rank_models(scores: np.ndarray, ascending: bool = False) -> np.ndarray:
    """
    对模型进行排名
    
    Args:
        scores: 分数数组
        ascending: 是否升序排名
    
    Returns:
        排名数组（1-based）
    """
    if ascending:
        ranks = stats.rankdata(scores, method='min')
    else:
        ranks = stats.rankdata(-scores, method='min')
    
    return ranks.astype(int)


def calculate_pareto_frontier(x: np.ndarray, y: np.ndarray,
                              maximize_x: bool = True,
                              maximize_y: bool = True) -> np.ndarray:
    """
    计算Pareto前沿
    
    Args:
        x, y: 两个目标的值
        maximize_x: 是否最大化x
        maximize_y: 是否最大化y
    
    Returns:
        Pareto前沿点的索引
    """
    # 转换为最大化问题
    x_transformed = x if maximize_x else -x
    y_transformed = y if maximize_y else -y
    
    # 找到Pareto前沿
    is_pareto = np.ones(len(x), dtype=bool)
    
    for i in range(len(x)):
        if is_pareto[i]:
            # 检查是否被其他点支配
            dominated = (x_transformed >= x_transformed[i]) & (y_transformed >= y_transformed[i])
            dominated[i] = False
            strictly_better = (x_transformed > x_transformed[i]) | (y_transformed > y_transformed[i])
            is_pareto[dominated & strictly_better] = False
    
    return np.where(is_pareto)[0]


def calculate_confidence_interval(data: np.ndarray, 
                                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    计算置信区间
    
    Args:
        data: 数据数组
        confidence: 置信水平
    
    Returns:
        (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean - interval, mean + interval


def perform_anova(groups: List[np.ndarray]) -> Tuple[float, float]:
    """
    执行单因素方差分析
    
    Args:
        groups: 多组数据列表
    
    Returns:
        (F_statistic, p_value)
    """
    f, p = stats.f_oneway(*groups)
    return f, p


def calculate_rank_correlation_matrix(df: pd.DataFrame, 
                                     metrics: List[str]) -> pd.DataFrame:
    """
    计算排名相关性矩阵
    
    Args:
        df: DataFrame包含模型和指标
        metrics: 指标列表
    
    Returns:
        相关性矩阵DataFrame
    """
    # 计算每个指标的排名
    rank_df = pd.DataFrame()
    for metric in metrics:
        rank_df[metric] = df[metric].rank(ascending=False)
    
    # 计算Spearman相关系数
    corr_matrix = rank_df.corr(method='spearman')
    
    return corr_matrix


def identify_outliers(data: np.ndarray, method: str = 'iqr',
                     threshold: float = 1.5) -> np.ndarray:
    """
    识别异常值
    
    Args:
        data: 数据数组
        method: 'iqr' 或 'zscore'
        threshold: 阈值（IQR倍数或Z分数）
    
    Returns:
        异常值的布尔数组
    """
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = (data < lower) | (data > upper)
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers


def calculate_percentage_improvement(baseline: float, 
                                    improved: float,
                                    higher_is_better: bool = True) -> float:
    """
    计算改进百分比
    
    Args:
        baseline: 基线值
        improved: 改进值
        higher_is_better: 是否越高越好
    
    Returns:
        改进百分比
    """
    if baseline == 0:
        return np.inf if improved > 0 else 0
    
    improvement = (improved - baseline) / baseline * 100
    
    if not higher_is_better:
        improvement = -improvement
    
    return improvement
