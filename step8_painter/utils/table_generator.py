"""
Table Generator - 表格生成工具
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
import logging

from config import TABLE_FORMATS, LATEX_TABLE_TEMPLATE, get_table_path

logger = logging.getLogger(__name__)


class TableGenerator:
    """表格生成器"""
    
    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            from config import TABLES_DIR
            output_dir = TABLES_DIR
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_table(self, df: pd.DataFrame, name: str, 
                   caption: str = "", label: str = "",
                   formats: List[str] = None):
        """
        保存表格为多种格式
        
        Args:
            df: DataFrame
            name: 表格名称
            caption: 表格标题（LaTeX）
            label: 表格标签（LaTeX）
            formats: 输出格式列表
        """
        if formats is None:
            formats = TABLE_FORMATS
        
        saved_files = []
        
        for fmt in formats:
            if fmt == 'csv':
                filepath = self.output_dir / f"{name}.csv"
                df.to_csv(filepath, index=False)
                saved_files.append(filepath)
                
            elif fmt == 'latex':
                filepath = self.output_dir / f"{name}.tex"
                latex_content = self._generate_latex_table(df, caption, label)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                saved_files.append(filepath)
                
            elif fmt == 'markdown':
                filepath = self.output_dir / f"{name}.md"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(df.to_markdown(index=False))
                saved_files.append(filepath)
                
            elif fmt == 'excel':
                filepath = self.output_dir / f"{name}.xlsx"
                df.to_excel(filepath, index=False)
                saved_files.append(filepath)
        
        for filepath in saved_files:
            logger.info(f"Saved table: {filepath}")
        
        return saved_files
    
    def _generate_latex_table(self, df: pd.DataFrame, 
                             caption: str, label: str) -> str:
        """生成LaTeX表格"""
        # 生成表格内容
        latex_body = df.to_latex(index=False, escape=False, 
                                 column_format='l' + 'c' * (len(df.columns) - 1))
        
        # 应用模板
        latex_content = LATEX_TABLE_TEMPLATE.format(
            caption=caption,
            label=label,
            content=latex_body
        )
        
        return latex_content
    
    def format_numeric_columns(self, df: pd.DataFrame, 
                              columns: List[str], 
                              format_str: str = '.4f') -> pd.DataFrame:
        """格式化数值列"""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:{format_str}}" if pd.notna(x) else "")
        return df
    
    def add_ranking_column(self, df: pd.DataFrame, 
                          metric_col: str, 
                          ascending: bool = False,
                          rank_col_name: str = 'Rank') -> pd.DataFrame:
        """添加排名列"""
        df = df.copy()
        df[rank_col_name] = df[metric_col].rank(ascending=ascending, method='min').astype(int)
        return df
    
    def highlight_best(self, df: pd.DataFrame, 
                      metric_cols: List[str],
                      higher_is_better: Dict[str, bool]) -> pd.DataFrame:
        """高亮最佳值（用于LaTeX）"""
        df = df.copy()
        
        for col in metric_cols:
            if col not in df.columns:
                continue
            
            is_higher_better = higher_is_better.get(col, True)
            
            if is_higher_better:
                best_idx = df[col].idxmax()
            else:
                best_idx = df[col].idxmin()
            
            # 添加LaTeX粗体标记
            df.loc[best_idx, col] = f"\\textbf{{{df.loc[best_idx, col]}}}"
        
        return df


def generate_traditional_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成传统指标表格"""
    table_df = df[['model', 'precision', 'recall', 'f1']].copy()
    table_df.columns = ['Model', 'Precision', 'Recall', 'F1-Score']
    
    # 格式化模型名称
    from config import get_model_name
    table_df['Model'] = table_df['Model'].apply(get_model_name)
    
    # 格式化数值
    for col in ['Precision', 'Recall', 'F1-Score']:
        table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}")
    
    return table_df


def generate_time_aware_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成时间感知指标表格"""
    table_df = df[['model', 'DR@5s', 'ADD', 'MTBFA']].copy()
    table_df.columns = ['Model', 'DR@5s', 'ADD (s)', 'MTBFA (h)']
    
    # 格式化模型名称
    from config import get_model_name
    table_df['Model'] = table_df['Model'].apply(get_model_name)
    
    # 格式化数值
    table_df['DR@5s'] = table_df['DR@5s'].apply(lambda x: f"{x:.4f}")
    table_df['ADD (s)'] = table_df['ADD (s)'].apply(lambda x: f"{x:.2f}")
    table_df['MTBFA (h)'] = table_df['MTBFA (h)'].apply(lambda x: f"{x:.2f}")
    
    return table_df


def generate_fp_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成FP窗口vs事件对比表格"""
    table_df = df[['Model', 'Precision', 'FP Windows', 'FP Events', 
                   'Aggregation', 'MTBFA (h)']].copy()
    
    return table_df


def generate_efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成计算效率表格"""
    # 选择CUDA设备的数据
    df_cuda = df[df['device'] == 'cuda'].copy()
    
    table_df = df_cuda[['model', 'mean_latency_ms', 'throughput_windows_per_sec',
                        'total_params', 'model_size_mb']].copy()
    table_df.columns = ['Model', 'Latency (ms)', 'Throughput (win/s)', 
                       'Parameters (M)', 'Size (MB)']
    
    # 格式化模型名称
    from config import get_model_name
    table_df['Model'] = table_df['Model'].apply(get_model_name)
    
    # 格式化数值
    table_df['Latency (ms)'] = table_df['Latency (ms)'].apply(lambda x: f"{x:.2f}")
    table_df['Throughput (win/s)'] = table_df['Throughput (win/s)'].apply(lambda x: f"{x:.1f}")
    table_df['Parameters (M)'] = (table_df['Parameters (M)'] / 1e6).apply(lambda x: f"{x:.2f}")
    table_df['Size (MB)'] = table_df['Size (MB)'].apply(lambda x: f"{x:.2f}")
    
    return table_df
