"""
Section B: Overall Performance - Traditional vs. Time-Aware Metrics
整体性能对比：传统指标 vs 时间感知指标
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from config import (
    MODELS, MODEL_COLORS, FIGURE_SIZES, FIGURES_DIR,
    get_model_name, get_model_color, get_figure_path
)
from data_loader import PaperDataLoader
from utils.plot_styles import setup_paper_style, save_figure, add_value_labels
from utils.table_generator import (
    TableGenerator, generate_traditional_metrics_table,
    generate_time_aware_metrics_table, generate_fp_comparison_table
)
from utils.statistics import calculate_correlation, rank_models

logger = logging.getLogger(__name__)


class SectionBPlotter:
    """Section B 图表生成器"""
    
    def __init__(self, data_loader: PaperDataLoader = None):
        if data_loader is None:
            data_loader = PaperDataLoader()
        self.loader = data_loader
        self.output_dir = FIGURES_DIR / 'section_b'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        setup_paper_style('small_8px')
        
    def generate_all(self):
        """生成所有Section B的图表和表格"""
        logger.info("="*80)
        logger.info("Generating Section B: Overall Performance")
        logger.info("="*80)
        
        # 生成表格
        self.generate_tables()
        
        # 生成图表
        self.plot_traditional_metrics_comparison()  # Figure B1
        self.plot_time_aware_metrics_comparison()   # Figure B2
        self.plot_precision_mtbfa_paradox()         # Figure B3
        self.plot_recall_dr_comparison()            # Figure B3b: Recall vs DR@5s/DR@15s
        self.plot_fp_aggregation_analysis()         # Figure B4
        self.plot_metrics_correlation_heatmap()     # Figure B5: 传统指标与时间感知指标相关性
        
        logger.info("✅ Section B complete!")
    
    def generate_tables(self):
        """生成所有表格"""
        logger.info("\n📊 Generating tables...")
        
        table_gen = TableGenerator()
        
        # Table 1: 传统指标
        trad = self.loader.load_traditional_metrics()
        table1 = generate_traditional_metrics_table(trad)
        table_gen.save_table(table1, 'table1_traditional_metrics',
                           caption='Traditional Performance Metrics',
                           label='traditional_metrics')
        
        # Table 2: 时间感知指标（包含DR@15s）
        time_aware = self.loader.load_time_aware_metrics()
        table2 = self._generate_time_aware_table_with_dr15s(time_aware)
        table_gen.save_table(table2, 'table2_time_aware_metrics',
                           caption='Time-Aware Performance Metrics',
                           label='time_aware_metrics')
        
        # Table 3: Precision vs MTBFA排名对比
        fp_comp = self.loader.load_fp_window_event_comparison()
        table3 = generate_fp_comparison_table(fp_comp)
        table_gen.save_table(table3, 'table3_precision_mtbfa_comparison',
                           caption='Precision vs MTBFA: Window-level vs Event-level Analysis',
                           label='fp_comparison')
    
    def _generate_time_aware_table_with_dr15s(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成包含DR@15s的时间感知指标表格"""
        from config import get_model_name
        
        table_df = df[['model', 'DR@5s', 'DR@15s', 'ADD', 'MTBFA']].copy()
        table_df.columns = ['Model', 'DR@5s', 'DR@15s', 'ADD (s)', 'MTBFA (h)']
        
        # 格式化模型名称
        table_df['Model'] = table_df['Model'].apply(get_model_name)
        
        # 格式化数值
        table_df['DR@5s'] = table_df['DR@5s'].apply(lambda x: f"{x:.4f}")
        table_df['DR@15s'] = table_df['DR@15s'].apply(lambda x: f"{x:.4f}")
        table_df['ADD (s)'] = table_df['ADD (s)'].apply(lambda x: f"{x:.2f}")
        table_df['MTBFA (h)'] = table_df['MTBFA (h)'].apply(lambda x: f"{x:.2f}")
        
        return table_df
    
    def plot_traditional_metrics_comparison(self):
        """Figure B1: 传统指标对比柱状图（拆分为独立图）"""
        logger.info("\n📈 Generating Figure B1: Traditional Metrics Comparison...")
        
        df = self.loader.load_traditional_metrics()
        
        metrics = ['precision', 'recall', 'f1']
        titles = ['Precision', 'Recall', 'F1-Score']
        
        for metric, title in zip(metrics, titles):
            fig, ax = plt.subplots()
            
            # 准备数据
            models = [get_model_name(m) for m in df['model']]
            values = df[metric].values
            colors = [get_model_color(m) for m in df['model']]
            
            # 绘制柱状图
            bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            add_value_labels(ax, bars, format_str='.4f')
            
            # 设置标签
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Comparison')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加基准线
            ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='0.9 threshold')
            if metric == 'precision':
                ax.legend()
            
            plt.tight_layout()
            save_figure(fig, f'fig_b1_{metric}', self.output_dir)
            plt.close()
    
    def plot_time_aware_metrics_comparison(self):
        """Figure B2: 时间感知指标对比柱状图（拆分为独立图）"""
        logger.info("\n📈 Generating Figure B2: Time-Aware Metrics Comparison...")
        
        df = self.loader.load_time_aware_metrics()
        models = [get_model_name(m) for m in df['model']]
        colors = [get_model_color(m) for m in df['model']]
        
        # DR@5s
        fig, ax = plt.subplots()
        values = df['DR@5s'].values
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
        add_value_labels(ax, bars, format_str='.4f')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('DR@5s')
        ax.set_title('Detection Rate @ 5s')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_figure(fig, 'fig_b2_dr_at_5s', self.output_dir)
        plt.close()
        
        # ADD
        fig, ax = plt.subplots()
        values = df['ADD'].values
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
        add_value_labels(ax, bars, format_str='.2f')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('ADD (seconds)')
        ax.set_title('Average Detection Delay')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_figure(fig, 'fig_b2_add', self.output_dir)
        plt.close()
        
        # MTBFA
        fig, ax = plt.subplots()
        values = df['MTBFA'].values
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, edgecolor='black')
        add_value_labels(ax, bars, format_str='.2f')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('MTBFA (hours)')
        ax.set_title('Mean Time Between False Alarms')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        save_figure(fig, 'fig_b2_mtbfa', self.output_dir)
        plt.close()
    
    def plot_precision_mtbfa_paradox(self):
        """Figure B3: Precision vs MTBFA散点图（展示悖论）"""
        logger.info("\n📈 Generating Figure B3: Precision vs MTBFA Paradox...")
        
        # 加载数据
        trad = self.loader.load_traditional_metrics()
        time_aware = self.loader.load_time_aware_metrics()
        merged = trad.merge(time_aware[['model', 'MTBFA']], on='model')
        
        fig, ax = plt.subplots()
        
        # 绘制散点并添加到图例
        for _, row in merged.iterrows():
            model = row['model']
            x = row['precision']
            y = row['MTBFA']
            color = get_model_color(model)
            ax.scatter(x, y, s=40, c=color, alpha=0.8, edgecolors='black',
                      linewidth=1, zorder=3, label=get_model_name(model))
        
        # 计算相关性
        r, p = calculate_correlation(merged['precision'].values, merged['MTBFA'].values)
        
        # 添加趋势线
        z = np.polyfit(merged['precision'].values, merged['MTBFA'].values, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(merged['precision'].min(), merged['precision'].max(), 100)
        ax.plot(x_line, p_fit(x_line), 'r--', alpha=0.5, 
               label=f'Trend (r={r:.3f}, p={p:.4f})')
        
        ax.set_xlabel('Precision')
        ax.set_ylabel('MTBFA (hours)')
        ax.set_title('Precision vs MTBFA')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=2)
        
        plt.tight_layout()
        save_figure(fig, 'fig_b3_precision_mtbfa_paradox', self.output_dir)
        plt.close()
    
    def plot_recall_dr_comparison(self):
        """Figure B3b: Recall vs DR@5s 和 DR@15s 散点图（拆分为独立图）"""
        logger.info("\n📈 Generating Figure B3b: Recall vs DR@5s and DR@15s...")
        
        # 加载数据
        trad = self.loader.load_traditional_metrics()
        time_aware = self.loader.load_time_aware_metrics()
        merged = trad.merge(time_aware[['model', 'DR@5s', 'DR@15s']], on='model')
        
        # 图1：Recall vs DR@5s
        fig, ax = plt.subplots()
        
        # 绘制散点并添加到图例
        for _, row in merged.iterrows():
            model = row['model']
            x = row['recall']
            y = row['DR@5s']
            color = get_model_color(model)
            ax.scatter(x, y, s=100, c=color, alpha=0.7, edgecolors='black', 
                      linewidth=1.5, zorder=3, label=get_model_name(model))
        
        # 计算相关性
        r1, p1 = calculate_correlation(merged['recall'].values, merged['DR@5s'].values)
        
        # 添加趋势线
        z1 = np.polyfit(merged['recall'].values, merged['DR@5s'].values, 1)
        p_fit1 = np.poly1d(z1)
        x_line1 = np.linspace(merged['recall'].min(), merged['recall'].max(), 100)
        ax.plot(x_line1, p_fit1(x_line1), 'r--', alpha=0.5, 
               label=f'Trend (r={r1:.3f}, p={p1:.4f})')
        
        ax.set_xlabel('Recall (Window-level)')
        ax.set_ylabel('DR@5s (Event-level)')
        ax.set_title('Recall vs DR@5s')
        ax.grid(True, alpha=0.3)

        # 添加阈值线
        # 水平虚线：DR@5s=85% 可部署性阈值
        ax.axhline(y=0.85, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Deployability (DR@5s=85%)')
        # 垂直虚线：Recall=0.85 传统阈值
        ax.axvline(x=0.85, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Traditional (Recall=85%)')

        ax.legend(loc='best')
        # 自动调整坐标轴范围，确保阈值线可见
        x_margin = (merged['recall'].max() - merged['recall'].min()) * 0.1
        y_margin = (merged['DR@5s'].max() - merged['DR@5s'].min()) * 0.1
        x_min = min(merged['recall'].min() - x_margin, 0.83)
        x_max = max(merged['recall'].max() + x_margin, 0.87)
        y_min = min(merged['DR@5s'].min() - y_margin, 0.83)
        y_max = max(merged['DR@5s'].max() + y_margin, 0.87)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        plt.tight_layout()
        save_figure(fig, 'fig_b3b_recall_vs_dr5s', self.output_dir)
        plt.close()
        
        # 图2：Recall vs DR@15s
        fig, ax = plt.subplots()
        
        # 绘制散点并添加到图例
        for _, row in merged.iterrows():
            model = row['model']
            x = row['recall']
            y = row['DR@15s']
            color = get_model_color(model)
            ax.scatter(x, y, s=100, c=color, alpha=0.7, edgecolors='black', 
                      linewidth=1.5, zorder=3, label=get_model_name(model))
        
        # 计算相关性
        r2, p2 = calculate_correlation(merged['recall'].values, merged['DR@15s'].values)
        
        # 添加趋势线
        z2 = np.polyfit(merged['recall'].values, merged['DR@15s'].values, 1)
        p_fit2 = np.poly1d(z2)
        x_line2 = np.linspace(merged['recall'].min(), merged['recall'].max(), 100)
        ax.plot(x_line2, p_fit2(x_line2), 'r--', alpha=0.5, 
               label=f'Trend (r={r2:.3f}, p={p2:.4f})')
        
        ax.set_xlabel('Recall (Window-level)')
        ax.set_ylabel('DR@15s (Event-level)')
        ax.set_title('Recall vs DR@15s')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        # 自动调整坐标轴范围
        x_margin = (merged['recall'].max() - merged['recall'].min()) * 0.1
        y_margin = (merged['DR@15s'].max() - merged['DR@15s'].min()) * 0.1
        ax.set_xlim([merged['recall'].min() - x_margin, merged['recall'].max() + x_margin])
        ax.set_ylim([merged['DR@15s'].min() - y_margin, merged['DR@15s'].max() + y_margin])
        
        plt.tight_layout()
        save_figure(fig, 'fig_b3b_recall_vs_dr15s', self.output_dir)
        plt.close()
    
    def plot_fp_aggregation_analysis(self):
        """Figure B4: FP窗口数 vs FP事件数对比（拆分为独立图）"""
        logger.info("\n📈 Generating Figure B4: FP Aggregation Analysis...")
        
        df = self.loader.load_fp_window_event_comparison()
        models = [get_model_name(m) for m in df['Model']]
        
        # 左图：FP窗口 vs FP事件对比
        fig, ax = plt.subplots()
        fp_windows = df['FP Windows'].values
        fp_events = df['FP Events'].values
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, fp_windows, width, label='FP Windows',
                       color='#EE6677', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, fp_events, width, label='FP Events',
                       color='#4477AA', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Count')
        ax.set_title('FP Windows vs FP Events')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_figure(fig, 'fig_b4_fp_windows_vs_events', self.output_dir)
        plt.close()
        
        # 右图：聚合率
        fig, ax = plt.subplots()
        aggregation = df['Aggregation'].astype(float).values
        colors = [get_model_color(m.lower()) for m in df['Model']]
        
        bars = ax.bar(range(len(models)), aggregation, color=colors, 
                      alpha=0.8, edgecolor='black')
        add_value_labels(ax, bars, format_str='.2f')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Aggregation Ratio\n(FP Events / FP Windows)')
        ax.set_title('FP Window Aggregation Rate')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_figure(fig, 'fig_b4_aggregation_rate', self.output_dir)
        plt.close()
    
    def plot_metrics_correlation_heatmap(self):
        """Figure B5: 传统指标与时间感知指标的相关性热图"""
        logger.info("\n📈 Generating Figure B5: Metrics Correlation Heatmap...")
        
        # 加载数据
        trad = self.loader.load_traditional_metrics()
        time_aware = self.loader.load_time_aware_metrics()
        
        # 合并数据
        merged = trad.merge(time_aware[['model', 'DR@5s', 'DR@15s', 'ADD', 'MTBFA']], on='model')
        
        # 选择需要计算相关性的指标
        metrics_df = merged[['precision', 'recall', 'f1', 'DR@5s', 'DR@15s', 'ADD', 'MTBFA']]
        
        # 计算相关性矩阵
        corr_matrix = metrics_df.corr()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制热图
        sns.heatmap(corr_matrix, 
                    annot=True,  # 显示数值
                    fmt='.3f',   # 格式化为3位小数
                    cmap='RdBu_r',  # 红蓝配色
                    center=0,    # 以0为中心
                    square=True,  # 方形单元格
                    linewidths=0.5,  # 网格线宽度
                    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                    vmin=-1, vmax=1,  # 相关系数范围
                    ax=ax)
        
        # 设置标签
        ax.set_title('Correlation between Traditional and Time-Aware Metrics\n(Based on 7 Models)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 调整标签
        ax.set_xticklabels(['Precision', 'Recall', 'F1', 'DR@5s', 'DR@15s', 'ADD', 'MTBFA'],
                          rotation=45, ha='right')
        ax.set_yticklabels(['Precision', 'Recall', 'F1', 'DR@5s', 'DR@15s', 'ADD', 'MTBFA'],
                          rotation=0)
        
        # 添加分组线条以区分传统指标和时间感知指标
        ax.axhline(y=3, color='black', linewidth=2)
        ax.axvline(x=3, color='black', linewidth=2)
        
        # 添加文本标注
        ax.text(1.5, -0.8, 'Traditional Metrics', ha='center', fontsize=11, fontweight='bold')
        ax.text(5, -0.8, 'Time-Aware Metrics', ha='center', fontsize=11, fontweight='bold')
        ax.text(-0.8, 1.5, 'Traditional\nMetrics', va='center', fontsize=11, fontweight='bold')
        ax.text(-0.8, 5, 'Time-Aware\nMetrics', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        save_figure(fig, 'fig_b5_metrics_correlation', self.output_dir)
        plt.close()
        
        # 输出关键发现
        logger.info("\n📊 Key Correlations:")
        logger.info(f"  Precision vs MTBFA: {corr_matrix.loc['precision', 'MTBFA']:.3f}")
        logger.info(f"  Recall vs DR@5s: {corr_matrix.loc['recall', 'DR@5s']:.3f}")
        logger.info(f"  Recall vs DR@15s: {corr_matrix.loc['recall', 'DR@15s']:.3f}")
        logger.info(f"  F1 vs ADD: {corr_matrix.loc['f1', 'ADD']:.3f}")


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(levelname)s: %(message)s')
    
    plotter = SectionBPlotter()
    plotter.generate_all()


if __name__ == '__main__':
    main()
