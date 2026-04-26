"""
Section E: Trade-off Analysis
权衡分析：检测速度 vs 误报频率 vs 准确率
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import seaborn as sns

from config import *
from data_loader import PaperDataLoader

# 导入样式文件路径
STYLE_FILE = '../styles/sci_small_8px.mplstyle'


class SectionETradeoff:
    """Section E: 权衡分析"""
    
    def __init__(self, output_dir='output/figures/section_e'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.data_loader = PaperDataLoader()
        
    def generate_all(self):
        """生成所有Section E的图表"""
        print("\n" + "="*60)
        print("Section E: Trade-off Analysis")
        print("="*60)
        
        # 加载数据
        time_aware = self.data_loader.load_time_aware_metrics()
        
        if time_aware.empty:
            print("❌ 无法加载时间感知指标数据")
            return
            
        # Figure E1: Pareto前沿
        self.plot_pareto_frontier(time_aware)
        
        # Figure E2: ADD vs DR@5s相关性
        self.plot_add_dr_correlation(time_aware)
        
        # Figure E3: 3D权衡分析
        self.plot_3d_tradeoff(time_aware)
        
        # Figure E4: 场景化模型推荐
        self.plot_scenario_recommendations(time_aware)
        
        print("\n✅ Section E 所有图表生成完成！")
        
    def plot_pareto_frontier(self, df):
        """Figure E1: Pareto前沿（DR@5s vs MTBFA）"""
        print("\n📊 生成 Figure E1: Pareto前沿...")
        
        plt.style.use(STYLE_FILE)
        fig, ax = plt.subplots()
        
        # 提取数据
        models = []
        dr_5s = []
        mtbfa = []
        
        for model in MODEL_ORDER:
            if model in df['model'].values:
                model_data = df[df['model'] == model].iloc[0]
                models.append(model)
                dr_5s.append(model_data['DR@5s'] * 100)
                mtbfa.append(model_data['MTBFA'])
        
        # 绘制所有点并添加到图例
        for i, model in enumerate(models):
            ax.scatter(mtbfa[i], dr_5s[i],
                      color=MODEL_COLORS[model],
                      s=40, alpha=0.8,
                      edgecolors='black', linewidth=1,
                      label=MODEL_LABELS[model], zorder=3)
        
        # 找到Pareto前沿
        points = np.array(list(zip(mtbfa, dr_5s)))
        pareto_indices = self._find_pareto_frontier(points)
        pareto_points = points[pareto_indices]
        
        # 排序并绘制Pareto前沿
        sorted_indices = np.argsort(pareto_points[:, 0])
        pareto_sorted = pareto_points[sorted_indices]
        
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
               'r--', linewidth=2, alpha=0.5,
               label='Pareto Frontier', zorder=2)

        # 添加可部署区域（DR@5s >= 85%, MTBFA >= 0.1h）
        deploy_dr_threshold = 85  # DR@5s = 85%
        deploy_mtbfa_threshold = 0.1  # MTBFA = 0.1h

        # 获取坐标轴范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # 绘制可部署区域边界虚线
        ax.axhline(y=deploy_dr_threshold, color='green', linestyle='--',
                   linewidth=1.5, alpha=0.7, zorder=1)
        ax.axvline(x=deploy_mtbfa_threshold, color='green', linestyle='--',
                   linewidth=1.5, alpha=0.7, zorder=1)

        # 填充可部署区域（右上角）
        ax.fill_between([deploy_mtbfa_threshold, xlim[1]],
                        deploy_dr_threshold, ylim[1],
                        color='green', alpha=0.1, zorder=0,
                        label='Deployable Region')

        # 添加区域标注
        # ax.text(xlim[1] * 0.95, ylim[1] * 0.95,
        #         'Deployable\nRegion',
        #         ha='right', va='top',
        #         fontsize=9, style='italic', color='green',
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='green'))

        # 恢复坐标轴范围
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('MTBFA (hours)')
        ax.set_ylabel('DR@5s (%)')
        ax.set_title('Trade-off: Detection Speed vs False Alarm Frequency')

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', framealpha=0.9, ncol=1, fontsize=7)

        plt.tight_layout()
        self._save_figure(fig, 'fig_e1_pareto_frontier')
        plt.close()
        
    def plot_add_dr_correlation(self, df):
        """Figure E2: ADD vs DR@5s相关性"""
        print("\n📊 生成 Figure E2: ADD vs DR@5s相关性...")
        
        plt.style.use(STYLE_FILE)
        fig, ax = plt.subplots()
        
        # 提取数据
        models = []
        add_values = []
        dr_5s = []
        
        for model in MODEL_ORDER:
            if model in df['model'].values:
                model_data = df[df['model'] == model].iloc[0]
                models.append(model)
                add_values.append(model_data['ADD'])
                dr_5s.append(model_data['DR@5s'] * 100)
        
        # 绘制散点图并添加到图例
        for i, model in enumerate(models):
            ax.scatter(add_values[i], dr_5s[i],
                      color=MODEL_COLORS[model],
                      s=40, alpha=0.8,
                      edgecolors='black', linewidth=1,
                      label=MODEL_LABELS[model], zorder=3)
        
        # 计算相关系数
        correlation = np.corrcoef(add_values, dr_5s)[0, 1]
        
        # 添加趋势线
        z = np.polyfit(add_values, dr_5s, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(add_values), max(add_values), 100)
        ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5,
               label=f'Trend (r={correlation:.3f})')
        
        ax.set_xlabel('ADD (seconds)')
        ax.set_ylabel('DR@5s (%)')
        ax.set_title('Correlation: Average Detection Delay vs Detection Rate')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)
        
        # 添加相关性说明
        ax.text(0.02, 0.98, 
               f'Pearson Correlation: {correlation:.3f}\n' +
               ('Strong negative correlation' if correlation < -0.7 else
                'Moderate negative correlation' if correlation < -0.4 else
                'Weak correlation'),
               transform=ax.transAxes, ha='left', va='top',
               style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_e2_add_dr_correlation')
        plt.close()
        
    def plot_3d_tradeoff(self, df):
        """Figure E3: 3D权衡分析（DR@5s, ADD, MTBFA）"""
        print("\n📊 生成 Figure E3: 3D权衡分析...")
        
        plt.style.use(STYLE_FILE)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取数据
        models = []
        dr_5s = []
        add_values = []
        mtbfa = []
        
        for model in MODEL_ORDER:
            if model in df['model'].values:
                model_data = df[df['model'] == model].iloc[0]
                models.append(model)
                dr_5s.append(model_data['DR@5s'] * 100)
                add_values.append(model_data['ADD'])
                mtbfa.append(model_data['MTBFA'])
        
        # 绘制3D散点图并添加到图例
        for i, model in enumerate(models):
            ax.scatter(dr_5s[i], add_values[i], mtbfa[i],
                      color=MODEL_COLORS[model],
                      s=40, alpha=0.8,
                      edgecolors='black', linewidth=1,
                      label=MODEL_LABELS[model])
        
        ax.set_xlabel('DR@5s (%)', labelpad=10)
        ax.set_ylabel('ADD (s)', labelpad=10)
        ax.set_zlabel('MTBFA (hours)', labelpad=10)
        ax.set_title('3D Trade-off Analysis')
        
        ax.legend(loc='upper left', framealpha=0.9)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_e3_3d_tradeoff')
        plt.close()
        
    def plot_scenario_recommendations(self, df):
        """Figure E4: 场景化模型推荐（拆分为独立图）"""
        print("\n📊 生成 Figure E4: 场景化模型推荐...")
        
        plt.style.use(STYLE_FILE)
        
        scenarios = [
            {
                'name': 'Critical Safety (Minimize ADD)',
                'filename': 'critical_safety',
                'metric': 'ADD',
                'ascending': True
            },
            {
                'name': 'Operational Stability (Maximize MTBFA)',
                'filename': 'operational_stability',
                'metric': 'MTBFA',
                'ascending': False
            },
            {
                'name': 'Balanced Performance (Maximize DR@5s)',
                'filename': 'balanced_performance',
                'metric': 'DR@5s',
                'ascending': False
            },
            {
                'name': 'Overall Best (Composite Score)',
                'filename': 'overall_best',
                'metric': 'composite',
                'ascending': False
            }
        ]
        
        for scenario in scenarios:
            fig, ax = plt.subplots()
            
            # 计算综合得分（如果需要）
            if scenario['metric'] == 'composite':
                df_copy = df.copy()
                # 归一化各指标
                df_copy['dr_norm'] = (df_copy['DR@5s'] - df_copy['DR@5s'].min()) / \
                                     (df_copy['DR@5s'].max() - df_copy['DR@5s'].min())
                df_copy['add_norm'] = 1 - (df_copy['ADD'] - df_copy['ADD'].min()) / \
                                      (df_copy['ADD'].max() - df_copy['ADD'].min())
                df_copy['mtbfa_norm'] = (df_copy['MTBFA'] - df_copy['MTBFA'].min()) / \
                                        (df_copy['MTBFA'].max() - df_copy['MTBFA'].min())
                df_copy['composite'] = (df_copy['dr_norm'] + df_copy['add_norm'] + df_copy['mtbfa_norm']) / 3
                df_sorted = df_copy.sort_values('composite', ascending=False)
            else:
                df_sorted = df.sort_values(scenario['metric'], ascending=scenario['ascending'])
            
            # 绘制排名
            models = df_sorted['model'].values[:7]  # Top 7
            if scenario['metric'] == 'composite':
                values = df_sorted['composite'].values[:7] * 100
                ylabel = 'Composite Score'
            elif scenario['metric'] == 'ADD':
                values = df_sorted[scenario['metric']].values[:7]
                ylabel = 'ADD (seconds)'
            elif scenario['metric'] == 'MTBFA':
                values = df_sorted[scenario['metric']].values[:7]
                ylabel = 'MTBFA (hours)'
            else:
                values = df_sorted[scenario['metric']].values[:7] * 100
                ylabel = 'DR@5s (%)'
            
            colors = [MODEL_COLORS[m] for m in models]
            labels = [MODEL_LABELS[m] for m in models]
            
            bars = ax.barh(range(len(models)), values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(labels)
            ax.set_xlabel(ylabel)
            ax.set_title(scenario['name'])
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val, i, f' {val:.2f}', 
                       va='center', ha='left')
            
            # 标注推荐
            ax.text(0.98, 0.98, '⭐ Recommended', 
                   transform=ax.transAxes, ha='right', va='top',
                   color='red',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            
            plt.tight_layout()
            self._save_figure(fig, f'fig_e4_{scenario["filename"]}')
            plt.close()
        
    def _find_pareto_frontier(self, points):
        """找到Pareto前沿点"""
        pareto_indices = []
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i != j:
                    # 如果另一个点在两个维度上都不差于当前点，且至少一个维度更好
                    if (other[0] >= point[0] and other[1] >= point[1] and 
                        (other[0] > point[0] or other[1] > point[1])):
                        dominated = True
                        break
            if not dominated:
                pareto_indices.append(i)
        return pareto_indices
        
    def _save_figure(self, fig, name):
        """保存图表"""
        for fmt in OUTPUT_FORMATS:
            filepath = os.path.join(self.output_dir, f'{name}.{fmt}')
            fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
            print(f"  ✅ 保存: {filepath}")


if __name__ == '__main__':
    generator = SectionETradeoff()
    generator.generate_all()
