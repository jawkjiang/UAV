"""
Section D: False Alarm Frequency Analysis
误报频率分析 - MTBFA和误报持续时长
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from config import (
    MODELS, MODEL_DISPLAY_NAMES, MODEL_COLORS, FIGURE_SIZES,
    FIGURES_DIR
)
from utils.plot_styles import setup_paper_style, save_figure
from utils.table_generator import TableGenerator
from data_loader import DataLoader


class SectionDFalseAlarms:
    """Section D: 误报频率分析"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.output_dir = FIGURES_DIR / 'section_d'
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置绘图风格
        setup_paper_style('small_8px')
    
    def generate_all(self):
        """生成Section D的所有图表"""
        print("\n" + "="*80)
        print("Section D: False Alarm Frequency Analysis")
        print("="*80)
        
        # 加载数据
        time_aware_df = self.data_loader.load_time_aware_metrics()
        fp_duration_dict = self.data_loader.load_fp_duration_data()
        
        if time_aware_df is None or time_aware_df.empty:
            print("❌ Time-aware metrics data not found!")
            return
        
        # 转换DataFrame为Dict格式
        time_aware_data = self._df_to_dict(time_aware_df)
        
        # Table 5: MTBFA详细分析
        self.generate_mtbfa_table(time_aware_data)
        
        # Figure D1: 误报事件持续时长分布
        if fp_duration_dict:
            self.plot_fp_duration_distribution(fp_duration_dict)
        
        # Figure D2: 误报持续时长箱线图
        if fp_duration_dict:
            self.plot_fp_duration_boxplot(fp_duration_dict)
        
        # Figure D3: 误报持续时长CDF
        if fp_duration_dict:
            self.plot_fp_duration_cdf(fp_duration_dict)
        
        # Figure D4: 100小时运行场景
        self.plot_100hour_scenario(time_aware_data)
        
        print("\n✅ Section D completed!")
    
    def _df_to_dict(self, df: pd.DataFrame) -> Dict:
        """将DataFrame转换为Dict格式"""
        data = {}
        for _, row in df.iterrows():
            model = row['model']
            data[model] = {
                'dr_at_5s': row.get('DR@5s', 0),
                'dr_at_10s': row.get('DR@10s', 0),
                'add': row.get('ADD', 0),
                'mtbfa': row.get('MTBFA', 0),
                'fp_events': row.get('FP_events', 0),
                'fp_windows': row.get('FP_windows', 0)
            }
        return data
    
    def generate_mtbfa_table(self, data: Dict):
        """
        Table 5: MTBFA详细分析表
        """
        print("\n📊 Generating Table 5: MTBFA Detailed Analysis...")
        
        table_data = []
        
        for model in MODELS:
            if model not in data:
                continue
            
            model_data = data[model]
            
            # 计算MTBFA相关指标
            mtbfa_hours = model_data.get('mtbfa', 0) / 3600  # 转换为小时
            fp_events = model_data.get('fp_events', 0)
            fp_windows = model_data.get('fp_windows', 0)
            
            # 计算100小时内的预期误报次数
            expected_fps_100h = 100 / mtbfa_hours if mtbfa_hours > 0 else float('inf')
            
            row = {
                'Model': MODEL_DISPLAY_NAMES[model],
                'MTBFA (hours)': f"{mtbfa_hours:.2f}",
                'FP Events': f"{fp_events}",
                'FP Windows': f"{fp_windows}",
                'Aggregation Ratio': f"{fp_windows/fp_events:.2f}" if fp_events > 0 else "N/A",
                'Expected FPs in 100h': f"{expected_fps_100h:.1f}" if expected_fps_100h != float('inf') else "∞"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # 保存表格
        table_gen = TableGenerator()
        table_gen.save_table(
            df, 'table5_mtbfa_detailed_analysis',
            caption='Mean Time Between False Alarms (MTBFA) Detailed Analysis',
            label='tab:mtbfa_analysis'
        )
    
    def plot_fp_duration_distribution(self, fp_data: Dict):
        """
        Figure D1: 误报事件持续时长分布（拆分为独立图）
        """
        print("\n📊 Generating Figure D1: FP Duration Distribution...")
        
        for model in MODELS:
            if model not in fp_data:
                continue
            
            # 从DataFrame中提取durations
            df = fp_data[model]
            if 'duration' not in df.columns or df.empty:
                continue
            
            durations = df['duration'].values
            
            fig, ax = plt.subplots()
            
            # 绘制直方图
            ax.hist(durations, bins=30, color=MODEL_COLORS[model],
                   alpha=0.7, edgecolor='black')
            
            # 添加统计信息
            mean_dur = np.mean(durations)
            median_dur = np.median(durations)
            
            ax.axvline(mean_dur, color='red', linestyle='--', 
                      label=f'Mean: {mean_dur:.1f}s')
            ax.axvline(median_dur, color='blue', linestyle='--',
                      label=f'Median: {median_dur:.1f}s')
            
            ax.set_xlabel('Duration (seconds)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{MODEL_DISPLAY_NAMES[model]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_figure(fig, f'fig_d1_fp_duration_{model}', self.output_dir)
            plt.close()
    
    def plot_fp_duration_boxplot(self, fp_data: Dict):
        """
        Figure D2: 误报持续时长箱线图
        """
        print("\n📊 Generating Figure D2: FP Duration Boxplot...")
        
        fig, ax = plt.subplots()
        
        # 准备数据
        durations_list = []
        labels = []
        colors = []
        
        for model in MODELS:
            if model in fp_data:
                df = fp_data[model]
                if 'duration' in df.columns and not df.empty:
                    durations = df['duration'].values
                    durations_list.append(durations)
                    labels.append(MODEL_DISPLAY_NAMES[model])
                    colors.append(MODEL_COLORS[model])
        
        if not durations_list:
            print("⚠️  No FP duration data available for boxplot")
            return
        
        # 绘制箱线图
        bp = ax.boxplot(durations_list, tick_labels=labels, patch_artist=True,
                       showmeans=True, meanline=True,
                       flierprops=dict(marker='o', markersize=3, alpha=0.5),
                       medianprops=dict(color='#FF6B35', linewidth=1),
                       meanprops=dict(color='#2E8B57', linestyle='--', linewidth=1))
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#FF6B35', linewidth=1, label='Median'),
            Line2D([0], [0], color='#2E8B57', linewidth=1, linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('FP Event Duration (seconds)')
        ax.set_title('False Positive Event Duration Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_figure(fig, 'fig_d2_fp_duration_boxplot', self.output_dir)
        plt.close()
    
    def plot_fp_duration_cdf(self, fp_data: Dict):
        """
        Figure D3: 误报持续时长累积分布函数(CDF)
        """
        print("\n📊 Generating Figure D3: FP Duration CDF...")
        
        fig, ax = plt.subplots()
        
        for model in MODELS:
            if model not in fp_data:
                continue
            
            df = fp_data[model]
            if 'duration' not in df.columns or df.empty:
                continue
            
            durations = df['duration'].values
            
            # 计算CDF
            sorted_durations = np.sort(durations)
            cdf = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
            
            ax.plot(sorted_durations, cdf * 100,
                   label=MODEL_DISPLAY_NAMES[model],
                   color=MODEL_COLORS[model],
                   linewidth=2)
        
        # 添加关键阈值线
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50th percentile')
        ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90th percentile')
        
        ax.set_xlabel('FP Event Duration (seconds)')
        ax.set_ylabel('Cumulative Probability (%)')
        ax.set_title('Cumulative Distribution of FP Event Duration')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 105)
        
        save_figure(fig, 'fig_d3_fp_duration_cdf', self.output_dir)
        plt.close()
    
    def plot_100hour_scenario(self, data: Dict):
        """
        Figure D4: 100小时运行场景
        展示在100小时运行期间，不同模型的预期误报次数
        """
        print("\n📊 Generating Figure D4: 100-Hour Operation Scenario...")
        
        fig, ax = plt.subplots()
        
        models_list = []
        fp_counts = []
        
        for model in MODELS:
            if model not in data:
                continue
            
            model_data = data[model]
            mtbfa_hours = model_data.get('mtbfa', 0) / 3600
            
            if mtbfa_hours > 0:
                expected_fps = 100 / mtbfa_hours
                models_list.append(MODEL_DISPLAY_NAMES[model])
                fp_counts.append(expected_fps)
        
        # 绘制柱状图
        bars = ax.bar(range(len(models_list)), fp_counts,
                     color=[MODEL_COLORS[m.lower().replace('-', '_')] 
                           for m in models_list])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        # 添加可接受阈值线
        ax.axhline(y=10, color='orange', linestyle='--',
                  label='Acceptable: 10 FPs/100h')
        ax.axhline(y=5, color='green', linestyle='--',
                  label='Good: 5 FPs/100h')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Expected False Alarms in 100 Hours')
        ax.set_title('Expected False Alarm Frequency in 100-Hour Operation')
        ax.set_xticks(range(len(models_list)))
        ax.set_xticklabels(models_list, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_figure(fig, 'fig_d4_100hour_scenario', self.output_dir)
        plt.close()


if __name__ == '__main__':
    # 测试
    data_loader = DataLoader()
    section_d = SectionDFalseAlarms(data_loader)
    section_d.generate_all()
