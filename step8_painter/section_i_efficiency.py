"""
Section I: Computational Efficiency Analysis
计算效率分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from data_loader import PaperDataLoader

# 导入样式文件路径
STYLE_FILE = '../styles/sci_small_8px.mplstyle'


class SectionIEfficiency:
    """Section I: 计算效率分析"""
    
    def __init__(self, output_dir='output/figures/section_i'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_loader = PaperDataLoader()
        
    def generate_all(self):
        """生成所有Section I的图表"""
        print("\n" + "="*60)
        print("Section I: Computational Efficiency Analysis")
        print("="*60)
        
        # 加载数据
        efficiency_data = self.data_loader.load_efficiency_metrics()
        time_aware = self.data_loader.load_time_aware_metrics()
        
        if efficiency_data is None or efficiency_data.empty:
            print("❌ 无法加载计算效率数据")
            return
            
        # Table 6: 计算效率对比表格
        self.generate_efficiency_table(efficiency_data)
        
        # Figure I1: 推理延迟对比
        self.plot_inference_latency(efficiency_data)
        
        # Figure I2: 模型复杂度对比
        self.plot_model_complexity(efficiency_data)
        
        # Figure I3: 性能-效率权衡
        self.plot_performance_efficiency_tradeoff(efficiency_data, time_aware)
        
        # Figure I4: 综合评分雷达图
        self.plot_comprehensive_radar(efficiency_data, time_aware)
        
        print("\n✅ Section I 所有图表生成完成！")
        
    def generate_efficiency_table(self, df):
        """Table 6: 计算效率对比表格"""
        print("\n📊 生成 Table 6: 计算效率对比表格...")
        
        # 创建表格
        table_data = []
        
        for model in MODEL_ORDER:
            if model in df['model'].values:
                model_data = df[df['model'] == model].iloc[0]
                
                row = {
                    'Model': MODEL_LABELS[model],
                    'Parameters': f"{model_data.get('parameters', 0)/1e6:.2f}M",
                    'Training Time (min)': f"{model_data.get('training_time_minutes', 0):.1f}",
                    'Inference Time (ms)': f"{model_data.get('inference_time_ms', 0):.2f}",
                    'Memory (MB)': f"{model_data.get('memory_mb', 0):.1f}",
                    'FLOPs': f"{model_data.get('flops', 0)/1e9:.2f}G"
                }
                table_data.append(row)
        
        table_df = pd.DataFrame(table_data)
        
        # 保存CSV
        csv_path = os.path.join('output/tables', 'table6_efficiency_comparison.csv')
        os.makedirs('output/tables', exist_ok=True)
        table_df.to_csv(csv_path, index=False)
        print(f"  ✅ 保存CSV: {csv_path}")
        
        # 保存LaTeX
        latex_path = os.path.join('output/tables', 'table6_efficiency_comparison.tex')
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Computational Efficiency Comparison}\n")
            f.write("\\label{tab:efficiency_comparison}\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\hline\n")
            f.write("Model & Parameters & Training Time & Inference Time & Memory & FLOPs \\\\\n")
            f.write(" & (M) & (min) & (ms) & (MB) & (G) \\\\\n")
            f.write("\\hline\n")
            
            for _, row in table_df.iterrows():
                f.write(f"{row['Model']} & {row['Parameters']} & {row['Training Time (min)']} & ")
                f.write(f"{row['Inference Time (ms)']} & {row['Memory (MB)']} & {row['FLOPs']} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"  ✅ 保存LaTeX: {latex_path}")
        
    def plot_inference_latency(self, df):
        """图I1: 推理延迟对比"""
        print("\n📊 生成 Figure I1: 推理延迟对比...")
        
        plt.style.use(STYLE_FILE)
        fig, ax = plt.subplots()
        
        # 提取数据
        models = []
        latencies = []
        
        for model in MODEL_ORDER:
            if model in df['model'].values:
                model_data = df[df['model'] == model].iloc[0]
                models.append(model)
                latencies.append(model_data.get('inference_time_ms', 0))
        
        # 绘制柱状图
        colors = [MODEL_COLORS[m] for m in models]
        labels = [MODEL_LABELS[m] for m in models]
        
        bars = ax.bar(range(len(models)), latencies, color=colors, 
                     alpha=0.7, edgecolor='black')
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Latency Comparison')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加数值标签
        for bar, val in zip(bars, latencies):
            ax.text(bar.get_x() + bar.get_width()/2, val, 
                   f'{val:.2f}', ha='center', va='bottom')
        
        # 添加实时性要求线
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Real-time Threshold (100ms)')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_i1_inference_latency')
        plt.close()
        
    def plot_model_complexity(self, df):
        """Figure I2: 模型复杂度对比（拆分为独立图）"""
        print("\n📊 生成 Figure I2: 模型复杂度对比...")
        
        plt.style.use(STYLE_FILE)
        
        # 参数量
        fig, ax = plt.subplots()
        models = []
        params = []
        
        for model in MODEL_ORDER:
            if model in df['model'].values:
                model_data = df[df['model'] == model].iloc[0]
                models.append(model)
                params.append(model_data.get('parameters', 0) / 1e6)  # 转换为百万
        
        colors = [MODEL_COLORS[m] for m in models]
        labels = [MODEL_LABELS[m] for m in models]
        
        bars = ax.bar(range(len(models)), params, color=colors, 
                       alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Parameters (Million)')
        ax.set_title('Model Parameters')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for bar, val in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, val, 
                    f'{val:.2f}M', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_i2_model_params')
        plt.close()
        
        # FLOPs
        fig, ax = plt.subplots()
        flops = []
        
        for model in models:
            model_data = df[df['model'] == model].iloc[0]
            flops.append(model_data.get('flops', 0) / 1e9)  # 转换为GFLOPs
        
        bars = ax.bar(range(len(models)), flops, color=colors, 
                       alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('FLOPs (GFLOPs)')
        ax.set_title('Computational Complexity')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for bar, val in zip(bars, flops):
            ax.text(bar.get_x() + bar.get_width()/2, val, 
                    f'{val:.2f}G', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_i2_model_flops')
        plt.close()
        
    def plot_performance_efficiency_tradeoff(self, efficiency_df, performance_df):
        """图I3: 性能-效率权衡"""
        print("\n📊 生成 Figure I3: 性能-效率权衡...")
        
        plt.style.use(STYLE_FILE)
        fig, ax = plt.subplots()
        
        # 合并数据
        models = []
        inference_times = []
        dr_5s = []
        
        for model in MODEL_ORDER:
            if model in efficiency_df['model'].values and model in performance_df['model'].values:
                eff_data = efficiency_df[efficiency_df['model'] == model].iloc[0]
                perf_data = performance_df[performance_df['model'] == model].iloc[0]
                
                models.append(model)
                inference_times.append(eff_data.get('inference_time_ms', 0))
                dr_5s.append(perf_data['DR@5s'] * 100)
        
        # 绘制散点图并添加到图例
        for i, model in enumerate(models):
            ax.scatter(inference_times[i], dr_5s[i], 
                      color=MODEL_COLORS[model],
                      s=100, alpha=0.7, 
                      edgecolors='black', linewidth=1.5,
                      label=MODEL_LABELS[model], zorder=3)
        
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('DR@5s (%)')
        ax.set_title('Performance-Efficiency Trade-off')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)
        
        # 添加理想区域标注
        ax.axvspan(0, 50, alpha=0.1, color='green', zorder=1)
        ax.axhspan(90, 100, alpha=0.1, color='green', zorder=1)
        ax.text(0.02, 0.98, 'Ideal Region\n(Fast & Accurate)', 
               transform=ax.transAxes, ha='left', va='top',
               style='italic',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_i3_performance_efficiency_tradeoff')
        plt.close()
        
    def plot_comprehensive_radar(self, efficiency_df, performance_df):
        """图I4: 综合评分雷达图"""
        print("\n📊 生成 Figure I4: 综合评分雷达图...")
        
        plt.style.use(STYLE_FILE)
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        
        # 定义评估维度
        categories = ['DR@5s', 'MTBFA', 'Speed\n(1/ADD)', 'Efficiency\n(1/Latency)', 'Lightness\n(1/Params)']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # 为每个模型计算归一化得分
        for model in MODEL_ORDER[:5]:  # 只显示前5个模型，避免过于拥挤
            if model not in efficiency_df['model'].values or model not in performance_df['model'].values:
                continue
                
            eff_data = efficiency_df[efficiency_df['model'] == model].iloc[0]
            perf_data = performance_df[performance_df['model'] == model].iloc[0]
            
            # 提取原始值
            dr = perf_data['DR@5s']
            mtbfa = perf_data['MTBFA']
            add = perf_data['ADD']
            latency = eff_data.get('inference_time_ms', 100)
            params = eff_data.get('parameters', 1e6)
            
            # 归一化到0-1（越大越好）
            values = [
                dr,  # DR@5s已经是0-1
                min(mtbfa / 100, 1),  # MTBFA归一化到100小时
                min(10 / add, 1),  # Speed: 1/ADD，归一化
                min(100 / latency, 1),  # Efficiency: 1/latency
                min(1e6 / params, 1)  # Lightness: 1/params
            ]
            values += values[:1]
            
            # 绘制
            ax.plot(angles, values, 'o-', linewidth=2, 
                   color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.7)
            ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS[model])
        
        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True, alpha=0.3)
        
        ax.set_title('Comprehensive Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_i4_comprehensive_radar')
        plt.close()
        
    def _save_figure(self, fig, name):
        """保存图表"""
        for fmt in OUTPUT_FORMATS:
            filepath = os.path.join(self.output_dir, f'{name}.{fmt}')
            fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
            print(f"  ✅ 保存: {filepath}")


if __name__ == '__main__':
    generator = SectionIEfficiency()
    generator.generate_all()
