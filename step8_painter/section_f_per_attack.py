"""
Section F: Per-Attack-Type Performance Analysis
分攻击类型性能分析
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


class SectionFPerAttack:
    """Section F: 分攻击类型性能分析"""
    
    def __init__(self, output_dir='output/figures/section_f'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_loader = PaperDataLoader()
        
        # GPS欺骗攻击类型定义
        self.attack_types = {
            'step': 'Step Attack',
            'delay': 'Delay Attack',
            'takeover': 'Takeover Attack',
            'drift': 'Drift Attack'
        }
        
    def generate_all(self):
        """生成所有Section F的图表"""
        print("\n" + "="*60)
        print("Section F: Per-Attack-Type Performance Analysis")
        print("="*60)
        
        # 加载数据
        per_attack_dict = self.data_loader.load_per_attack_metrics()
        
        if per_attack_dict is None or len(per_attack_dict) == 0:
            print("❌ 无法加载分攻击类型数据")
            return
        
        # 将dict转换为DataFrame
        per_attack_data = self._convert_dict_to_dataframe(per_attack_dict)
            
        # Table 5: 分攻击类型性能表格
        self.generate_per_attack_table(per_attack_data)
        
        # Figure F1: 分攻击类型热力图（DR@5s）
        self.plot_heatmap_dr(per_attack_data)
        
        # Figure F2: 分攻击类型热力图（ADD）
        self.plot_heatmap_add(per_attack_data)
        
        # Figure F3: 攻击检测难度对比
        self.plot_attack_difficulty(per_attack_data)
        
        # Figure F4: 模型-攻击性能矩阵
        self.plot_performance_matrix(per_attack_data)
        
        print("\n✅ Section F 所有图表生成完成！")
    
    def _convert_dict_to_dataframe(self, data_dict):
        """将dict格式的per-attack数据转换为DataFrame"""
        rows = []
        for model, attacks in data_dict.items():
            for attack_type, metrics in attacks.items():
                row = {
                    'model': model,
                    'attack_type': attack_type,
                    'DR@5s': metrics.get('DR@5s', 0),
                    'ADD': metrics.get('ADD', 0)
                }
                rows.append(row)
        return pd.DataFrame(rows)
        
    def generate_per_attack_table(self, df):
        """Table 5: 分攻击类型性能表格"""
        print("\n📊 生成 Table 5: 分攻击类型性能表格...")
        
        # 创建表格
        table_data = []
        
        for model in MODEL_ORDER:
            model_data = df[df['model'] == model]
            if model_data.empty:
                continue
                
            row = {'Model': MODEL_LABELS[model]}
            
            for attack_key, attack_name in self.attack_types.items():
                attack_data = model_data[model_data['attack_type'] == attack_key]
                if not attack_data.empty:
                    dr = attack_data['DR@5s'].values[0] * 100
                    add = attack_data['ADD'].values[0]
                    row[f'{attack_name} DR@5s'] = f'{dr:.1f}%'
                    row[f'{attack_name} ADD'] = f'{add:.2f}s'
                else:
                    row[f'{attack_name} DR@5s'] = 'N/A'
                    row[f'{attack_name} ADD'] = 'N/A'
            
            table_data.append(row)
        
        table_df = pd.DataFrame(table_data)
        
        # 保存CSV
        csv_path = os.path.join('output/tables', 'table5_per_attack_performance.csv')
        os.makedirs('output/tables', exist_ok=True)
        table_df.to_csv(csv_path, index=False)
        print(f"  ✅ 保存CSV: {csv_path}")
        
        # 保存LaTeX
        latex_path = os.path.join('output/tables', 'table5_per_attack_performance.tex')
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Per-Attack-Type Performance Analysis}\n")
            f.write("\\label{tab:per_attack_performance}\n")
            f.write("\\begin{tabular}{l" + "cc" * len(self.attack_types) + "}\n")
            f.write("\\hline\n")
            
            # 表头
            header = "Model"
            for attack_name in self.attack_types.values():
                header += f" & \\multicolumn{{2}}{{c}}{{{attack_name}}}"
            f.write(header + " \\\\\n")
            
            subheader = ""
            for _ in self.attack_types:
                subheader += " & DR@5s & ADD"
            f.write(subheader + " \\\\\n")
            f.write("\\hline\n")
            
            # 数据行
            for _, row in table_df.iterrows():
                line = row['Model']
                for attack_name in self.attack_types.values():
                    dr = row[f'{attack_name} DR@5s']
                    add = row[f'{attack_name} ADD']
                    line += f" & {dr} & {add}"
                f.write(line + " \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"  ✅ 保存LaTeX: {latex_path}")
        
    def plot_heatmap_dr(self, df):
        """Figure F1: 分攻击类型热力图（DR@5s）"""
        print("\n📊 生成 Figure F1: DR@5s热力图...")
        
        plt.style.use(STYLE_FILE)
        fig, ax = plt.subplots()
        
        # 准备数据矩阵
        matrix_data = []
        model_labels = []
        
        for model in MODEL_ORDER:
            model_data = df[df['model'] == model]
            if model_data.empty:
                continue
                
            row = []
            for attack_key in self.attack_types.keys():
                attack_data = model_data[model_data['attack_type'] == attack_key]
                if not attack_data.empty:
                    dr = attack_data['DR@5s'].values[0] * 100
                    row.append(dr)
                else:
                    row.append(0)
            
            matrix_data.append(row)
            model_labels.append(MODEL_LABELS[model])
        
        matrix = np.array(matrix_data)
        
        # 绘制热力图
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # 设置刻度
        ax.set_xticks(range(len(self.attack_types)))
        ax.set_xticklabels(list(self.attack_types.values()), rotation=45, ha='right')
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels)
        
        # 添加数值标签
        for i in range(len(model_labels)):
            for j in range(len(self.attack_types)):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black")
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Detection Rate @ 5s (%)', rotation=270, labelpad=20)
        
        ax.set_title('Detection Rate @ 5s by Attack Type')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_f1_heatmap_dr_at_5s')
        plt.close()
        
    def plot_heatmap_add(self, df):
        """Figure F2: 分攻击类型热力图（ADD）"""
        print("\n📊 生成 Figure F2: ADD热力图...")
        
        plt.style.use(STYLE_FILE)
        fig, ax = plt.subplots()
        
        # 准备数据矩阵
        matrix_data = []
        model_labels = []
        
        for model in MODEL_ORDER:
            model_data = df[df['model'] == model]
            if model_data.empty:
                continue
                
            row = []
            for attack_key in self.attack_types.keys():
                attack_data = model_data[model_data['attack_type'] == attack_key]
                if not attack_data.empty:
                    add = attack_data['ADD'].values[0]
                    row.append(add)
                else:
                    row.append(0)
            
            matrix_data.append(row)
            model_labels.append(MODEL_LABELS[model])
        
        matrix = np.array(matrix_data)
        
        # 绘制热力图（ADD越小越好，所以用反向颜色）
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        
        # 设置刻度
        ax.set_xticks(range(len(self.attack_types)))
        ax.set_xticklabels(list(self.attack_types.values()), rotation=45, ha='right')
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels)
        
        # 添加数值标签
        for i in range(len(model_labels)):
            for j in range(len(self.attack_types)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}s',
                             ha="center", va="center", color="black")
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Detection Delay (seconds)', rotation=270, labelpad=20)
        
        ax.set_title('Average Detection Delay by Attack Type')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_f2_heatmap_add')
        plt.close()
        
    def plot_attack_difficulty(self, df):
        """Figure F3: 攻击检测难度对比（拆分为独立图）"""
        print("\n📊 生成 Figure F3: 攻击检测难度对比...")
        
        plt.style.use(STYLE_FILE)
        
        # 左图：平均DR@5s
        fig, ax = plt.subplots()
        attack_dr_means = []
        attack_labels = []
        
        for attack_key, attack_name in self.attack_types.items():
            attack_data = df[df['attack_type'] == attack_key]
            if not attack_data.empty:
                mean_dr = attack_data['DR@5s'].mean() * 100
                attack_dr_means.append(mean_dr)
                attack_labels.append(attack_name)
        
        colors_attack = plt.cm.Set3(range(len(attack_labels)))
        bars = ax.bar(range(len(attack_labels)), attack_dr_means, 
                       color=colors_attack, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(attack_labels)))
        ax.set_xticklabels(attack_labels, rotation=45, ha='right')
        ax.set_ylabel('Average DR@5s (%)')
        ax.set_title('Detection Difficulty by Attack Type (Higher = Easier)')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加数值标签
        for bar, val in zip(bars, attack_dr_means):
            ax.text(bar.get_x() + bar.get_width()/2, val, 
                    f'{val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_f3_attack_difficulty_dr')
        plt.close()
        
        # 右图：平均ADD
        fig, ax = plt.subplots()
        attack_add_means = []
        
        for attack_key in self.attack_types.keys():
            attack_data = df[df['attack_type'] == attack_key]
            if not attack_data.empty:
                mean_add = attack_data['ADD'].mean()
                attack_add_means.append(mean_add)
        
        bars = ax.bar(range(len(attack_labels)), attack_add_means, 
                       color=colors_attack, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(attack_labels)))
        ax.set_xticklabels(attack_labels, rotation=45, ha='right')
        ax.set_ylabel('Average ADD (seconds)')
        ax.set_title('Detection Speed by Attack Type (Lower = Faster)')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加数值标签
        for bar, val in zip(bars, attack_add_means):
            ax.text(bar.get_x() + bar.get_width()/2, val, 
                    f'{val:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        self._save_figure(fig, 'fig_f3_attack_difficulty_add')
        plt.close()
        
    def plot_performance_matrix(self, df):
        """Figure F4: 模型-攻击性能矩阵（拆分为独立图）"""
        print("\n📊 生成 Figure F4: 模型-攻击性能矩阵...")
        
        plt.style.use(STYLE_FILE)
        
        attack_keys = list(self.attack_types.keys())
        
        for attack_key in attack_keys:
            fig, ax = plt.subplots()
            attack_name = self.attack_types[attack_key]
            
            # 提取该攻击类型的数据
            attack_data = df[df['attack_type'] == attack_key]
            
            models = []
            dr_values = []
            add_values = []
            
            for model in MODEL_ORDER:
                model_data = attack_data[attack_data['model'] == model]
                if not model_data.empty:
                    models.append(model)
                    dr_values.append(model_data['DR@5s'].values[0] * 100)
                    add_values.append(model_data['ADD'].values[0])
            
            # 绘制散点图
            for i, model in enumerate(models):
                ax.scatter(add_values[i], dr_values[i], 
                          color=MODEL_COLORS[model],
                          s=150, alpha=0.7, 
                          edgecolors='black',
                          label=MODEL_LABELS[model])
            
            ax.set_xlabel('ADD (seconds)')
            ax.set_ylabel('DR@5s (%)')
            ax.set_title(attack_name)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', framealpha=0.9)
            
            plt.tight_layout()
            self._save_figure(fig, f'fig_f4_performance_{attack_key}')
            plt.close()
        
    def _save_figure(self, fig, name):
        """保存图表"""
        for fmt in OUTPUT_FORMATS:
            filepath = os.path.join(self.output_dir, f'{name}.{fmt}')
            fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
            print(f"  ✅ 保存: {filepath}")


if __name__ == '__main__':
    generator = SectionFPerAttack()
    generator.generate_all()
