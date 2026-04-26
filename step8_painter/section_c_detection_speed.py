"""
Section C: Detection Speed Analysis
检测速度分析 - DR@Δt曲线和关键时间阈值
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


class SectionCDetectionSpeed:
    """Section C: 检测速度分析"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.output_dir = FIGURES_DIR / 'section_c'
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置绘图风格
        setup_paper_style('small_8px')
    
    def generate_all(self):
        """生成Section C的所有图表"""
        print("\n" + "="*80)
        print("Section C: Detection Speed Analysis")
        print("="*80)
        
        # 加载数据
        time_aware_df = self.data_loader.load_time_aware_metrics()
        if time_aware_df is None or time_aware_df.empty:
            print("❌ Time-aware metrics data not found!")
            return
        
        # 转换DataFrame为Dict格式
        time_aware_data = self._df_to_dict(time_aware_df)
        
        # Figure C1: DR@Δt曲线
        self.plot_dr_at_delta_t_curves(time_aware_data)
        
        # Figure C2: 关键时间阈值对比
        self.plot_critical_time_thresholds(time_aware_data)
        
        # Figure C3: 检测速度分布
        self.plot_detection_speed_distribution(time_aware_data)
        
        # Table 4: 检测速度统计
        self.generate_detection_speed_table(time_aware_data)
        
        print("\n✅ Section C completed!")
    
    def _df_to_dict(self, df: pd.DataFrame) -> Dict:
        """将DataFrame转换为Dict格式，保留所有DR@Δt数据"""
        data = {}
        for _, row in df.iterrows():
            model = row['model']
            model_data = {
                'add': row.get('ADD', 0),
                'mtbfa': row.get('MTBFA', 0),
                'n_false_alarms': row.get('n_false_alarms', 0)
            }
            
            # 提取所有DR@Δt列
            for col in df.columns:
                if col.startswith('DR@'):
                    # 从列名提取时间值，如 DR@0.5s -> 0.5
                    time_str = col.replace('DR@', '').replace('s', '')
                    try:
                        time_val = float(time_str)
                        model_data[f'dr_at_{time_str}s'] = row[col]
                    except ValueError:
                        continue
            
            data[model] = model_data
        return data
    
    def plot_dr_at_delta_t_curves(self, data: Dict):
        """
        Figure C1: DR@Δt曲线
        展示不同模型在不同时间阈值下的检测率
        """
        print("\n📊 Generating Figure C1: DR@Δt Curves...")
        
        fig, ax = plt.subplots()
        
        # 时间阈值范围（0-30秒，0-5秒更密集采样）
        # 0-5秒：每0.5秒一个点（高密度）
        # 5-15秒：每1秒一个点（中密度）
        # 15-30秒：每2秒一个点（正常密度）
        time_thresholds_1 = np.arange(0, 5.5, 0.5)  # 0, 0.5, 1, ..., 5
        time_thresholds_2 = np.arange(6, 16, 1)     # 6, 7, ..., 15
        time_thresholds_3 = np.arange(17, 31, 2)    # 17, 19, ..., 29
        time_thresholds = np.concatenate([time_thresholds_1, time_thresholds_2, time_thresholds_3])
        
        for model in MODELS:
            if model not in data:
                continue
            
            model_data = data[model]
            dr_values = []
            
            # 计算每个时间阈值下的DR
            for t in time_thresholds:
                # 尝试直接查找精确值
                dr_key = f'dr_at_{t}s'
                if dr_key in model_data:
                    dr_values.append(model_data[dr_key] * 100)
                else:
                    # 使用插值
                    dr_values.append(self._interpolate_dr(model_data, t))
            
            # 绘制曲线和标记点
            ax.plot(time_thresholds, dr_values, linewidth=2,
                   color=MODEL_COLORS[model], alpha=0.8,
                   label=MODEL_DISPLAY_NAMES[model])
            
            # 在关键时间点添加强调标记
            key_times = [0.5, 1, 2, 3, 5, 10, 15]
            for kt in key_times:
                if kt in time_thresholds:
                    idx = np.where(time_thresholds == kt)[0]
                    if len(idx) > 0:
                        markersize = 4 if kt <= 5 else (3 if kt <= 10 else 2)
                        ax.plot(kt, dr_values[idx[0]], marker='o', 
                               markersize=markersize,
                               color=MODEL_COLORS[model])
        
        # 添加关键时间点的垂直线和区域着色
        # ax.axvspan(0, 5, alpha=0.1, color='red', label='Critical Zone (0-5s)')
        # ax.axvspan(5, 15, alpha=0.1, color='orange', label='Acceptable Zone (5-15s)')
        # ax.axvline(x=5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        # ax.axvline(x=15, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Time Threshold Δt (seconds)')
        ax.set_ylabel('Detection Rate DR@Δt (%)')
        ax.set_title('Detection Rate vs Time Threshold (0-30s)')
        ax.legend(loc='lower right', ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 105)
        
        # 设置更详细的x轴刻度
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_xticklabels([0, 5, 10, 15, 20, 25, 30])
        
        save_figure(fig, 'fig_c1_dr_at_delta_t_curves', self.output_dir)
        plt.close()
    
    def plot_critical_time_thresholds(self, data: Dict):
        """
        Figure C2: 关键时间阈值对比
        对比不同模型达到特定DR所需的时间
        """
        print("\n📊 Generating Figure C2: Critical Time Thresholds...")
        
        fig, ax = plt.subplots()
        
        # 目标检测率
        target_drs = [0.5, 0.7, 0.9, 0.95]
        target_labels = ['50%', '70%', '90%', '95%']
        
        x = np.arange(len(target_drs))
        width = 0.12
        
        for i, model in enumerate(MODELS):
            if model not in data:
                continue
            
            model_data = data[model]
            times = []
            
            for target_dr in target_drs:
                # 找到达到目标DR所需的时间
                time_needed = self._find_time_for_dr(model_data, target_dr)
                times.append(time_needed)
            
            offset = (i - len(MODELS)/2) * width
            bars = ax.bar(x + offset, times, width,
                         label=MODEL_DISPLAY_NAMES[model],
                         color=MODEL_COLORS[model])
        
        ax.set_xlabel('Target Detection Rate')
        ax.set_ylabel('Time Required (seconds)')
        ax.set_title('Time Required to Reach Target Detection Rate')
        ax.set_xticks(x)
        ax.set_xticklabels(target_labels)
        ax.legend(loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        save_figure(fig, 'fig_c2_critical_time_thresholds', self.output_dir)
        plt.close()
    
    def plot_detection_speed_distribution(self, data: Dict):
        """
        Figure C3: 检测速度分布（拆分为独立图）
        展示ADD（平均检测延迟）的分布
        """
        print("\n📊 Generating Figure C3: Detection Speed Distribution...")
        
        # 提取ADD数据
        models_list = []
        add_values = []
        
        for model in MODELS:
            if model in data and 'add' in data[model]:
                models_list.append(MODEL_DISPLAY_NAMES[model])
                add_values.append(data[model]['add'])
        
        # 柱状图
        fig, ax = plt.subplots()
        bars = ax.bar(range(len(models_list)), add_values,
                      color=[MODEL_COLORS[m.lower().replace('-', '_')] 
                            for m in models_list])
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Detection Delay (seconds)')
        ax.set_title('Average Detection Delay (ADD)')
        ax.set_xticks(range(len(models_list)))
        ax.set_xticklabels(models_list, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        save_figure(fig, 'fig_c3_detection_speed_bar', self.output_dir)
        plt.close()
        
        # 箱线图（如果有详细数据）
        fig, ax = plt.subplots()
        ax.boxplot([add_values], tick_labels=['All Models'])
        ax.set_ylabel('Detection Delay (seconds)')
        ax.set_title('Detection Delay Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_figure(fig, 'fig_c3_detection_speed_box', self.output_dir)
        plt.close()
    
    def generate_detection_speed_table(self, data: Dict):
        """
        Table 4: 检测速度统计表
        """
        print("\n📊 Generating Table 4: Detection Speed Statistics...")
        
        table_data = []
        
        for model in MODELS:
            if model not in data:
                continue
            
            model_data = data[model]
            
            row = {
                'Model': MODEL_DISPLAY_NAMES[model],
                'DR@0.5s (%)': f"{model_data.get('dr_at_0.5s', 0) * 100:.2f}",
                'DR@1s (%)': f"{model_data.get('dr_at_1s', 0) * 100:.2f}",
                'DR@2s (%)': f"{model_data.get('dr_at_2s', 0) * 100:.2f}",
                'DR@3s (%)': f"{model_data.get('dr_at_3s', 0) * 100:.2f}",
                'DR@5s (%)': f"{model_data.get('dr_at_5s', 0) * 100:.2f}",
                'DR@10s (%)': f"{model_data.get('dr_at_10s', 0) * 100:.2f}",
                'DR@15s (%)': f"{model_data.get('dr_at_15s', 0) * 100:.2f}",
                'ADD (s)': f"{model_data.get('add', 0):.2f}",
                'Time to 90% DR (s)': f"{self._find_time_for_dr(model_data, 0.9):.1f}",
                'Time to 95% DR (s)': f"{self._find_time_for_dr(model_data, 0.95):.1f}"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # 保存表格
        table_gen = TableGenerator()
        table_gen.save_table(
            df, 'table4_detection_speed_statistics',
            caption='Detection Speed Statistics for Different Models',
            label='tab:detection_speed'
        )
    
    def _interpolate_dr(self, model_data: Dict, time_threshold: float) -> float:
        """插值计算特定时间阈值下的DR"""
        # 查找所有可用的时间点
        time_dr_map = {}
        for key in model_data.keys():
            if key.startswith('dr_at_') and key.endswith('s'):
                try:
                    time_str = key.replace('dr_at_', '').replace('s', '')
                    t = float(time_str)
                    time_dr_map[t] = model_data[key]
                except:
                    pass
        
        if not time_dr_map:
            return 0.0
        
        available_times = sorted(time_dr_map.keys())
        
        # 如果时间阈值超出范围，返回边界值
        if time_threshold <= available_times[0]:
            return time_dr_map[available_times[0]] * 100
        if time_threshold >= available_times[-1]:
            return time_dr_map[available_times[-1]] * 100
        
        # 线性插值
        for i in range(len(available_times) - 1):
            t1, t2 = available_times[i], available_times[i+1]
            if t1 <= time_threshold <= t2:
                dr1 = time_dr_map[t1] * 100
                dr2 = time_dr_map[t2] * 100
                ratio = (time_threshold - t1) / (t2 - t1)
                return dr1 + ratio * (dr2 - dr1)
        
        return 0.0
    
    def _find_time_for_dr(self, model_data: Dict, target_dr: float) -> float:
        """找到达到目标DR所需的时间"""
        # 收集所有时间点和对应的DR
        time_dr_pairs = []
        for key in model_data.keys():
            if key.startswith('dr_at_') and key.endswith('s'):
                try:
                    time_str = key.replace('dr_at_', '').replace('s', '')
                    t = float(time_str)
                    dr = model_data[key]
                    time_dr_pairs.append((t, dr))
                except:
                    pass
        
        if not time_dr_pairs:
            return 30.0  # 默认最大值改为30秒
        
        time_dr_pairs.sort()
        
        # 如果目标DR低于最小DR，返回最早时间
        if target_dr <= time_dr_pairs[0][1]:
            return time_dr_pairs[0][0]
        
        # 如果目标DR高于最大DR，返回最大时间（但不超过30秒）
        if target_dr >= time_dr_pairs[-1][1]:
            return min(time_dr_pairs[-1][0], 30.0)
        
        # 线性插值找到时间
        for i in range(len(time_dr_pairs) - 1):
            t1, dr1 = time_dr_pairs[i]
            t2, dr2 = time_dr_pairs[i+1]
            if dr1 <= target_dr <= dr2:
                ratio = (target_dr - dr1) / (dr2 - dr1)
                return t1 + ratio * (t2 - t1)
        
        return 30.0


if __name__ == '__main__':
    # 测试
    data_loader = DataLoader()
    section_c = SectionCDetectionSpeed(data_loader)
    section_c.generate_all()
