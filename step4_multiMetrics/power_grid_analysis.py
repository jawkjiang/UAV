"""
电力系统无人机巡检场景 - 分析和可视化模块
Power Grid UAV Inspection - Analysis and Visualization Module

生成综合排名和可视化图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class PowerGridAnalyzer:
    """电力巡检场景分析器"""
    
    def __init__(self, metrics_dir='./output/power_grid_analysis'):
        self.metrics_dir = metrics_dir
        self.viz_dir = os.path.join(metrics_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 加载所有指标
        self.metrics = self.load_all_metrics()
    
    def load_all_metrics(self) -> dict:
        """加载所有已计算的指标"""
        print("加载指标数据...")
        
        metrics = {}
        metric_files = {
            'cost_sensitive': 'metrics1_cost_sensitive.csv',
            'detection_delay': 'metrics2_detection_delay.csv',
            'reliability': 'metrics3_reliability.csv',
            'robustness': 'metrics3_robustness.csv',
            'risk_weighted': 'metrics4_risk_weighted.csv',
            'edge_deployment': 'metrics5_edge_deployment.csv'
        }
        
        for key, filename in metric_files.items():
            filepath = os.path.join(self.metrics_dir, filename)
            if os.path.exists(filepath):
                metrics[key] = pd.read_csv(filepath)
                print(f"  ✓ {filename}: {len(metrics[key])} 条记录")
            else:
                print(f"  ✗ {filename}: 文件不存在")
        
        return metrics
    
    def generate_attack_type_rankings(self):
        """生成按攻击类型分组的排名表"""
        print("\n生成按攻击类型分组的排名...")
        
        if 'cost_sensitive' not in self.metrics:
            return None
        
        df = self.metrics['cost_sensitive']
        
        # 为每种攻击类型生成排名
        attack_rankings = {}
        
        for attack_type in df['attack_type'].unique():
            attack_df = df[df['attack_type'] == attack_type]
            
            # 按模型排名（EOC越低越好）
            eoc_rank = attack_df.set_index('model_type')['EOC'].rank(ascending=True)
            f1_rank = attack_df.set_index('model_type')['F_1.0'].rank(ascending=False)
            fnr_rank = attack_df.set_index('model_type')['FNR'].rank(ascending=True)
            fpr_rank = attack_df.set_index('model_type')['FPR'].rank(ascending=True)
            
            attack_rankings[attack_type] = pd.DataFrame({
                'EOC_rank': eoc_rank,
                'F1_rank': f1_rank,
                'FNR_rank': fnr_rank,
                'FPR_rank': fpr_rank,
                'avg_rank': (eoc_rank + f1_rank + fnr_rank + fpr_rank) / 4
            }).sort_values('avg_rank')
        
        # 保存每种攻击类型的排名
        for attack_type, ranking_df in attack_rankings.items():
            output_path = os.path.join(self.metrics_dir, f'ranking_{attack_type}.csv')
            ranking_df.to_csv(output_path)
            print(f"  ✓ {attack_type}: {output_path}")
        
        return attack_rankings
    
    def generate_comprehensive_ranking(self):
        """生成综合排名表（保留用于整体对比）"""
        print("\n生成综合排名...")
        
        rankings = {}
        
        # 1. 代价权衡排名 (EOC越低越好)
        if 'cost_sensitive' in self.metrics:
            df = self.metrics['cost_sensitive']
            eoc_by_model = df.groupby('model_type')['EOC'].mean()
            rankings['EOC_rank'] = eoc_by_model.rank(ascending=True)
            
            # F_5.0排名 (越高越好)
            f5_by_model = df.groupby('model_type')['F_5.0'].mean()
            rankings['F5_rank'] = f5_by_model.rank(ascending=False)
        
        # 2. 时效性排名 (MTTD越低越好)
        if 'detection_delay' in self.metrics:
            df = self.metrics['detection_delay']
            # 过滤掉inf值
            df_clean = df[df['MTTD'] != float('inf')]
            mttd_by_model = df_clean.groupby('model_type')['MTTD'].mean()
            rankings['MTTD_rank'] = mttd_by_model.rank(ascending=True)
            
            # RRR@5排名 (越高越好)
            rrr5_by_model = df_clean.groupby('model_type')['RRR@5'].mean()
            rankings['RRR5_rank'] = rrr5_by_model.rank(ascending=False)
        
        # 3. 可靠性排名
        if 'reliability' in self.metrics:
            df = self.metrics['reliability']
            # Availability排名 (越高越好)
            avail_by_model = df.groupby('model_type')['Availability'].mean()
            rankings['Availability_rank'] = avail_by_model.rank(ascending=False)
        
        # 3.5 鲁棒性排名
        if 'robustness' in self.metrics:
            df = self.metrics['robustness']
            df_indexed = df.set_index('model_type')
            rankings['Robustness_rank'] = df_indexed['Robustness'].rank(ascending=False)
        
        # 4. 风险加权排名
        if 'risk_weighted' in self.metrics:
            df = self.metrics['risk_weighted']
            df_indexed = df.set_index('model_type')
            
            # RWDR排名 (越高越好)
            rankings['RWDR_rank'] = df_indexed['RWDR'].rank(ascending=False)
            
            # CMR排名 (越低越好)
            rankings['CMR_rank'] = df_indexed['CMR'].rank(ascending=True)
            
            # WCP排名 (越高越好)
            rankings['WCP_rank'] = df_indexed['WCP'].rank(ascending=False)
        
        # 5. 边缘部署排名
        if 'edge_deployment' in self.metrics:
            df = self.metrics['edge_deployment']
            df_indexed = df.set_index('model_type')
            
            # PER排名 (越高越好)
            rankings['PER_rank'] = df_indexed['PER'].rank(ascending=False)
            
            # Memory Score排名 (越高越好)
            rankings['Memory_rank'] = df_indexed['Memory_Score'].rank(ascending=False)
        
        # 合并所有排名
        ranking_df = pd.DataFrame(rankings)
        
        # 计算平均排名
        ranking_df['avg_rank'] = ranking_df.mean(axis=1)
        ranking_df = ranking_df.sort_values('avg_rank')
        
        # 保存
        output_path = os.path.join(self.metrics_dir, 'comprehensive_rankings.csv')
        ranking_df.to_csv(output_path)
        print(f"✓ 综合排名已保存: {output_path}")
        
        # 生成按攻击类型的排名
        attack_rankings = self.generate_attack_type_rankings()
        
        return ranking_df, attack_rankings
    
    def calculate_discrimination_ability(self):
        """计算各指标的区分度"""
        print("\n计算指标区分度...")
        
        discrimination = []
        
        # 指标体系1
        if 'cost_sensitive' in self.metrics:
            df = self.metrics['cost_sensitive']
            for metric in ['EOC', 'F_0.5', 'F_1.0', 'F_2.0', 'F_5.0']:
                if metric in df.columns:
                    values = df.groupby('model_type')[metric].mean()
                    cv = values.std() / values.mean() if values.mean() != 0 else 0
                    discrimination.append({
                        'metric_system': '代价权衡',
                        'metric': metric,
                        'CV': cv,
                        'rating': self._rate_cv(cv)
                    })
        
        # 指标体系2
        if 'detection_delay' in self.metrics:
            df = self.metrics['detection_delay']
            df_clean = df[df['MTTD'] != float('inf')]
            for metric in ['MTTD', 'P90_delay', 'RRR@5', 'RRR@10']:
                if metric in df_clean.columns:
                    values = df_clean.groupby('model_type')[metric].mean()
                    cv = values.std() / values.mean() if values.mean() != 0 else 0
                    discrimination.append({
                        'metric_system': '时效性',
                        'metric': metric,
                        'CV': cv,
                        'rating': self._rate_cv(cv)
                    })
        
        # 指标体系4
        if 'risk_weighted' in self.metrics:
            df = self.metrics['risk_weighted']
            for metric in ['RWDR', 'CMR', 'WCP']:
                if metric in df.columns:
                    values = df[metric]
                    cv = values.std() / values.mean() if values.mean() != 0 else 0
                    discrimination.append({
                        'metric_system': '风险加权',
                        'metric': metric,
                        'CV': cv,
                        'rating': self._rate_cv(cv)
                    })
        
        # 指标体系5
        if 'edge_deployment' in self.metrics:
            df = self.metrics['edge_deployment']
            for metric in ['PER', 'Memory_Score', 'Energy_Score']:
                if metric in df.columns:
                    values = df[metric]
                    cv = values.std() / values.mean() if values.mean() != 0 else 0
                    discrimination.append({
                        'metric_system': '边缘部署',
                        'metric': metric,
                        'CV': cv,
                        'rating': self._rate_cv(cv)
                    })
        
        disc_df = pd.DataFrame(discrimination)
        output_path = os.path.join(self.metrics_dir, 'discrimination_analysis.csv')
        disc_df.to_csv(output_path, index=False)
        print(f"✓ 区分度分析已保存: {output_path}")
        
        return disc_df
    
    @staticmethod
    def _rate_cv(cv: float) -> str:
        """评价变异系数"""
        if cv > 0.15:
            return "⭐⭐⭐⭐⭐ 优秀"
        elif cv > 0.10:
            return "⭐⭐⭐⭐ 良好"
        elif cv > 0.05:
            return "⭐⭐⭐ 中等"
        else:
            return "⭐⭐ 偏低"
    
    def plot_cost_sensitive_comparison(self):
        """可视化1: 代价权衡对比图"""
        if 'cost_sensitive' not in self.metrics:
            return
        
        print("生成可视化1: 代价权衡对比...")
        
        df = self.metrics['cost_sensitive']
        
        # 计算不同C_FN值下的EOC
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: 按模型的平均EOC
        eoc_by_model = df.groupby('model_type')['EOC'].mean().sort_values()
        
        axes[0].barh(range(len(eoc_by_model)), eoc_by_model.values, color='steelblue')
        axes[0].set_yticks(range(len(eoc_by_model)))
        axes[0].set_yticklabels(eoc_by_model.index)
        axes[0].set_xlabel('Expected Operational Cost (EOC)', fontsize=11)
        axes[0].set_title('各模型的期望运营代价 (C_FP=1, C_FN=50)', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # 子图2: F_beta对比
        f_cols = ['F_0.5', 'F_1.0', 'F_2.0', 'F_5.0']
        f_by_model = df.groupby('model_type')[f_cols].mean()
        
        x = np.arange(len(f_by_model))
        width = 0.2
        
        for i, col in enumerate(f_cols):
            axes[1].bar(x + i*width, f_by_model[col], width, label=col)
        
        axes[1].set_xlabel('模型', fontsize=11)
        axes[1].set_ylabel('F-Score', fontsize=11)
        axes[1].set_title('不同β值的F-Score对比', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(f_by_model.index, rotation=45, ha='right')
        axes[1].legend(loc='lower right')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'cost_sensitive_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def plot_detection_delay_distribution(self):
        """可视化2: 检测延迟分布图"""
        if 'detection_delay' not in self.metrics:
            return
        
        print("生成可视化2: 检测延迟分布...")
        
        df = self.metrics['detection_delay']
        df_clean = df[df['MTTD'] != float('inf')]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: 箱线图
        delay_cols = ['P50_delay', 'P90_delay', 'P95_delay']
        delay_by_model = df_clean.groupby('model_type')[delay_cols].mean()
        
        bp = axes[0].boxplot([delay_by_model[col] for col in delay_cols],
                             labels=['P50', 'P90', 'P95'],
                             patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        axes[0].set_ylabel('检测延迟 (秒)', fontsize=11)
        axes[0].set_title('检测延迟分位数分布', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 子图2: 各模型MTTD对比
        mttd_by_model = df_clean.groupby('model_type')['MTTD'].mean().sort_values()
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(mttd_by_model)))
        axes[1].barh(range(len(mttd_by_model)), mttd_by_model.values, color=colors)
        axes[1].set_yticks(range(len(mttd_by_model)))
        axes[1].set_yticklabels(mttd_by_model.index)
        axes[1].set_xlabel('平均检测延迟 (秒)', fontsize=11)
        axes[1].set_title('各模型的MTTD对比', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'detection_delay_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def plot_risk_weighted_performance(self):
        """可视化3: 风险加权性能雷达图"""
        if 'risk_weighted' not in self.metrics:
            return
        
        print("生成可视化3: 风险加权性能...")
        
        df = self.metrics['risk_weighted']
        
        # 准备雷达图数据
        categories = ['RWDR', '1-CMR', 'WCP', 'BCP', 'AVG_F1']
        num_vars = len(categories)
        
        # 归一化数据
        df_norm = df.copy()
        df_norm['1-CMR'] = 1 - df_norm['CMR']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # 绘制每个模型
        colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
        
        for idx, (_, row) in enumerate(df_norm.iterrows()):
            values = [row['RWDR'], row['1-CMR'], row['WCP'], row['BCP'], row['AVG_F1']]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model_type'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('风险加权性能雷达图', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'risk_weighted_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def plot_edge_deployment_suitability(self):
        """可视化4: 边缘部署适配性散点图"""
        if 'edge_deployment' not in self.metrics:
            return
        
        print("生成可视化4: 边缘部署适配性...")
        
        df = self.metrics['edge_deployment']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 散点图: X=模型大小, Y=AUC, 气泡大小=参数量
        sizes = (df['model_params'] / df['model_params'].max()) * 1000
        
        scatter = ax.scatter(df['model_size_MB'], df['mean_AUC'], 
                           s=sizes, alpha=0.6, c=df['PER'], cmap='RdYlGn')
        
        # 添加模型标签
        for _, row in df.iterrows():
            ax.annotate(row['model_type'], 
                       (row['model_size_MB'], row['mean_AUC']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        ax.set_xlabel('模型大小 (MB)', fontsize=12)
        ax.set_ylabel('平均AUC', fontsize=12)
        ax.set_title('边缘部署适配性分析\n(气泡大小=参数量, 颜色=性能效率比)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('性能效率比 (PER)', fontsize=10)
        
        # 添加理想区域标注
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='高性能阈值')
        ax.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='轻量级阈值')
        ax.legend(loc='lower left')
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'edge_deployment_suitability.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def plot_attack_severity_analysis(self):
        """可视化6: 攻击严重度分析"""
        if 'cost_sensitive' not in self.metrics:
            return
        
        print("生成可视化6: 攻击严重度分析...")
        
        df = self.metrics['cost_sensitive']
        
        # 按攻击类型计算平均召回率
        recall_by_attack = df.groupby('attack_type')['FNR'].mean().sort_values(ascending=False)
        
        # 定义攻击权重
        attack_weights = {
            'replay_same_hard': 5,
            'replay_other_soft': 4,
            'delay': 3,
            'drift_ramp': 2,
            'drift_sigmoid': 2,
            'step': 1,
            'takeover_step': 1,
            'takeover_ramp': 1
        }
        
        weights = [attack_weights.get(attack, 1) for attack in recall_by_attack.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(recall_by_attack)), recall_by_attack.values)
        
        # 根据权重着色
        colors = plt.cm.Reds(np.array(weights) / max(weights))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(range(len(recall_by_attack)))
        ax.set_yticklabels(recall_by_attack.index)
        ax.set_xlabel('平均漏报率 (FNR)', fontsize=11)
        ax.set_title('各攻击类型的检测难度分析\n(颜色深度=攻击严重度)', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=plt.cm.Reds(1.0), label='高危 (权重5)'),
            Patch(facecolor=plt.cm.Reds(0.6), label='中危 (权重2-4)'),
            Patch(facecolor=plt.cm.Reds(0.2), label='低危 (权重1)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'attack_severity_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def plot_comprehensive_ranking_heatmap(self, ranking_df):
        """可视化5: 综合排名热力图"""
        print("生成可视化5: 综合排名热力图...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 准备数据 (排除平均排名列)
        data = ranking_df.drop(columns=['avg_rank'])
        
        # 绘制热力图
        sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   cbar_kws={'label': '排名 (1=最佳)'}, 
                   linewidths=0.5, ax=ax)
        
        ax.set_xlabel('指标', fontsize=12, fontweight='bold')
        ax.set_ylabel('模型', fontsize=12, fontweight='bold')
        ax.set_title('综合排名热力图', fontsize=14, fontweight='bold', pad=15)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'comprehensive_ranking.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def plot_attack_type_performance(self):
        """可视化7: 各攻击类型下的模型性能对比"""
        if 'cost_sensitive' not in self.metrics:
            return
        
        print("生成可视化7: 各攻击类型性能对比...")
        
        df = self.metrics['cost_sensitive']
        
        # 定义攻击分组
        easy_attacks = ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']
        hard_attacks = ['replay_same_hard', 'replay_other_soft']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 子图1: F1分数对比（所有攻击）
        attack_types = df['attack_type'].unique()
        models = df['model_type'].unique()
        
        f1_matrix = df.pivot_table(values='F_1.0', index='model_type', columns='attack_type')
        
        sns.heatmap(f1_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                   vmin=0, vmax=1, ax=axes[0, 0], cbar_kws={'label': 'F1 Score'})
        axes[0, 0].set_title('各攻击类型的F1分数', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('攻击类型', fontsize=10)
        axes[0, 0].set_ylabel('模型', fontsize=10)
        
        # 子图2: 漏报率对比（所有攻击）
        fnr_matrix = df.pivot_table(values='FNR', index='model_type', columns='attack_type')
        
        sns.heatmap(fnr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   vmin=0, vmax=1, ax=axes[0, 1], cbar_kws={'label': 'FNR (漏报率)'})
        axes[0, 1].set_title('各攻击类型的漏报率', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('攻击类型', fontsize=10)
        axes[0, 1].set_ylabel('模型', fontsize=10)
        
        # 子图3: 误报率对比（所有攻击）
        fpr_matrix = df.pivot_table(values='FPR', index='model_type', columns='attack_type')
        
        sns.heatmap(fpr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   vmin=0, vmax=1, ax=axes[1, 0], cbar_kws={'label': 'FPR (误报率)'})
        axes[1, 0].set_title('各攻击类型的误报率', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('攻击类型', fontsize=10)
        axes[1, 0].set_ylabel('模型', fontsize=10)
        
        # 子图4: 分组对比（易检测 vs 难检测）
        df_easy = df[df['attack_type'].isin(easy_attacks)]
        df_hard = df[df['attack_type'].isin(hard_attacks)]
        
        easy_f1 = df_easy.groupby('model_type')['F_1.0'].mean()
        hard_f1 = df_hard.groupby('model_type')['F_1.0'].mean()
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x - width/2, easy_f1, width, label='易检测攻击', color='green', alpha=0.7)
        bars2 = axes[1, 1].bar(x + width/2, hard_f1, width, label='重放攻击', color='red', alpha=0.7)
        
        axes[1, 1].set_xlabel('模型', fontsize=10)
        axes[1, 1].set_ylabel('平均F1分数', fontsize=10)
        axes[1, 1].set_title('易检测攻击 vs 重放攻击性能对比', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].set_ylim(0, 1.1)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'attack_type_performance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def generate_attack_type_report(self, attack_rankings):
        """生成按攻击类型的详细分析报告"""
        if not attack_rankings:
            return
        
        print("\n生成分攻击类型分析报告...")
        
        report_lines = [
            "# 各攻击类型详细性能分析",
            "",
            "## 📊 概述",
            "",
            "本报告按攻击类型分别展示各模型的性能，避免混合评估导致的偏差。",
            "",
            "---",
            ""
        ]
        
        # 定义攻击分组
        easy_attacks = {
            'step': '突变攻击',
            'drift_ramp': '线性漂移',
            'drift_sigmoid': 'S型漂移',
            'delay': '延迟攻击',
            'takeover_step': '接管突变',
            'takeover_ramp': '接管渐变'
        }
        
        hard_attacks = {
            'replay_same_hard': '同飞行硬重放',
            'replay_other_soft': '跨飞行软重放'
        }
        
        # 易检测攻击分析
        report_lines.extend([
            "## ✅ 易检测攻击类型（高性能）",
            "",
            "以下攻击类型特征明显，所有模型均可达到优秀性能：",
            ""
        ])
        
        for attack_type, name_cn in easy_attacks.items():
            if attack_type in attack_rankings:
                ranking = attack_rankings[attack_type]
                best_model = ranking.index[0]
                best_avg_rank = ranking.iloc[0]['avg_rank']
                
                report_lines.extend([
                    f"### {attack_type} ({name_cn})",
                    "",
                    f"- **最佳模型**: {best_model} (平均排名: {best_avg_rank:.2f})",
                    f"- **推荐**: 所有模型均可部署",
                    f"- **详细排名**: 见 `ranking_{attack_type}.csv`",
                    ""
                ])
        
        # 难检测攻击分析
        report_lines.extend([
            "---",
            "",
            "## ❌ 难检测攻击类型（重放攻击）",
            "",
            "⚠️ **警告**: 重放攻击检测性能较差，不建议单独部署当前模型。",
            ""
        ])
        
        for attack_type, name_cn in hard_attacks.items():
            if attack_type in attack_rankings:
                ranking = attack_rankings[attack_type]
                best_model = ranking.index[0]
                best_avg_rank = ranking.iloc[0]['avg_rank']
                worst_model = ranking.index[-1]
                worst_avg_rank = ranking.iloc[-1]['avg_rank']
                
                report_lines.extend([
                    f"### {attack_type} ({name_cn})",
                    "",
                    f"- **相对最佳**: {best_model} (平均排名: {best_avg_rank:.2f})",
                    f"- **表现最差**: {worst_model} (平均排名: {worst_avg_rank:.2f})",
                    f"- **建议**: 需要配合其他检测手段或人工介入",
                    f"- **详细排名**: 见 `ranking_{attack_type}.csv`",
                    ""
                ])
        
        report_lines.extend([
            "---",
            "",
            "## 💡 部署建议",
            "",
            "### 分层部署策略",
            "",
            "```",
            "第一层：快速检测（易检测攻击）",
            "├─ 检测目标: Step, Drift, Delay, Takeover",
            "├─ 推荐模型: TCN, CNN-LSTM, BiLSTM",
            "├─ 阈值设置: 0.5 (标准)",
            "├─ 性能指标: F1 > 0.95, FNR < 5%",
            "└─ 部署状态: ✅ 可直接部署",
            "",
            "第二层：重放检测（需要增强）",
            "├─ 检测目标: Replay attacks",
            "├─ 当前限制: F1 < 0.22, FNR > 30%",
            "├─ 建议方案:",
            "│   ├─ 人工复核机制",
            "│   ├─ 多传感器融合",
            "│   ├─ 统计异常检测",
            "│   └─ 行为分析算法",
            "└─ 部署状态: ❌ 需要专门研发",
            "```",
            "",
            "### 各模型适用场景",
            ""
        ])
        
        # 添加各模型在不同攻击上的表现总结
        if 'cost_sensitive' in self.metrics:
            df = self.metrics['cost_sensitive']
            
            for model in df['model_type'].unique():
                model_df = df[df['model_type'] == model]
                
                # 计算易检测和难检测攻击的平均F1
                easy_f1 = model_df[model_df['attack_type'].isin(easy_attacks.keys())]['F_1.0'].mean()
                hard_f1 = model_df[model_df['attack_type'].isin(hard_attacks.keys())]['F_1.0'].mean()
                
                report_lines.extend([
                    f"**{model}**:",
                    f"- 易检测攻击: F1 = {easy_f1:.3f}",
                    f"- 重放攻击: F1 = {hard_f1:.3f}",
                    f"- 性能差距: {abs(easy_f1 - hard_f1):.3f}",
                    ""
                ])
        
        report_lines.extend([
            "---",
            "",
            "## 📈 关键结论",
            "",
            "1. **性能分化严重**: 易检测攻击F1 > 0.9，重放攻击F1 < 0.2",
            "2. **不宜混合评估**: 统一指标会掩盖重放攻击的严重问题",
            "3. **分层部署必要**: 第一层可直接部署，第二层需要专门研发",
            "4. **当前可落地**: 针对易检测攻击的检测系统可以部署",
            "5. **未来方向**: 重点攻克重放攻击检测技术",
            ""
        ])
        
        # 保存报告
        output_path = os.path.join(self.metrics_dir, 'ATTACK_TYPE_ANALYSIS.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 分攻击类型报告已保存: {output_path}")
        """可视化6: 攻击严重度分析"""
        if 'cost_sensitive' not in self.metrics:
            return
        
        print("生成可视化6: 攻击严重度分析...")
        
        df = self.metrics['cost_sensitive']
        
        # 按攻击类型计算平均召回率
        recall_by_attack = df.groupby('attack_type')['FNR'].mean().sort_values(ascending=False)
        
        # 定义攻击权重
        attack_weights = {
            'replay_same_hard': 5,
            'replay_other_soft': 4,
            'delay': 3,
            'drift_ramp': 2,
            'drift_sigmoid': 2,
            'step': 1,
            'takeover_step': 1,
            'takeover_ramp': 1
        }
        
        weights = [attack_weights.get(attack, 1) for attack in recall_by_attack.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(recall_by_attack)), recall_by_attack.values)
        
        # 根据权重着色
        colors = plt.cm.Reds(np.array(weights) / max(weights))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(range(len(recall_by_attack)))
        ax.set_yticklabels(recall_by_attack.index)
        ax.set_xlabel('平均漏报率 (FNR)', fontsize=11)
        ax.set_title('各攻击类型的检测难度分析\n(颜色深度=攻击严重度)', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=plt.cm.Reds(1.0), label='高危 (权重5)'),
            Patch(facecolor=plt.cm.Reds(0.6), label='中危 (权重2-4)'),
            Patch(facecolor=plt.cm.Reds(0.2), label='低危 (权重1)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        output_path = os.path.join(self.viz_dir, 'attack_severity_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {output_path}")
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("\n" + "="*80)
        print("生成可视化图表...")
        print("="*80)
        
        self.plot_cost_sensitive_comparison()
        self.plot_detection_delay_distribution()
        self.plot_risk_weighted_performance()
        self.plot_edge_deployment_suitability()
        self.plot_attack_severity_analysis()
        self.plot_attack_type_performance()
        
        # 生成排名并绘制热力图
        ranking_df, attack_rankings = self.generate_comprehensive_ranking()
        self.plot_comprehensive_ranking_heatmap(ranking_df)
        
        # 生成分攻击类型报告
        self.generate_attack_type_report(attack_rankings)
        
        print("\n✓ 所有可视化图表生成完成!")


def main():
    """主函数"""
    print("="*80)
    print("电力系统无人机巡检场景 - 分析和可视化")
    print("Power Grid UAV Inspection - Analysis and Visualization")
    print("="*80)
    
    # 初始化分析器
    analyzer = PowerGridAnalyzer(metrics_dir='./output/power_grid_analysis')
    
    # 生成综合排名和按攻击类型的排名
    ranking_df, attack_rankings = analyzer.generate_comprehensive_ranking()
    
    # 计算区分度
    disc_df = analyzer.calculate_discrimination_ability()
    
    # 生成所有可视化
    analyzer.generate_all_visualizations()
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
    print(f"\n查看结果:")
    print(f"  - 综合排名: {analyzer.metrics_dir}/comprehensive_rankings.csv")
    print(f"  - 区分度分析: {analyzer.metrics_dir}/discrimination_analysis.csv")
    print(f"  - 可视化图表: {analyzer.viz_dir}/")
    print(f"  - 综合报告: {analyzer.metrics_dir}/POWER_GRID_REPORT.md")


if __name__ == '__main__':
    main()
