"""
风险分析模块 - 可视化和报告生成
Risk Analysis Module - Visualization and Report Generation

实现电力巡检场景的风险导向评估可视化和报告
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RiskAnalyzer:
    """风险分析器 - 生成可视化和报告"""
    
    def __init__(self, metrics_dir='./output/power_grid_analysis'):
        self.metrics_dir = metrics_dir
        self.viz_dir = os.path.join(metrics_dir, 'visualizations')
        self.risk_viz_dir = os.path.join(self.viz_dir, 'risk_analysis')
        
        # 创建输出目录
        os.makedirs(self.risk_viz_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载风险指标数据"""
        print("\n加载风险分析数据...")
        
        # 加载风险导向指标
        risk_metrics_path = os.path.join(self.metrics_dir, 'metrics6_advanced_risk.csv')
        if os.path.exists(risk_metrics_path):
            self.risk_metrics = pd.read_csv(risk_metrics_path)
            print(f"✓ 加载风险指标: {len(self.risk_metrics)} 条记录")
        else:
            print(f"⚠️ 未找到风险指标文件: {risk_metrics_path}")
            self.risk_metrics = None
        
        # 加载传统指标用于对比
        cost_metrics_path = os.path.join(self.metrics_dir, 'metrics1_cost_sensitive.csv')
        if os.path.exists(cost_metrics_path):
            self.cost_metrics = pd.read_csv(cost_metrics_path)
            print(f"✓ 加载传统指标: {len(self.cost_metrics)} 条记录")
        else:
            self.cost_metrics = None
    
    def generate_magnitude_vs_tpr_plot(self):
        """生成幅度vs TPR可视化"""
        if self.risk_metrics is None:
            print("⚠️ 跳过幅度vs TPR可视化（无数据）")
            return
        
        print("\n生成幅度vs TPR可视化...")
        
        # 获取所有攻击类型
        attack_types = self.risk_metrics['attack_type'].unique()
        n_attacks = len(attack_types)
        
        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, attack in enumerate(attack_types):
            ax = axes[idx]
            attack_data = self.risk_metrics[self.risk_metrics['attack_type'] == attack]
            
            # 获取所有模型
            models = attack_data['model_type'].unique()
            
            # 为每个模型绘制条形图（低/中/高幅度的TPR）
            x = np.arange(3)  # 3个幅度bin
            width = 0.1
            
            for i, model in enumerate(models):
                model_data = attack_data[attack_data['model_type'] == model].iloc[0]
                tprs = [
                    model_data['MSDR_low'] if pd.notna(model_data['MSDR_low']) else 0,
                    model_data['MSDR_mid'] if pd.notna(model_data['MSDR_mid']) else 0,
                    model_data['MSDR_high'] if pd.notna(model_data['MSDR_high']) else 0
                ]
                
                offset = (i - len(models)/2) * width
                ax.bar(x + offset, tprs, width, label=model, alpha=0.8)
            
            # 添加临界阈值线
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target: 95%')
            
            ax.set_xlabel('Magnitude Range')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{attack}')
            ax.set_xticks(x)
            ax.set_xticklabels(['<5m', '5-10m', '>10m'])
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
        
        # 删除多余的子图
        for idx in range(n_attacks, 8):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        save_path = os.path.join(self.risk_viz_dir, 'magnitude_vs_tpr.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {save_path}")
    
    def generate_rwmr_comparison(self):
        """生成RWMR vs FNR对比图"""
        if self.risk_metrics is None:
            print("⚠️ 跳过RWMR对比可视化（无数据）")
            return
        
        print("\n生成RWMR vs FNR对比可视化...")
        
        # 按模型和攻击类型统计
        comparison_data = []
        
        for attack in self.risk_metrics['attack_type'].unique():
            attack_data = self.risk_metrics[self.risk_metrics['attack_type'] == attack]
            
            for _, row in attack_data.iterrows():
                comparison_data.append({
                    'attack_type': attack,
                    'model_type': row['model_type'],
                    'metric': 'RWMR',
                    'value': row['RWMR']
                })
                comparison_data.append({
                    'attack_type': attack,
                    'model_type': row['model_type'],
                    'metric': 'FNR',
                    'value': row['traditional_FNR']
                })
        
        df = pd.DataFrame(comparison_data)
        
        # 创建分组柱状图
        fig, ax = plt.subplots(figsize=(16, 8))
        
        attack_types = df['attack_type'].unique()
        models = df['model_type'].unique()
        
        x = np.arange(len(attack_types))
        width = 0.08
        
        colors = {'RWMR': '#e74c3c', 'FNR': '#3498db'}
        
        for i, model in enumerate(models):
            for j, metric in enumerate(['RWMR', 'FNR']):
                values = []
                for attack in attack_types:
                    mask = (df['attack_type'] == attack) & \
                           (df['model_type'] == model) & \
                           (df['metric'] == metric)
                    val = df[mask]['value'].values
                    values.append(val[0] if len(val) > 0 else 0)
                
                offset = (i * 2 + j - len(models)) * width
                alpha = 0.9 if metric == 'RWMR' else 0.5
                ax.bar(x + offset, values, width, 
                      label=f'{model} - {metric}' if i == 0 else '',
                      color=colors[metric], alpha=alpha)
        
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Miss Rate')
        ax.set_title('Risk-Weighted Miss Rate (RWMR) vs Traditional FNR Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(attack_types, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.risk_viz_dir, 'rwmr_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {save_path}")
    
    def generate_risk_score_heatmap(self):
        """生成风险分数热力图"""
        if self.risk_metrics is None:
            print("⚠️ 跳过风险分数热力图（无数据）")
            return
        
        print("\n生成风险分数热力图...")
        
        # 创建透视表
        pivot_data = self.risk_metrics.pivot_table(
            index='model_type',
            columns='attack_type',
            values='avg_risk_score',
            aggfunc='mean'
        )
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                   cbar_kws={'label': 'Average Risk Score'},
                   ax=ax)
        
        ax.set_title('Average Risk Score by Model and Attack Type')
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Model Type')
        
        plt.tight_layout()
        save_path = os.path.join(self.risk_viz_dir, 'risk_score_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {save_path}")
    
    def generate_ctdr_analysis(self):
        """生成CTDR关键阈值检出率分析"""
        if self.risk_metrics is None:
            print("⚠️ 跳过CTDR分析（无数据）")
            return
        
        print("\n生成CTDR分析可视化...")
        
        # 过滤有CTDR数据的记录
        ctdr_data = self.risk_metrics[self.risk_metrics['CTDR'].notna()].copy()
        
        if len(ctdr_data) == 0:
            print("⚠️ 无CTDR数据可用")
            return
        
        # 按模型汇总CTDR
        model_ctdr = ctdr_data.groupby('model_type')['CTDR'].mean().sort_values()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(model_ctdr)), model_ctdr.values, alpha=0.7)
        
        # 为不同性能区间着色
        colors = ['#e74c3c' if x < 0.9 else '#f39c12' if x < 0.95 else '#27ae60' 
                 for x in model_ctdr.values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='Target: 95%')
        ax.axvline(x=0.99, color='green', linestyle='--', alpha=0.5, label='Excellent: 99%')
        
        ax.set_yticks(range(len(model_ctdr)))
        ax.set_yticklabels(model_ctdr.index)
        ax.set_xlabel('Critical Threshold Detection Rate (CTDR)')
        ax.set_title('Detection Rate for High-Risk Attacks\n(Magnitude >10m or Duration >20s)')
        ax.set_xlim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = os.path.join(self.risk_viz_dir, 'ctdr_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {save_path}")
    
    def generate_risk_report(self):
        """生成风险分析报告"""
        if self.risk_metrics is None:
            print("⚠️ 跳过报告生成（无数据）")
            return
        
        print("\n生成风险分析报告...")
        
        report_lines = []
        
        # 标题
        report_lines.extend([
            "# 电力巡检场景GPS欺骗检测风险评估报告",
            "# Power Grid UAV Inspection - GPS Spoofing Detection Risk Assessment",
            "",
            "---",
            "",
            "## 📋 执行摘要",
            "",
            "本报告从电力系统安全角度评估各模型的检测性能，重点关注：",
            "1. 高风险攻击（大幅度偏移、长时间攻击）的检出能力",
            "2. 漏报造成的物理风险（碰撞、任务失效）",
            "3. 传统指标无法揭示的安全盲区",
            "",
            "---",
            "",
            "## 🎯 风险模型说明",
            "",
            "### 物理风险因子",
            "",
            "攻击风险分数 (ARS) = α×CR + β×MFR + γ×CLR + δ×SAR",
            "",
            "其中：",
            "- **碰撞风险 (CR)**: 偏移距离 × 风险权重（基于安全走廊宽度）",
            "- **任务失效风险 (MFR)**: 持续时间 × 遗漏检测权重",
            "- **操作失控风险 (CLR)**: 攻击类型固有风险（突变vs渐变）",
            "- **隐蔽累积风险 (SAR)**: 特殊攻击类型加权（重放、延迟等）",
            "",
            "### 安全关键阈值",
            "",
            "- **幅度阈值**: 10米（超过可能撞击高压线/铁塔）",
            "- **时长阈值**: 20秒（遗漏大段线路检测）",
            "",
            "---",
            "",
            "## 📊 创新指标说明",
            "",
            "### 1. RWMR (Risk-Weighted Miss Rate) - 风险加权漏报率",
            "",
            "```",
            "RWMR = Σ(未检出攻击的风险分数) / Σ(所有攻击的风险分数)",
            "```",
            "",
            "**意义**:",
            "- 传统FNR只看漏了多少个（数量）",
            "- RWMR看漏掉的攻击造成的风险有多大（危害）",
            "- 如果漏掉的都是低风险攻击 → RWMR低（可接受）",
            "- 如果漏掉的都是高风险攻击 → RWMR高（危险）",
            "",
            "### 2. MSDR (Magnitude-Stratified Detection Rate) - 按幅度分层检出率",
            "",
            "按攻击幅度分层统计：",
            "- **小幅度** (<5m): 低风险区域",
            "- **中幅度** (5-10m): 中等风险",  
            "- **大幅度** (>10m): 高风险（必须100%检出）",
            "",
            "**期望**: 大幅度攻击必须接近100%检出（安全要求）",
            "",
            "### 3. DS-FNR (Duration-Sensitive False Negative Rate) - 时长敏感漏报率",
            "",
            "```",
            "DS-FNR = Σ(FN样本的持续时间) / Σ(所有攻击样本的持续时间)",
            "```",
            "",
            "**vs 传统FNR**: FNR = FN数量 / 攻击数量",
            "",
            "**意义**:",
            "- 漏了1个30秒攻击 vs 漏了3个5秒攻击",
            "- FNR相同（都是漏了样本）",
            "- 但DS-FNR前者更高（30秒 vs 15秒）→ 更危险",
            "",
            "### 4. CTDR (Critical Threshold Detection Rate) - 关键阈值检出率",
            "",
            "```",
            "CTDR = 检出的高危攻击 / 所有高危攻击",
            "```",
            "",
            "**高危攻击定义**: 幅度>10米 或 时长>20秒",
            "",
            "**vs TPR**: TPR = 检出的所有攻击 / 所有攻击",
            "",
            "**强调**: 不是所有攻击都同等重要，高危攻击必须检出",
            "",
            "---",
            "",
        ])
        
        # 关键发现
        report_lines.extend([
            "## 🔍 关键发现",
            "",
        ])
        
        # 发现1: 模型排名变化
        if self.cost_metrics is not None:
            # 按模型汇总RWMR和F1
            model_comparison = []
            
            for model in self.risk_metrics['model_type'].unique():
                rwmr = self.risk_metrics[self.risk_metrics['model_type'] == model]['RWMR'].mean()
                
                # 从cost_metrics获取F1
                f1_data = self.cost_metrics[self.cost_metrics['model_type'] == model]['F_1.0']
                f1 = f1_data.mean() if len(f1_data) > 0 else 0
                
                model_comparison.append({
                    'model': model,
                    'F1': f1,
                    'RWMR': rwmr
                })
            
            df_comp = pd.DataFrame(model_comparison).sort_values('F1', ascending=False)
            
            report_lines.extend([
                "### 发现1: 传统最优模型在高风险场景下表现不同",
                "",
                "| 模型 | F1 (传统) | 排名 | RWMR (风险) | 排名 | 排名变化 |",
                "|------|-----------|------|-------------|------|---------|"
            ])
            
            df_comp['F1_rank'] = range(1, len(df_comp) + 1)
            df_comp = df_comp.sort_values('RWMR')
            df_comp['RWMR_rank'] = range(1, len(df_comp) + 1)
            df_comp['rank_change'] = df_comp['F1_rank'] - df_comp['RWMR_rank']
            
            for _, row in df_comp.iterrows():
                change_icon = "↑" if row['rank_change'] > 0 else "↓" if row['rank_change'] < 0 else "-"
                report_lines.append(
                    f"| {row['model']} | {row['F1']:.4f} | #{int(row['F1_rank'])} | "
                    f"{row['RWMR']:.4f} | #{int(row['RWMR_rank'])} | {change_icon}{abs(int(row['rank_change']))} |"
                )
            
            report_lines.extend(["", "→ **风险导向评估改变了模型选择！**", ""])
        
        # 发现2: 高风险攻击的检出能力
        report_lines.extend([
            "### 发现2: 大幅度攻击检出能力",
            "",
        ])
        
        # 统计MSDR_high
        msdr_high_data = self.risk_metrics[self.risk_metrics['MSDR_high'].notna()]
        if len(msdr_high_data) > 0:
            model_msdr_high = msdr_high_data.groupby('model_type')['MSDR_high'].mean().sort_values(ascending=False)
            
            report_lines.append("**大幅度攻击(>10m)平均检出率**:")
            report_lines.append("")
            for model, rate in model_msdr_high.items():
                status = "✅" if rate >= 0.95 else "⚠️" if rate >= 0.90 else "❌"
                report_lines.append(f"- {status} {model}: {rate:.2%}")
            report_lines.append("")
        
        # 发现3: CTDR分析
        ctdr_data = self.risk_metrics[self.risk_metrics['CTDR'].notna()]
        if len(ctdr_data) > 0:
            report_lines.extend([
                "### 发现3: 关键阈值检出率 (CTDR)",
                "",
                "**超过安全阈值的攻击检出率**:",
                ""
            ])
            
            model_ctdr = ctdr_data.groupby('model_type')['CTDR'].mean().sort_values(ascending=False)
            
            for model, rate in model_ctdr.items():
                status = "✅" if rate >= 0.95 else "⚠️" if rate >= 0.90 else "❌"
                report_lines.append(f"- {status} {model}: {rate:.2%}")
            
            report_lines.append("")
            report_lines.append(f"→ **推荐**: 选择CTDR ≥ 95%的模型用于高风险区域")
            report_lines.append("")
        
        # 部署建议
        report_lines.extend([
            "---",
            "",
            "## 💡 部署建议",
            "",
            "### 场景A: 高压线密集区（碰撞风险优先）",
            "",
            "**优先指标**: MSDR_high, CTDR",
            "",
        ])
        
        if len(msdr_high_data) > 0:
            best_model_msdr = model_msdr_high.idxmax()
            report_lines.append(f"**推荐模型**: {best_model_msdr} (大幅度检出率: {model_msdr_high[best_model_msdr]:.2%})")
            report_lines.append("")
        
        report_lines.extend([
            "### 场景B: 长距离巡检（任务失效风险优先）",
            "",
            "**优先指标**: DS-FNR",
            "",
        ])
        
        # 找DS-FNR最低的模型
        model_dsfnr = self.risk_metrics.groupby('model_type')['DS_FNR'].mean().sort_values()
        best_model_dsfnr = model_dsfnr.idxmin()
        report_lines.append(f"**推荐模型**: {best_model_dsfnr} (时长敏感漏报率: {model_dsfnr[best_model_dsfnr]:.2%})")
        report_lines.extend(["", "### 场景C: 综合场景", "", "**优先指标**: RWMR", ""])
        
        # 找RWMR最低的模型
        model_rwmr = self.risk_metrics.groupby('model_type')['RWMR'].mean().sort_values()
        best_model_rwmr = model_rwmr.idxmin()
        report_lines.append(f"**推荐模型**: {best_model_rwmr} (风险加权漏报率: {model_rwmr[best_model_rwmr]:.2%})")
        
        report_lines.extend([
            "",
            "---",
            "",
            "## 📁 数据文件",
            "",
            f"- 风险指标详细数据: `metrics6_advanced_risk.csv`",
            f"- 可视化图表: `visualizations/risk_analysis/`",
            "",
            "---",
            "",
            "*报告生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*",
            ""
        ])
        
        # 保存报告
        report_path = os.path.join(self.metrics_dir, 'RISK_ANALYSIS.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 保存: {report_path}")
    
    def analyze_all(self):
        """运行所有分析和可视化"""
        print("="*80)
        print("风险导向评估 - 完整分析")
        print("="*80)
        
        # 生成可视化
        self.generate_magnitude_vs_tpr_plot()
        self.generate_rwmr_comparison()
        self.generate_risk_score_heatmap()
        self.generate_ctdr_analysis()
        
        # 生成报告
        self.generate_risk_report()
        
        print("\n" + "="*80)
        print("✓ 风险分析完成!")
        print(f"✓ 可视化保存在: {self.risk_viz_dir}")
        print(f"✓ 报告保存在: {os.path.join(self.metrics_dir, 'RISK_ANALYSIS.md')}")
        print("="*80)


def main():
    """主函数"""
    analyzer = RiskAnalyzer()
    analyzer.analyze_all()


if __name__ == '__main__':
    main()
