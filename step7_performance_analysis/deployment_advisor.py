"""
部署建议模块
基于性能指标和部署场景，生成部署建议
"""

import pandas as pd
from typing import Dict, List
import config


class DeploymentAdvisor:
    """部署建议生成器"""
    
    def __init__(self, comprehensive_df: pd.DataFrame):
        self.df = comprehensive_df
    
    def recommend_for_scenario(self, scenario_name: str) -> List[Dict]:
        """
        为特定场景推荐模型
        Args:
            scenario_name: 'edge_device', 'edge_server', 'cloud'
        Returns:
            推荐模型列表（按优先级排序）
        """
        scenario = config.DEPLOYMENT_SCENARIOS[scenario_name]
        constraints = scenario['constraints']
        
        # 筛选满足约束的模型
        candidates = self.df[
            (self.df['latency_p95_ms'] <= constraints['max_latency_ms']) &
            (self.df['params_millions'] <= constraints['max_params_m']) &
            (self.df['memory_mb'] <= constraints['max_memory_mb']) &
            (self.df['dr_5s'] >= constraints['min_dr_5s'])
        ].copy()
        
        if len(candidates) == 0:
            return []
        
        # 按综合得分排序
        candidates = candidates.sort_values('composite_score', ascending=False)
        
        recommendations = []
        for _, row in candidates.iterrows():
            rec = {
                'model': row['model'],
                'composite_score': row['composite_score'],
                'dr_5s': row['dr_5s'],
                'latency_p95_ms': row['latency_p95_ms'],
                'params_millions': row['params_millions'],
                'reason': self._generate_reason(row, scenario_name)
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_reason(self, row: pd.Series, scenario: str) -> str:
        """生成推荐理由"""
        reasons = []
        
        if row['dr_5s'] >= 0.98:
            reasons.append("优秀的检测率")
        if row['latency_p95_ms'] < 50:
            reasons.append("极低延迟")
        if row['params_millions'] < 1.0:
            reasons.append("轻量级模型")
        
        return "; ".join(reasons) if reasons else "满足基本要求"
    
    def generate_full_report(self) -> str:
        """生成完整的部署建议报告"""
        report = []
        report.append("="*80)
        report.append(" " * 25 + "DEPLOYMENT RECOMMENDATIONS")
        report.append("="*80)
        
        for scenario_name, scenario in config.DEPLOYMENT_SCENARIOS.items():
            report.append(f"\n{'='*80}")
            report.append(f"Scenario: {scenario['name']}")
            report.append(f"{'='*80}")
            
            constraints = scenario['constraints']
            report.append("\nConstraints:")
            report.append(f"  • Max Latency: {constraints['max_latency_ms']}ms")
            report.append(f"  • Max Parameters: {constraints['max_params_m']}M")
            report.append(f"  • Max Memory: {constraints['max_memory_mb']}MB")
            report.append(f"  • Min DR@5s: {constraints['min_dr_5s']*100:.1f}%")
            
            recommendations = self.recommend_for_scenario(scenario_name)
            
            if recommendations:
                report.append(f"\n✅ Recommended Models ({len(recommendations)} found):")
                for i, rec in enumerate(recommendations, 1):
                    report.append(f"\n  {i}. {rec['model'].upper()}")
                    report.append(f"     Score: {rec['composite_score']:.3f}")
                    report.append(f"     DR@5s: {rec['dr_5s']*100:.1f}%")
                    report.append(f"     Latency: {rec['latency_p95_ms']:.2f}ms")
                    report.append(f"     Params: {rec['params_millions']:.2f}M")
                    report.append(f"     Reason: {rec['reason']}")
            else:
                report.append("\n❌ No models meet all constraints for this scenario")
                report.append("   Consider relaxing constraints or model optimization")
        
        return "\n".join(report)
