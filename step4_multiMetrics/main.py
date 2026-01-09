"""
主入口脚本 - 电力系统无人机巡检场景指标分析
Main Entry Point for Power Grid UAV Inspection Metrics Analysis

一键运行所有分析流程
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import power_grid_metrics
import power_grid_analysis
import risk_analysis


def main():
    """主函数 - 运行完整分析流程"""
    
    print("="*80)
    print("电力系统无人机巡检场景 - 完整分析流程")
    print("Power Grid UAV Inspection - Complete Analysis Pipeline")
    print("="*80)
    print()
    
    # 步骤1: 计算所有指标
    print("📊 步骤 1/3: 计算指标体系...")
    print("-"*80)
    try:
        power_grid_metrics.main()
        print("\n✅ 步骤1完成: 指标计算成功")
    except Exception as e:
        print(f"\n❌ 步骤1失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    
    # 步骤2: 生成分析和可视化
    print("📈 步骤 2/3: 生成分析和可视化...")
    print("-"*80)
    try:
        power_grid_analysis.main()
        print("\n✅ 步骤2完成: 分析和可视化成功")
    except Exception as e:
        print(f"\n❌ 步骤2失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    
    # 步骤3: 风险导向评估 (新增)
    print("🎯 步骤 3/3: 风险导向评估...")
    print("-"*80)
    try:
        risk_analysis.main()
        print("\n✅ 步骤3完成: 风险分析成功")
    except Exception as e:
        print(f"\n❌ 步骤3失败: {e}")
        import traceback
        traceback.print_exc()
        # 风险分析失败不中断整体流程
    
    # 完成
    print("\n" + "="*80)
    print("🎉 分析完成!")
    print("="*80)
    print()
    print("📁 查看结果:")
    print("  - 指标数据: ./output/power_grid_analysis/metrics*.csv")
    print("  - 综合排名: ./output/power_grid_analysis/comprehensive_rankings.csv")
    print("  - 区分度分析: ./output/power_grid_analysis/discrimination_analysis.csv")
    print("  - 分析报告: ./output/power_grid_analysis/POWER_GRID_REPORT.md")
    print("  - 风险评估: ./output/power_grid_analysis/RISK_ANALYSIS.md")
    print("  - 可视化图表: ./output/power_grid_analysis/visualizations/")
    print()
    print("💡 推荐查看:")
    print("  1. RISK_ANALYSIS.md - 风险导向评估报告 (新增)")
    print("  2. POWER_GRID_REPORT.md - 综合分析报告")
    print("  3. metrics6_advanced_risk.csv - 风险导向指标 (RWMR, MSDR, DS-FNR, CTDR)")
    print("  4. visualizations/risk_analysis/ - 风险分析可视化图表")
    print()


if __name__ == '__main__':
    main()
