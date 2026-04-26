"""
Step7主流程
"""

import pandas as pd
import os
import config
import inference_profiler
import model_analyzer
import comprehensive_evaluation
import deployment_advisor
import visualizations


def main():
    """主流程"""
    print("\n" + "="*80)
    print(" " * 20 + "STEP7: Performance Analysis")
    print("="*80)
    
    config.ensure_output_dirs()
    
    # ========================================================================
    # Step 1: 推理性能测试
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 1: Inference Performance Profiling")
    print("-"*80)
    
    inference_results = inference_profiler.profile_all_models()
    
    # 保存推理性能结果
    perf_rows = []
    for model, devices in inference_results.items():
        for device, metrics in devices.items():
            perf_rows.append({
                'model': model,
                'device': device,
                'mean_latency_ms': metrics['latency']['mean_ms'],
                'p95_latency_ms': metrics['latency']['p95_ms'],
                'p99_latency_ms': metrics['latency']['p99_ms'],
                'throughput_windows_per_sec': metrics['throughput']['batch_1']['throughput_samples_per_sec'],
                'meets_real_time': metrics['real_time_capability']['meets_real_time']
            })
    
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(
        os.path.join(config.OUTPUT_DIR, 'performance_metrics.csv'),
        index=False
    )
    print(f"\n✓ Saved: performance_metrics.csv")
    
    # ========================================================================
    # Step 2: 模型复杂度分析
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 2: Model Complexity Analysis")
    print("-"*80)
    
    complexity_results = model_analyzer.analyze_all_models()
    
    # 保存复杂度结果
    complex_rows = []
    for model, metrics in complexity_results.items():
        complex_rows.append({
            'model': model,
            'total_params': metrics['parameters']['total'],
            'params_millions': metrics['parameters']['total'] / 1e6,
            'model_size_mb': metrics['model_size_mb'],
            'flops': metrics['flops'],
            'flops_millions': metrics['flops'] / 1e6,
            'memory_mb': metrics['memory']['total_per_sample_mb']
        })
    
    complex_df = pd.DataFrame(complex_rows)
    complex_df.to_csv(
        os.path.join(config.OUTPUT_DIR, 'model_complexity.csv'),
        index=False
    )
    print(f"\n✓ Saved: model_complexity.csv")
    
    # ========================================================================
    # Step 3: 综合评估
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 3: Comprehensive Evaluation")
    print("-"*80)
    
    step6_metrics = comprehensive_evaluation.load_step6_metrics()
    
    comprehensive_df = comprehensive_evaluation.integrate_all_metrics(
        step6_metrics,
        inference_results,
        complexity_results
    )
    
    comprehensive_df.to_csv(
        os.path.join(config.OUTPUT_DIR, 'comprehensive_report.csv'),
        index=False
    )
    print(f"\n✓ Saved: comprehensive_report.csv")
    
    # 打印排名
    print("\n" + "="*80)
    print("MODEL RANKINGS (by Composite Score)")
    print("="*80)
    ranked = comprehensive_df.sort_values('composite_score', ascending=False)
    for i, row in ranked.iterrows():
        print(f"\n{int(row['rank'])}. {row['model'].upper()}")
        print(f"   Composite Score: {row['composite_score']:.3f}")
        print(f"   DR@5s: {row['dr_5s']*100:.1f}%")
        print(f"   Latency (P95): {row['latency_p95_ms']:.2f}ms")
        print(f"   Parameters: {row['params_millions']:.2f}M")
    
    # ========================================================================
    # Step 4: 部署建议
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 4: Deployment Recommendations")
    print("-"*80)
    
    advisor = deployment_advisor.DeploymentAdvisor(comprehensive_df)
    report = advisor.generate_full_report()
    
    report_path = os.path.join(config.OUTPUT_DIR, 'deployment_recommendations.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n✓ Saved: deployment_recommendations.txt")
    
    # ========================================================================
    # Step 5: 可视化
    # ========================================================================
    print("\n" + "-"*80)
    print("STEP 5: Visualizations")
    print("-"*80)
    
    visualizations.generate_all_visualizations(comprehensive_df)
    
    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "="*80)
    print("✅ Step7 Analysis Complete!")
    print("="*80)
    
    print("\n📁 Generated Files:")
    print(f"  • {config.OUTPUT_DIR}/performance_metrics.csv")
    print(f"  • {config.OUTPUT_DIR}/model_complexity.csv")
    print(f"  • {config.OUTPUT_DIR}/comprehensive_report.csv")
    print(f"  • {config.OUTPUT_DIR}/deployment_recommendations.txt")
    print(f"  • {config.VISUALIZATION_DIR}/ (5 charts)")
    
    return 0


if __name__ == "__main__":
    exit(main())
