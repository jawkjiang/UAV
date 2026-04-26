"""
补充数据提取脚本
为额外的可视化图表提取所需数据
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

OUTPUT_DIR = "paper_figures/supplementary_data"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print(" "*15 + "SUPPLEMENTARY DATA EXTRACTION")
print("="*70)

# ===========================================================================
# 1. 提取攻击轨迹示例（用于Fig 6）
# ===========================================================================
def extract_attack_trajectories():
    """提取6种攻击类型的代表性轨迹"""
    print("\n📍 Extracting attack trajectories for Fig 6...")
    
    test_df = pd.read_csv('step5_timegan/output/test_with_attacks.csv')
    attack_info = pd.read_csv('step5_timegan/output/test_attack_info.csv')
    
    trajectories = {}
    
    for attack_type in ['step', 'drift_ramp', 'drift_sigmoid', 'delay', 'takeover_step', 'takeover_ramp']:
        # 找到该攻击类型的第一个flight
        attack_flights = attack_info[attack_info['attack_type'] == attack_type]
        if len(attack_flights) == 0:
            print(f"   ⚠ No flights found for {attack_type}")
            continue
        
        flight_id = attack_flights.iloc[0]['flight']
        attack_start = attack_flights.iloc[0]['attack_start_time']
        
        # 提取轨迹数据
        flight_data = test_df[test_df['flight'] == flight_id].copy()
        
        # 计算相对时间（从起飞开始）
        flight_data['time_relative'] = flight_data['time'] - flight_data['time'].min()
        
        # 估算攻击结束时间（从数据中推断，攻击通常持续到flight结束）
        attack_end = flight_data['time'].max()  # 简化处理
        
        # 保存轨迹
        trajectory = flight_data[['time_relative', 'position_x', 'position_y', 'position_z',
                                  'velocity_x', 'velocity_y', 'velocity_z']].copy()
        trajectory['is_attack'] = (flight_data['time'] >= attack_start)
        trajectory['attack_type'] = attack_type
        
        trajectories[attack_type] = {
            'flight_id': flight_id,
            'attack_start_time': attack_start,
            'attack_end_time': attack_end,
            'data': trajectory
        }
        
        # 保存CSV
        trajectory.to_csv(f'{OUTPUT_DIR}/trajectory_{attack_type}.csv', index=False)
        print(f"   ✓ Saved trajectory for {attack_type} (flight {flight_id})")
    
    # 保存元信息
    meta_info = pd.DataFrame([
        {
            'attack_type': k,
            'flight_id': v['flight_id'],
            'attack_start': v['attack_start_time'],
            'attack_end': v['attack_end_time'],
            'duration': v['attack_end_time'] - v['attack_start_time']
        }
        for k, v in trajectories.items()
    ])
    meta_info.to_csv(f'{OUTPUT_DIR}/attack_trajectories_metadata.csv', index=False)
    print(f"   ✓ Saved metadata")
    
    return trajectories


# ===========================================================================
# 2. 提取详细延迟分布（用于Fig 11箱线图）
# ===========================================================================
def extract_delay_distribution():
    """提取每个攻击的检测延迟（用于箱线图）"""
    print("\n⏱️ Extracting delay distribution for Fig 11...")
    
    try:
        detailed_delays = pd.read_csv('step6_newMetrcisWithTimegan/output/time_aware_metrics/detailed_delays.csv')
        print(f"   ✓ Found detailed_delays.csv with {len(detailed_delays)} records")
        
        # 按模型分组统计
        delay_stats = []
        for model in detailed_delays['model'].unique():
            model_delays = detailed_delays[detailed_delays['model'] == model]['delay']
            
            delay_stats.append({
                'model': model,
                'count': len(model_delays),
                'mean': model_delays.mean(),
                'median': model_delays.median(),
                'std': model_delays.std(),
                'min': model_delays.min(),
                'q25': model_delays.quantile(0.25),
                'q75': model_delays.quantile(0.75),
                'max': model_delays.max()
            })
        
        delay_stats_df = pd.DataFrame(delay_stats)
        delay_stats_df.to_csv(f'{OUTPUT_DIR}/delay_distribution_stats.csv', index=False)
        
        # 保存原始数据的副本
        detailed_delays.to_csv(f'{OUTPUT_DIR}/detailed_delays_all_models.csv', index=False)
        
        print(f"   ✓ Saved delay distribution statistics")
        return detailed_delays
        
    except FileNotFoundError:
        print("   ⚠ detailed_delays.csv not found")
        print("   ℹ️ You need to run step6 evaluation to generate this file")
        return None


# ===========================================================================
# 3. 选择一个代表性案例用于时序可视化（Fig 1 & Fig 12）
# ===========================================================================
def select_representative_case():
    """选择一个典型的攻击案例用于时序可视化"""
    print("\n🎯 Selecting representative case for Fig 1 & Fig 12...")
    
    test_df = pd.read_csv('step5_timegan/output/test_with_attacks.csv')
    attack_info = pd.read_csv('step5_timegan/output/test_attack_info.csv')
    
    # 选择一个中等难度的攻击（DR@5s在0.9-0.95之间）
    # 优先选择delay或drift攻击（更有代表性）
    
    candidate_types = ['delay', 'drift_ramp', 'step']
    selected_case = None
    
    for attack_type in candidate_types:
        attacks = attack_info[attack_info['attack_type'] == attack_type]
        if len(attacks) > 0:
            # 选择第一个
            case = attacks.iloc[0]
            flight_id = case['flight']
            
            # 提取完整flight数据
            flight_data = test_df[test_df['flight'] == flight_id].copy()
            flight_data['time_relative'] = flight_data['time'] - flight_data['time'].min()
            
            # 估算攻击结束时间
            attack_end = flight_data['time'].max()
            
            selected_case = {
                'flight_id': flight_id,
                'attack_type': attack_type,
                'attack_start_time': case['attack_start_time'],
                'attack_end_time': attack_end,
                'data': flight_data
            }
            
            # 保存数据
            flight_data.to_csv(f'{OUTPUT_DIR}/representative_case_flight_{flight_id}.csv', index=False)
            
            # 保存元信息
            with open(f'{OUTPUT_DIR}/representative_case_metadata.json', 'w') as f:
                json.dump({
                    'flight_id': int(flight_id),
                    'attack_type': attack_type,
                    'attack_start_time': float(case['attack_start_time']),
                    'attack_end_time': float(attack_end),
                    'duration': float(attack_end - case['attack_start_time']),
                    'total_points': len(flight_data)
                }, f, indent=2)
            
            print(f"   ✓ Selected flight {flight_id} with {attack_type} attack")
            print(f"   ✓ Attack duration: {attack_end - case['attack_start_time']:.2f}s")
            print(f"   ✓ Total data points: {len(flight_data)}")
            break
    
    if selected_case:
        print(f"\n   ℹ️ To visualize this case with model predictions:")
        print(f"      1. Load step5 models")
        print(f"      2. Run predictions on this flight")
        print(f"      3. Save prediction probabilities")
        print(f"      → I can create a script for this if needed")
    
    return selected_case


# ===========================================================================
# 4. 统计测试集中的攻击分布
# ===========================================================================
def analyze_attack_distribution():
    """分析测试集中的攻击类型分布（用于Table 2补充）"""
    print("\n📊 Analyzing attack distribution...")
    
    attack_info = pd.read_csv('step5_timegan/output/test_attack_info.csv')
    
    # 按攻击类型统计
    distribution = attack_info['attack_type'].value_counts().reset_index()
    distribution.columns = ['attack_type', 'count']
    distribution['percentage'] = (distribution['count'] / len(attack_info) * 100).round(1)
    
    distribution.to_csv(f'{OUTPUT_DIR}/test_attack_distribution.csv', index=False)
    print(f"   ✓ Saved attack distribution")
    print(f"\n{distribution.to_string(index=False)}")
    
    return distribution


# ===========================================================================
# 5. 检查是否有baseline实验数据（用于Fig 15消融实验）
# ===========================================================================
def check_baseline_data():
    """检查是否存在baseline（无TimeGAN）实验数据"""
    print("\n🔍 Checking for baseline experiment data...")
    
    # 可能的baseline路径
    baseline_paths = [
        'step3b_multiModelGeneral/output',
        'step1_benchmark/output',
        'baseline_no_timegan/output'
    ]
    
    found_baseline = False
    for path in baseline_paths:
        if Path(path).exists():
            print(f"   ✓ Found potential baseline data: {path}")
            
            # 尝试读取指标
            for model in ['lstm', 'gru', 'cnn_lstm']:
                metrics_file = f'{path}/test_metrics_{model}.json'
                if Path(metrics_file).exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        print(f"      • {model}: F1={metrics.get('f1', 'N/A'):.3f}")
                        found_baseline = True
    
    if not found_baseline:
        print("   ⚠ No baseline data found")
        print("   ℹ️ To create Fig 15 (TimeGAN ablation), you would need to:")
        print("      1. Train models without TimeGAN data augmentation")
        print("      2. Compare with current results")
        print("      → Skip this figure for now or use step3b as baseline")
    
    return found_baseline


# ===========================================================================
# 主函数
# ===========================================================================
def main():
    """执行所有数据提取"""
    
    # 1. 攻击轨迹
    trajectories = extract_attack_trajectories()
    
    # 2. 延迟分布
    delays = extract_delay_distribution()
    
    # 3. 代表性案例
    case = select_representative_case()
    
    # 4. 攻击分布
    distribution = analyze_attack_distribution()
    
    # 5. Baseline检查
    baseline = check_baseline_data()
    
    print("\n" + "="*70)
    print("✅ Data extraction completed!")
    print("="*70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}/")
    print("\n📋 Summary:")
    print(f"   • Attack trajectories: {len(trajectories)} types")
    print(f"   • Delay distribution: {'Available' if delays is not None else 'Not found'}")
    print(f"   • Representative case: {'Selected' if case else 'Not found'}")
    print(f"   • Attack distribution: {len(distribution)} types")
    print(f"   • Baseline data: {'Available' if baseline else 'Not available'}")
    
    print("\n💡 Next steps:")
    print("   1. Check extracted data in paper_figures/supplementary_data/")
    print("   2. Use these data to create additional figures (Fig 1, 6, 11, 12)")
    print("   3. I can create visualization scripts for these if needed")
    print()


if __name__ == "__main__":
    main()
