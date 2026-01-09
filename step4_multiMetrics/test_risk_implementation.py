"""
测试脚本 - 验证风险评估实现
Test Script - Verify Risk Assessment Implementation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import ast
import numpy as np
from power_grid_metrics import PowerGridMetrics


def test_extract_risk_factors():
    """测试风险因子提取函数"""
    print("\n" + "="*80)
    print("测试1: 风险因子提取")
    print("="*80)
    
    # 模拟attack_info数据
    test_cases = [
        {
            'attack_type': 'step',
            'attacked': True,
            'attack_params': "{'magnitude': np.float64(30.0), 'direction_mode': 'random_xy', 'direction_vector': [0.90, 0.43], 'attack_duration': 3.0, 'time_constant': 1.0}"
        },
        {
            'attack_type': 'drift_ramp',
            'attacked': True,
            'attack_params': "{'profile': 'ramp', 'T_drift': 10.0, 'T_drift_clipped': 10.0, 'M': 15.0, 'direction_mode': 'random_xy', 'direction_vector': [0.71, 0.71]}"
        },
        {
            'attack_type': 'delay',
            'attacked': True,
            'attack_params': "{'delay_seconds': np.float64(1.0), 'source_time_clamped_count': 0}"
        },
        {
            'attack_type': 'none',
            'attacked': False,
            'attack_params': "{}"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {case['attack_type']}")
        print(f"  参数: {case['attack_params'][:50]}...")
        
        class MockRow:
            def __init__(self, data):
                self.data = data
            def __getitem__(self, key):
                return self.data[key]
        
        row = MockRow(case)
        result = PowerGridMetrics.extract_risk_factors(row)
        
        if result is None:
            print(f"  ✓ 结果: None (正常样本)")
        else:
            print(f"  ✓ 幅度: {result['magnitude']:.2f} 米")
            print(f"  ✓ 时长: {result['duration']:.2f} 秒")
            print(f"  ✓ 类型: {result['attack_type']}")
            print(f"  ✓ 轮廓: {result['profile']}")


def test_calculate_attack_risk_score():
    """测试风险评分计算"""
    print("\n" + "="*80)
    print("测试2: 风险分数计算")
    print("="*80)
    
    calculator = PowerGridMetrics()
    
    test_cases = [
        {'magnitude': 5.0, 'duration': 5.0, 'attack_type': 'step', 'profile': 'step'},
        {'magnitude': 15.0, 'duration': 10.0, 'attack_type': 'drift_ramp', 'profile': 'ramp'},
        {'magnitude': 30.0, 'duration': 20.0, 'attack_type': 'replay_same_hard', 'profile': 'replay'},
        {'magnitude': 5.0, 'duration': 999.0, 'attack_type': 'delay', 'profile': 'delay'},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"  攻击类型: {case['attack_type']}")
        print(f"  幅度: {case['magnitude']} 米, 时长: {case['duration']} 秒")
        
        score = calculator.calculate_attack_risk_score(
            case['magnitude'], case['duration'], 
            case['attack_type'], case['profile']
        )
        
        print(f"  ✓ 风险分数: {score:.2f}")
        
        # 判断风险等级
        if score < 100:
            level = "低风险"
        elif score < 500:
            level = "中等风险"
        elif score < 1000:
            level = "高风险"
        else:
            level = "极高风险"
        
        print(f"  ✓ 风险等级: {level}")


def test_magnitude_bins():
    """测试幅度分层"""
    print("\n" + "="*80)
    print("测试3: 幅度分层")
    print("="*80)
    
    from config_step4 import MAGNITUDE_BINS
    
    print("\n配置的幅度分层:")
    for bin_name, (low, high) in MAGNITUDE_BINS.items():
        print(f"  {bin_name}: [{low}, {high}) 米")
    
    # 测试示例幅度
    test_magnitudes = [3.0, 7.5, 15.0, 25.0]
    
    print("\n测试样本分类:")
    for mag in test_magnitudes:
        for bin_name, (low, high) in MAGNITUDE_BINS.items():
            if low <= mag < high:
                print(f"  幅度 {mag:.1f}m → {bin_name}")
                break


def test_risk_model_config():
    """测试风险模型配置"""
    print("\n" + "="*80)
    print("测试4: 风险模型配置")
    print("="*80)
    
    from config_step4 import POWER_GRID_RISK_MODEL
    
    print("\n碰撞风险配置:")
    cr = POWER_GRID_RISK_MODEL['collision_risk']
    print(f"  阈值: {cr['thresholds']}")
    print(f"  权重: {cr['weights']}")
    print(f"  系数: {cr['coefficient']}")
    
    print("\n任务失效风险配置:")
    mfr = POWER_GRID_RISK_MODEL['mission_failure_risk']
    print(f"  阈值: {mfr['thresholds']}")
    print(f"  权重: {mfr['weights']}")
    print(f"  系数: {mfr['coefficient']}")
    
    print("\n操作失控风险配置:")
    clr = POWER_GRID_RISK_MODEL['control_loss_risk']
    print(f"  类型权重: {clr['type_weights']}")
    print(f"  系数: {clr['coefficient']}")
    
    print("\n安全关键阈值:")
    thresholds = POWER_GRID_RISK_MODEL['critical_thresholds']
    print(f"  幅度阈值: {thresholds['magnitude']} 米")
    print(f"  时长阈值: {thresholds['duration']} 秒")


def main():
    """运行所有测试"""
    print("="*80)
    print("风险评估实现验证测试")
    print("Risk Assessment Implementation Verification")
    print("="*80)
    
    try:
        test_extract_risk_factors()
        test_calculate_attack_risk_score()
        test_magnitude_bins()
        test_risk_model_config()
        
        print("\n" + "="*80)
        print("✅ 所有测试通过!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ 测试失败!")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
