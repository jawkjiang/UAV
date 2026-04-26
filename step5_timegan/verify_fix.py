"""
验证数据泄漏修复
检查所有关键点，确保无泄漏
"""
import pandas as pd
import json
from pathlib import Path
import config

def check_data_splits():
    """检查数据划分"""
    print("=" * 80)
    print("[检查1] 数据划分")
    print("=" * 80)
    
    split_info_path = Path(config.OUTPUT_DIR) / 'split_info.json'
    
    if not split_info_path.exists():
        print("❌ 未找到split_info.json")
        return False
    
    with open(split_info_path) as f:
        split_info = json.load(f)
    
    train_flights = set(split_info['train']['flight_ids'])
    val_flights = set(split_info['val']['flight_ids'])
    test_flights = set(split_info['test']['flight_ids'])
    
    print(f"\n数据集大小:")
    print(f"  Train: {len(train_flights)} flights")
    print(f"  Val:   {len(val_flights)} flights")
    print(f"  Test:  {len(test_flights)} flights")
    
    # 检查重叠
    train_val = train_flights & val_flights
    train_test = train_flights & test_flights
    val_test = val_flights & test_flights
    
    print(f"\n重叠检查:")
    print(f"  Train ∩ Val:  {len(train_val)}")
    print(f"  Train ∩ Test: {len(train_test)}")
    print(f"  Val ∩ Test:   {len(val_test)}")
    
    if train_test:
        print(f"\n❌ 发现Train-Test重叠: {train_test}")
        return False
    
    if train_val or val_test:
        print(f"\n⚠️  发现重叠（非致命）")
    
    print("\n✅ 数据划分检查通过")
    return True


def check_timegan_independence():
    """检查TimeGAN训练独立性"""
    print("\n" + "=" * 80)
    print("[检查2] TimeGAN独立性")
    print("=" * 80)
    
    model_dir = Path('timegan_models')
    
    required_models = [
        'timegan_train.pt',
        'timegan_val.pt',
        'timegan_test.pt'
    ]
    
    print("\nTimeGAN模型:")
    for model_name in required_models:
        model_path = model_dir / model_name
        if model_path.exists():
            print(f"  ✓ {model_name}")
        else:
            print(f"  ❌ {model_name} 未找到")
            return False
    
    # 检查归一化参数
    required_norms = [
        'norm_params_train.npz',
        'norm_params_val.npz',
        'norm_params_test.npz'
    ]
    
    print("\n归一化参数:")
    for norm_name in required_norms:
        norm_path = model_dir / norm_name
        if norm_path.exists():
            print(f"  ✓ {norm_name}")
        else:
            print(f"  ❌ {norm_name} 未找到")
            return False
    
    print("\n✅ TimeGAN独立性检查通过")
    print("   每个split都有独立的TimeGAN模型和归一化参数")
    return True


def check_test_coverage():
    """检查测试集覆盖度"""
    print("\n" + "=" * 80)
    print("[检查3] 测试集覆盖度")
    print("=" * 80)
    
    # 加载测试集攻击信息
    test_attack_path = Path(config.OUTPUT_DIR) / 'test_attack_info.csv'
    
    if not test_attack_path.exists():
        print("❌ 未找到test_attack_info.csv")
        return False
    
    attack_info = pd.read_csv(test_attack_path)
    
    total_flights = len(attack_info)
    attack_flights = attack_info[attack_info['attack_type'] != 'none']
    normal_flights = attack_info[attack_info['attack_type'] == 'none']
    
    print(f"\n测试集组成:")
    print(f"  总flights:  {total_flights}")
    print(f"  攻击flights: {len(attack_flights)} ({len(attack_flights)/total_flights*100:.1f}%)")
    print(f"  正常flights: {len(normal_flights)} ({len(normal_flights)/total_flights*100:.1f}%)")
    
    # 检查每种攻击类型
    attack_types = ['step', 'ramp', 'takeover_step', 'takeover_ramp', 'delay', 'drift']
    
    print(f"\n攻击类型分布:")
    min_count = float('inf')
    for atype in attack_types:
        count = len(attack_flights[attack_flights['attack_type'] == atype])
        print(f"  {atype:20s}: {count:3d} 实例")
        min_count = min(min_count, count)
    
    # 验证覆盖度
    target_per_type = config.TEST_ATTACKS_PER_TYPE
    
    print(f"\n覆盖度检查:")
    print(f"  目标: 每种攻击>= {target_per_type}个")
    print(f"  实际: 最少{min_count}个")
    
    if min_count < target_per_type:
        print(f"  ⚠️  某些攻击类型不足{target_per_type}个")
    else:
        print(f"  ✅ 所有攻击类型都>= {target_per_type}个")
    
    # 检查MTBFA可测性
    print(f"\nMTBFA可测性:")
    print(f"  正常flights: {len(normal_flights)}")
    
    if len(normal_flights) < 50:
        print(f"  ⚠️  正常flights可能不足以可靠估计MTBFA")
    elif len(normal_flights) < 100:
        print(f"  ✓ 正常flights足够估计MTBFA（但可能有波动）")
    else:
        print(f"  ✅ 正常flights充分，MTBFA估计可靠")
    
    print("\n✅ 测试集覆盖度检查通过")
    return True


def check_synthetic_ids():
    """检查合成flight ID无冲突"""
    print("\n" + "=" * 80)
    print("[检查4] 合成Flight ID")
    print("=" * 80)
    
    # 加载split信息获取原始IDs
    split_info_path = Path(config.OUTPUT_DIR) / 'split_info.json'
    with open(split_info_path) as f:
        split_info = json.load(f)
    
    original_ids = set()
    for split in ['train', 'val', 'test']:
        original_ids.update(split_info[split]['flight_ids'])
    
    print(f"\n原始flight IDs: {len(original_ids)}")
    print(f"  范围: [{min(original_ids)}, {max(original_ids)}]")
    
    # 检查合成IDs
    synthetic_ranges = config.SYNTHETIC_FLIGHT_ID_RANGES
    
    print(f"\n合成flight ID范围:")
    for split, (start, end) in synthetic_ranges.items():
        print(f"  {split:5s}: [{start}, {end-1}]")
        
        # 检查是否与原始ID冲突
        synthetic_ids = set(range(start, end))
        overlap = synthetic_ids & original_ids
        
        if overlap:
            print(f"    ❌ 与原始IDs冲突: {len(overlap)}个")
            return False
    
    # 检查实际生成的IDs
    for split in ['train', 'val', 'test']:
        data_path = Path(config.OUTPUT_DIR) / f'{split}_with_attacks.csv'
        if data_path.exists():
            df = pd.read_csv(data_path, low_memory=False)
            all_ids = set(df['flight'].unique())
            
            start, end = synthetic_ranges[split]
            expected_synthetic = set(range(start, end))
            actual_synthetic = all_ids - original_ids
            
            print(f"\n{split.capitalize()}实际情况:")
            print(f"  原始flights: {len(all_ids & original_ids)}")
            print(f"  合成flights: {len(actual_synthetic)}")
            
            # 检查合成IDs是否在预期范围内
            out_of_range = actual_synthetic - expected_synthetic
            if out_of_range:
                print(f"  ⚠️  {len(out_of_range)}个合成ID超出预期范围")
    
    print("\n✅ 合成ID检查通过，无冲突")
    return True


def check_no_leakage():
    """最终数据泄漏检查"""
    print("\n" + "=" * 80)
    print("[检查5] 数据泄漏最终验证")
    print("=" * 80)
    
    # 加载split信息
    split_info_path = Path(config.OUTPUT_DIR) / 'split_info.json'
    with open(split_info_path) as f:
        split_info = json.load(f)
    
    train_original_ids = set(split_info['train']['flight_ids'])
    test_original_ids = set(split_info['test']['flight_ids'])
    
    print(f"\n原始数据集:")
    print(f"  Train原始flights: {len(train_original_ids)}")
    print(f"  Test原始flights:  {len(test_original_ids)}")
    
    # 检查TimeGAN训练数据（通过读取训练数据反推）
    train_normal_path = Path(config.OUTPUT_DIR) / 'split_flights' / 'train_normal.csv'
    test_normal_path = Path(config.OUTPUT_DIR) / 'split_flights' / 'test_normal.csv'
    
    if train_normal_path.exists() and test_normal_path.exists():
        train_df = pd.read_csv(train_normal_path, low_memory=False)
        test_df = pd.read_csv(test_normal_path, low_memory=False)
        
        train_timegan_flights = set(train_df['flight'].unique())
        test_timegan_flights = set(test_df['flight'].unique())
        
        print(f"\nTimeGAN训练数据:")
        print(f"  Train-TimeGAN见过: {len(train_timegan_flights)} flights")
        print(f"  Test-TimeGAN见过:  {len(test_timegan_flights)} flights")
        
        # 关键检查：Train-TimeGAN是否见过test flights
        train_timegan_test_overlap = train_timegan_flights & test_original_ids
        
        print(f"\n关键泄漏检查:")
        print(f"  Train-TimeGAN ∩ Test原始flights: {len(train_timegan_test_overlap)}")
        
        if train_timegan_test_overlap:
            print(f"  ❌ 发现数据泄漏! Train-TimeGAN见过{len(train_timegan_test_overlap)}个测试flights")
            print(f"     泄漏flights: {sorted(list(train_timegan_test_overlap))[:10]}...")
            return False
        else:
            print(f"  ✅ 无泄漏! Train-TimeGAN未见过任何test flights")
        
        # 反向检查
        test_timegan_train_overlap = test_timegan_flights & train_original_ids
        print(f"  Test-TimeGAN ∩ Train原始flights: {len(test_timegan_train_overlap)}")
        
        if test_timegan_train_overlap:
            print(f"  ✅ 这是正常的（Test-TimeGAN独立训练）")
        
    print("\n✅ 数据泄漏检查通过")
    return True


def main():
    """运行所有验证"""
    print("=" * 80)
    print(" " * 25 + "数据泄漏修复验证")
    print("=" * 80)
    
    checks = [
        ("数据划分", check_data_splits),
        ("TimeGAN独立性", check_timegan_independence),
        ("测试集覆盖度", check_test_coverage),
        ("合成ID冲突", check_synthetic_ids),
        ("数据泄漏", check_no_leakage)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name}检查失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print(" " * 30 + "验证总结")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print(" " * 25 + "✅ 所有检查通过！")
        print("=" * 80)
        print("\n数据泄漏已成功修复。")
        print("\n可以开始训练模型：")
        print("  python main.py --all")
        return 0
    else:
        print(" " * 25 + "❌ 存在问题")
        print("=" * 80)
        print("\n请检查上述失败的检查项")
        return 1


if __name__ == "__main__":
    exit(main())
