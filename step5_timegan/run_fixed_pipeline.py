"""
完整流程控制脚本
按顺序执行所有步骤，并验证结果
"""
import subprocess
import sys
from pathlib import Path
import time

def run_step(step_num, script_name, description):
    """运行单个步骤"""
    print("\n" + "=" * 80)
    print(f"步骤 {step_num}: {description}")
    print("=" * 80)
    print(f"执行: {script_name}")
    
    start_time = time.time()
    
    # 运行脚本
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ 步骤{step_num}失败！退出码: {result.returncode}")
        return False
    
    print(f"\n✅ 步骤{step_num}完成 (耗时: {elapsed:.1f}秒)")
    return True


def main():
    """执行完整的修复流程"""
    print("=" * 80)
    print(" " * 20 + "Step5 TimeGAN 数据泄漏修复流程")
    print("=" * 80)
    print("\n关键改进:")
    print("  ✓ 先划分数据，再训练TimeGAN")
    print("  ✓ 每个split独立的TimeGAN")
    print("  ✓ 测试集充分(53攻击 + 211正常)")
    print("  ✓ 彻底避免数据泄漏")
    
    # 检查output_backup是否存在
    if not Path('output_backup_20260109').exists():
        print("\n⚠️  警告: 未找到备份文件夹 output_backup_20260109")
        print("   当前output文件夹将被覆盖")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    total_start = time.time()
    
    # 步骤1：划分原始数据
    if not run_step(1, 'split_original_first.py', '划分原始数据'):
        return
    
    # 步骤2：训练独立TimeGANs
    if not run_step(2, 'train_independent_timegans.py', '训练独立TimeGAN模型'):
        return
    
    # 步骤3：生成合成数据
    if not run_step(3, 'generate_per_split.py', '生成合成数据'):
        return
    
    # 步骤4：注入攻击
    if not run_step(4, 'inject_attacks_stratified.py', '分层注入攻击'):
        return
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 80)
    print(" " * 25 + "流程执行完成")
    print("=" * 80)
    print(f"\n总耗时: {total_elapsed/60:.1f}分钟")
    
    # 运行验证
    print("\n" + "=" * 80)
    print("运行验证检查...")
    print("=" * 80)
    
    result = subprocess.run(
        [sys.executable, 'verify_fix.py'],
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print(" " * 25 + "✅ 所有检查通过！")
        print("=" * 80)
        print("\n数据泄漏已修复，可以开始训练模型：")
        print("  python main.py --all")
    else:
        print("\n⚠️  验证过程中发现问题，请检查输出")
    
    # 生成执行报告
    import json
    from datetime import datetime
    
    report = {
        'execution_date': datetime.now().isoformat(),
        'total_time_minutes': total_elapsed / 60,
        'steps_completed': 4,
        'verification_passed': result.returncode == 0,
        'output_files': {
            'train': 'output/train_with_attacks.csv',
            'val': 'output/val_with_attacks.csv',
            'test': 'output/test_with_attacks.csv'
        }
    }
    
    report_path = Path('output') / 'pipeline_execution_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n执行报告已保存: {report_path}")


if __name__ == "__main__":
    main()
