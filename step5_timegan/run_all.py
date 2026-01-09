"""
一键运行Step5完整流程
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示进度"""
    print("\n" + "=" * 80)
    print(f">>> {description}")
    print("=" * 80)
    print(f"命令: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ 错误: {description} 失败")
        return False
    
    print(f"\n✓ 完成: {description}")
    return True


def main():
    """完整流程"""
    print("=" * 80)
    print(" " * 20 + "Step5: TimeGAN增强 - 完整流程")
    print("=" * 80)
    
    steps = [
        ("python prepare_normal_data.py", "1. 提取正常飞行数据"),
        ("python train_timegan.py", "2. 训练TimeGAN模型"),
        ("python generate_synthetic_flights.py", "3. 生成合成飞行"),
        ("python merge_and_split.py", "4. 合并和划分数据集"),
        ("python inject_attacks.py", "5. 注入攻击"),
        ("python main.py --model lstm", "6. 训练LSTM模型"),
        ("python compare_with_baseline.py --model lstm", "7. 对比分析"),
    ]
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print("\n流程中断")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print(" " * 25 + "🎉 全部完成!")
    print("=" * 80)
    print("\n查看结果:")
    print("  - 模型: output/best_model_lstm.pth")
    print("  - 指标: output/test_metrics_lstm.json")
    print("  - 对比: output/comparison_report.json")
    print("  - 图表: output/comparison_lstm.png")


if __name__ == "__main__":
    main()
