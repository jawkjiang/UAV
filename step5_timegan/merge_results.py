"""
合并所有已训练模型的结果，生成对比报告
无需重新训练
"""
import json
import numpy as np
from pathlib import Path

def convert_to_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

output_dir = Path('output')

# 所有模型类型
ALL_MODEL_TYPES = ['lstm', 'gru', 'bilstm', 'cnn', 'cnn_lstm', 'tcn', 'transformer']

all_results = {}

for model_type in ALL_MODEL_TYPES:
    metrics_file = output_dir / f'test_metrics_{model_type}.json'
    history_file = output_dir / f'training_history_{model_type}.json'
    
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        best_epoch = 0
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                best_epoch = len(history.get('train_loss', []))
        
        all_results[model_type] = {
            'metrics': metrics,
            'best_epoch': best_epoch
        }
        print(f"✓ 已加载 {model_type} 的结果")
    else:
        print(f"✗ 未找到 {model_type} 的结果文件")
        all_results[model_type] = {'error': 'metrics file not found'}

# 保存合并结果
serializable_results = convert_to_serializable(all_results)

with open(output_dir / 'all_models_comparison.json', 'w') as f:
    json.dump(serializable_results, f, indent=2)

print(f"\n✓ 已保存对比结果: {output_dir / 'all_models_comparison.json'}")

# 打印对比表
print("\n" + "=" * 80)
print("模型性能对比")
print("=" * 80)
print(f"{'模型':<15} {'PR-AUC':<10} {'ROC-AUC':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'训练轮数':<10}")
print("-" * 80)

for model_type, result in all_results.items():
    if 'error' not in result:
        m = result['metrics']
        epoch = result['best_epoch']
        print(f"{model_type:<15} "
              f"{m.get('pr_auc', 0):<10.4f} "
              f"{m.get('roc_auc', 0):<10.4f} "
              f"{m.get('f1', 0):<10.4f} "
              f"{m.get('precision', 0):<10.4f} "
              f"{m.get('recall', 0):<10.4f} "
              f"{epoch:<10}")
    else:
        print(f"{model_type:<15} ERROR: {result['error']}")

print("\n最佳模型（按PR-AUC）:")
valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
if valid_results:
    best_model = max(valid_results.items(), 
                     key=lambda x: x[1]['metrics'].get('pr_auc', 0))
    print(f"  {best_model[0].upper()}: PR-AUC = {best_model[1]['metrics']['pr_auc']:.4f}")
