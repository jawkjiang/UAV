"""
重新生成heatmap with正确的攻击类型
"""
import sys
import json
import config
import visualizations

# 加载所有模型的指标
all_metrics = {}
models = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']

for model in models:
    json_path = f'output/time_aware_metrics/{model}_metrics.json'
    try:
        with open(json_path, 'r') as f:
            all_metrics[model] = json.load(f)
        print(f"Loaded {model}")
    except FileNotFoundError:
        print(f"Warning: {model}_metrics.json not found")
        continue

if all_metrics:
    print(f"\nRegenerating visualizations with correct attack types: {config.ATTACK_TYPES}")
    visualizations.plot_per_attack_heatmap(all_metrics, 'output/per_attack_heatmap_fixed.png')
    print("Done!")
else:
    print("No metrics found!")
