"""
模型复杂度分析模块
功能：
1. 统计参数量
2. 计算FLOPs
3. 测量模型文件大小
4. 估算内存占用
"""

import os
import torch
import numpy as np
from typing import Dict
import sys
sys.path.append('../step3b_multiModelGeneral')

from model import create_model
import config


class ModelAnalyzer:
    """模型复杂度分析器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # 从路径提取模型类型
        # Handle best_model_cnn_lstm.pth -> cnn_lstm
        import os
        filename = os.path.basename(model_path)  # best_model_cnn_lstm.pth
        model_type = filename.replace('best_model_', '').replace('.pth', '')  # cnn_lstm
        
        self.model = create_model(
            model_type=model_type,
            n_features=config.N_FEATURES,
            window_size=config.WINDOW_SIZE
        )
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle both formats: direct state_dict or wrapped in a dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def count_parameters(self) -> Dict[str, int]:
        """
        统计参数量
        Returns:
            {
                'total': 总参数,
                'trainable': 可训练参数,
                'non_trainable': 不可训练参数
            }
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def get_model_size(self) -> float:
        """
        获取模型文件大小（MB）
        """
        size_bytes = os.path.getsize(self.model_path)
        return size_bytes / (1024 * 1024)  # 转换为MB
    
    def estimate_flops(self) -> int:
        """
        估算FLOPs（粗略估计）
        使用thop库或手动计算
        """
        try:
            from thop import profile
            
            dummy_input = torch.randn(
                1, config.WINDOW_SIZE, config.N_FEATURES
            )
            
            flops, params = profile(
                self.model, inputs=(dummy_input,), verbose=False
            )
            
            return int(flops)
        except ImportError:
            # 如果没有thop，返回粗略估计
            params = self.count_parameters()['total']
            # 假设每个参数涉及1次乘法和1次加法
            return params * config.WINDOW_SIZE * 2
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        估算内存占用（MB）
        Returns:
            {
                'model_memory': 模型参数内存,
                'activation_memory': 激活值内存（单样本）,
                'total_per_sample': 每样本总内存
            }
        """
        # 模型参数内存
        param_memory = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)
        
        # 激活值内存（粗略估计）
        # 假设中间激活值约为输入的10倍
        input_size = config.WINDOW_SIZE * config.N_FEATURES * 4  # float32
        activation_memory = input_size * 10 / (1024 * 1024)
        
        return {
            'model_memory_mb': float(param_memory),
            'activation_memory_mb': float(activation_memory),
            'total_per_sample_mb': float(param_memory + activation_memory)
        }
    
    def get_model_summary(self) -> str:
        """生成模型摘要字符串"""
        params = self.count_parameters()
        size = self.get_model_size()
        memory = self.estimate_memory_usage()
        flops = self.estimate_flops()
        
        summary = f"""
Model Summary:
  Parameters: {params['total']:,} ({params['total']/1e6:.2f}M)
  Model Size: {size:.2f} MB
  FLOPs: {flops:,} ({flops/1e6:.2f}M)
  Memory (inference): {memory['total_per_sample_mb']:.2f} MB/sample
"""
        return summary


def analyze_all_models() -> Dict[str, Dict]:
    """
    分析所有模型的复杂度
    Returns:
        {
            'model_name': {
                'parameters': {...},
                'model_size_mb': ...,
                'flops': ...,
                'memory': {...}
            }
        }
    """
    results = {}
    
    for model_name in config.MODEL_NAMES:
        print(f"\nAnalyzing: {model_name.upper()}")
        
        model_path = os.path.join(
            config.STEP5_OUTPUT, f'best_model_{model_name}.pth'
        )
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
        
        analyzer = ModelAnalyzer(model_path)
        
        results[model_name] = {
            'parameters': analyzer.count_parameters(),
            'model_size_mb': analyzer.get_model_size(),
            'flops': analyzer.estimate_flops(),
            'memory': analyzer.estimate_memory_usage()
        }
        
        print(analyzer.get_model_summary())
    
    return results
