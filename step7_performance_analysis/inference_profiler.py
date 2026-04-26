"""
推理性能测试模块
功能：
1. 测量单窗口推理延迟（CPU/GPU）
2. 测量批量推理吞吐量
3. 计算延迟分布（P50/P90/P95/P99）
4. 评估实时性能力
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import sys
import os
sys.path.append('../step3b_multiModelGeneral')

from model import create_model
import config


class InferenceProfiler:
    """推理性能分析器"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化
        Args:
            model_path: 模型文件路径
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str):
        """加载模型"""
        # 从路径提取模型类型
        # Handle best_model_cnn_lstm.pth -> cnn_lstm
        import os
        filename = os.path.basename(model_path)  # best_model_cnn_lstm.pth
        model_type = filename.replace('best_model_', '').replace('.pth', '')  # cnn_lstm
        
        model = create_model(
            model_type=model_type,
            n_features=config.N_FEATURES,
            window_size=config.WINDOW_SIZE
        )
        model = model.to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle both formats: direct state_dict or wrapped in a dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def warmup(self, n_iterations: int = None):
        """预热（避免首次推理的开销）"""
        if n_iterations is None:
            n_iterations = config.WARMUP_ITERATIONS
            
        dummy_input = torch.randn(
            1, config.WINDOW_SIZE, config.N_FEATURES
        ).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = self.model(dummy_input)
                
        # GPU同步
        if self.device == 'cuda':
            torch.cuda.synchronize()
    
    def measure_single_window_latency(
        self, n_iterations: int = None
    ) -> Dict[str, float]:
        """
        测量单窗口推理延迟
        Returns:
            {
                'mean_ms': 平均延迟,
                'std_ms': 标准差,
                'min_ms': 最小延迟,
                'max_ms': 最大延迟,
                'p50_ms': 50th百分位,
                'p90_ms': 90th百分位,
                'p95_ms': 95th百分位,
                'p99_ms': 99th百分位
            }
        """
        if n_iterations is None:
            n_iterations = config.BENCHMARK_ITERATIONS
            
        latencies = []
        
        dummy_input = torch.randn(
            1, config.WINDOW_SIZE, config.N_FEATURES
        ).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_iterations):
                # 开始计时
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                # 推理
                _ = self.model(dummy_input)
                
                # 结束计时
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # 转换为ms
        
        latencies = np.array(latencies)
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p90_ms': float(np.percentile(latencies, 90)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99))
        }
    
    def measure_batch_throughput(
        self, batch_size: int, n_batches: int = 100
    ) -> Dict[str, float]:
        """
        测量批量推理吞吐量
        Args:
            batch_size: 批大小
            n_batches: 测试批次数
        Returns:
            {
                'throughput_samples_per_sec': 吞吐量（样本/秒）,
                'avg_batch_latency_ms': 平均批延迟（ms）
            }
        """
        dummy_input = torch.randn(
            batch_size, config.WINDOW_SIZE, config.N_FEATURES
        ).to(self.device)
        
        batch_times = []
        
        with torch.no_grad():
            for _ in range(n_batches):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                
                _ = self.model(dummy_input)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                batch_times.append(end - start)
        
        avg_batch_time = np.mean(batch_times)
        throughput = batch_size / avg_batch_time
        
        return {
            'throughput_samples_per_sec': float(throughput),
            'avg_batch_latency_ms': float(avg_batch_time * 1000)
        }
    
    def evaluate_real_time_capability(self) -> Dict[str, bool]:
        """
        评估实时性能力
        Returns:
            {
                'meets_real_time': 是否满足实时要求（<100ms）,
                'meets_near_real_time': 是否满足准实时（<500ms）,
                'suitable_for_offline': 是否适合离线分析
            }
        """
        latency = self.measure_single_window_latency()
        
        return {
            'meets_real_time': latency['p95_ms'] < config.LATENCY_THRESHOLDS['real_time'],
            'meets_near_real_time': latency['p95_ms'] < config.LATENCY_THRESHOLDS['near_real_time'],
            'suitable_for_offline': latency['p95_ms'] < config.LATENCY_THRESHOLDS['offline']
        }


def profile_all_models(test_data: np.ndarray = None) -> Dict[str, Dict]:
    """
    测试所有模型的推理性能
    Args:
        test_data: 测试数据（可选，用于真实数据测试）
    Returns:
        {
            'model_name': {
                'cpu': {...},
                'cuda': {...}
            }
        }
    """
    results = {}
    
    for model_name in config.MODEL_NAMES:
        print(f"\n{'='*60}")
        print(f"Profiling: {model_name.upper()}")
        print('='*60)
        
        model_path = os.path.join(
            config.STEP5_OUTPUT, f'best_model_{model_name}.pth'
        )
        
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
        
        results[model_name] = {}
        
        # 测试不同设备
        for device in config.DEVICES:
            if device == 'cuda' and not torch.cuda.is_available():
                print(f"  ⚠️  CUDA not available, skipping GPU test")
                continue
                
            print(f"\n  Device: {device.upper()}")
            
            profiler = InferenceProfiler(model_path, device=device)
            
            # 预热
            print("  Warming up...")
            profiler.warmup()
            
            # 单窗口延迟
            print("  Measuring single-window latency...")
            latency = profiler.measure_single_window_latency()
            
            # 批量吞吐量（测试多个batch size）
            print("  Measuring batch throughput...")
            throughput_results = {}
            for bs in config.BATCH_SIZES:
                tp = profiler.measure_batch_throughput(bs)
                throughput_results[f'batch_{bs}'] = tp
            
            # 实时性能力
            rt_capability = profiler.evaluate_real_time_capability()
            
            results[model_name][device] = {
                'latency': latency,
                'throughput': throughput_results,
                'real_time_capability': rt_capability
            }
            
            print(f"    ✓ Mean latency: {latency['mean_ms']:.2f}ms")
            print(f"    ✓ P95 latency: {latency['p95_ms']:.2f}ms")
            print(f"    ✓ Real-time capable: {rt_capability['meets_real_time']}")
    
    return results
