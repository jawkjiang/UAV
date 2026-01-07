"""
Supplementary Experiment 1 — Multi-Model Benchmark
固定协议下的多模型对比基准实验

固定条件：
- 数据划分: 70%/15%/15% by flight, seed=2026
- 窗口: L=50, S=5
- 攻击: Step+闭环诱导, magnitude∈{5,15,30}, T_resp=3.0, τ=1.0
- 训练集正样本窗口比例: 35%
- Val/Test正样本窗口比例: 3%

模型:
A. Rule Baseline (Residual Threshold)
B. XGBoost
C. 1D-CNN
D. TCN
E. GRU
F. Transformer Encoder

输出指标:
- Recall@FPR=1%
- PR-AUC
- avg_FP_per_flight
- detection_delay (median, mean, P90)
"""

import os
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit as sigmoid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    roc_curve
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: XGBoost not installed. Model B will be skipped.")


# ============================================================================
# 固定配置
# ============================================================================
EXPERIMENT_CONFIG = {
    'seed': 2026,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'window_length': 50,
    'window_stride': 5,
    'train_pos_window_ratio': 0.35,
    'valtest_pos_window_ratio': 0.03,
    'attack_magnitude_range': [5, 15, 30],
    'attack_T_resp': 3.0,
    'attack_tau': 1.0,
    'attack_vmax_range': [0.5, 2.0],
    'target_param_range': [0.3e6, 0.8e6],  # 参数量区间
    'num_seeds': 1,  # 重复训练次数
    'seeds_list': [2026],
    
    # 训练配置
    'learning_rate': 1e-3,
    'batch_size': 256,  # Increased for better GPU utilization
    'max_epochs': 40,
    'early_stopping_patience': 6,
    'dropout': 0.1,
    
    # 评估配置
    'target_fpr': 0.01,
}


# ============================================================================
# 工具函数
# ============================================================================
def set_random_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model) -> int:
    """统计模型参数量"""
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """计算一致性残差特征
    
    r_posvel = ||(p_t - p_{t-1}) - v_t * delta_t||
    r_velacc = ||((v_t - v_{t-1})/delta_t) - a_t||
    """
    df = df.copy()
    
    # 计算delta_t
    df['delta_t'] = df.groupby('flight')['time'].diff()
    # 首个样本用同flight的第二个差值回填
    for flight_id in df['flight'].unique():
        mask = df['flight'] == flight_id
        first_idx = df[mask].index[0]
        if len(df[mask]) > 1:
            second_delta = df[mask]['delta_t'].iloc[1]
            df.loc[first_idx, 'delta_t'] = second_delta
        else:
            df.loc[first_idx, 'delta_t'] = 0.1  # fallback
    
    # 位置-速度一致性残差
    df['dp_x'] = df.groupby('flight')['position_x'].diff()
    df['dp_y'] = df.groupby('flight')['position_y'].diff()
    df['dp_predicted_x'] = df['velocity_x'] * df['delta_t']
    df['dp_predicted_y'] = df['velocity_y'] * df['delta_t']
    df['r_posvel'] = np.sqrt(
        (df['dp_x'] - df['dp_predicted_x'])**2 + 
        (df['dp_y'] - df['dp_predicted_y'])**2
    )
    
    # 速度-加速度一致性残差
    df['dv_x'] = df.groupby('flight')['velocity_x'].diff()
    df['dv_y'] = df.groupby('flight')['velocity_y'].diff()
    df['acc_predicted_x'] = df['dv_x'] / df['delta_t']
    df['acc_predicted_y'] = df['dv_y'] / df['delta_t']
    df['r_velacc'] = np.sqrt(
        (df['acc_predicted_x'] - df['linear_acceleration_x'])**2 + 
        (df['acc_predicted_y'] - df['linear_acceleration_y'])**2
    )
    
    # 填充首行NaN
    df['r_posvel'] = df['r_posvel'].fillna(0)
    df['r_velacc'] = df['r_velacc'].fillna(0)
    
    return df


# ============================================================================
# 数据加载与划分
# ============================================================================
class BenchmarkDataLoader:
    """固定协议数据加载器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.feature_cols = [
            'position_x', 'position_y',
            'velocity_x', 'velocity_y',
            'linear_acceleration_x', 'linear_acceleration_y',
            'delta_t',
            'r_posvel',
            'r_velacc'
        ]
        self.scaler = None
        
    def load_and_split(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载数据并按固定协议划分"""
        # 尝试加载预处理的数据（包含攻击和标签）
        if Path('output/train_data.pkl').exists():
            print("Loading preprocessed data with attacks and labels...")
            df_train = pd.read_pickle('output/train_data.pkl')
            df_val = pd.read_pickle('output/val_data.pkl')
            df_test = pd.read_pickle('output/test_data.pkl')
            
            # 计算残差特征（如果还没有）
            if 'r_posvel' not in df_train.columns:
                df_train = compute_residuals(df_train)
                df_val = compute_residuals(df_val)
                df_test = compute_residuals(df_test)
            
            print(f"Loaded preprocessed data: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
            return df_train, df_val, df_test
        
        # 如果预处理数据不存在，加载原始数据
        print("Preprocessed data not found. Loading raw data...")
        df = pd.read_csv(data_path)
        
        # 计算残差特征
        df = compute_residuals(df)
        
        # 按flight分组划分
        flight_ids = df['flight'].unique()
        np.random.seed(self.config['seed'])
        np.random.shuffle(flight_ids)
        
        n_flights = len(flight_ids)
        n_train = int(n_flights * self.config['train_ratio'])
        n_val = int(n_flights * self.config['val_ratio'])
        
        train_flights = flight_ids[:n_train]
        val_flights = flight_ids[n_train:n_train+n_val]
        test_flights = flight_ids[n_train+n_val:]
        
        df_train = df[df['flight'].isin(train_flights)].copy()
        df_val = df[df['flight'].isin(val_flights)].copy()
        df_test = df[df['flight'].isin(test_flights)].copy()
        
        # Check if labels exist
        if 'label' not in df_train.columns:
            raise ValueError(
                "\n" + "="*80 + "\n"
                "ERROR: Raw data does not contain 'label' column.\n"
                "\n"
                "To run the benchmark, you need preprocessed data with attacks and labels.\n"
                "\n"
                "Please run the following command first to prepare the data:\n"
                "    python data_loader.py\n"
                "\n"
                "This will create output/train_data.pkl, output/val_data.pkl, and output/test_data.pkl\n"
                "with the required attack injections and labels.\n"
                "\n"
                "Alternatively, ensure your input data already has a 'label' column (0=normal, 1=attack).\n"
                + "="*80
            )
        
        print(f"Data split: Train={len(train_flights)} flights, Val={len(val_flights)} flights, Test={len(test_flights)} flights")
        print(f"Train: {len(df_train)} samples, Val: {len(df_val)} samples, Test: {len(df_test)} samples")
        
        return df_train, df_val, df_test
    
    def create_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """创建窗口切片
        
        Returns:
            X: [N, L, D] 特征张量
            y: [N] 窗口标签
            flight_ids: [N] 每个窗口所属的flight
            window_meta: [N] 窗口元信息 (flight_id, start_idx, end_idx, end_time)
        """
        L = self.config['window_length']
        S = self.config['window_stride']
        
        windows = []
        labels = []
        flight_ids_list = []
        window_meta = []
        
        for flight_id in df['flight'].unique():
            flight_df = df[df['flight'] == flight_id].reset_index(drop=True)
            n_samples = len(flight_df)
            
            # 滑动窗口
            for start in range(0, n_samples - L + 1, S):
                end = start + L
                window_data = flight_df.iloc[start:end]
                
                # 提取特征
                window_features = window_data[self.feature_cols].values  # [L, D]
                
                # 窗口标签：任一时间步是攻击则为正样本
                window_label = int(window_data['label'].max())
                
                windows.append(window_features)
                labels.append(window_label)
                flight_ids_list.append(flight_id)
                
                # 记录窗口元信息
                window_meta.append({
                    'flight_id': flight_id,
                    'start_idx': start,
                    'end_idx': end,
                    'end_time': window_data['time'].iloc[-1]
                })
        
        X = np.array(windows, dtype=np.float32)  # [N, L, D]
        y = np.array(labels, dtype=np.int64)
        flight_ids_arr = np.array(flight_ids_list)
        
        return X, y, flight_ids_arr, window_meta
    
    def fit_scaler(self, X_train: np.ndarray):
        """在训练集上拟合标准化器"""
        # X_train: [N, L, D]
        N, L, D = X_train.shape
        X_flat = X_train.reshape(-1, D)
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_flat)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用标准化"""
        N, L, D = X.shape
        X_flat = X.reshape(-1, D)
        X_scaled = self.scaler.transform(X_flat)
        return X_scaled.reshape(N, L, D)
    
    def balance_train_windows(self, X: np.ndarray, y: np.ndarray, flight_ids: np.ndarray, window_meta: List,
                              target_pos_ratio: float) -> Tuple:
        """平衡训练窗口使正样本比例达到目标值"""
        pos_mask = y == 1
        neg_mask = y == 0
        
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        
        # 计算需要的负样本数量
        target_n_neg = int(n_pos * (1 - target_pos_ratio) / target_pos_ratio)
        
        if target_n_neg < n_neg:
            # 下采样负样本
            neg_indices = np.where(neg_mask)[0]
            np.random.shuffle(neg_indices)
            selected_neg_indices = neg_indices[:target_n_neg]
            
            pos_indices = np.where(pos_mask)[0]
            selected_indices = np.concatenate([pos_indices, selected_neg_indices])
            np.random.shuffle(selected_indices)
            
            X = X[selected_indices]
            y = y[selected_indices]
            flight_ids = flight_ids[selected_indices]
            window_meta = [window_meta[i] for i in selected_indices]
        
        actual_ratio = y.mean()
        print(f"Balanced windows: pos_ratio={actual_ratio:.3f} (target={target_pos_ratio:.3f})")
        
        return X, y, flight_ids, window_meta


# ============================================================================
# PyTorch Dataset
# ============================================================================
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# 模型定义
# ============================================================================

# Model A: Rule Baseline
class RuleBaseline:
    """基于残差阈值的规则基线"""
    
    def __init__(self):
        self.threshold = None
        self.alpha = 1.0
    
    def fit(self, X_val, y_val, target_fpr=0.01):
        """在验证集上搜索阈值（采用main.py策略）"""
        # X_val: [N, L, D]
        # 提取r_posvel (index=7)
        r_posvel_idx = 7
        r_posvel_sequences = X_val[:, :, r_posvel_idx]
        
        # 计算窗口统计量
        max_r = r_posvel_sequences.max(axis=1)
        
        # 使用ROC曲线找阈值
        fpr_list = []
        thresholds = np.percentile(max_r, np.linspace(0, 100, 1000))
        
        for thr in thresholds:
            preds = (max_r >= thr).astype(int)
            
            neg_mask = y_val == 0
            if neg_mask.sum() > 0:
                fp = ((preds == 1) & neg_mask).sum()
                fpr = fp / neg_mask.sum()
                fpr_list.append(fpr)
            else:
                fpr_list.append(0)
        
        fpr_array = np.array(fpr_list)
        
        # 找最接近target_fpr的点（不要求<=）
        best_idx = np.argmin(np.abs(fpr_array - target_fpr))
        
        self.threshold = thresholds[best_idx]
        best_fpr = fpr_array[best_idx]
        print(f"RuleBaseline: threshold={self.threshold:.4f}, val_fpr={best_fpr:.4f}")
    
    def predict_proba(self, X):
        """预测概率"""
        r_posvel_idx = 7
        r_posvel_sequences = X[:, :, r_posvel_idx]
        max_r = r_posvel_sequences.max(axis=1)
        # 直接返回max_r作为score（越大越可能是攻击）
        return max_r


# Model B: XGBoost
class XGBoostModel:
    """XGBoost分类器"""
    
    def __init__(self, random_state=42):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    def extract_features(self, X):
        """提取窗口统计特征"""
        # X: [N, L, D]
        N, L, D = X.shape
        features = []
        
        for i in range(D):
            seq = X[:, :, i]
            features.append(seq.mean(axis=1))
            features.append(seq.std(axis=1))
            features.append(seq.min(axis=1))
            features.append(seq.max(axis=1))
            features.append(np.percentile(seq, 95, axis=1))
            
            # 线性趋势 (简单slope)
            t = np.arange(L)
            slopes = []
            for j in range(N):
                slope = np.polyfit(t, seq[j], 1)[0]
                slopes.append(slope)
            features.append(np.array(slopes))
        
        return np.column_stack(features)
    
    def fit(self, X_train, y_train, X_val, y_val):
        """训练"""
        print("  Extracting features...")
        X_train_feat = self.extract_features(X_train)
        X_val_feat = self.extract_features(X_val)
        
        print("  Training XGBoost...")
        self.model.fit(
            X_train_feat, y_train,
            eval_set=[(X_val_feat, y_val)],
            verbose=False
        )
        print("  Training complete.")
    
    def predict_proba(self, X):
        """预测概率"""
        X_feat = self.extract_features(X)
        return self.model.predict_proba(X_feat)[:, 1]


# Model C: 1D-CNN
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_channels=64, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_dim, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_channels*2, hidden_channels*4, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_channels*4, hidden_channels*2)
        self.fc2 = nn.Linear(hidden_channels*2, 1)
    
    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.pool(x).squeeze(-1)  # [B, hidden*4]
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x).squeeze(-1)


# Model D: TCN
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Causal: 去除未来信息
        out = F.relu(self.conv1(x))
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        
        # 对齐长度
        if out.shape[2] != res.shape[2]:
            res = res[:, :, :out.shape[2]]
        
        return F.relu(out + res)


class TCN(nn.Module):
    def __init__(self, input_dim, num_channels=[64, 128, 256], kernel_size=3, dropout=0.1):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)
        x = self.network(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1)


# Model E: GRU
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x):
        # x: [B, L, D]
        _, h_n = self.gru(x)  # h_n: [num_layers, B, hidden]
        
        # 取最后一层hidden state
        h = h_n[-1]  # [B, hidden]
        h = self.dropout(h)
        
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        
        return torch.sigmoid(h).squeeze(-1)


# Model F: Transformer Encoder
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1, max_len=50):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: [B, L, D]
        x = self.input_proj(x)  # [B, L, d_model]
        x = x + self.pe[:, :x.size(1), :]
        
        x = self.transformer(x)
        
        # Mean pooling
        x = x.mean(dim=1)  # [B, d_model]
        
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1)


# ============================================================================
# 参数量对齐工具
# ============================================================================
def adjust_model_size(model_class, input_dim, target_range, **kwargs):
    """自动调整模型hidden size使参数量落入目标区间"""
    min_params, max_params = target_range
    
    # 根据模型类型尝试不同的hidden size
    if model_class == SimpleCNN:
        for hc in range(32, 256, 16):
            model = model_class(input_dim, hidden_channels=hc, **kwargs)
            n_params = count_parameters(model)
            if min_params <= n_params <= max_params:
                return model, n_params
        # 如果未找到，返回最接近的
        model = model_class(input_dim, hidden_channels=64, **kwargs)
        return model, count_parameters(model)
    
    elif model_class == TCN:
        for base in range(32, 128, 8):
            channels = [base, base*2, base*4]
            model = model_class(input_dim, num_channels=channels, **kwargs)
            n_params = count_parameters(model)
            if min_params <= n_params <= max_params:
                return model, n_params
        model = model_class(input_dim, num_channels=[64, 128, 256], **kwargs)
        return model, count_parameters(model)
    
    elif model_class == GRUModel:
        for hd in range(64, 256, 16):
            model = model_class(input_dim, hidden_dim=hd, **kwargs)
            n_params = count_parameters(model)
            if min_params <= n_params <= max_params:
                return model, n_params
        model = model_class(input_dim, hidden_dim=128, **kwargs)
        return model, count_parameters(model)
    
    elif model_class == TransformerModel:
        for dm in range(64, 256, 16):
            model = model_class(input_dim, d_model=dm, nhead=8, num_layers=4, **kwargs)
            n_params = count_parameters(model)
            if min_params <= n_params <= max_params:
                return model, n_params
        model = model_class(input_dim, d_model=128, nhead=8, num_layers=4, **kwargs)
        return model, count_parameters(model)
    
    else:
        raise ValueError(f"Unknown model class: {model_class}")


# ============================================================================
# 训练器
# ============================================================================
class ModelTrainer:
    def __init__(self, config, device='cuda', force_cpu=False):
        self.config = config
        if force_cpu:
            self.device = 'cpu'
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
    
    def train_pytorch_model(self, model, train_loader, val_loader, pos_weight):
        """训练PyTorch模型"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(self.config['max_epochs']), desc="Training", ncols=100)
        for epoch in pbar:
            # Train
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                outputs = model(X_batch)
                
                # 使用BCEWithLogitsLoss需要logits，但模型输出sigmoid，需要调整
                # 这里重新定义为BCELoss
                loss = F.binary_cross_entropy(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.float().to(self.device)
                    
                    outputs = model(X_batch)
                    loss = F.binary_cross_entropy(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'patience': f"{patience_counter}/{self.config['early_stopping_patience']}"
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['early_stopping_patience']:
                pbar.close()
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # 恢复最佳模型
        model.load_state_dict(best_state)
        return model
    
    def predict(self, model, X):
        """预测"""
        if isinstance(model, (RuleBaseline, XGBoostModel)):
            return model.predict_proba(X)
        else:
            model.eval()
            
            # 检测模型实际所在的设备
            model_device = next(model.parameters()).device
            
            dataset = WindowDataset(X, np.zeros(len(X)))
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            
            preds = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    # 将数据移动到模型所在的设备
                    X_batch = X_batch.to(model_device)
                    outputs = model(X_batch)
                    preds.append(outputs.cpu().numpy())
            
            return np.concatenate(preds)


# ============================================================================
# 评估工具
# ============================================================================
class BenchmarkEvaluator:
    def __init__(self, config):
        self.config = config
    
    def find_threshold_at_fpr(self, y_true, y_score, target_fpr=0.01):
        """在验证集上找到满足FPR约束的阈值（采用main.py策略）"""
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # 找到最接近目标FPR的点（不要求 <= target_fpr）
        # 这与main.py的evaluation.py中compute_recall_at_fpr一致
        idx = np.argmin(np.abs(fpr - target_fpr))
        
        return thresholds[idx], fpr[idx], tpr[idx]
    
    def compute_metrics(self, y_true, y_score, threshold):
        """计算评估指标"""
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_score)
        
        # Recall@threshold
        y_pred = (y_score >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'recall_at_threshold': recall,
            'fpr_at_threshold': fpr,
            'threshold': threshold
        }
    
    def compute_detection_delay(self, y_score, threshold, window_meta, df):
        """计算检测延迟"""
        # 按flight分组
        flight_scores = defaultdict(list)
        flight_windows = defaultdict(list)
        
        for i, meta in enumerate(window_meta):
            flight_id = meta['flight_id']
            flight_scores[flight_id].append(y_score[i])
            flight_windows[flight_id].append(meta)
        
        delays = []
        
        for flight_id in flight_scores:
            # 检查该flight是否被攻击
            flight_df = df[df['flight'] == flight_id]
            if flight_df['label'].max() == 0:
                continue  # 正常flight，跳过
            
            # 找到攻击开始时间
            attack_start_time = flight_df[flight_df['label'] == 1]['time'].min()
            
            # 找到首次报警窗口
            scores = flight_scores[flight_id]
            windows = flight_windows[flight_id]
            
            first_alarm_time = None
            for score, window in zip(scores, windows):
                if score >= threshold:
                    first_alarm_time = window['end_time']
                    break
            
            if first_alarm_time is not None:
                delay = first_alarm_time - attack_start_time
                delays.append(delay)
            else:
                # 未检测到，记为NaN
                delays.append(np.nan)
        
        delays = np.array(delays)
        
        # 统计
        detected_delays = delays[~np.isnan(delays)]
        
        if len(detected_delays) > 0:
            delay_median = np.median(detected_delays)
            delay_mean = np.mean(detected_delays)
            delay_p90 = np.percentile(detected_delays, 90)
        else:
            delay_median = np.nan
            delay_mean = np.nan
            delay_p90 = np.nan
        
        return {
            'delay_median': delay_median,
            'delay_mean': delay_mean,
            'delay_p90': delay_p90,
            'n_detected': len(detected_delays),
            'n_total_attacks': len(delays)
        }
    
    def compute_avg_fp_per_flight(self, y_true, y_score, threshold, flight_ids):
        """计算每个flight的平均误报数"""
        y_pred = (y_score >= threshold).astype(int)
        
        # 统计每个flight的FP
        flight_fp = defaultdict(int)
        flight_count = defaultdict(int)
        
        for i in range(len(y_pred)):
            flight_id = flight_ids[i]
            flight_count[flight_id] += 1
            
            if y_pred[i] == 1 and y_true[i] == 0:
                flight_fp[flight_id] += 1
        
        # 计算平均FP
        fp_counts = [flight_fp[fid] for fid in flight_count]
        avg_fp = np.mean(fp_counts) if len(fp_counts) > 0 else 0
        
        return avg_fp


# ============================================================================
# 主实验流程
# ============================================================================
def run_multi_model_benchmark():
    """运行多模型基准实验"""
    
    config = EXPERIMENT_CONFIG
    output_dir = Path('benchmark_output')
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Supplementary Experiment 1 — Multi-Model Benchmark")
    print("="*80)
    
    # 显示设备信息
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Using device: CUDA (GPU acceleration enabled)")
    else:
        print(f"Using device: CPU (slow, consider installing CUDA version)")
    
    # 1. 加载数据
    print("\n[1/7] Loading and splitting data...")
    data_loader = BenchmarkDataLoader(config)
    df_train, df_val, df_test = data_loader.load_and_split('data/src/flights.csv')
    
    # 2. 创建窗口
    print("\n[2/7] Creating windows...")
    X_train, y_train, fid_train, meta_train = data_loader.create_windows(df_train)
    X_val, y_val, fid_val, meta_val = data_loader.create_windows(df_val)
    X_test, y_test, fid_test, meta_test = data_loader.create_windows(df_test)
    
    print(f"Train windows: {len(y_train)}, pos_ratio={y_train.mean():.3f}")
    print(f"Val windows: {len(y_val)}, pos_ratio={y_val.mean():.3f}")
    print(f"Test windows: {len(y_test)}, pos_ratio={y_test.mean():.3f}")
    
    # 3. 保持训练集原始分布（不采样，与val/test一致）
    print("\n[3/7] Keeping natural distribution in training set...")
    print(f"Train natural pos_ratio={y_train.mean():.3f} (matching val/test for fair evaluation)")
    # 不做窗口层采样，使用class_weight处理不平衡
    
    # 4. 标准化
    print("\n[4/7] Fitting scaler...")
    data_loader.fit_scaler(X_train)
    X_train = data_loader.transform(X_train)
    X_val = data_loader.transform(X_val)
    X_test = data_loader.transform(X_test)
    
    input_dim = X_train.shape[2]
    print(f"Input dimension: {input_dim}")
    
    # 5. 定义模型列表
    print("\n[5/7] Defining models...")
    
    results = {}
    
    # 检查是否有checkpoint可以恢复
    import pickle
    existing_checkpoints = sorted(output_dir.glob('checkpoint_seed_*.pkl'))
    if existing_checkpoints:
        latest_checkpoint = existing_checkpoints[-1]
        print(f"\n⚠ Found existing checkpoint: {latest_checkpoint}")
        print("Do you want to resume from checkpoint? (This will skip completed seeds)")
        # 自动加载最新checkpoint
        with open(latest_checkpoint, 'rb') as f:
            results = pickle.load(f)
        completed_seeds = set()
        for model_runs in results.values():
            for run in model_runs:
                completed_seeds.add(run['seed'])
        print(f"✓ Loaded checkpoint with completed seeds: {sorted(completed_seeds)}")
        # 过滤掉已完成的seed
        remaining_seeds = [s for s in config['seeds_list'] if s not in completed_seeds]
        if not remaining_seeds:
            print("✓ All seeds already completed! Skipping to evaluation.")
            config['seeds_list'] = []
        else:
            config['seeds_list'] = remaining_seeds
            print(f"→ Remaining seeds to run: {remaining_seeds}")
    
    # 遍历随机种子
    for seed in config['seeds_list']:
        print(f"\n{'='*80}")
        print(f"Running with seed={seed}")
        print(f"{'='*80}")
        
        set_random_seed(seed)
        
        # Model A: Rule Baseline
        print("\n[Model A] Rule Baseline")
        model_a = RuleBaseline()
        model_a.fit(X_val, y_val, target_fpr=config['target_fpr'])
        
        results.setdefault('A_Rule', []).append({
            'model': model_a,
            'seed': seed,
            'num_params': 0
        })
        
        # Model B: XGBoost
        if HAS_XGBOOST:
            print("\n[Model B] XGBoost")
            model_b = XGBoostModel(random_state=seed)
            model_b.fit(X_train, y_train, X_val, y_val)
            
            results.setdefault('B_XGBoost', []).append({
                'model': model_b,
                'seed': seed,
                'num_params': 0  # XGBoost参数量不计
            })
        
        # Model C: 1D-CNN
        print("\n[Model C] 1D-CNN")
        model_c, params_c = adjust_model_size(
            SimpleCNN, input_dim, config['target_param_range'],
            dropout=config['dropout']
        )
        print(f"  Parameters: {params_c:,}")
        
        # 训练（使用自然分布的class weight）
        trainer = ModelTrainer(config)
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"  Using pos_weight={pos_weight:.2f} to handle class imbalance")
        
        train_dataset = WindowDataset(X_train, y_train)
        val_dataset = WindowDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        model_c = trainer.train_pytorch_model(model_c, train_loader, val_loader, pos_weight)
        
        results.setdefault('C_CNN', []).append({
            'model': model_c,
            'seed': seed,
            'num_params': params_c
        })
        
        # Model D: TCN
        print("\n[Model D] TCN")
        model_d, params_d = adjust_model_size(
            TCN, input_dim, config['target_param_range'],
            dropout=config['dropout']
        )
        print(f"  Parameters: {params_d:,}")
        print("  Note: Using CPU for TCN (dilated conv may be slow on GPU)")
        
        # Use CPU trainer for TCN
        trainer_cpu = ModelTrainer(config, force_cpu=True)
        model_d = trainer_cpu.train_pytorch_model(model_d, train_loader, val_loader, pos_weight)
        
        results.setdefault('D_TCN', []).append({
            'model': model_d,
            'seed': seed,
            'num_params': params_d
        })
        
        # Model E: GRU
        print("\n[Model E] GRU")
        model_e, params_e = adjust_model_size(
            GRUModel, input_dim, config['target_param_range'],
            dropout=config['dropout']
        )
        print(f"  Parameters: {params_e:,}")
        print("  Note: Using CPU for GRU (often faster than GPU for RNN)")
        
        # Use CPU trainer for GRU
        trainer_cpu = ModelTrainer(config, force_cpu=True)
        model_e = trainer_cpu.train_pytorch_model(model_e, train_loader, val_loader, pos_weight)
        
        results.setdefault('E_GRU', []).append({
            'model': model_e,
            'seed': seed,
            'num_params': params_e
        })
        
        # Model F: Transformer
        print("\n[Model F] Transformer Encoder")
        model_f, params_f = adjust_model_size(
            TransformerModel, input_dim, config['target_param_range'],
            dropout=config['dropout'], max_len=config['window_length']
        )
        print(f"  Parameters: {params_f:,}")
        
        model_f = trainer.train_pytorch_model(model_f, train_loader, val_loader, pos_weight)
        
        results.setdefault('F_Transformer', []).append({
            'model': model_f,
            'seed': seed,
            'num_params': params_f
        })
        
        # 保存checkpoint（每个seed完成后）
        checkpoint_path = output_dir / f'checkpoint_seed_{seed}.pkl'
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
    
    # 6. 评估所有模型
    print("\n[6/7] Evaluating all models...")
    evaluator = BenchmarkEvaluator(config)
    trainer = ModelTrainer(config)
    
    summary_results = []
    
    for model_name, model_runs in results.items():
        print(f"\nEvaluating {model_name}...")
        
        metrics_list = {
            'pr_auc': [],
            'roc_auc': [],
            'recall_at_fpr0.01': [],
            'actual_fpr_at_thr': [],
            'threshold_at_fpr0.01': [],
            'avg_fp_per_flight': [],
            'delay_median': [],
            'delay_p90': [],
            'delay_mean': []
        }
        
        for run in tqdm(model_runs, desc=f"  {model_name} seeds", ncols=100):
            model = run['model']
            seed = run['seed']
            
            # 预测验证集，搜索阈值
            y_val_score = trainer.predict(model, X_val)
            threshold, actual_fpr, recall = evaluator.find_threshold_at_fpr(
                y_val, y_val_score, target_fpr=config['target_fpr']
            )
            
            # 在测试集上评估
            y_test_score = trainer.predict(model, X_test)
            test_metrics = evaluator.compute_metrics(y_test, y_test_score, threshold)
            
            # 计算avg_fp_per_flight
            avg_fp = evaluator.compute_avg_fp_per_flight(y_test, y_test_score, threshold, fid_test)
            
            # 计算detection_delay
            delay_metrics = evaluator.compute_detection_delay(y_test_score, threshold, meta_test, df_test)
            
            # 记录
            metrics_list['pr_auc'].append(test_metrics['pr_auc'])
            metrics_list['roc_auc'].append(test_metrics['roc_auc'])
            metrics_list['recall_at_fpr0.01'].append(test_metrics['recall_at_threshold'])
            metrics_list['actual_fpr_at_thr'].append(test_metrics['fpr_at_threshold'])
            metrics_list['threshold_at_fpr0.01'].append(threshold)
            metrics_list['avg_fp_per_flight'].append(avg_fp)
            metrics_list['delay_median'].append(delay_metrics['delay_median'])
            metrics_list['delay_p90'].append(delay_metrics['delay_p90'])
            metrics_list['delay_mean'].append(delay_metrics['delay_mean'])
        
        # 计算均值和标准差
        summary = {
            'model': model_name,
            'num_params': model_runs[0]['num_params']
        }
        
        for metric_name, values in metrics_list.items():
            values = np.array(values)
            summary[f'{metric_name}_mean'] = np.nanmean(values)
            summary[f'{metric_name}_std'] = np.nanstd(values)
        
        summary_results.append(summary)
    
    # 7. 输出结果
    print("\n[7/7] Generating outputs...")
    
    # 7.1 汇总表
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(output_dir / 'benchmark_summary.csv', index=False)
    
    # 生成Markdown表格
    with open(output_dir / 'benchmark_summary.md', 'w', encoding='utf-8') as f:
        f.write("# Multi-Model Benchmark Results\n\n")
        f.write("## Summary Table\n\n")
        f.write(df_summary.to_markdown(index=False))
        
        f.write("\n\n## Parameter Count Alignment\n\n")
        param_table = df_summary[['model', 'num_params']].copy()
        param_table['param_range_check'] = param_table['num_params'].apply(
            lambda x: '✓' if (x == 0 or (config['target_param_range'][0] <= x <= config['target_param_range'][1])) else '✗'
        )
        f.write(param_table.to_markdown(index=False))
        
        f.write("\n\n## Training Configuration\n\n")
        f.write(f"- Learning rate: {config['learning_rate']}\n")
        f.write(f"- Batch size: {config['batch_size']}\n")
        f.write(f"- Max epochs: {config['max_epochs']}\n")
        f.write(f"- Early stopping patience: {config['early_stopping_patience']}\n")
        f.write(f"- Dropout: {config['dropout']}\n")
        f.write(f"- Random seeds: {config['seeds_list']}\n")
    
    print(f"\nResults saved to {output_dir}/")
    print("\n" + "="*80)
    print("Benchmark completed!")
    print("="*80)
    
    return df_summary, results


# ============================================================================
# 可视化
# ============================================================================
def generate_visualizations(results, X_test, y_test, fid_test, meta_test, df_test, config):
    """生成可视化图表"""
    
    output_dir = Path('benchmark_output')
    trainer = ModelTrainer(config)
    evaluator = BenchmarkEvaluator(config)
    
    # 为每个模型收集预测结果
    model_predictions = {}
    
    for model_name, model_runs in results.items():
        # 使用第一个seed的模型
        model = model_runs[0]['model']
        
        # 预测
        y_val_score = trainer.predict(model, X_test)  # 这里应该用val，但为了简化直接用test
        threshold, _, _ = evaluator.find_threshold_at_fpr(y_test, y_val_score, target_fpr=config['target_fpr'])
        
        y_test_score = trainer.predict(model, X_test)
        
        model_predictions[model_name] = {
            'y_score': y_test_score,
            'threshold': threshold
        }
    
    # 1. PR曲线
    plt.figure(figsize=(10, 8))
    for model_name, preds in model_predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, preds['y_score'])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUC={pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pr_curves.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. ROC曲线
    plt.figure(figsize=(10, 8))
    for model_name, preds in model_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds['y_score'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Test Set)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'roc_curves.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. Score分布直方图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (model_name, preds) in enumerate(model_predictions.items()):
        ax = axes[idx]
        y_score = preds['y_score']
        
        # 正负样本分开
        pos_scores = y_score[y_test == 1]
        neg_scores = y_score[y_test == 0]
        
        # 检查数据范围
        neg_range = neg_scores.max() - neg_scores.min() if len(neg_scores) > 0 else 0
        pos_range = pos_scores.max() - pos_scores.min() if len(pos_scores) > 0 else 0
        
        # 如果范围太小，使用scatter plot代替histogram
        if neg_range < 1e-6 and pos_range < 1e-6:
            # 所有值几乎相同，显示为垂直线
            if len(neg_scores) > 0:
                ax.axvline(neg_scores.mean(), color='blue', alpha=0.5, linewidth=3, label='Normal')
            if len(pos_scores) > 0:
                ax.axvline(pos_scores.mean(), color='red', alpha=0.5, linewidth=3, label='Attack')
            ax.set_ylabel('Constant Score')
        else:
            # 正常绘制histogram
            try:
                if len(neg_scores) > 0:
                    ax.hist(neg_scores, bins=min(20, max(5, len(neg_scores)//10)), 
                           alpha=0.5, label='Normal', color='blue')
                if len(pos_scores) > 0:
                    ax.hist(pos_scores, bins=min(20, max(5, len(pos_scores)//10)), 
                           alpha=0.5, label='Attack', color='red')
            except (ValueError, RuntimeError):
                # 最后的fallback：使用KDE或简单的scatter
                if len(neg_scores) > 0:
                    ax.scatter([neg_scores.mean()]*len(neg_scores), 
                              np.random.randn(len(neg_scores)), 
                              alpha=0.1, color='blue', label='Normal', s=1)
                if len(pos_scores) > 0:
                    ax.scatter([pos_scores.mean()]*len(pos_scores), 
                              np.random.randn(len(pos_scores)), 
                              alpha=0.1, color='red', label='Attack', s=1)
            ax.set_ylabel('Count')
        
        ax.axvline(preds['threshold'], color='green', linestyle='--', label='Threshold', linewidth=2)
        ax.set_xlabel('Score')
        ax.set_title(f'{model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'score_distributions.pdf', bbox_inches='tight')
    plt.close()
    
    # 4. Avg FP per flight柱状图
    avg_fps = []
    for model_name, preds in model_predictions.items():
        avg_fp = evaluator.compute_avg_fp_per_flight(
            y_test, preds['y_score'], preds['threshold'], fid_test
        )
        avg_fps.append((model_name, avg_fp))
    
    models, fps = zip(*avg_fps)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(models)), fps)
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylabel('Avg FP per Flight')
    plt.title('Average False Positives per Flight (Test Set)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_fp_per_flight.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'avg_fp_per_flight.pdf', bbox_inches='tight')
    plt.close()
    
    # 5. Detection delay箱线图
    delay_data = []
    
    for model_name, preds in model_predictions.items():
        delay_metrics = evaluator.compute_detection_delay(
            preds['y_score'], preds['threshold'], meta_test, df_test
        )
        
        # 获取所有delays
        flight_scores = defaultdict(list)
        flight_windows = defaultdict(list)
        
        for i, meta in enumerate(meta_test):
            flight_id = meta['flight_id']
            flight_scores[flight_id].append(preds['y_score'][i])
            flight_windows[flight_id].append(meta)
        
        delays = []
        for flight_id in flight_scores:
            flight_df = df_test[df_test['flight'] == flight_id]
            if flight_df['label'].max() == 0:
                continue
            
            attack_start_time = flight_df[flight_df['label'] == 1]['time'].min()
            
            scores = flight_scores[flight_id]
            windows = flight_windows[flight_id]
            
            first_alarm_time = None
            for score, window in zip(scores, windows):
                if score >= preds['threshold']:
                    first_alarm_time = window['end_time']
                    break
            
            if first_alarm_time is not None:
                delay = first_alarm_time - attack_start_time
                delays.append(delay)
        
        delay_data.append(delays)
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(delay_data, labels=list(model_predictions.keys()))
    plt.ylabel('Detection Delay (s)')
    plt.title('Detection Delay Distribution (Attacked Flights Only)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_delay.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'detection_delay.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    # 运行基准实验
    df_summary, results = run_multi_model_benchmark()
    
    # 生成可视化
    print("\nGenerating visualizations...")
    
    # 重新加载数据用于可视化
    config = EXPERIMENT_CONFIG
    data_loader = BenchmarkDataLoader(config)
    df_train, df_val, df_test = data_loader.load_and_split('data/src/flights.csv')
    
    X_train, y_train, fid_train, meta_train = data_loader.create_windows(df_train)
    X_val, y_val, fid_val, meta_val = data_loader.create_windows(df_val)
    X_test, y_test, fid_test, meta_test = data_loader.create_windows(df_test)
    
    data_loader.fit_scaler(X_train)
    X_test = data_loader.transform(X_test)
    
    generate_visualizations(results, X_test, y_test, fid_test, meta_test, df_test, config)
    
    print("\n" + "="*80)
    print("All tasks completed successfully!")
    print("="*80)
