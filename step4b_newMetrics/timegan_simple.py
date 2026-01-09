"""
TimeGAN实现框架 - 用于GPS欺骗攻击窗口生成
简化版，适合快速实验（2-5分钟训练）
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class TimeGAN:
    """
    TimeGAN for GPS Spoofing Attack Synthesis
    
    简化版实现，针对50个时间步的窗口优化
    """
    
    def __init__(self, seq_len=50, feature_dim=6, hidden_dim=24, 
                 num_layers=3, device='cpu'):
        """
        Args:
            seq_len: 序列长度（窗口大小）
            feature_dim: 特征维度数
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
            device: 'cpu' or 'cuda'
        """
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # 1. Embedder: X -> H
        self.embedder = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(device)
        
        # 2. Recovery: H -> X
        self.recovery = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(device)
        self.recovery_linear = nn.Linear(hidden_dim, feature_dim).to(device)
        
        # 3. Generator: Z -> H
        self.generator = nn.GRU(
            input_size=hidden_dim,  # 噪声维度
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(device)
        
        # 4. Supervisor: H -> H (下一时刻)
        self.supervisor = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(device)
        
        # 5. Discriminator: H -> [0,1]
        self.discriminator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        ).to(device)
        self.discriminator_linear = nn.Linear(hidden_dim, 1).to(device)
        
    def train_autoencoder(self, real_data, epochs=100, batch_size=128, lr=0.001):
        """
        阶段1: 训练Embedder和Recovery（自编码器）
        
        Args:
            real_data: (n_samples, seq_len, feature_dim)
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
        """
        print("\n[阶段1] 训练Embedder和Recovery (自编码器)...")
        
        dataset = TensorDataset(torch.FloatTensor(real_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) + 
            list(self.recovery.parameters()) + 
            list(self.recovery_linear.parameters()),
            lr=lr
        )
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                X = batch[0].to(self.device)
                
                # Embed
                H, _ = self.embedder(X)
                
                # Recover
                X_hat, _ = self.recovery(H)
                X_hat = self.recovery_linear(X_hat)
                
                # 重构损失
                loss = criterion(X_hat, X)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    def train_supervisor(self, real_data, epochs=100, batch_size=128, lr=0.001):
        """
        阶段2: 训练Supervisor（学习时序动态）
        
        Args:
            real_data: (n_samples, seq_len, feature_dim)
        """
        print("\n[阶段2] 训练Supervisor (时序结构)...")
        
        dataset = TensorDataset(torch.FloatTensor(real_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(self.supervisor.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                X = batch[0].to(self.device)
                
                # Embed
                with torch.no_grad():
                    H, _ = self.embedder(X)
                
                # Supervisor预测下一时刻
                H_supervised, _ = self.supervisor(H[:, :-1, :])
                
                # 监督损失
                loss = criterion(H_supervised, H[:, 1:, :])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    def train_gan(self, real_data, epochs=200, batch_size=128, lr=0.001):
        """
        阶段3: 联合训练Generator和Discriminator
        
        Args:
            real_data: (n_samples, seq_len, feature_dim)
        """
        print("\n[阶段3] 联合训练Generator和Discriminator (GAN)...")
        
        dataset = TensorDataset(torch.FloatTensor(real_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器
        optimizer_G = torch.optim.Adam(
            list(self.generator.parameters()) + 
            list(self.supervisor.parameters()) +
            list(self.recovery.parameters()) +
            list(self.recovery_linear.parameters()),
            lr=lr
        )
        optimizer_D = torch.optim.Adam(
            list(self.discriminator.parameters()) + 
            list(self.discriminator_linear.parameters()),
            lr=lr
        )
        
        criterion = nn.BCEWithLogitsLoss()
        mse_criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_d_loss = 0
            total_g_loss = 0
            
            for batch in dataloader:
                X = batch[0].to(self.device)
                batch_size = X.size(0)
                
                # ===== 训练Discriminator =====
                optimizer_D.zero_grad()
                
                # 真实样本
                H_real, _ = self.embedder(X)
                D_real, _ = self.discriminator(H_real)
                D_real = self.discriminator_linear(D_real).squeeze()
                
                # 生成样本
                Z = torch.randn(batch_size, self.seq_len, self.hidden_dim).to(self.device)
                H_fake, _ = self.generator(Z)
                H_fake_supervised, _ = self.supervisor(H_fake)
                D_fake, _ = self.discriminator(H_fake_supervised.detach())
                D_fake = self.discriminator_linear(D_fake).squeeze()
                
                # 判别损失
                d_loss_real = criterion(D_real, torch.ones_like(D_real))
                d_loss_fake = criterion(D_fake, torch.zeros_like(D_fake))
                d_loss = d_loss_real + d_loss_fake
                
                d_loss.backward()
                optimizer_D.step()
                
                # ===== 训练Generator =====
                optimizer_G.zero_grad()
                
                # 生成样本
                Z = torch.randn(batch_size, self.seq_len, self.hidden_dim).to(self.device)
                H_fake, _ = self.generator(Z)
                H_fake_supervised, _ = self.supervisor(H_fake)
                
                # 判别器分数
                D_fake, _ = self.discriminator(H_fake_supervised)
                D_fake = self.discriminator_linear(D_fake).squeeze()
                
                # Recovery损失
                X_fake, _ = self.recovery(H_fake_supervised)
                X_fake = self.recovery_linear(X_fake)
                
                # 生成器损失
                g_loss_adv = criterion(D_fake, torch.ones_like(D_fake))  # 骗过判别器
                g_loss_supervised = mse_criterion(H_fake[:, 1:, :], H_fake_supervised[:, :-1, :])  # 时序一致性
                g_loss = g_loss_adv + 100 * g_loss_supervised
                
                g_loss.backward()
                optimizer_G.step()
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
            
            if (epoch + 1) % 40 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, D_loss: {total_d_loss/len(dataloader):.4f}, "
                      f"G_loss: {total_g_loss/len(dataloader):.4f}")
    
    def generate(self, n_samples):
        """
        生成合成样本
        
        Args:
            n_samples: 生成样本数量
        
        Returns:
            生成的样本 (n_samples, seq_len, feature_dim)
        """
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            # 随机噪声
            Z = torch.randn(n_samples, self.seq_len, self.hidden_dim).to(self.device)
            
            # 生成隐藏表示
            H, _ = self.generator(Z)
            H, _ = self.supervisor(H)
            
            # 恢复到原始空间
            X, _ = self.recovery(H)
            X = self.recovery_linear(X)
        
        return X.cpu().numpy()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'embedder': self.embedder.state_dict(),
            'recovery': self.recovery.state_dict(),
            'recovery_linear': self.recovery_linear.state_dict(),
            'generator': self.generator.state_dict(),
            'supervisor': self.supervisor.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'discriminator_linear': self.discriminator_linear.state_dict(),
        }, path)
        print(f"✓ 模型已保存: {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.recovery.load_state_dict(checkpoint['recovery'])
        self.recovery_linear.load_state_dict(checkpoint['recovery_linear'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.supervisor.load_state_dict(checkpoint['supervisor'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.discriminator_linear.load_state_dict(checkpoint['discriminator_linear'])
        print(f"✓ 模型已加载: {path}")


def train_timegan_for_attack_type(attack_windows, attack_type, save_path='timegan_models'):
    """
    为单个攻击类型训练TimeGAN
    
    Args:
        attack_windows: (n_samples, 50, 6) 攻击窗口数据
        attack_type: 攻击类型名称
        save_path: 模型保存路径
    
    Returns:
        trained TimeGAN model
    """
    print("=" * 70)
    print(f"训练TimeGAN - 攻击类型: {attack_type}")
    print("=" * 70)
    print(f"训练样本数: {len(attack_windows)}")
    print(f"窗口大小: {attack_windows.shape[1]}")
    print(f"特征数: {attack_windows.shape[2]}")
    
    # 归一化
    mean = attack_windows.mean(axis=(0, 1))
    std = attack_windows.std(axis=(0, 1)) + 1e-6
    attack_windows_norm = (attack_windows - mean) / std
    
    # 初始化TimeGAN
    timegan = TimeGAN(
        seq_len=attack_windows.shape[1],
        feature_dim=attack_windows.shape[2],
        hidden_dim=24,
        num_layers=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 三阶段训练
    timegan.train_autoencoder(attack_windows_norm, epochs=100, batch_size=128)
    timegan.train_supervisor(attack_windows_norm, epochs=100, batch_size=128)
    timegan.train_gan(attack_windows_norm, epochs=200, batch_size=128)
    
    # 保存模型和统计信息
    import os
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, f'timegan_{attack_type}.pt')
    timegan.save(model_file)
    
    # 保存归一化参数
    np.savez(os.path.join(save_path, f'norm_params_{attack_type}.npz'),
             mean=mean, std=std)
    
    return timegan, mean, std


def generate_synthetic_attacks(timegan, n_samples, mean, std):
    """
    使用训练好的TimeGAN生成合成攻击样本
    
    Args:
        timegan: 训练好的TimeGAN模型
        n_samples: 生成数量
        mean, std: 归一化参数
    
    Returns:
        生成的攻击窗口 (n_samples, 50, 6)
    """
    # 生成
    synthetic_norm = timegan.generate(n_samples)
    
    # 反归一化
    synthetic = synthetic_norm * std + mean
    
    return synthetic


# 使用示例
if __name__ == "__main__":
    import os
    os.makedirs('timegan_models', exist_ok=True)
    
    # 模拟攻击窗口数据
    np.random.seed(42)
    attack_type = 'step'
    n_samples = 2000
    seq_len = 50
    feature_dim = 6
    
    # 模拟step攻击特征（阶跃变化）
    attack_windows = np.random.randn(n_samples, seq_len, feature_dim)
    for i in range(n_samples):
        step_point = np.random.randint(20, 40)
        attack_windows[i, step_point:, :] += np.random.randn(1, feature_dim) * 2
    
    print("=" * 70)
    print("TimeGAN训练示例")
    print("=" * 70)
    print(f"攻击类型: {attack_type}")
    print(f"训练样本: {n_samples}")
    print(f"窗口大小: {seq_len}")
    print(f"特征维度: {feature_dim}")
    
    # 训练
    timegan, mean, std = train_timegan_for_attack_type(
        attack_windows, attack_type, save_path='timegan_models'
    )
    
    # 生成
    print("\n生成合成样本...")
    synthetic_attacks = generate_synthetic_attacks(timegan, n_samples=500, mean=mean, std=std)
    print(f"✓ 已生成 {len(synthetic_attacks)} 个合成攻击窗口")
    
    print("\n✓ 完成！模型已保存到 timegan_models/")
