"""
TimeGAN模型 - 针对UAV飞行轨迹优化
基于step4b的实现，专门用于正常飞行数据生成
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class TimeGAN:
    """
    TimeGAN for Normal Flight Synthesis
    学习正常飞行的时序动态，生成合成正常飞行轨迹
    """
    
    def __init__(self, seq_len, feature_dim, hidden_dim=32, 
                 num_layers=3, device='cpu'):
        """
        Args:
            seq_len: 序列长度
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
            input_size=hidden_dim,
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
        """阶段1: 训练Embedder和Recovery（自编码器）"""
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
        """阶段2: 训练Supervisor（学习时序动态）"""
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
        """阶段3: 联合训练Generator和Discriminator"""
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
                
                # 生成器损失
                g_loss_adv = criterion(D_fake, torch.ones_like(D_fake))
                g_loss_supervised = mse_criterion(H_fake[:, 1:, :], H_fake_supervised[:, :-1, :])
                g_loss = g_loss_adv + 100 * g_loss_supervised
                
                g_loss.backward()
                optimizer_G.step()
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
            
            if (epoch + 1) % 40 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, D_loss: {total_d_loss/len(dataloader):.4f}, "
                      f"G_loss: {total_g_loss/len(dataloader):.4f}")
    
    def generate(self, n_samples):
        """生成合成样本"""
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
        checkpoint = torch.load(path, map_location=self.device)
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.recovery.load_state_dict(checkpoint['recovery'])
        self.recovery_linear.load_state_dict(checkpoint['recovery_linear'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.supervisor.load_state_dict(checkpoint['supervisor'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.discriminator_linear.load_state_dict(checkpoint['discriminator_linear'])
        print(f"✓ 模型已加载: {path}")
