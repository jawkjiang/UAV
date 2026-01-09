"""
Training Module

Train GPS spoofing detection model with weighted loss.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc
import config


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance.
    """
    
    def __init__(self, pos_weight=1.0):
        """
        Initialize loss.
        
        Args:
            pos_weight: Weight for positive class
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, outputs, targets):
        """
        Compute weighted BCE loss.
        
        Args:
            outputs: Model predictions [batch_size, 1]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Loss value
        """
        outputs = outputs.squeeze()
        
        # BCE with logit is more stable, but we use sigmoid in model
        # So we use standard BCE
        bce = -(targets * torch.log(outputs + 1e-7) + 
                (1 - targets) * torch.log(1 - outputs + 1e-7))
        
        # Apply weight to positive samples
        weights = torch.where(targets == 1, 
                             torch.tensor(self.pos_weight, device=targets.device),
                             torch.tensor(1.0, device=targets.device))
        
        weighted_bce = bce * weights
        
        return weighted_bce.mean()


def train_epoch(model, train_loader, criterion, optimizer, device, debug=False):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        debug: Whether to print debug info
    
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Debug: 检查第一个batch的输入数据
        if debug and batch_idx == 0:
            print(f"\n  [DEBUG] 第一个batch:")
            print(f"    输入shape: {data.shape}")
            print(f"    输入范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"    输入均值/std: {data.mean():.4f} / {data.std():.4f}")
            print(f"    是否有NaN: {torch.isnan(data).any().item()}")
            print(f"    是否有Inf: {torch.isinf(data).any().item()}")
            print(f"    标签: {target[:10].cpu().numpy()}")
        
        optimizer.zero_grad()
        
        output = model(data)
        
        # Debug: 检查模型输出
        if debug and batch_idx == 0:
            print(f"\n  [DEBUG] 模型输出:")
            print(f"    输出shape: {output.shape}")
            print(f"    输出范围: [{output.min():.4f}, {output.max():.4f}]")
            print(f"    输出均值/std: {output.mean():.4f} / {output.std():.4f}")
            print(f"    是否有NaN: {torch.isnan(output).any().item()}")
            print(f"    是否有Inf: {torch.isinf(output).any().item()}")
            print(f"    前10个输出: {output[:10].squeeze().detach().cpu().numpy()}")
        
        loss = criterion(output, target)
        
        # Debug: 检查损失
        if debug and batch_idx == 0:
            print(f"\n  [DEBUG] 损失:")
            print(f"    损失值: {loss.item():.4f}")
            print(f"    是否有NaN: {torch.isnan(loss).item()}")
        
        loss.backward()
        
        # Debug: 检查梯度
        if debug and batch_idx == 0:
            print(f"\n  [DEBUG] 梯度检查:")
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"    ⚠️  {name}: 梯度有NaN/Inf!")
            total_norm = total_norm ** 0.5
            print(f"    梯度总范数: {total_norm:.4f}")
            if total_norm > 100:
                print(f"    ⚠️  梯度可能爆炸! (norm={total_norm:.2f})")
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 如果出现NaN，立即停止
        if torch.isnan(loss).item():
            print(f"\n❌ 错误: 第{batch_idx}个batch出现NaN损失，停止训练")
            break
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device, debug=False):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to use
        debug: Whether to print debug info
    
    Returns:
        Tuple of (avg_loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
            # Debug: 检查第一个batch
            if debug and batch_idx == 0:
                print(f"\n  [DEBUG] 验证集第一个batch:")
                print(f"    输出范围: [{output.min():.4f}, {output.max():.4f}]")
                print(f"    是否有NaN: {torch.isnan(output).any().item()}")
                print(f"    是否有Inf: {torch.isinf(output).any().item()}")
    
    avg_loss = total_loss / len(data_loader)
    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Debug: 检查最终预测
    if debug:
        print(f"\n  [DEBUG] 所有预测汇总:")
        print(f"    预测数量: {len(predictions)}")
        print(f"    预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"    是否有NaN: {np.isnan(predictions).any()}")
        print(f"    是否有Inf: {np.isinf(predictions).any()}")
        if np.isnan(predictions).any():
            nan_count = np.isnan(predictions).sum()
            print(f"    ⚠️  NaN数量: {nan_count} / {len(predictions)}")
    
    return avg_loss, predictions, targets


def compute_pr_auc(targets, predictions):
    """
    Compute Precision-Recall AUC.
    
    Args:
        targets: Ground truth labels
        predictions: Predicted probabilities
    
    Returns:
        PR-AUC score
    """
    precision, recall, _ = precision_recall_curve(targets, predictions)
    pr_auc = auc(recall, precision)
    return pr_auc


def train_model(model, train_loader, val_loader, 
                pos_weight=1.0,
                num_epochs=config.NUM_EPOCHS,
                learning_rate=config.LEARNING_RATE,
                patience=config.EARLY_STOPPING_PATIENCE,
                device='cuda',
                save_path=None):
    """
    Train model with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        pos_weight: Weight for positive class
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        device: Device to use
        save_path: Path to save best model
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pr_auc': []
    }
    
    best_val_pr_auc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Training (debug第一个epoch)
        debug_mode = (epoch == 0)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, debug=debug_mode)
        print(f"  训练损失: {train_loss:.4f}")
        
        # Validation
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device, debug=debug_mode)
        print(f"  验证损失: {val_loss:.4f}")
        
        # 在计算PR-AUC前检查数据
        if debug_mode:
            print(f"\n  [DEBUG] 计算PR-AUC前检查:")
            print(f"    val_targets: {len(val_targets)}, 范围[{val_targets.min():.4f}, {val_targets.max():.4f}]")
            print(f"    val_preds: {len(val_preds)}, 范围[{val_preds.min():.4f}, {val_preds.max():.4f}]")
            print(f"    val_preds是否有NaN: {np.isnan(val_preds).any()}")
            print(f"    val_preds是否有Inf: {np.isinf(val_preds).any()}")
            
            if np.isnan(val_preds).any() or np.isinf(val_preds).any():
                print(f"\n❌ 发现NaN/Inf，停止训练")
                break
        
        val_pr_auc = compute_pr_auc(val_targets, val_preds)
        print(f"  验证PR-AUC: {val_pr_auc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pr_auc'].append(val_pr_auc)
        
        # Learning rate scheduling
        scheduler.step(val_pr_auc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val PR-AUC={val_pr_auc:.4f}")
        
        # Early stopping
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            
            # Save best model
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_pr_auc': val_pr_auc,
                }, save_path)
                print(f"  Saved best model (PR-AUC={val_pr_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\nTraining completed. Best Val PR-AUC: {best_val_pr_auc:.4f}")
    
    return history


if __name__ == '__main__':
    import pickle
    import json
    from model import GPSSpoofingDetector
    from window_creation import WindowDataset
    
    print("="*80)
    print("Model Training")
    print("="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load windows data
    print("\n[1] Loading windows data...")
    with open(os.path.join(config.OUTPUT_DIR, 'windows_data.pkl'), 'rb') as f:
        windows_data = pickle.load(f)
    
    train_windows = windows_data['train']['windows']
    train_labels = windows_data['train']['labels']
    val_windows = windows_data['val']['windows']
    val_labels = windows_data['val']['labels']
    
    print(f"    Train: {len(train_labels)} windows")
    print(f"    Val:   {len(val_labels)} windows")
    
    # Create datasets
    train_dataset = WindowDataset(train_windows, train_labels)
    val_dataset = WindowDataset(val_windows, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Compute positive weight
    pos_weight = train_dataset.get_positive_weight()
    print(f"\n[2] Positive class weight: {pos_weight:.2f}")
    
    # Create model
    print("\n[3] Creating model...")
    n_features = train_windows.shape[2]
    model = GPSSpoofingDetector(n_features=n_features)
    model = model.to(device)
    
    print(f"    Model created with n_features={n_features}")
    print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n[4] Training model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        pos_weight=pos_weight,
        patience=config.EARLY_STOPPING_PATIENCE,
        save_path=os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    )
    
    # Save training history
    print("\n[5] Saving training history...")
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"    ✓ Saved to {config.OUTPUT_DIR}/training_history.json")
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
